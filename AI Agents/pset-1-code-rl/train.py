import asyncio
import datetime
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import chz
import datasets
import numpy as np
import tinker
import torch
from transformers import AutoTokenizer

from tinker_utils.checkpoint import get_last_checkpoint, save_checkpoint
from tinker_utils.cli import LogdirBehavior, check_log_dir
from tinker_utils.data import build_question
from tinker_utils.env import CodeEnv, StopCondition
from tinker_utils.log import setup_logging
from tinker_utils.renderers import get_renderer

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None  # kept for compatibility

    # Logging
    log_path: str = os.path.join(
        os.path.expanduser("~/code-rl-logs"),
        datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
    )
    log_dir_behavior: LogdirBehavior = "ask"
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Model
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    max_tokens: int = 24576

    # Optimization
    batch_size: int = 128
    group_size: int = 8
    learning_rate: float = 4e-5
    max_steps: int = -1  # -1 = unlimited

    # Reward / env
    format_coef: float = 0.1
    reward_timeout: int = 6

    # Sampling
    temperature: float = 1.0

    # Logging cadence
    save_every: int = 10  # 0 = disabled
    eval_every: int = 10  # -1 = disabled


def should_skip(advantages: list[float]) -> bool:
    """Skip if all advantages are effectively zero (all rewards equal)."""
    return all(abs(a) < 1e-8 for a in advantages)


def compute_advantages(rewards: list[float]) -> list[float]:
    """Group-relative advantages A_i = R_i - mean(R)."""
    if not rewards:
        return []
    mean_reward = float(np.mean(rewards))
    return [r - mean_reward for r in rewards]


def make_datum(
    tokens: list[int],
    logprobs: list[float],
    ob_len: int,
    advantage: float,
) -> tinker.types.Datum:
    """Construct a Tinker training datum."""
    return tinker.types.Datum(
        tokens=tokens,
        logprobs=logprobs,
        observation_len=ob_len,
        advantage=advantage,
    )


def train_step(
    training_client: tinker.TrainingClient,
    datums: list[tinker.types.Datum],
    adam_params: tinker.types.AdamParams,
) -> None:
    """Perform a single GRPO update on collected datums."""
    if not datums:
        return
    training_client.train(
        data=datums,
        adam_params=adam_params,
    ).result()


def sample_completions(
    sampler: tinker.SamplingClient,
    prompt: tinker.ModelInput,
    group_size: int,
    temperature: float,
    stop_condition: StopCondition,
) -> list[tuple[list[int], list[float]]]:
    """Sample multiple completions (with logprobs) for one prompt (sync API)."""
    sampling_params = tinker.types.SamplingParams(
        max_tokens=2048,
        temperature=temperature,
        stop_sequences=stop_condition,
        include_logprobs=True,
    )

    futures = [
        sampler.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=sampling_params,
        )
        for _ in range(group_size)
    ]

    results = [f.result() for f in futures]

    completions: list[tuple[list[int], list[float]]] = []
    for result in results:
        if not result.sequences:
            continue
        seq = result.sequences[0]
        tokens = list(seq.tokens)
        logprobs = list(seq.logprobs or [])
        if not tokens or not logprobs:
            continue
        logprobs = logprobs[: len(tokens)]
        completions.append((tokens, logprobs))

    return completions


async def evaluate_completions(
    env: CodeEnv,
    completions: list[tuple[list[int], list[float]]],
) -> list[tuple[float, list[int], list[float]]]:
    """Run environment on all completions and return rewards."""
    tasks = [env.step(tokens) for tokens, _ in completions]
    step_results = await asyncio.gather(*tasks)

    out: list[tuple[float, list[int], list[float]]] = []
    for (tokens, logprobs), step_result in zip(completions, step_results):
        out.append((step_result.reward, tokens, logprobs))
    return out


def normalize_tests(example: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Normalize DeepCoder-Preview tests into LCB-style list[{'input', 'output'}]."""
    tests_raw = example.get("tests", [])
    if not tests_raw:
        return None

    # Common pattern: {'samples': [...]}
    if isinstance(tests_raw, dict):
        tests = tests_raw.get("samples", [])
    else:
        tests = tests_raw

    if not isinstance(tests, list):
        return None

    # If tests are already dicts with input/output, accept them
    if tests and isinstance(tests[0], dict) and "input" in tests[0] and "output" in tests[0]:
        return tests

    # Unknown format -> skip
    return None


async def process_batch(
    batch: list[dict[str, Any]],
    sampler: tinker.SamplingClient,
    tokenizer: Any,
    renderer: Any,
    config: Config,
    executor: ThreadPoolExecutor,
) -> list[tinker.types.Datum]:
    """Build GRPO datums for a batch of problems."""
    all_datums: list[tinker.types.Datum] = []

    for example in batch:
        question = build_question(example)
        if not question:
            continue

        tests = normalize_tests(example)
        if not tests:
            continue

        env = CodeEnv(
            problem=question,
            tests=tests,
            renderer=renderer,
            format_coef=config.format_coef,
            reward_timeout=config.reward_timeout,
        )

        messages = [{"role": "user", "content": question}]
        prompt = renderer.build_generation_prompt(messages)
        stop_condition = env.stop_condition

        completions = sample_completions(
            sampler=sampler,
            prompt=prompt,
            group_size=config.group_size,
            temperature=config.temperature,
            stop_condition=stop_condition,
        )
        if not completions:
            continue

        rewards_data = await evaluate_completions(env, completions)
        if not rewards_data:
            continue

        rewards = [r for r, _, _ in rewards_data]
        advantages = compute_advantages(rewards)

        if should_skip(advantages):
            continue

        ob_len = len(prompt.tokens()) if hasattr(prompt, "tokens") else 0

        for (reward, tokens, logprobs), adv in zip(rewards_data, advantages):
            _ = reward  # used indirectly via adv
            datum = make_datum(tokens=tokens, logprobs=logprobs, ob_len=ob_len, advantage=adv)
            all_datums.append(datum)

    return all_datums


async def evaluate_model(
    test_dataset: datasets.Dataset,
    sampler: tinker.SamplingClient,
    tokenizer: Any,
    renderer: Any,
    config: Config,
    num_samples: int = 100,
) -> dict[str, float]:
    """Simple evaluation loop on a subset of the test set."""
    logger.info(f"Evaluating on {num_samples} test samples...")

    successes = 0
    format_errors = 0
    total_reward = 0.0

    if len(test_dataset) == 0:
        return {
            "eval_accuracy": 0.0,
            "eval_avg_reward": 0.0,
            "eval_format_errors": 0.0,
            "eval_samples": 0.0,
        }

    indices = np.random.choice(
        len(test_dataset),
        size=min(num_samples, len(test_dataset)),
        replace=False,
    )

    for idx in indices:
        example = test_dataset[int(idx)]
        question = build_question(example)
        if not question:
            continue

        tests = normalize_tests(example)
        if not tests:
            continue

        env = CodeEnv(
            problem=question,
            tests=tests,
            renderer=renderer,
            format_coef=config.format_coef,
            reward_timeout=config.reward_timeout,
        )

        messages = [{"role": "user", "content": question}]
        prompt = renderer.build_generation_prompt(messages)

        sampling_params = tinker.types.SamplingParams(
            max_tokens=2048,
            temperature=0.2,
            stop_sequences=env.stop_condition,
            include_logprobs=False,
        )

        try:
            future = sampler.sample(
                prompt=prompt,
                num_samples=1,
                sampling_params=sampling_params,
            )
            result = future.result()

            if result.sequences:
                seq = result.sequences[0]
                tokens = list(seq.tokens)
                if tokens:
                    step_result = await env.step(tokens)
                    total_reward += step_result.reward

                    if step_result.metrics.get("correct", 0) == 1:
                        successes += 1
                    if step_result.metrics.get("format", 0) == -1:
                        format_errors += 1

        except Exception as e:
            logger.warning(f"Evaluation error: {e}")
            continue

    n = float(num_samples) if num_samples > 0 else 1.0
    accuracy = successes / n
    avg_reward = total_reward / n

    return {
        "eval_accuracy": accuracy,
        "eval_avg_reward": avg_reward,
        "eval_format_errors": float(format_errors),
        "eval_samples": float(num_samples),
    }


async def main_async(config: Config):
    """Main async GRPO training loop."""
    # Logging setup
    check_log_dir(config.log_path, config.log_dir_behavior)
    logger_obj = setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
    )

    # Tokenizer and renderer
    logger.info(f"Loading tokenizer for {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    renderer = get_renderer("qwen3_instruct", tokenizer)

    # Tinker service client and LoRA training client
    logger.info("Initializing Tinker ServiceClient...")
    service_client = tinker.ServiceClient()
    logger.info("Creating LoRA TrainingClient...")
    training_client: tinker.TrainingClient = (
        await service_client.create_lora_training_client_async(
            base_model=config.model_name,
            rank=config.lora_rank,
        )
    )

    # Resume from checkpoint if available
    checkpoint = get_last_checkpoint(config.log_path)
    start_global_step = 0
    if checkpoint and "state_path" in checkpoint:
        logger.info(f"Loading checkpoint from {checkpoint['state_path']}")
        await training_client.load_state_async(checkpoint["state_path"])
        start_global_step = int(checkpoint.get("loop_state", {}).get("global_step", 0))

    # Sampling client for generation
    sampler: tinker.SamplingClient = await training_client.save_weights_and_get_sampling_client_async()

    # Datasets: DeepCoder-Preview as in assignment
    logger.info("Loading datasets...")
    train_dataset = datasets.concatenate_datasets(
        [
            cast(
                datasets.Dataset,
                datasets.load_dataset(
                    "agentica-org/DeepCoder-Preview-Dataset",
                    name=name,
                    split="train",
                ),
            )
            for name in ("primeintellect", "taco", "lcbv5")
        ]
    )

    test_dataset = datasets.concatenate_datasets(
        [
            cast(
                datasets.Dataset,
                datasets.load_dataset(
                    "agentica-org/DeepCoder-Preview-Dataset",
                    name=name,
                    split="test",
                ),
            )
            for name in ("codeforces", "lcbv5")
        ]
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # AdamParams: only learning_rate (batch_size is not a valid field in this version)
    adam_params = tinker.types.AdamParams(
        learning_rate=config.learning_rate,
    )

    executor = ThreadPoolExecutor(max_workers=4)

    logger.info("Starting training...")
    global_step = start_global_step
    step_in_epoch = 0

    while config.max_steps < 0 or global_step < config.max_steps:
        train_dataset = train_dataset.shuffle(seed=42 + global_step)

        for i in range(0, len(train_dataset), config.batch_size):
            # HF datasets slice returns a dict of columns -> convert to list of row dicts
            batch_ds = train_dataset[i : i + config.batch_size]
            batch = [
                dict(zip(batch_ds.keys(), values))
                for values in zip(*batch_ds.values())
            ]

            datums = await process_batch(
                batch=batch,
                sampler=sampler,
                tokenizer=tokenizer,
                renderer=renderer,
                config=config,
                executor=executor,
            )

            if datums:
                train_step(training_client, datums, adam_params)
                metrics = {
                    "step": global_step,
                    "datums_collected": len(datums),
                    "learning_rate": config.learning_rate,
                }
                logger_obj.log_metrics(metrics, step=global_step)
                logger.info(f"Step {global_step}: processed {len(datums)} datums")

            global_step += 1
            step_in_epoch += 1

            if config.eval_every > 0 and step_in_epoch % config.eval_every == 0:
                eval_metrics = await evaluate_model(
                    test_dataset=test_dataset,
                    sampler=sampler,
                    tokenizer=tokenizer,
                    renderer=renderer,
                    config=config,
                    num_samples=50,
                )
                logger_obj.log_metrics(eval_metrics, step=global_step)
                logger.info(f"Evaluation at step {global_step}: {eval_metrics}")

            if config.save_every > 0 and step_in_epoch % config.save_every == 0:
                name = f"checkpoint_step_{global_step}"
                save_checkpoint(
                    training_client=training_client,
                    name=name,
                    log_path=config.log_path,
                    loop_state={"global_step": global_step},
                    kind="state",
                )
                logger.info(f"Saved checkpoint: {name}")

            if config.max_steps > 0 and global_step >= config.max_steps:
                break

        if config.max_steps > 0 and global_step >= config.max_steps:
            break

    final_name = f"checkpoint_final_{global_step}"
    save_checkpoint(
        training_client=training_client,
        name=final_name,
        log_path=config.log_path,
        loop_state={"global_step": global_step},
        kind="state",
    )
    logger.info(f"Saved final checkpoint: {final_name}")

    executor.shutdown()
    logger_obj.close()
    logger.info("Training completed!")


def main(config: Config):
    asyncio.run(main_async(config))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
