# AI Agents @ Dartmouth College
## Problem Set 1, Part II: Code RL Post-Training

This codebase implements an RL post-training loop for a code-generating language model using Tinker and a sandboxed Python execution environment. The goal is to train Qwen 3 4B Instruct (with LoRA) on the DeepCoder-Preview dataset to generate Python code that passes all provided test cases.

## Sandbox configuration

Code execution is handled via a sandbox compatible with the LiveCodeBench utilities in `tinker_utils/lcb.py`. A base configuration for Sandbox Fusion is provided in `sandbox_config/local.yaml` (or `local.yaml` if you placed it at the project root).

To run the sandbox with Docker:
```bash
docker run -it \
    -p 8080:8080 \
    -v ./sandbox_config/local.yaml:/root/sandbox/sandbox/configs/local.yaml \
    volcengine/sandbox-fusion:server-20250609
```

The environment uses the following configuration in `tinker_utils/env.py`:
```python
SANDBOX_URL = os.getenv("SANDBOX_URL", "http://localhost:8080/run_code")
SANDBOX_MAX_CONCURRENCY = int(os.getenv("SANDBOX_MAX_CONCURRENCY", "4"))
SANDBOX_CLIENT_TIMEOUT_SECONDS = int(os.getenv("SANDBOX_CLIENT_TIMEOUT_SECONDS", "6000"))
```

You can override these via environment variables when launching `train.py`.

## Tinker utilities

The `tinker_utils` package provides most of the infrastructure needed for the assignment:

- `tinker_utils/checkpoint.py`
    Saving and loading training checkpoints (model state paths + loop state) via a JSONL file (`checkpoints.jsonl`).
- `tinker_utils/data.py`
    Utilities for turning dataset examples into natural language questions/prompts, including handling optional starter code.
- `tinker_utils/env.py`
    The `CodeEnv` environment that:
    - Renders prompts via a `Renderer`
    - Extracts Python code blocks from model outputs
    - Executes code in the sandbox
    - Computes rewards from format + correctness (see `CodeEnv.step`)
- `tinker_utils/log.py`
    A multi-backend logger (`JsonLogger`, `PrettyPrintLogger`, optional `WandbLogger`) combined via `MultiplexLogger`, plus `setup_logging` to initialize logging and save config/metrics.
- `tinker_utils/lcb.py`
    Utilities for working with LiveCodeBench-style test cases and sandbox test code (`TEST_CODE`, `TEST_UTIL`).
- `tinker_utils/renderers.py`
    Message rendering and tokenization for different chat formats, including a factory `get_renderer("qwen3_instruct", tokenizer)`.
- `tinker_utils/qwen.py`
    Qwen-specific renderers compatible with Qwen chat templates.

## Training loop (train.py)

The implemented `train.py` wires these components into a GRPO-style RL post-training pipeline. At a high level, it:

- Loads configuration via a `Config` class decorated with `@chz.chz` (enabling CLI overrides).
- Loads the `Qwen/Qwen3-4B-Instruct-2507` tokenizer and the `qwen3_instruct` renderer.
- Creates a Tinker `ServiceClient`, then a LoRA `TrainingClient` and `SamplingClient` for Qwen 3 4B Instruct.
- Loads the DeepCoder-Preview train/test splits:
    - Train: `primeintellect`, `taco`, `lcbv5` (split `train`)
    - Test: `codeforces`, `lcbv5` (split `test`)
- Runs an async GRPO training loop that:
    - Samples `group_size` completions per prompt via `SamplingClient` (with `SamplingParams`).
    - Evaluates completions in `CodeEnv` to obtain scalar rewards.
    - Computes group-relative advantages $A_i = R_i - \bar{R}$.
    - Skips degenerate groups where all rewards are identical (all advantages $\approx 0$).
    - Calls `training_client.train(...)` with `tinker.types.Datum` objects (tokens, logprobs, observation length, advantage).
- Periodically:
    - Logs training metrics (e.g. `datums_collected`, `learning_rate`).
    - Evaluates on a random subset of the test set via `evaluate_model`, logging `eval_accuracy`, `eval_avg_reward`, and `eval_format_errors`.
    - Saves checkpoints at `save_every` steps and at completion via `save_checkpoint`.

The key helper functions in `train.py` include:

```python
def should_skip(advantages: list[float]) -> bool:
        """Skip group when all advantages are ~0 (all rewards equal)."""


def compute_advantages(rewards: list[float]) -> list[float]:
        """Compute group-relative advantages A_i = R_i - mean(R)."""


def make_datum(
        tokens: list[int],
        logprobs: list[float],
        ob_len: int,
        advantage: float,
) -> tinker.types.Datum:
        """Pack a single GRPO sample into a Tinker datum."""


def train_step(
        training_client: tinker.TrainingClient,
        datums: list[tinker.types.Datum],
        adam_params: tinker.types.AdamParams,
) -> None:
        """Run one GRPO update on the collected datums."""
```

In addition, the async helpers:

- `sample_completions(...)`: uses `asyncio.gather` to sample multiple completions in parallel from the sampler.
- `evaluate_completions(...)`: uses `asyncio.gather` to run environment evaluation in parallel for each completion.
- `process_batch(...)`: ties sampling, evaluation, advantage computation, skipping of degenerate groups, and datum construction together for each batch.

## Requirements and running

Dependencies are specified in `requirements.txt`, including `tinker`, `datasets`, `torch`, `transformers`, `aiohttp`, `chz`, and logging utilities.

Before running:

1. Start the sandbox (see “Sandbox configuration” above).
2. Set your Tinker API key:

```bash
export TINKER_API_KEY="your_tinker_api_key_here"
```

From the project root (where `train.py` lives), run:

```bash
python train.py
```

You can adjust hyperparameters (e.g., `max_steps`, `batch_size`, `eval_every`, `save_every`) via `chz` overrides using `key=value` syntax:

```bash
python train.py max_steps=5 batch_size=4 group_size=2
```

The implementation is designed to:

- Train for a configurable number of steps (or effectively full epochs if `max_steps = -1`).
- Log training and evaluation metrics at regular intervals.
- Save checkpoints periodically and at completion.
- Handle sampling and environment execution with `asyncio`.
- Skip degenerate groups where all rewards are identical.

We also provide a number of utilities in `tinker_utils` to get things moving quickly. These include:

- **Module:** `tinker_utils/checkpoint.py` Handles saving and loading training checkpoints with full optimizer state.
- **Module:** `tinker_utils/data.py` has utilities for preparing dataset examples into model prompts.
- **Module:** `tinker_utils/env.py` defines the code generation environment with sandbox execution.
- **Module:** `tinker_utils/log.py` provides a multi-backend logging system supporting JSON files, console output, and Weights & Biases.
- **Module:** `tinker_utils/lcb.py` provides utilities for working with LiveCodeBench dataset format and test execution.
- **Module:** `tinker_utils/renderers.py` has conversation formatting and token rendering for different model architectures.
- **Module:** `tinker_utils/qwen.py` has specialized renderers for Qwen3 models.


The starter code in `train.py` includes the following function signatures, which you should fill out and compose into a functional training loop. This is intended to help scaffold your implementation, and to make us easier to test things.

```python
def should_skip(advantages: list[float]) -> bool:
    # Should we skip this training step?

def compute_advantages(
    rewards: list[float]
) -> list[float]:
    # Compute advantages from rewards


def make_datum(
    tokens: list[int],
    logprobs: list[float],
    ob_len: int,
    advantage: float
) -> tinker.types.Datum:
    # Make a training datapoint for Tinker


def train_step(
    training_client: tinker.TrainingClient,
    datums: list[tinker.types.Datum],
    adam_params: tinker.types.AdamParams
) -> None:
    # Run one training step
```

We recommend making good use of `asyncio.gather()`, e.g. to sample multiple results simultaneously. Look at the documentation for Tinker's `async` APIs.
