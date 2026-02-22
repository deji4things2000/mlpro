import asyncio
import json
import logging
import random
import hydra
import pandas as pd
from pydantic import BaseModel, Field
from pathlib import Path
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from agents.base import AgentConfig
from agents.llm_agent import LLMAgent
from envs.negotiation import NegotiationEnv, NegotiationConfig
from envs.public_goods import PublicGoodsConfig
from envs.public_goods import PublicGoodsEnv
from envs.collaboration import CollaborationConfig, CollaborationEnv
from utils.parallel import gather_limited
from utils.parallel import to_thread_map
from utils.io import log_stage, resolve_path, load_json, write_jsonl


class NumericAction(BaseModel):
    rationale: str = Field(description="Internal monologue explaining the strategic calculation.")
    action: int = Field(description="The numeric action to take in the scenario.")


class TextAction(BaseModel):
    rationale: str = Field(description="Internal monologue explaining the strategic calculation.")
    message: str = Field(description="The natural language message to broadcast to the team.")


logging.getLogger("LiteLLM").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    base_dir = Path(get_original_cwd())
    log_stage("Loading input data")
    output_dir = resolve_path(base_dir, str(cfg.output.base_dir))
    simulation_dir = output_dir / "simulation"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    simulation_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    population = load_json(resolve_path(base_dir, str(cfg.data.population_path)))
    personas = list(population["personas"])
    groups = chunk_groups(personas, int(cfg.simulation.group_size), int(cfg.simulation.seed))
    if int(cfg.simulation.max_groups) > 0:
        groups = groups[:int(cfg.simulation.max_groups)]
    log_stage(f"Prepared {len(groups)} groups and {len(personas)} personas")

    context_conditions = [str(value) for value in cfg.simulation.context_conditions]
    step_records = []
    dialogue_records = []
    for condition_index, context_condition in enumerate(context_conditions):
        condition_seed_offset = condition_index * 10_000_000
        log_stage(f"Running iteration [{context_condition}]")
        public_goods_steps, public_goods_dialogues = asyncio.run(run_scenario("public_goods", groups, cfg, condition_seed_offset + 0, context_condition))
        negotiation_steps, negotiation_dialogues = asyncio.run(run_scenario("negotiation", groups, cfg, condition_seed_offset + 1_000_000, context_condition))
        collab_steps, collab_dialogues = asyncio.run(run_scenario("collaboration", groups, cfg, condition_seed_offset + 2_000_000, context_condition))

        step_records.extend(public_goods_steps)
        step_records.extend(negotiation_steps)
        step_records.extend(collab_steps)

        dialogue_records.extend(public_goods_dialogues)
        dialogue_records.extend(negotiation_dialogues)
        dialogue_records.extend(collab_dialogues)
    log_stage(f"Collected {len(step_records)} step records and {len(dialogue_records)} dialogue records")

    steps_jsonl = simulation_dir / "step_records.jsonl"
    dialogues_jsonl = simulation_dir / "dialogue_records.jsonl"
    write_jsonl(steps_jsonl, step_records)
    write_jsonl(dialogues_jsonl, dialogue_records)

    steps_table = tables_dir / "simulation_steps.csv"
    dialogues_table = tables_dir / "dialogue_log.csv"
    pd.DataFrame(serialize_step_rows(step_records)).to_csv(steps_table, index=False)
    pd.DataFrame(dialogue_records).to_csv(dialogues_table, index=False)

    manifest = {
        "metadata": {
            "n_personas": int(len(personas)),
            "n_groups": int(len(groups)),
            "group_size": int(cfg.simulation.group_size),
            "seed": int(cfg.simulation.seed),
            "model": str(cfg.agent.model),
            "context_conditions": context_conditions,
            "scenarios": ["public_goods", "negotiation", "collaboration"]
        },
        "artifacts": {
            "step_records_jsonl": str(steps_jsonl),
            "dialogue_records_jsonl": str(dialogues_jsonl),
            "simulation_steps_csv": str(steps_table),
            "dialogue_log_csv": str(dialogues_table)
        }
    }
    manifest_path = output_dir / "simulation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log_stage(f"Wrote simulation manifest to {manifest_path}")

    print(f"Simulation ready for analysis with {manifest['metadata']['n_groups']} groups")


def chunk_groups(personas: list[dict], group_size: int, seed: int) -> list[list[dict]]:
    rng = random.Random(seed)
    shuffled = list(personas)
    rng.shuffle(shuffled)
    groups = [shuffled[index:index + group_size] for index in range(0, len(shuffled), group_size)]
    if len(groups[-1]) == group_size:
        return groups
    deficit = group_size - len(groups[-1])
    groups[-1].extend(shuffled[:deficit])
    return groups


def format_persona(context_condition: str, cfg: DictConfig, persona: dict) -> str:
    traits = persona["traits"]
    values = {
        "name": persona["name"],
        "life_story": persona["life_story"],
        "O": float(traits["O"]),
        "C": float(traits["C"]),
        "E": float(traits["E"]),
        "A": float(traits["A"]),
        "N": float(traits["N"])
    }
    if context_condition == "life_story_only":
        return str(cfg.agent.persona_template_life_story_only).format(**values)
    if context_condition == "bfi_only":
        return str(cfg.agent.persona_template_bfi_only).format(**values)
    if context_condition == "life_story_and_bfi":
        return str(cfg.agent.persona_template_life_story_and_bfi).format(**values)
    raise ValueError(f"Unknown context condition {context_condition}")


def build_agents(group: list[dict], cfg: DictConfig, context_condition: str, scenario: str) -> dict[str, LLMAgent]:
    response_model = TextAction if scenario == "collaboration" else NumericAction

    agents = {}
    for persona in group:
        config = AgentConfig(
            name=persona["name"],
            persona=format_persona(context_condition, cfg, persona),
            temperature=float(cfg.agent.temperature),
            use_memory=True,
            use_reflection=False,
            system_prompt=str(cfg.agent.system_prompt)
        )
        agents[persona["name"]] = LLMAgent(config, model=str(cfg.agent.model), response_model=response_model)

    return agents


def build_env(scenario: str, agent_ids: list[str], cfg: DictConfig, seed: int):
    if scenario == "public_goods":
        public_goods_cfg = cfg.simulation.public_goods
        return PublicGoodsEnv(
            agent_ids,
            PublicGoodsConfig(
                num_rounds=int(public_goods_cfg.num_rounds),
                endowment=int(public_goods_cfg.endowment),
                threshold=int(public_goods_cfg.threshold),
                catastrophe_risk=float(public_goods_cfg.catastrophe_risk),
                success_bonus=int(public_goods_cfg.success_bonus),
                risk_increase_per_round=float(public_goods_cfg.risk_increase_per_round),
                contribution_fatigue_penalty=float(public_goods_cfg.contribution_fatigue_penalty)
            ),
            seed=seed
        )
    if scenario == "negotiation":
        negotiation_cfg = cfg.simulation.negotiation
        return NegotiationEnv(
            agent_ids,
            NegotiationConfig(
                total_resource=int(negotiation_cfg.total_resource),
                num_rounds=int(negotiation_cfg.num_rounds),
                disagreement_penalty=float(negotiation_cfg.disagreement_penalty),
                scarcity_growth_per_round=float(negotiation_cfg.scarcity_growth_per_round),
                disagreement_penalty_growth_per_round=float(negotiation_cfg.disagreement_penalty_growth_per_round)
            ),
            seed=seed
        )
    if scenario == "collaboration":
        collab_cfg = cfg.simulation.collaboration
        return CollaborationEnv(
            agent_ids,
            CollaborationConfig(
                num_rounds=int(collab_cfg.num_rounds)
            ),
            seed=seed
        )
    raise ValueError(f"Unknown scenario {scenario}")


async def run_episode(
    env,
    agents: dict[str, LLMAgent],
    max_steps: int,
    max_parallel_agents: int,
    scenario: str,
    group_id: int,
    context_condition: str
) -> tuple[list[dict], list[dict]]:
    observations = env.reset()
    done = False
    step = 0
    step_records = []
    dialogue_records = []
    while not done and step < max_steps:
        agent_ids = list(agents.keys())

        def act(agent_id: str) -> dict:
            return agents[agent_id].act_with_trace(observations[agent_id])

        traces_list = await to_thread_map(act, agent_ids, max_concurrency=max_parallel_agents)
        traces = {agent_id: trace for agent_id, trace in zip(agent_ids, traces_list)}

        actions = {agent_id: traces[agent_id]["action"] for agent_id in agent_ids}
        result = env.step(actions)

        info = dict(result.info)
        info["scenario"] = scenario
        info["context_condition"] = context_condition
        info["group_id"] = group_id
        info["step"] = step + 1
        info["done"] = bool(result.done)
        info["rewards"] = dict(result.rewards)
        step_records.append(info)

        parsed_map = {}

        if scenario == "public_goods":
            parsed_map = dict(info.get("actions", {}))
        elif scenario == "negotiation":
            parsed_map = dict(info.get("demands", {}))

        for agent_id in agent_ids:
            trace = traces[agent_id]
            prompt_messages = list(trace["messages"])
            reward = float(result.rewards[agent_id])

            action_parsed = parsed_map.get(agent_id, 0.0)

            dialogue_records.append(
                {
                    "scenario": scenario,
                    "context_condition": context_condition,
                    "group_id": group_id,
                    "round": int(info["round"]),
                    "step": int(info["step"]),
                    "agent_id": agent_id,
                    "observation": observations[agent_id],
                    "prompt_system": str(prompt_messages[0]["content"]),
                    "prompt_user": str(prompt_messages[1]["content"]),
                    "action_text": trace["action_text"],
                    "action_parsed": float(action_parsed) if scenario != "collaboration" else "NaN",
                    "reward": reward,
                    "done": bool(result.done),
                    "gm_events": list(info["gm_events"])
                }
            )
            agents[agent_id].add_memory(
                f"Observation: {observations[agent_id]} | Action Output: {trace['action_text']} | Reward: {reward}"
            )
        observations = result.observations
        done = bool(result.done)
        step += 1
    return step_records, dialogue_records


async def run_scenario(
    scenario: str,
    groups: list[list[dict]],
    cfg: DictConfig,
    seed_offset: int,
    context_condition: str
) -> tuple[list[dict], list[dict]]:
    async def run_group(group: list[dict], group_index: int) -> tuple[list[dict], list[dict]]:
        rng = random.Random(int(cfg.simulation.seed) + seed_offset + group_index)

        agent_ids = [persona["name"] for persona in group]
        env = build_env(scenario, agent_ids, cfg, seed=rng.randint(0, 2**31 - 1))
        agents = build_agents(group, cfg, context_condition, scenario)

        return await run_episode(
            env,
            agents,
            max_steps=int(cfg.simulation.max_steps),
            max_parallel_agents=int(cfg.simulation.parallel.max_parallel_agents),
            scenario=scenario,
            group_id=group_index,
            context_condition=context_condition
        )

    tasks = [run_group(group, group_index + 1) for group_index, group in enumerate(groups)]
    grouped_results = await gather_limited(tasks, max_concurrency=int(cfg.simulation.parallel.max_parallel_groups))
    step_records = []
    dialogue_records = []
    for group_steps, group_dialogues in grouped_results:
        step_records.extend(group_steps)
        dialogue_records.extend(group_dialogues)

    return step_records, dialogue_records


def serialize_step_rows(step_records: list[dict]) -> list[dict]:
    rows = []
    for record in step_records:
        row = {}
        for key, value in record.items():
            if isinstance(value, dict) or isinstance(value, list):
                row[key] = json.dumps(value)
            else:
                row[key] = value
        rows.append(row)
    return rows


if __name__ == "__main__":
    main()
