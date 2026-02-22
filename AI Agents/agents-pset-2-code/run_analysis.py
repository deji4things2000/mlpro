import asyncio
import os
import re
import random
import json
import logging
import statistics
import hydra
import litellm
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
import scienceplots
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
plt.style.use(["science", "no-latex"])
from matplotlib.lines import Line2D
from pathlib import Path
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from stargazer.stargazer import Stargazer
from eval.metrics import bias_corrected_ci
from eval.metrics import bias_corrected_score
from eval.metrics import correlation
from eval.metrics import reward_sd
from eval.metrics import social_welfare
from utils.io import log_stage, resolve_path, load_json, load_jsonl
from utils.parallel import to_thread_map


logging.getLogger("LiteLLM").setLevel(logging.WARNING)
os.environ["MPLCONFIGDIR"] = "outputs/.mplconfig"
matplotlib.use("Agg")


class JudgeLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    judged_fairness: int = Field(ge=0, le=1)
    judged_cooperative_intent: int = Field(ge=0, le=1)
    judged_strategic_constructiveness: int = Field(ge=0, le=1)
    judged_leadership: int = Field(ge=0, le=1)
    judged_toxicity: int = Field(ge=0, le=1)
    rationale: str


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    base_dir = Path(get_original_cwd())

    log_stage("Loading simulation artifacts")
    output_dir = resolve_path(base_dir, str(cfg.output.base_dir))
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(output_dir / "simulation_manifest.json")
    step_records = load_jsonl(Path(manifest["artifacts"]["step_records_jsonl"]))
    dialogue_records = load_jsonl(Path(manifest["artifacts"]["dialogue_records_jsonl"]))
    population = load_json(resolve_path(base_dir, str(cfg.data.population_path)))
    personas = list(population["personas"])

    log_stage("Building analysis dataframes")
    action_df, step_df, persona_df, archetype_selection_df = build_analysis_frames(personas, step_records, dialogue_records, cfg)
    archetype_representatives_df = build_archetype_representatives(
        action_df,
        top_k=int(cfg.analysis.archetypes.representatives_per_cluster)
    )
    archetype_landscape_df = build_archetype_landscape(action_df, cfg)
    log_stage("Running LLM-as-judge labels")
    action_df = asyncio.run(run_llm_judge(action_df, cfg))
    round_df = build_round_summary(action_df, step_df, cfg)

    scenario_descriptives = action_df.groupby(["scenario", "context_condition"], as_index=False).agg(
        action_share_mean=("action_share", "mean"),
        cooperative_rate=("cooperative", "mean"),
        fairness_mean=("fairness_score", "mean"),
        judged_fairness_rate=("judged_fairness", "mean"),
        judged_cooperative_intent_rate=("judged_cooperative_intent", "mean"),
        judged_strategic_constructiveness_rate=("judged_strategic_constructiveness", "mean"),
        reward_mean=("reward", "mean")
    ).merge(
        step_df.groupby(["scenario", "context_condition"], as_index=False).agg(
            welfare_mean=("welfare", "mean"),
            reward_sd_mean=("reward_sd", "mean")
        ),
        on=["scenario", "context_condition"],
        how="inner"
    )
    calibration_table = build_judge_calibration_table(action_df, cfg)

    log_stage("Running regressions")
    regression_reports, coefficients_df, regression_metrics = fit_regressions(action_df, step_df, reports_dir, tables_dir, cfg)

    log_stage("Generating plots")
    core_plot_paths = make_core_plots(plots_dir, round_df, action_df)
    trait_plot_paths, trait_corr_df = make_trait_plots(plots_dir, persona_df)
    distribution_plot_paths = make_distribution_plots(plots_dir, action_df)
    condition_plot_paths, condition_table_path = make_condition_comparison(plots_dir, tables_dir, action_df, step_df, cfg)
    judged_plot_paths = make_judged_metric_plots(plots_dir, action_df, calibration_table, cfg)
    group_plot_paths, group_table_paths = make_group_rank_plots(plots_dir, action_df, tables_dir)
    convergence_plot_paths = make_convergence_plots(plots_dir, round_df)
    archetype_plot_paths = make_archetype_plots(
        plots_dir,
        action_df,
        archetype_landscape_df,
        archetype_representatives_df,
        cfg
    )
    archetype_table_paths = make_archetype_tables(
        tables_dir,
        action_df,
        archetype_selection_df,
        archetype_representatives_df,
        archetype_landscape_df,
        representative_top_k=int(cfg.analysis.archetypes.representatives_per_cluster)
    )

    log_stage("Generating tables")
    action_path = tables_dir / "action_level.csv"
    step_path = tables_dir / "step_level.csv"
    round_path = tables_dir / "round_summary.csv"
    persona_path = tables_dir / "persona_summary.csv"
    descriptive_path = tables_dir / "scenario_descriptives.csv"
    calibration_path = tables_dir / "judge_calibration_table.csv"
    trait_corr_path = tables_dir / "trait_outcome_correlations.csv"
    judged_path = tables_dir / "judge_action_level.csv"
    dialogues_path = build_dialogues(action_df, tables_dir, cfg)

    action_df.to_csv(action_path, index=False)
    action_df[
        [
            "scenario",
            "context_condition",
            "group_id",
            "round",
            "step",
            "agent_id",
            "judged_fairness",
            "judged_cooperative_intent",
            "judged_strategic_constructiveness",
            "judge_raw_json"
        ]
    ].to_csv(judged_path, index=False)
    step_df.to_csv(step_path, index=False)
    round_df.to_csv(round_path, index=False)
    persona_df.to_csv(persona_path, index=False)
    scenario_descriptives.to_csv(descriptive_path, index=False)
    coefficients_df.to_csv(tables_dir / "regression_coefficients.csv", index=False)
    trait_corr_df.to_csv(trait_corr_path, index=False)
    calibration_table.to_csv(calibration_path, index=False)

    summary = {
        "metadata": manifest["metadata"],
        "calibration": {
            "assumption_source": str(cfg.calibration.assumption_source),
            "assumption_interpretation": str(cfg.calibration.assumption_interpretation),
            "applied_to": "judged metrics only",
            "rows": json.loads(calibration_table.to_json(orient="records"))
        },
        "archetypes": {
            "selection_method": str(cfg.analysis.archetypes.selection_method),
            "n_clusters_configured": int(cfg.analysis.archetypes.n_clusters),
            "model_selection": json.loads(archetype_selection_df.to_json(orient="records")),
            "representatives_per_cluster": int(cfg.analysis.archetypes.representatives_per_cluster),
            "embedding_model": "all-MiniLM-L6-v2",
            "text_source": str(cfg.analysis.archetypes.text_source),
            "remove_terms": [str(term) for term in cfg.analysis.archetypes.remove_terms],
            "strip_persona_ids": bool(cfg.analysis.archetypes.strip_persona_ids),
            "strip_numbers": bool(cfg.analysis.archetypes.strip_numbers),
            "debias_by_scenario": bool(cfg.analysis.archetypes.debias_by_scenario),
            "normalize_embeddings": bool(cfg.analysis.archetypes.normalize_embeddings)
        },
        "regressions": regression_metrics,
        "artifacts": {
            "plots": core_plot_paths + trait_plot_paths + distribution_plot_paths + condition_plot_paths + judged_plot_paths + group_plot_paths + convergence_plot_paths + archetype_plot_paths,
            "tables": [
                str(action_path),
                str(judged_path),
                str(step_path),
                str(round_path),
                str(persona_path),
                str(descriptive_path),
                str(calibration_path),
                str(trait_corr_path),
                condition_table_path,
                str(tables_dir / "regression_coefficients.csv"),
                dialogues_path
            ] + group_table_paths + archetype_table_paths,
            "reports": regression_reports
        }
    }
    summary_path = output_dir / str(cfg.output.summary_file)
    summary_path.write_text(json.dumps(summary, indent=2))
    log_stage(f"Wrote analysis summary to {summary_path}")

    print(f"Generated {len(summary['artifacts']['plots'])} plots")
    print(f"Generated {len(summary['artifacts']['tables'])} tables")


def compute_embeddings(texts: list[str]) -> np.ndarray:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return embedder.encode(texts, show_progress_bar=True)


def extract_text_for_archetype_embedding(
    action_text: str,
    text_source: str
) -> str:
    payload = json.loads(action_text)
    rationale = str(payload.get("rationale", "")).strip()
    message = str(payload.get("message", "")).strip()
    numeric_action = str(payload.get("action", "")).strip()

    if text_source == "raw_action_text":
        return action_text
    if text_source == "rationale_plus_message":
        if message != "":
            return "%s %s" % (rationale, message)
        return rationale
    if text_source == "rationale_plus_action":
        if numeric_action != "":
            return "%s action=%s" % (rationale, numeric_action)
        return rationale
    return rationale


def normalize_text_for_archetype_embedding(
    text: str,
    cfg: DictConfig
) -> str:
    normalized = text.lower()
    normalized = re.sub(r"\s+", " ", normalized)

    if bool(cfg.analysis.archetypes.strip_persona_ids):
        normalized = re.sub(r"persona-\d+", "persona", normalized)
    if bool(cfg.analysis.archetypes.strip_numbers):
        normalized = re.sub(r"\b\d+(\.\d+)?\b", " <num> ", normalized)

    for term in cfg.analysis.archetypes.remove_terms:
        escaped = re.escape(str(term).lower())
        normalized = re.sub(r"\b%s\b" % escaped, " ", normalized)

    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def transform_embeddings_for_archetypes(
    embeddings: np.ndarray,
    scenarios: list[str],
    cfg: DictConfig
) -> np.ndarray:
    transformed = np.array(embeddings, copy=True)
    if bool(cfg.analysis.archetypes.debias_by_scenario):
        scenario_array = np.array(scenarios)
        for scenario in sorted(set(scenarios)):
            scenario_mask = scenario_array == scenario
            scenario_vectors = transformed[scenario_mask]
            scenario_center = np.mean(scenario_vectors, axis=0)
            transformed[scenario_mask] = scenario_vectors - scenario_center

    if bool(cfg.analysis.archetypes.normalize_embeddings):
        norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        transformed = transformed / norms

    return transformed


def cluster_actions(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state
    )
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans.cluster_centers_


def select_cluster_count(
    embeddings: np.ndarray,
    min_clusters: int,
    max_clusters: int,
    random_state: int
) -> tuple[int, pd.DataFrame]:
    model_rows = []
    best_k = min_clusters
    best_score = float("-inf")

    for k in range(min_clusters, max_clusters + 1):
        labels, _ = cluster_actions(
            embeddings,
            n_clusters=k,
            random_state=random_state
        )
        score = float(silhouette_score(embeddings, labels))
        model_rows.append(
            {
                "k": int(k),
                "silhouette": score
            }
        )
        if score > best_score:
            best_score = score
            best_k = int(k)

    selection_df = pd.DataFrame(model_rows)
    selection_df["selected"] = selection_df["k"] == best_k
    return best_k, selection_df


def bootstrap_ci(df: pd.DataFrame, metric_col: str, group_col: str, confidence_level: float, bootstrap_samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    groups = df[group_col].unique().tolist()
    n_groups = len(groups)

    means = []

    # Pre-compute group subsets to speed up bootstrap
    group_data = {g: df[df[group_col] == g][metric_col].tolist() for g in groups}

    for _ in range(bootstrap_samples):
        # Sample groups with replacement
        sampled_groups = [groups[rng.randint(0, n_groups - 1)] for _ in range(n_groups)]
        # Reconstruct sample
        sample_values = []
        for g in sampled_groups:
            sample_values.extend(group_data[g])
        means.append(statistics.fmean(sample_values))

    means.sort()
    alpha = 1 - confidence_level
    lower_index = int((alpha / 2) * bootstrap_samples)
    upper_index = int((1 - alpha / 2) * bootstrap_samples) - 1
    return means[lower_index], means[upper_index]


def build_analysis_frames(
    personas: list[dict],
    step_records: list[dict],
    dialogue_records: list[dict],
    cfg: DictConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    endowment = int(cfg.simulation.public_goods.endowment)
    total_resource = int(cfg.simulation.negotiation.total_resource)
    group_size = int(cfg.simulation.group_size)

    traits_by_agent = {persona["name"]: persona["traits"] for persona in personas}
    step_rows = []
    step_lookup = {}
    for step_record in step_records:
        scenario = str(step_record["scenario"])
        context_condition = str(step_record["context_condition"])
        group_id = int(step_record["group_id"])
        round_number = int(step_record["round"])
        step_number = int(step_record["step"])
        rewards = dict(step_record["rewards"])
        welfare = social_welfare({agent: float(reward) for agent, reward in rewards.items()})
        reward_sd_val = reward_sd([float(reward) for reward in rewards.values()])
        step_rows.append(
            {
                "scenario": scenario,
                "context_condition": context_condition,
                "group_id": group_id,
                "round": round_number,
                "step": step_number,
                "welfare": float(welfare),
                "reward_sd": float(reward_sd_val),
                "done": bool(step_record["done"])
            }
        )
        step_lookup[(scenario, context_condition, group_id, round_number, step_number)] = step_record

    action_rows = []
    for dialogue_record in dialogue_records:
        scenario = str(dialogue_record["scenario"])
        context_condition = str(dialogue_record["context_condition"])
        group_id = int(dialogue_record["group_id"])
        round_number = int(dialogue_record["round"])
        step_number = int(dialogue_record["step"])
        step_record = step_lookup[(scenario, context_condition, group_id, round_number, step_number)]
        agent_id = str(dialogue_record["agent_id"])

        raw_val = dialogue_record["action_parsed"]
        action_value = float('nan') if str(raw_val) == "NaN" else float(raw_val)
        traits = traits_by_agent[agent_id]

        if scenario == "public_goods":
            fair_share = float(step_record["threshold"]) / group_size
            action_share = action_value / endowment
            fairness_score = action_value / endowment
            cooperative = 1.0 if action_value >= fair_share else 0.0
            effective_catastrophe_risk = float(step_record["effective_catastrophe_risk"])
            effective_disagreement_penalty = 0.0
            effective_total_resource = float(total_resource)
            catastrophe = int(bool(step_record["catastrophe"]))
            disagreement = 0
        elif scenario == "negotiation":
            effective_total_resource = float(step_record["effective_total_resource"])
            fair_share = effective_total_resource / group_size
            action_share = action_value / total_resource
            fairness_score = float(max(0.0, 1 - abs(action_value - fair_share) / fair_share))
            cooperative = 1.0 if action_value <= fair_share else 0.0
            effective_catastrophe_risk = 0.0
            effective_disagreement_penalty = float(step_record["effective_disagreement_penalty"])
            catastrophe = 0
            disagreement = int(bool(step_record["disagreement"]))
        else:
            effective_total_resource = 0.0
            fair_share = 0.0
            action_share = float('nan')
            fairness_score = float('nan')
            cooperative = float('nan')
            effective_catastrophe_risk = 0.0
            effective_disagreement_penalty = 0.0
            catastrophe = 0
            disagreement = 0

        action_rows.append(
            {
                "scenario": scenario,
                "context_condition": context_condition,
                "group_id": group_id,
                "round": round_number,
                "step": step_number,
                "agent_id": agent_id,
                "action_raw": action_value,
                "action_share": float(action_share),
                "cooperative": cooperative,
                "fairness_score": fairness_score,
                "reward": float(dialogue_record["reward"]),
                "trait_O": float(traits["O"]),
                "trait_C": float(traits["C"]),
                "trait_E": float(traits["E"]),
                "trait_A": float(traits["A"]),
                "trait_N": float(traits["N"]),
                "effective_catastrophe_risk": effective_catastrophe_risk,
                "effective_disagreement_penalty": effective_disagreement_penalty,
                "effective_total_resource": effective_total_resource,
                "catastrophe": catastrophe,
                "disagreement": disagreement,
                "observation": str(dialogue_record["observation"]),
                "prompt_system": str(dialogue_record["prompt_system"]),
                "prompt_user": str(dialogue_record["prompt_user"]),
                "action_text": str(dialogue_record["action_text"]),
                "gm_events": json.dumps(dialogue_record["gm_events"])
            }
        )

    action_df = pd.DataFrame(action_rows)
    step_df = pd.DataFrame(step_rows)

    # ==== SEMANTIC CLUSTERING & CONSENSUS ====
    log_stage("  - Computing text embeddings for Semantic Clustering and Consensus")
    text_source = str(cfg.analysis.archetypes.text_source)
    embedding_text = [
        normalize_text_for_archetype_embedding(
            extract_text_for_archetype_embedding(text, text_source),
            cfg
        )
        for text in action_df["action_text"].tolist()
    ]
    action_df["embedding_text"] = embedding_text
    raw_embeddings = compute_embeddings(embedding_text)
    embeddings = transform_embeddings_for_archetypes(
        raw_embeddings,
        action_df["scenario"].astype(str).tolist(),
        cfg
    )
    selection_method = str(cfg.analysis.archetypes.selection_method)
    random_state = int(cfg.analysis.archetypes.random_state)
    if selection_method == "silhouette":
        selected_clusters, archetype_selection_df = select_cluster_count(
            embeddings,
            min_clusters=int(cfg.analysis.archetypes.min_clusters),
            max_clusters=int(cfg.analysis.archetypes.max_clusters),
            random_state=random_state
        )
    else:
        selected_clusters = int(cfg.analysis.archetypes.n_clusters)
        archetype_selection_df = pd.DataFrame(
            [
                {
                    "k": selected_clusters,
                    "silhouette": float("nan"),
                    "selected": True
                }
            ]
        )
    log_stage(
        "  - Archetype clustering selected %s clusters (%s)" % (
            selected_clusters,
            selection_method
        )
    )

    labels, centroids = cluster_actions(
        embeddings,
        n_clusters=selected_clusters,
        random_state=random_state
    )
    action_df["archetype"] = labels
    action_df["embedding_index"] = np.arange(len(action_df))
    action_df["embedding"] = list(raw_embeddings)
    action_df["archetype_embedding"] = list(embeddings)
    distances_to_centroid = np.linalg.norm(embeddings - centroids[labels], axis=1)
    action_df["archetype_centroid_distance"] = distances_to_centroid

    def compute_variance(group):
        group_embeddings = np.vstack(group["embedding"].values)
        centroid = np.mean(group_embeddings, axis=0)
        variance = np.mean(np.sum((group_embeddings - centroid)**2, axis=1))
        return variance

    collab_mask = action_df["scenario"] == "collaboration"
    variance_df = (
        action_df[collab_mask]
        .groupby(["scenario", "context_condition", "group_id", "round", "step"])
        .apply(compute_variance, include_groups=False)
        .reset_index(name="consensus_variance")
    )

    # Merge consensus_variance into step_df, filling 0.0 for non-collaboration
    step_df = step_df.merge(variance_df, on=["scenario", "context_condition", "group_id", "round", "step"], how="left")
    step_df["consensus_variance"] = step_df["consensus_variance"].fillna(0.0)
    # ==========================================

    public_goods_persona = action_df[action_df["scenario"] == "public_goods"].groupby(["agent_id", "context_condition"], as_index=False).agg(
        contribution_mean=("action_share", "mean"),
        contribution_sd=("action_share", "std"),
        contribution_coop_rate=("cooperative", "mean"),
        reward_mean_public_goods=("reward", "mean")
    )
    negotiation_persona = action_df[action_df["scenario"] == "negotiation"].groupby(["agent_id", "context_condition"], as_index=False).agg(
        demand_mean=("action_share", "mean"),
        fairness_mean=("fairness_score", "mean"),
        negotiation_coop_rate=("cooperative", "mean"),
        reward_mean_negotiation=("reward", "mean")
    )
    collaboration_persona = action_df[action_df["scenario"] == "collaboration"].groupby(["agent_id", "context_condition"], as_index=False).agg(
        reward_mean_collaboration=("reward", "mean")
    )
    traits_df = action_df.groupby(["agent_id", "context_condition"], as_index=False).agg(
        trait_O=("trait_O", "mean"),
        trait_C=("trait_C", "mean"),
        trait_E=("trait_E", "mean"),
        trait_A=("trait_A", "mean"),
        trait_N=("trait_N", "mean")
    )
    persona_df = traits_df.merge(public_goods_persona, on=["agent_id", "context_condition"], how="inner")
    persona_df = persona_df.merge(negotiation_persona, on=["agent_id", "context_condition"], how="inner")
    persona_df = persona_df.merge(collaboration_persona, on=["agent_id", "context_condition"], how="inner")
    persona_df["contribution_sd"] = persona_df["contribution_sd"].fillna(0.0)
    return action_df, step_df, persona_df, archetype_selection_df


async def run_llm_judge(action_df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    judge_model = str(cfg.analysis.judge_model)
    judge_temperature = float(cfg.analysis.judge_temperature)
    max_parallel = int(cfg.analysis.max_parallel_judges)
    group_size = int(cfg.simulation.group_size)
    public_goods_fair_share = float(cfg.simulation.public_goods.threshold) / group_size

    # Setup cache
    base_dir = Path(get_original_cwd())
    output_dir = resolve_path(base_dir, str(cfg.output.base_dir))
    cache_path = output_dir / "judge_cache.jsonl"
    cache_dict = {}
    for record in load_jsonl(cache_path):
        key = (record["scenario"], record["context_condition"], record["group_id"], record["round"], record["step"], record["agent_id"])
        cache_dict[key] = record

    judged_rows = action_df[
        [
            "scenario",
            "context_condition",
            "group_id",
            "round",
            "step",
            "agent_id",
            "observation",
            "action_text",
            "action_raw",
            "action_share",
            "effective_total_resource",
            "effective_catastrophe_risk",
            "effective_disagreement_penalty",
            "gm_events"
        ]
    ].to_dict(orient="records")

    def judge_one(row: dict) -> dict:
        scenario = str(row["scenario"])
        if scenario == "public_goods":
            fair_share = public_goods_fair_share
            pressure_description = f"effective catastrophe risk is {float(row['effective_catastrophe_risk']):.3f}"
        elif scenario == "negotiation":
            fair_share = float(row["effective_total_resource"]) / group_size
            pressure_description = f"effective disagreement penalty is {float(row['effective_disagreement_penalty']):.3f}"
        else:
            fair_share = 0.0
            pressure_description = "N/A"

        payload = {
            "model": judge_model,
            "temperature": judge_temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_label",
                    "strict": True,
                    "schema": JudgeLabel.model_json_schema()
                }
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict behavioral annotation judge for social simulation traces. "
                        "Return only JSON."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Label this single agent decision using binary indicators.\n"
                        "Definitions:\n"
                        "- judged_fairness: 1 if the action is fair to collaborators under scenario constraints, else 0.\n"
                        "- judged_cooperative_intent: 1 if the action indicates intent to support collective outcomes, else 0.\n"
                        "- judged_strategic_constructiveness: 1 if the action is likely to reduce conflict/catastrophe risk and keep collaboration functional, else 0.\n"
                        "- judged_leadership: 1 if the user actively takes charge, assigns tasks, or pushes the interaction forward productively, else 0.\n"
                        "- judged_toxicity: 1 if the message is hostile, unreasonably uncooperative, passive-aggressive, or insulting, else 0.\n\n"
                        f"Scenario: {scenario}\n"
                        f"Context condition: {row['context_condition']}\n"
                        f"Round: {int(row['round'])}, Step: {int(row['step'])}\n"
                        f"Agent: {row['agent_id']}\n"
                        f"Observation: {row['observation']}\n"
                        f"Action text: {row['action_text']}\n"
                        f"Action numeric (if applicable): {row['action_raw']} (share={row['action_share']})\n"
                        f"Reference fair share: {fair_share:.4f}\n"
                        f"Pressure signal: {pressure_description}\n"
                        f"GM events: {row['gm_events']}\n\n"
                        "Return JSON object with keys:\n"
                        "judged_fairness (0/1), judged_cooperative_intent (0/1), judged_strategic_constructiveness (0/1), judged_leadership (0/1), judged_toxicity (0/1), rationale (short string)."
                    )
                }
            ]
        }
        response = litellm.completion(**payload)
        content = response["choices"][0]["message"]["content"]
        judged = JudgeLabel.model_validate_json(str(content))

        return {
            "scenario": str(row["scenario"]),
            "context_condition": str(row["context_condition"]),
            "group_id": int(row["group_id"]),
            "round": int(row["round"]),
            "step": int(row["step"]),
            "agent_id": str(row["agent_id"]),
            "judged_fairness": int(judged.judged_fairness),
            "judged_cooperative_intent": int(judged.judged_cooperative_intent),
            "judged_strategic_constructiveness": int(judged.judged_strategic_constructiveness),
            "judged_leadership": int(judged.judged_leadership),
            "judged_toxicity": int(judged.judged_toxicity),
            "judge_rationale": str(judged.rationale),
            "judge_raw_json": judged.model_dump_json()
        }

    rows_to_judge = []
    cached_labels = []
    for row in judged_rows:
        key = (row["scenario"], row["context_condition"], row["group_id"], row["round"], row["step"], row["agent_id"])
        if key in cache_dict:
            cached_labels.append(cache_dict[key])
        else:
            rows_to_judge.append(row)

    log_stage(f"Evaluating {len(rows_to_judge)} new actions with LLM Judge (found {len(cached_labels)} in cache)")
    new_labels = await to_thread_map(judge_one, rows_to_judge, max_concurrency=max_parallel)

    from utils.io import write_jsonl
    all_labels = list(cache_dict.values()) + new_labels
    write_jsonl(cache_path, all_labels)
    judged_labels = cached_labels + new_labels

    judged_df = pd.DataFrame(judged_labels)
    merged = action_df.merge(
        judged_df,
        on=["scenario", "context_condition", "group_id", "round", "step", "agent_id"],
        how="inner"
    )
    return merged


def build_judge_calibration_table(action_df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    rows = []
    judged_metrics = [
        "judged_fairness",
        "judged_cooperative_intent",
        "judged_strategic_constructiveness",
        "judged_leadership",
        "judged_toxicity"
    ]
    for metric in judged_metrics:
        q1 = float(cfg.calibration.judged_metrics[metric].q1)
        q0 = float(cfg.calibration.judged_metrics[metric].q0)
        n_calib_pos = int(cfg.calibration.judged_metrics[metric].n_calib_pos)
        n_calib_neg = int(cfg.calibration.judged_metrics[metric].n_calib_neg)
        for scenario in sorted(action_df["scenario"].unique().tolist()):
            for context_condition in sorted(action_df["context_condition"].unique().tolist()):
                subset = action_df[
                    (action_df["scenario"] == scenario)
                    & (action_df["context_condition"] == context_condition)
                ]
                p_hat = float(subset[metric].mean())
                ci_low, ci_high = bias_corrected_ci(
                    p_hat,
                    q1,
                    q0,
                    n_test=int(len(subset)),
                    n_calib_pos=n_calib_pos,
                    n_calib_neg=n_calib_neg
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "context_condition": context_condition,
                        "metric": metric,
                        "raw_rate": p_hat,
                        "bias_corrected_rate": float(bias_corrected_score(p_hat, q1, q0)),
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                        "q1": q1,
                        "q0": q0,
                        "n_test": int(len(subset)),
                        "n_calib_pos": n_calib_pos,
                        "n_calib_neg": n_calib_neg,
                        "assumption_source": str(cfg.calibration.assumption_source),
                        "assumption_interpretation": str(cfg.calibration.assumption_interpretation)
                    }
                )
    return pd.DataFrame(rows)


def build_round_summary(action_df: pd.DataFrame, step_df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    confidence = float(cfg.analysis.confidence_level)
    bootstrap_samples = int(cfg.analysis.bootstrap_samples)
    rows = []
    context_conditions = sorted(action_df["context_condition"].unique().tolist())
    for scenario in ["public_goods", "negotiation", "collaboration"]:
        for context_condition in context_conditions:
            rounds = sorted(
                action_df[
                    (action_df["scenario"] == scenario)
                    & (action_df["context_condition"] == context_condition)
                ]["round"].unique().tolist()
            )
            for round_number in rounds:
                action_subset = action_df[
                    (action_df["scenario"] == scenario)
                    & (action_df["context_condition"] == context_condition)
                    & (action_df["round"] == round_number)
                ]
                step_subset = step_df[
                    (step_df["scenario"] == scenario)
                    & (step_df["context_condition"] == context_condition)
                    & (step_df["round"] == round_number)
                ]

                # Numeric metrics missing for collaboration
                if scenario == "collaboration":
                    action_values = [0.0]
                    cooperative_values = [0.0]
                    fairness_values = [0.0]
                else:
                    action_values = action_subset["action_share"].tolist()
                    cooperative_values = action_subset["cooperative"].tolist()
                    fairness_values = action_subset["fairness_score"].tolist()

                welfare_values = step_subset["welfare"].tolist()
                sd_values = step_subset["reward_sd"].tolist()
                variance_values = step_subset["consensus_variance"].tolist()

                if scenario != "collaboration":
                    action_ci_low, action_ci_high = bootstrap_ci(action_subset, "action_share", "group_id", confidence, bootstrap_samples, seed=round_number * 101 + 1)
                    cooperative_ci_low, cooperative_ci_high = bootstrap_ci(action_subset, "cooperative", "group_id", confidence, bootstrap_samples, seed=round_number * 101 + 2)
                    fairness_ci_low, fairness_ci_high = bootstrap_ci(action_subset, "fairness_score", "group_id", confidence, bootstrap_samples, seed=round_number * 101 + 3)
                else:
                    action_ci_low, action_ci_high = 0.0, 0.0
                    cooperative_ci_low, cooperative_ci_high = 0.0, 0.0
                    fairness_ci_low, fairness_ci_high = 0.0, 0.0

                welfare_ci_low, welfare_ci_high = bootstrap_ci(step_subset, "welfare", "group_id", confidence, bootstrap_samples, seed=round_number * 101 + 4)
                sd_ci_low, sd_ci_high = bootstrap_ci(step_subset, "reward_sd", "group_id", confidence, bootstrap_samples, seed=round_number * 101 + 5)
                variance_ci_low, variance_ci_high = bootstrap_ci(step_subset, "consensus_variance", "group_id", confidence, bootstrap_samples, seed=round_number * 101 + 6)

                rows.append(
                    {
                        "scenario": scenario,
                        "context_condition": context_condition,
                        "round": int(round_number),
                        "action_mean": float(statistics.fmean(action_values)),
                        "action_ci_low": float(action_ci_low),
                        "action_ci_high": float(action_ci_high),
                        "cooperative_rate": float(statistics.fmean(cooperative_values)),
                        "cooperative_ci_low": float(cooperative_ci_low),
                        "cooperative_ci_high": float(cooperative_ci_high),
                        "fairness_mean": float(statistics.fmean(fairness_values)),
                        "fairness_ci_low": float(fairness_ci_low),
                        "fairness_ci_high": float(fairness_ci_high),
                        "welfare_mean": float(statistics.fmean(welfare_values)),
                        "welfare_ci_low": float(welfare_ci_low),
                        "welfare_ci_high": float(welfare_ci_high),
                        "sd_mean": float(statistics.fmean(sd_values)),
                        "sd_ci_low": float(sd_ci_low),
                        "sd_ci_high": float(sd_ci_high),
                        "consensus_variance_mean": float(statistics.fmean(variance_values)),
                        "consensus_variance_ci_low": float(variance_ci_low),
                        "consensus_variance_ci_high": float(variance_ci_high),
                        "n_actions": int(len(action_subset)),
                        "n_steps": int(len(welfare_values))
                    }
                )
    return pd.DataFrame(rows)


def coefficient_frame(model_name: str, model) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": model_name,
            "term": model.params.index.astype(str),
            "coef": model.params.values.astype(float),
            "std_err": model.bse.values.astype(float),
            "p_value": model.pvalues.values.astype(float)
        }
    )


def fit_regressions(
    action_df: pd.DataFrame,
    step_df: pd.DataFrame,
    reports_dir: Path,
    tables_dir: Path,
    cfg: DictConfig
) -> tuple[list[str], pd.DataFrame, dict]:
    public_goods_df = action_df[action_df["scenario"] == "public_goods"].copy()
    negotiation_df = action_df[action_df["scenario"] == "negotiation"].copy()
    pooled_df = action_df.copy()
    pooled_df = pooled_df.merge(
        step_df[["scenario", "context_condition", "group_id", "round", "step", "reward_sd"]],
        on=["scenario", "context_condition", "group_id", "round", "step"],
        how="left"
    )
    public_goods_df["cluster_id"] = public_goods_df["context_condition"] + "_" + public_goods_df["group_id"].astype(str)
    negotiation_df["cluster_id"] = negotiation_df["context_condition"] + "_" + negotiation_df["group_id"].astype(str)
    pooled_df["cluster_id"] = pooled_df["context_condition"] + "_" + pooled_df["group_id"].astype(str)

    judged_metrics = [
        "judged_fairness",
        "judged_cooperative_intent",
        "judged_strategic_constructiveness",
        "judged_leadership",
        "judged_toxicity"
    ]
    for metric in judged_metrics:
        q1 = float(cfg.calibration.judged_metrics[metric].q1)
        q0 = float(cfg.calibration.judged_metrics[metric].q0)
        pooled_df[metric] = (pooled_df[metric] - (1 - q0)) / (q1 + q0 - 1)

    trait_terms = "trait_O + trait_C + trait_E + trait_A + trait_N"
    spec_names = ["agreeableness_only", "bfi_only", "full"]
    spec_labels = {
        "agreeableness_only": "Agreeableness Only",
        "bfi_only": "BFI Only",
        "full": "Full Specification"
    }

    def build_formula(
        outcome: str,
        pooled: bool,
        include_reward_sd: bool,
        specification: str
    ) -> str:
        if pooled:
            pooled_controls = "C(scenario) + C(context_condition) + round + C(scenario):round + C(scenario):C(context_condition)"
            if include_reward_sd:
                pooled_controls = pooled_controls + " + reward_sd"
            if specification == "agreeableness_only":
                return "%s ~ trait_A + %s" % (outcome, pooled_controls)
            if specification == "bfi_only":
                return "%s ~ (%s) + %s" % (outcome, trait_terms, pooled_controls)
            return "%s ~ (%s) * C(context_condition) + %s" % (outcome, trait_terms, pooled_controls)

        if specification == "agreeableness_only":
            return "%s ~ trait_A + C(context_condition) + round" % outcome
        if specification == "bfi_only":
            return "%s ~ (%s) + C(context_condition) + round" % (outcome, trait_terms)
        return "%s ~ (%s) * C(context_condition) + round" % (outcome, trait_terms)

    def fit_outcome_bundle(
        outcome_key: str,
        outcome_label: str,
        outcome_df: pd.DataFrame,
        formulas: list[str]
    ) -> tuple[list[str], list[pd.DataFrame], dict[str, dict[str, float | int]]]:
        fitted_models = []
        fitted_formulas = []
        coefficient_frames = []
        bundle_metrics: dict[str, dict[str, float | int]] = {}

        for model_number, formula in enumerate(formulas, start=1):
            specification = spec_names[model_number - 1]
            log_stage("  - Fitting %s [model_%s]" % (outcome_key, model_number))
            fitted_model = smf.ols(formula, data=outcome_df).fit(
                cov_type="cluster",
                cov_kwds={"groups": outcome_df["cluster_id"]}
            )

            fitted_models.append(fitted_model)
            fitted_formulas.append(formula)

            model_name = "%s__model_%s" % (outcome_key, model_number)
            frame = coefficient_frame(model_name, fitted_model)
            frame["outcome"] = outcome_key
            frame["specification"] = specification
            frame["model_number"] = model_number
            coefficient_frames.append(frame)

            bundle_metrics["model_%s" % model_number] = {
                "specification": specification,
                "adj_r2": float(fitted_model.rsquared_adj),
                "n": int(fitted_model.nobs)
            }

        text_chunks = []
        for model_number, specification, formula, fitted_model in zip(
            range(1, len(fitted_models) + 1),
            spec_names,
            fitted_formulas,
            fitted_models
        ):
            text_chunks.append(
                "MODEL %s: %s\nFORMULA: %s\n%s" % (
                    model_number,
                    spec_labels[specification],
                    formula,
                    fitted_model.summary().as_text()
                )
            )
        outcome_text_path = reports_dir / ("models_%s.txt" % outcome_key)
        outcome_text_path.write_text("\n\n".join(text_chunks))

        stargazer = Stargazer(fitted_models)
        stargazer.title("%s (Grouped by Common Dependent Variable)" % outcome_label)
        stargazer_html_path = reports_dir / ("stargazer_%s.html" % outcome_key)
        stargazer_tex_path = reports_dir / ("stargazer_%s.tex" % outcome_key)
        stargazer_html_path.write_text(stargazer.render_html())
        stargazer_tex_path.write_text(stargazer.render_latex())

        return [str(outcome_text_path), str(stargazer_html_path), str(stargazer_tex_path)], coefficient_frames, bundle_metrics

    judged_labels = {
        "judged_fairness": "Judged Fairness",
        "judged_cooperative_intent": "Judged Cooperative Intent",
        "judged_strategic_constructiveness": "Judged Strategic Constructiveness",
        "judged_leadership": "Judged Leadership",
        "judged_toxicity": "Judged Toxicity"
    }
    outcome_specs = [
        {
            "outcome_key": "public_goods_action_share",
            "outcome_label": "Public-Goods Action Share",
            "outcome_column": "action_share",
            "frame_name": "public_goods",
            "subset_columns": [],
            "pooled": False,
            "include_reward_sd": False
        },
        {
            "outcome_key": "negotiation_fairness",
            "outcome_label": "Negotiation Fairness",
            "outcome_column": "fairness_score",
            "frame_name": "negotiation",
            "subset_columns": [],
            "pooled": False,
            "include_reward_sd": False
        },
        {
            "outcome_key": "pooled_cooperation",
            "outcome_label": "Pooled Cooperation",
            "outcome_column": "cooperative",
            "frame_name": "pooled",
            "subset_columns": ["cooperative"],
            "pooled": True,
            "include_reward_sd": False
        },
        {
            "outcome_key": "pooled_reward",
            "outcome_label": "Pooled Reward",
            "outcome_column": "reward",
            "frame_name": "pooled",
            "subset_columns": ["reward", "reward_sd"],
            "pooled": True,
            "include_reward_sd": True
        }
    ]
    for metric, label in judged_labels.items():
        outcome_specs.append(
            {
                "outcome_key": metric,
                "outcome_label": label,
                "outcome_column": metric,
                "frame_name": "pooled",
                "subset_columns": [metric],
                "pooled": True,
                "include_reward_sd": False
            }
        )

    outcome_formula_map: dict[str, list[str]] = {}
    for outcome_spec in outcome_specs:
        outcome_formula_map[str(outcome_spec["outcome_key"])] = [
            build_formula(
                outcome=str(outcome_spec["outcome_column"]),
                pooled=bool(outcome_spec["pooled"]),
                include_reward_sd=bool(outcome_spec["include_reward_sd"]),
                specification=specification
            )
            for specification in spec_names
        ]

    report_paths = []
    all_coefficient_frames = []
    metrics_by_outcome: dict[str, dict[str, dict[str, float | int]]] = {}
    for outcome_spec in outcome_specs:
        frame_name = str(outcome_spec["frame_name"])
        if frame_name == "public_goods":
            outcome_df = public_goods_df
        elif frame_name == "negotiation":
            outcome_df = negotiation_df
        else:
            subset_columns = [str(column) for column in outcome_spec["subset_columns"]]
            outcome_df = pooled_df.dropna(subset=subset_columns).copy()

        outcome_report_paths, outcome_frames, outcome_metrics = fit_outcome_bundle(
            outcome_key=str(outcome_spec["outcome_key"]),
            outcome_label=str(outcome_spec["outcome_label"]),
            outcome_df=outcome_df,
            formulas=outcome_formula_map[str(outcome_spec["outcome_key"])]
        )
        report_paths.extend(outcome_report_paths)
        all_coefficient_frames.extend(outcome_frames)
        metrics_by_outcome[str(outcome_spec["outcome_key"])] = outcome_metrics

    coefficients_df = pd.concat(all_coefficient_frames, ignore_index=True)
    coefficients_path = tables_dir / "regression_coefficients.csv"
    coefficients_df.to_csv(coefficients_path, index=False)

    metrics = {
        "by_outcome": metrics_by_outcome,
        "specifications": spec_names
    }
    return report_paths, coefficients_df, metrics


def make_core_plots(
    plots_dir: Path,
    round_df: pd.DataFrame,
    action_df: pd.DataFrame
) -> list[str]:
    plot_paths = []
    context_conditions = sorted(round_df["context_condition"].unique().tolist())
    color_map = plt.get_cmap("tab10")
    condition_colors = {condition: color_map(index % 10) for index, condition in enumerate(context_conditions)}

    plt.figure(figsize=(9, 5))
    for context_condition in context_conditions:
        public_goods_rounds = round_df[
            (round_df["scenario"] == "public_goods")
            & (round_df["context_condition"] == context_condition)
        ].sort_values("round")
        color = condition_colors[context_condition]
        plt.plot(public_goods_rounds["round"], public_goods_rounds["action_mean"], marker="o", color=color, label=context_condition)
        plt.fill_between(public_goods_rounds["round"], public_goods_rounds["action_ci_low"], public_goods_rounds["action_ci_high"], alpha=0.12, color=color)

    plt.xlabel("Round")
    plt.ylabel("Contribution share")
    plt.title("Public-Goods Contribution Dynamics")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(3, len(context_conditions)), frameon=False)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    public_goods_plot = plots_dir / "public_goods_contribution_dynamics.pdf"
    plt.savefig(public_goods_plot, dpi=300)
    plt.close()
    plot_paths.append(str(public_goods_plot))

    plt.figure(figsize=(9, 5))
    for context_condition in context_conditions:
        negotiation_rounds = round_df[
            (round_df["scenario"] == "negotiation")
            & (round_df["context_condition"] == context_condition)
        ].sort_values("round")
        color = condition_colors[context_condition]
        plt.plot(negotiation_rounds["round"], negotiation_rounds["fairness_mean"], marker="o", color=color, label=context_condition)
        plt.fill_between(negotiation_rounds["round"], negotiation_rounds["fairness_ci_low"], negotiation_rounds["fairness_ci_high"], alpha=0.12, color=color)
    plt.xlabel("Round")
    plt.ylabel("Fairness score")
    plt.title("Negotiation Fairness Dynamics")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(3, len(context_conditions)), frameon=False)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    negotiation_plot = plots_dir / "negotiation_fairness_dynamics.pdf"
    plt.savefig(negotiation_plot, dpi=300)
    plt.close()
    plot_paths.append(str(negotiation_plot))

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    scenario_titles = {"public_goods": "Public Goods", "negotiation": "Negotiation"}
    for row_index, scenario in enumerate(["public_goods", "negotiation"]):
        welfare_axis = axes[row_index, 0]
        sd_axis = axes[row_index, 1]
        for context_condition in context_conditions:
            color = condition_colors[context_condition]
            scenario_rounds = round_df[
                (round_df["scenario"] == scenario)
                & (round_df["context_condition"] == context_condition)
            ].sort_values("round")
            welfare_axis.plot(scenario_rounds["round"], scenario_rounds["welfare_mean"], marker="o", color=color)
            welfare_axis.fill_between(scenario_rounds["round"], scenario_rounds["welfare_ci_low"], scenario_rounds["welfare_ci_high"], alpha=0.12, color=color)
            sd_axis.plot(scenario_rounds["round"], scenario_rounds["sd_mean"], marker="o", color=color)
            sd_axis.fill_between(scenario_rounds["round"], scenario_rounds["sd_ci_low"], scenario_rounds["sd_ci_high"], alpha=0.12, color=color)
        welfare_axis.set_title(f"{scenario_titles[scenario]} Welfare")
        sd_axis.set_title(f"{scenario_titles[scenario]} Inequality")
        welfare_axis.set_ylabel("Welfare")
        sd_axis.set_ylabel("Reward SD")
    axes[1, 0].set_xlabel("Round")
    axes[1, 1].set_xlabel("Round")
    condition_handles = [Line2D([0], [0], color=condition_colors[condition], marker="o", linewidth=2, label=condition) for condition in context_conditions]
    figure.legend(handles=condition_handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=min(3, len(context_conditions)), frameon=False, title="Context condition")
    figure.tight_layout(rect=[0, 0.06, 1, 1])
    welfare_plot = plots_dir / "welfare_and_inequality.pdf"
    figure.savefig(welfare_plot, dpi=300)
    plt.close(figure)
    plot_paths.append(str(welfare_plot))

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row_index, scenario in enumerate(["public_goods", "negotiation"]):
        coop_axis = axes[row_index, 0]
        pressure_axis = axes[row_index, 1]
        for context_condition in context_conditions:
            color = condition_colors[context_condition]
            scenario_rounds = round_df[
                (round_df["scenario"] == scenario)
                & (round_df["context_condition"] == context_condition)
            ].sort_values("round")
            coop_axis.plot(scenario_rounds["round"], scenario_rounds["cooperative_rate"], marker="o", color=color)
            coop_axis.fill_between(scenario_rounds["round"], scenario_rounds["cooperative_ci_low"], scenario_rounds["cooperative_ci_high"], alpha=0.12, color=color)

            if scenario == "public_goods":
                pressure = action_df[
                    (action_df["scenario"] == "public_goods")
                    & (action_df["context_condition"] == context_condition)
                ].groupby("round", as_index=False).agg(
                    event_rate=("catastrophe", "mean"),
                    intensity=("effective_catastrophe_risk", "mean")
                )
                pressure_axis.plot(pressure["round"], pressure["event_rate"], color=color, marker="o", linestyle="-")
                pressure_axis.plot(pressure["round"], pressure["intensity"], color=color, marker="s", linestyle="--")
            else:
                pressure = action_df[
                    (action_df["scenario"] == "negotiation")
                    & (action_df["context_condition"] == context_condition)
                ].groupby("round", as_index=False).agg(
                    event_rate=("disagreement", "mean"),
                    intensity=("effective_disagreement_penalty", "mean")
                )
                scaled_intensity = pressure["intensity"] / float(pressure["intensity"].max())
                pressure_axis.plot(pressure["round"], pressure["event_rate"], color=color, marker="o", linestyle="-")
                pressure_axis.plot(pressure["round"], scaled_intensity, color=color, marker="s", linestyle="--")

        coop_axis.set_title(f"{scenario_titles[scenario]} Cooperation")
        pressure_axis.set_title(f"{scenario_titles[scenario]} Pressure")
        coop_axis.set_ylabel("Cooperation rate")
        pressure_axis.set_ylabel("Rate or scaled intensity")

    axes[1, 0].set_xlabel("Round")
    axes[1, 1].set_xlabel("Round")
    condition_handles = [Line2D([0], [0], color=condition_colors[condition], marker="o", linewidth=2, label=condition) for condition in context_conditions]
    metric_handles = [
        Line2D([0], [0], color="black", marker="o", linestyle="-", linewidth=2, label="Event rate"),
        Line2D([0], [0], color="black", marker="s", linestyle="--", linewidth=2, label="Risk or penalty")
    ]
    figure.legend(handles=condition_handles, loc="upper center", bbox_to_anchor=(0.34, 0.02), ncol=min(3, len(context_conditions)), frameon=False, title="Context condition")
    figure.legend(handles=metric_handles, loc="upper center", bbox_to_anchor=(0.82, 0.02), ncol=2, frameon=False, title="Line type")
    figure.tight_layout(rect=[0, 0.08, 1, 1])
    pressure_plot = plots_dir / "cooperation_and_pressure_dynamics.pdf"

    figure.savefig(pressure_plot, dpi=300)
    plt.close(figure)

    plot_paths.append(str(pressure_plot))

    return plot_paths


def make_trait_plots(plots_dir: Path, persona_df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    trait_specs = [
        ("trait_O", "Openness"),
        ("trait_C", "Conscientiousness"),
        ("trait_E", "Extraversion"),
        ("trait_A", "Agreeableness"),
        ("trait_N", "Neuroticism")
    ]
    plot_paths = []
    rows = []

    conditions = sorted(persona_df["context_condition"].unique().tolist())
    color_map = plt.get_cmap("tab10")
    condition_colors = {condition: color_map(index % 10) for index, condition in enumerate(conditions)}

    for trait_key, trait_name in trait_specs:
        plt.figure(figsize=(12, 5))
        contrib_axis = plt.subplot(1, 2, 1)
        fair_axis = plt.subplot(1, 2, 2)

        for condition in conditions:
            subset = persona_df[persona_df["context_condition"] == condition]
            x_values = subset[trait_key].tolist()
            y_contrib = subset["contribution_mean"].tolist()
            y_fairness = subset["fairness_mean"].tolist()
            corr_contrib = correlation(x_values, y_contrib)
            corr_fairness = correlation(x_values, y_fairness)
            rows.append(
                {
                    "trait": trait_key,
                    "trait_name": trait_name,
                    "context_condition": condition,
                    "corr_contribution": float(corr_contrib),
                    "corr_fairness": float(corr_fairness)
                }
            )

            color = condition_colors[condition]

            contrib_axis.scatter(x_values, y_contrib, alpha=0.3, color=color, label=condition)
            slope_contrib, intercept_contrib = statistics.linear_regression(x_values, y_contrib)
            sorted_x = sorted(x_values)
            fitted_contrib = [intercept_contrib + slope_contrib * x for x in sorted_x]
            contrib_axis.plot(sorted_x, fitted_contrib, color=color, linewidth=2)

            fair_axis.scatter(x_values, y_fairness, alpha=0.3, color=color, label=condition)
            slope_fairness, intercept_fairness = statistics.linear_regression(x_values, y_fairness)
            fitted_fairness = [intercept_fairness + slope_fairness * x for x in sorted_x]
            fair_axis.plot(sorted_x, fitted_fairness, color=color, linewidth=2)

        contrib_axis.set_xlabel(trait_name)
        contrib_axis.set_ylabel("Mean public-goods contribution")
        contrib_axis.set_title(f"{trait_name} vs Contribution")

        fair_axis.set_xlabel(trait_name)
        fair_axis.set_ylabel("Mean negotiation fairness")
        fair_axis.set_title(f"{trait_name} vs Fairness")

        handles = [Line2D([0], [0], color=condition_colors[condition], lw=2, label=condition) for condition in conditions]
        plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(-0.1, -0.15), ncol=len(conditions), frameon=False)
        plt.tight_layout(rect=[0, 0.08, 1, 1])

        trait_plot = plots_dir / f"{trait_key}_relationships.pdf"
        plt.savefig(trait_plot, dpi=300)
        plt.close()
        plot_paths.append(str(trait_plot))

    return plot_paths, pd.DataFrame(rows)



def make_distribution_plots(plots_dir: Path, action_df: pd.DataFrame) -> list[str]:
    plot_paths = []
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    public_goods_actions = action_df[action_df["scenario"] == "public_goods"]["action_share"]
    plt.hist(public_goods_actions, bins=20, alpha=0.8)
    plt.xlabel("Contribution share")
    plt.ylabel("Count")
    plt.title("Public-Goods Action Distribution")
    plt.subplot(1, 2, 2)
    negotiation_actions = action_df[action_df["scenario"] == "negotiation"]["action_share"]
    plt.hist(negotiation_actions, bins=20, alpha=0.8)
    plt.xlabel("Demand share")
    plt.ylabel("Count")
    plt.title("Negotiation Demand Distribution")
    plt.tight_layout()
    distribution_plot = plots_dir / "action_distributions.pdf"
    plt.savefig(distribution_plot, dpi=300)
    plt.close()
    plot_paths.append(str(distribution_plot))
    return plot_paths


def make_condition_comparison(
    plots_dir: Path,
    tables_dir: Path,
    action_df: pd.DataFrame,
    step_df: pd.DataFrame,
    cfg: DictConfig
) -> tuple[list[str], str]:
    confidence = float(cfg.analysis.confidence_level)
    bootstrap_samples = int(cfg.analysis.bootstrap_samples)

    rows = []
    for scenario in ["public_goods", "negotiation"]:
        for context_condition in sorted(action_df["context_condition"].unique().tolist()):
            action_subset = action_df[
                (action_df["scenario"] == scenario)
                & (action_df["context_condition"] == context_condition)
            ]
            step_subset = step_df[
                (step_df["scenario"] == scenario)
                & (step_df["context_condition"] == context_condition)
            ]
            action_values = action_subset["action_share"].tolist()
            coop_values = action_subset["cooperative"].tolist()
            fairness_values = action_subset["fairness_score"].tolist()
            welfare_values = step_subset["welfare"].tolist()
            rows.append(
                {
                    "scenario": scenario,
                    "context_condition": context_condition,
                    "action_mean": float(statistics.fmean(action_values)),
                    "coop_mean": float(statistics.fmean(coop_values)),
                    "fairness_mean": float(statistics.fmean(fairness_values)),
                    "welfare_mean": float(statistics.fmean(welfare_values)),
                    "action_ci_low": float(bootstrap_ci(action_subset, "action_share", "group_id", confidence, bootstrap_samples, 7001)[0]),
                    "action_ci_high": float(bootstrap_ci(action_subset, "action_share", "group_id", confidence, bootstrap_samples, 7001)[1]),
                    "coop_ci_low": float(bootstrap_ci(action_subset, "cooperative", "group_id", confidence, bootstrap_samples, 7002)[0]),
                    "coop_ci_high": float(bootstrap_ci(action_subset, "cooperative", "group_id", confidence, bootstrap_samples, 7002)[1]),
                    "fairness_ci_low": float(bootstrap_ci(action_subset, "fairness_score", "group_id", confidence, bootstrap_samples, 7003)[0]),
                    "fairness_ci_high": float(bootstrap_ci(action_subset, "fairness_score", "group_id", confidence, bootstrap_samples, 7003)[1]),
                    "welfare_ci_low": float(bootstrap_ci(step_subset, "welfare", "group_id", confidence, bootstrap_samples, 7004)[0]),
                    "welfare_ci_high": float(bootstrap_ci(step_subset, "welfare", "group_id", confidence, bootstrap_samples, 7004)[1])
                }
            )

    summary_df = pd.DataFrame(rows)
    table_path = tables_dir / "condition_comparison.csv"
    summary_df.to_csv(table_path, index=False)

    plot_paths = []
    for scenario in ["public_goods", "negotiation"]:
        scenario_df = summary_df[summary_df["scenario"] == scenario].sort_values("context_condition")
        labels = scenario_df["context_condition"].tolist()
        x_values = list(range(len(labels)))

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.errorbar(
            x_values,
            scenario_df["action_mean"],
            yerr=[
                scenario_df["action_mean"] - scenario_df["action_ci_low"],
                scenario_df["action_ci_high"] - scenario_df["action_mean"]
            ],
            fmt="o",
            capsize=4
        )
        plt.xticks(x_values, labels, rotation=20)
        plt.ylabel("Action share")
        plt.title(f"{scenario.replace('_', ' ').title()} Action by Condition")

        plt.subplot(1, 2, 2)
        plt.errorbar(
            x_values,
            scenario_df["coop_mean"],
            yerr=[
                scenario_df["coop_mean"] - scenario_df["coop_ci_low"],
                scenario_df["coop_ci_high"] - scenario_df["coop_mean"]
            ],
            fmt="o",
            capsize=4
        )
        plt.xticks(x_values, labels, rotation=20)
        plt.ylabel("Cooperation rate")
        plt.title(f"{scenario.replace('_', ' ').title()} Cooperation by Condition")
        plt.tight_layout()

        plot_path = plots_dir / f"{scenario}_condition_comparison.pdf"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        plot_paths.append(str(plot_path))

    return plot_paths, str(table_path)


def make_judged_metric_plots(
    plots_dir: Path,
    action_df: pd.DataFrame,
    calibration_table: pd.DataFrame,
    cfg: DictConfig
) -> list[str]:
    confidence = float(cfg.analysis.confidence_level)
    bootstrap_samples = int(cfg.analysis.bootstrap_samples)
    metrics = [
        ("judged_fairness", "Judged Fairness"),
        ("judged_cooperative_intent", "Judged Cooperative Intent"),
        ("judged_strategic_constructiveness", "Judged Strategic Constructiveness"),
        ("judged_leadership", "Judged Leadership"),
        ("judged_toxicity", "Judged Toxicity")
    ]
    conditions = sorted(action_df["context_condition"].unique().tolist())
    color_map = plt.get_cmap("tab10")
    condition_colors = {condition: color_map(index % 10) for index, condition in enumerate(conditions)}

    plot_paths = []

    round_rows = []
    for scenario in ["public_goods", "negotiation", "collaboration"]:
        for context_condition in conditions:
            subset = action_df[
                (action_df["scenario"] == scenario)
                & (action_df["context_condition"] == context_condition)
            ]
            rounds = sorted(subset["round"].unique().tolist())
            for round_number in rounds:
                round_subset = subset[subset["round"] == round_number]
                row = {
                    "scenario": scenario,
                    "context_condition": context_condition,
                    "round": int(round_number)
                }
                for metric_key, _ in metrics:
                    raw_mean = float(statistics.fmean(round_subset[metric_key].tolist()))
                    ci_low_raw, ci_high_raw = bootstrap_ci(round_subset, metric_key, "group_id", confidence, bootstrap_samples, seed=10000 + int(round_number))
                    bootstrap_var_p = ((ci_high_raw - ci_low_raw) / (2 * 1.96)) ** 2

                    q1 = float(cfg.calibration.judged_metrics[metric_key].q1)
                    q0 = float(cfg.calibration.judged_metrics[metric_key].q0)
                    n_pos = int(cfg.calibration.judged_metrics[metric_key].n_calib_pos)
                    n_neg = int(cfg.calibration.judged_metrics[metric_key].n_calib_neg)
                    ci_low, ci_high = bias_corrected_ci(
                        raw_mean, q1, q0, len(round_subset), n_pos, n_neg, var_p_override=bootstrap_var_p
                    )
                    row[f"{metric_key}_mean"] = float(bias_corrected_score(raw_mean, q1, q0))
                    row[f"{metric_key}_ci_low"] = float(ci_low)
                    row[f"{metric_key}_ci_high"] = float(ci_high)
                round_rows.append(row)

    judged_round_df = pd.DataFrame(round_rows)

    figure, axes = plt.subplots(3, 5, figsize=(20, 12), sharex=True, sharey=True)
    scenario_titles = {"public_goods": "Public Goods", "negotiation": "Negotiation", "collaboration": "Collaboration"}
    for row_index, scenario in enumerate(["public_goods", "negotiation", "collaboration"]):
        for column_index, (metric_key, metric_label) in enumerate(metrics):
            axis = axes[row_index, column_index]
            for condition in conditions:
                subset = judged_round_df[
                    (judged_round_df["scenario"] == scenario)
                    & (judged_round_df["context_condition"] == condition)
                ].sort_values("round")
                color = condition_colors[condition]
                axis.plot(subset["round"], subset[f"{metric_key}_mean"], marker="o", color=color)
                axis.fill_between(
                    subset["round"],
                    subset[f"{metric_key}_ci_low"],
                    subset[f"{metric_key}_ci_high"],
                    alpha=0.12,
                    color=color
                )
            axis.set_title(f"{scenario_titles[scenario]}: {metric_label}")
            if column_index == 0:
                axis.set_ylabel("Rate")
            if row_index == 2:
                axis.set_xlabel("Round")

    handles = [Line2D([0], [0], color=condition_colors[condition], marker="o", linewidth=2, label=condition) for condition in conditions]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=min(3, len(conditions)), frameon=False, title="Context condition")
    figure.tight_layout(rect=[0, 0.06, 1, 1])
    round_plot_path = plots_dir / "judged_metrics_by_round.pdf"
    figure.savefig(round_plot_path, dpi=300)
    plt.close(figure)
    plot_paths.append(str(round_plot_path))

    figure, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
    for axis_index, (metric_key, metric_label) in enumerate(metrics):
        axis = axes[axis_index]
        metric_rows = calibration_table[calibration_table["metric"] == metric_key]
        metric_rows = metric_rows.copy()
        metric_rows["label"] = metric_rows["scenario"] + " | " + metric_rows["context_condition"]
        metric_rows = metric_rows.sort_values("label").reset_index(drop=True)
        x_positions = list(range(len(metric_rows)))
        axis.errorbar(
            [position - 0.1 for position in x_positions],
            metric_rows["raw_rate"],
            fmt="o",
            color="#4C72B0",
            label="Raw judged rate"
        )
        axis.errorbar(
            [position + 0.1 for position in x_positions],
            metric_rows["bias_corrected_rate"],
            yerr=[
                metric_rows["bias_corrected_rate"] - metric_rows["ci_low"],
                metric_rows["ci_high"] - metric_rows["bias_corrected_rate"]
            ],
            fmt="o",
            color="#DD8452",
            capsize=4,
            label="Bias-corrected judged rate"
        )
        axis.set_title(metric_label)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(metric_rows["label"], rotation=30, ha="right")
        if axis_index == 0:
            axis.set_ylabel("Rate")
    legend_handles = [
        Line2D([0], [0], color="#4C72B0", marker="o", linestyle="None", label="Raw judged rate"),
        Line2D([0], [0], color="#DD8452", marker="o", linestyle="None", label="Bias-corrected judged rate (95% CI)")
    ]
    figure.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    figure.tight_layout(rect=[0, 0.08, 1, 1])
    corrected_plot_path = plots_dir / "judged_metrics_bias_correction.pdf"

    figure.savefig(corrected_plot_path, dpi=300)
    plt.close(figure)

    plot_paths.append(str(corrected_plot_path))

    return plot_paths


def make_group_rank_plots(plots_dir: Path, action_df: pd.DataFrame, tables_dir: Path) -> tuple[list[str], list[str]]:
    plot_paths = []
    table_paths = []
    conditions = sorted(action_df["context_condition"].unique().tolist())
    for context_condition in conditions:
        public_goods_group = action_df[
            (action_df["scenario"] == "public_goods")
            & (action_df["context_condition"] == context_condition)
        ].groupby("group_id", as_index=False).agg(
            group_mean=("action_share", "mean"),
            cooperative_rate=("cooperative", "mean"),
            n_rows=("agent_id", "count")
        ).sort_values("group_mean", ascending=False).reset_index(drop=True)
        public_goods_group["rank"] = public_goods_group.index + 1
        public_goods_group["context_condition"] = context_condition
        public_goods_table = tables_dir / f"public_goods_group_rankings_{context_condition}.csv"
        public_goods_group.to_csv(public_goods_table, index=False)
        table_paths.append(str(public_goods_table))

        plt.figure(figsize=(10, 4))
        labels_pg = [f"G{int(group_id)}" for group_id in public_goods_group["group_id"]]
        plt.bar(labels_pg, public_goods_group["group_mean"])
        plt.xlabel("Group ID (sorted by mean contribution)")
        plt.ylabel("Mean contribution share")
        plt.title(f"Public-Goods Group Means (sorted): {context_condition.replace('_', ' ').title().replace('Bfi', 'BFI').replace('And', '&')}")
        plt.xticks(rotation=90)
        plt.tight_layout()
        public_goods_plot = plots_dir / f"public_goods_group_ranked_means_{context_condition}.pdf"
        plt.savefig(public_goods_plot, dpi=300)
        plt.close()
        plot_paths.append(str(public_goods_plot))

        negotiation_group = action_df[
            (action_df["scenario"] == "negotiation")
            & (action_df["context_condition"] == context_condition)
        ].groupby("group_id", as_index=False).agg(
            group_mean=("fairness_score", "mean"),
            cooperative_rate=("cooperative", "mean"),
            n_rows=("agent_id", "count")
        ).sort_values("group_mean", ascending=False).reset_index(drop=True)
        negotiation_group["rank"] = negotiation_group.index + 1
        negotiation_group["context_condition"] = context_condition
        negotiation_table = tables_dir / f"negotiation_group_rankings_{context_condition}.csv"
        negotiation_group.to_csv(negotiation_table, index=False)
        table_paths.append(str(negotiation_table))

        plt.figure(figsize=(10, 4))
        labels_neg = [f"G{int(group_id)}" for group_id in negotiation_group["group_id"]]
        plt.bar(labels_neg, negotiation_group["group_mean"])
        plt.xlabel("Group ID (sorted by mean fairness)")
        plt.ylabel("Mean fairness score")
        plt.title(f"Negotiation Group Means (sorted): {context_condition.replace('_', ' ').title().replace('Bfi', 'BFI').replace('And', '&')}")
        plt.xticks(rotation=90)
        plt.tight_layout()
        negotiation_plot = plots_dir / f"negotiation_group_ranked_means_{context_condition}.pdf"

        plt.savefig(negotiation_plot, dpi=300)
        plt.close()

        plot_paths.append(str(negotiation_plot))

    return plot_paths, table_paths


def build_dialogues(action_df: pd.DataFrame, tables_dir: Path, cfg: DictConfig, top_n: int = 40) -> str:
    group_size = int(cfg.simulation.group_size)
    interesting = action_df.copy()
    interesting["distance_from_fair"] = 0.0

    public_goods_mask = interesting["scenario"] == "public_goods"
    public_goods_fair_share = float(cfg.simulation.public_goods.threshold) / group_size
    interesting.loc[public_goods_mask, "distance_from_fair"] = (interesting.loc[public_goods_mask, "action_raw"] - public_goods_fair_share).abs()

    negotiation_mask = interesting["scenario"] == "negotiation"
    interesting.loc[negotiation_mask, "distance_from_fair"] = (
        interesting.loc[negotiation_mask, "action_raw"] - interesting.loc[negotiation_mask, "effective_total_resource"] / group_size
    ).abs()

    sample = interesting.sort_values(
        ["scenario", "context_condition", "distance_from_fair", "group_id", "round"],
        ascending=[True, True, False, True, True]
    ).groupby(["scenario", "context_condition"], as_index=False).head(top_n)
    sample_path = tables_dir / "dialogue_samples.csv"
    sample[
        [
            "scenario",
            "context_condition",
            "group_id",
            "round",
            "agent_id",
            "action_raw",
            "distance_from_fair",
            "reward",
            "observation",
            "prompt_user",
            "action_text",
            "gm_events",
            "judged_fairness",
            "judged_cooperative_intent",
            "judged_strategic_constructiveness",
            "judge_rationale"
        ]
    ].to_csv(sample_path, index=False)
    return str(sample_path)


def make_convergence_plots(plots_dir: Path, round_df: pd.DataFrame) -> list[str]:
    plot_paths = []
    collab_round_df = round_df[round_df["scenario"] == "collaboration"]

    conditions = sorted(collab_round_df["context_condition"].unique().tolist())

    plt.figure(figsize=(6, 4))
    for condition in conditions:
        subset = collab_round_df[collab_round_df["context_condition"] == condition].sort_values("round")
        plt.plot(
            subset["round"],
            subset["consensus_variance_mean"],
            marker="o",
            label=condition.replace("_", " ").title().replace("Bfi", "BFI").replace("And", "&")
        )
        plt.fill_between(
            subset["round"],
            subset["consensus_variance_ci_low"],
            subset["consensus_variance_ci_high"],
            alpha=0.2
        )

    plt.title("Collaboration Consensus Convergence")
    plt.xlabel("Round")
    plt.ylabel("Mean Embedding Variance (Lower = Higher Consensus)")
    plt.xticks(sorted(collab_round_df["round"].unique().tolist()))
    plt.legend()
    plt.tight_layout()
    plot_path = plots_dir / "collaboration_consensus_convergence.pdf"

    plt.savefig(plot_path, dpi=300)
    plt.close()

    plot_paths.append(str(plot_path))

    return plot_paths


def truncate_text_for_label(
    text: str,
    max_chars: int
) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def build_archetype_representatives(
    action_df: pd.DataFrame,
    top_k: int
) -> pd.DataFrame:
    representative_columns = [
        "archetype",
        "archetype_label",
        "representative_rank",
        "archetype_centroid_distance",
        "embedding_index",
        "scenario",
        "context_condition",
        "group_id",
        "round",
        "step",
        "agent_id",
        "action_raw",
        "action_share",
        "cooperative",
        "fairness_score",
        "reward",
        "embedding_text",
        "action_text"
    ]
    representative_chunks = []
    for archetype_id in sorted(action_df["archetype"].unique().tolist()):
        cluster_subset = action_df[action_df["archetype"] == archetype_id].copy()
        cluster_subset = cluster_subset.sort_values(
            [
                "archetype_centroid_distance",
                "scenario",
                "context_condition",
                "group_id",
                "round",
                "step"
            ],
            ascending=[True, True, True, True, True, True]
        ).head(top_k)
        cluster_subset["representative_rank"] = np.arange(1, len(cluster_subset) + 1)
        cluster_subset["archetype_label"] = cluster_subset["archetype"].apply(
            lambda value: "Archetype %s" % int(value)
        )
        representative_chunks.append(cluster_subset[representative_columns])

    return pd.concat(representative_chunks, ignore_index=True)


def build_archetype_representatives_balanced_by_scenario(
    action_df: pd.DataFrame,
    top_k: int
) -> pd.DataFrame:
    representative_columns = [
        "archetype",
        "archetype_label",
        "representative_rank",
        "selection_mode",
        "target_scenario",
        "archetype_centroid_distance",
        "embedding_index",
        "scenario",
        "context_condition",
        "group_id",
        "round",
        "step",
        "agent_id",
        "action_raw",
        "action_share",
        "cooperative",
        "fairness_score",
        "reward",
        "embedding_text",
        "action_text"
    ]
    scenario_priority = ["public_goods", "negotiation", "collaboration"]
    observed_scenarios = sorted(action_df["scenario"].astype(str).unique().tolist())
    scenario_order = scenario_priority + [scenario for scenario in observed_scenarios if scenario not in scenario_priority]
    scenario_rank = {scenario: index for index, scenario in enumerate(scenario_order)}

    archetype_chunks = []
    for archetype_id in sorted(action_df["archetype"].unique().tolist()):
        cluster_subset = action_df[action_df["archetype"] == archetype_id].copy()
        cluster_subset = cluster_subset.sort_values(
            [
                "archetype_centroid_distance",
                "scenario",
                "context_condition",
                "group_id",
                "round",
                "step"
            ],
            ascending=[True, True, True, True, True, True]
        )

        selected_df = cluster_subset.groupby("scenario", as_index=False).head(1).copy()
        selected_df["scenario_rank"] = selected_df["scenario"].map(scenario_rank)
        selected_df = selected_df.sort_values(
            [
                "scenario_rank",
                "archetype_centroid_distance",
                "scenario",
                "context_condition",
                "group_id",
                "round",
                "step"
            ],
            ascending=[True, True, True, True, True, True, True]
        ).head(top_k)
        selected_df["selection_mode"] = "scenario_balanced"
        selected_df["target_scenario"] = selected_df["scenario"].astype(str)

        fallback_needed = top_k - len(selected_df)
        fallback_subset = cluster_subset[~cluster_subset["embedding_index"].isin(selected_df["embedding_index"].tolist())].head(fallback_needed).copy()
        fallback_subset["selection_mode"] = "fallback_global"
        fallback_subset["target_scenario"] = "fallback"

        selected_df = pd.concat(
            [
                selected_df.drop(columns=["scenario_rank"]),
                fallback_subset
            ],
            ignore_index=True
        ).head(top_k)
        selected_df["representative_rank"] = np.arange(1, len(selected_df) + 1)
        selected_df["archetype_label"] = selected_df["archetype"].apply(
            lambda value: "Archetype %s" % int(value)
        )
        archetype_chunks.append(selected_df[representative_columns])

    return pd.concat(archetype_chunks, ignore_index=True)


def build_archetype_landscape(
    action_df: pd.DataFrame,
    cfg: DictConfig
) -> pd.DataFrame:
    landscape_columns = [
        "embedding_index",
        "scenario",
        "context_condition",
        "group_id",
        "round",
        "step",
        "agent_id",
        "archetype",
        "tsne_x",
        "tsne_y",
        "embedding_text",
        "action_text"
    ]
    embeddings = np.vstack(action_df["archetype_embedding"].to_numpy())
    tsne = TSNE(
        n_components=2,
        perplexity=float(cfg.analysis.archetypes.tsne_perplexity),
        random_state=int(cfg.analysis.archetypes.random_state),
        init="pca",
        learning_rate="auto",
        max_iter=int(cfg.analysis.archetypes.tsne_iterations)
    )
    coords = tsne.fit_transform(embeddings)

    landscape_df = action_df[
        [
            "embedding_index",
            "scenario",
            "context_condition",
            "group_id",
            "round",
            "step",
            "agent_id",
            "archetype",
            "embedding_text",
            "action_text"
        ]
    ].copy()
    landscape_df["tsne_x"] = coords[:, 0]
    landscape_df["tsne_y"] = coords[:, 1]
    return landscape_df[landscape_columns]


def make_archetype_plots(
    plots_dir: Path,
    action_df: pd.DataFrame,
    landscape_df: pd.DataFrame,
    representatives_df: pd.DataFrame,
    cfg: DictConfig
) -> list[str]:
    plot_paths = []

    archetype_counts = action_df.groupby(["context_condition", "archetype"]).size().unstack(fill_value=0)
    archetype_pct = archetype_counts.div(archetype_counts.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(10, 6))
    archetype_pct.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="viridis")
    plt.title("Action Archetype Distribution by Context Condition")
    plt.xlabel("Context Condition")
    plt.ylabel("Percentage of Actions")
    plt.legend(title="Archetype ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    distribution_plot_path = plots_dir / "archetype_distribution.pdf"
    plt.savefig(distribution_plot_path, dpi=300)
    plt.close()
    plot_paths.append(str(distribution_plot_path))

    plt.figure(figsize=(11, 7))
    archetypes = sorted(landscape_df["archetype"].unique().tolist())
    for archetype_id in archetypes:
        subset = landscape_df[landscape_df["archetype"] == archetype_id]
        color = plt.cm.tab10(int(archetype_id) % 10)
        plt.scatter(
            subset["tsne_x"],
            subset["tsne_y"],
            s=16,
            alpha=0.55,
            color=color,
            label="Archetype %s" % int(archetype_id)
        )

    anchor_rows = representatives_df[representatives_df["representative_rank"] == 1]
    points_by_index = landscape_df.set_index("embedding_index")
    label_max_chars = int(cfg.analysis.archetypes.label_max_chars)
    for _, row in anchor_rows.iterrows():
        embedding_index = int(row["embedding_index"])
        point = points_by_index.loc[embedding_index]
        label_text = "A%s: %s" % (
            int(row["archetype"]),
            truncate_text_for_label(str(row["embedding_text"]), label_max_chars)
        )
        plt.annotate(
            label_text,
            (float(point["tsne_x"]), float(point["tsne_y"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "alpha": 0.8,
                "lw": 0.5
            }
        )

    plt.title("Archetype Landscape (t-SNE of Action Embeddings)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Archetype", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    landscape_plot_path = plots_dir / "archetype_landscape_tsne.pdf"
    plt.savefig(landscape_plot_path, dpi=300)
    plt.close()
    plot_paths.append(str(landscape_plot_path))

    print("\nArchetype Distribution (%):")
    print(archetype_pct.round(2).to_string())
    return plot_paths


def make_archetype_tables(
    tables_dir: Path,
    action_df: pd.DataFrame,
    archetype_selection_df: pd.DataFrame,
    representatives_df: pd.DataFrame,
    landscape_df: pd.DataFrame,
    representative_top_k: int
) -> list[str]:
    table_paths = []

    archetype_counts = action_df.groupby(["context_condition", "archetype"]).size().reset_index(name="count")
    total_counts = action_df.groupby("context_condition").size().reset_index(name="total")
    archetype_pct = archetype_counts.merge(total_counts, on="context_condition")
    archetype_pct["pct"] = archetype_pct["count"] / archetype_pct["total"] * 100

    distribution_path = tables_dir / "archetype_distribution.csv"
    archetype_pct.to_csv(distribution_path, index=False)
    table_paths.append(str(distribution_path))

    summary_df = action_df.groupby("archetype", as_index=False).agg(
        count=("archetype", "size"),
        action_share_mean=("action_share", "mean"),
        cooperative_rate=("cooperative", "mean"),
        fairness_mean=("fairness_score", "mean"),
        reward_mean=("reward", "mean")
    ).sort_values("count", ascending=False)
    summary_df["pct_total"] = summary_df["count"] / len(action_df) * 100

    scenario_mix = action_df.groupby(["archetype", "scenario"], as_index=False).size().rename(columns={"size": "count"})
    scenario_total = scenario_mix.groupby("archetype", as_index=False)["count"].sum().rename(columns={"count": "archetype_total"})
    scenario_mix = scenario_mix.merge(scenario_total, on="archetype", how="inner")
    scenario_mix["scenario_share"] = scenario_mix["count"] / scenario_mix["archetype_total"]
    dominant_scenario = scenario_mix.sort_values(["archetype", "scenario_share"], ascending=[True, False]).groupby("archetype", as_index=False).head(1)
    dominant_scenario = dominant_scenario.rename(
        columns={
            "scenario": "dominant_scenario",
            "scenario_share": "dominant_scenario_share"
        }
    )[["archetype", "dominant_scenario", "dominant_scenario_share"]]
    summary_df = summary_df.merge(dominant_scenario, on="archetype", how="left")

    summary_path = tables_dir / "archetype_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    table_paths.append(str(summary_path))

    selection_path = tables_dir / "archetype_model_selection.csv"
    archetype_selection_df.to_csv(selection_path, index=False)
    table_paths.append(str(selection_path))

    representatives_path = tables_dir / "archetype_representatives.csv"
    representatives_df.to_csv(representatives_path, index=False)
    table_paths.append(str(representatives_path))

    balanced_representatives_df = build_archetype_representatives_balanced_by_scenario(
        action_df,
        top_k=representative_top_k
    )
    balanced_representatives_path = tables_dir / "archetype_representatives_balanced.csv"
    balanced_representatives_df.to_csv(balanced_representatives_path, index=False)
    table_paths.append(str(balanced_representatives_path))

    landscape_path = tables_dir / "archetype_landscape_points.csv"
    landscape_df.to_csv(landscape_path, index=False)
    table_paths.append(str(landscape_path))

    scenario_mix_path = tables_dir / "archetype_scenario_mix.csv"
    scenario_mix.to_csv(scenario_mix_path, index=False)
    table_paths.append(str(scenario_mix_path))

    print("\nArchetype Representative Actions (scenario-balanced nearest to centroid):")
    scenario_priority = ["public_goods", "negotiation", "collaboration"]
    for archetype_id in sorted(balanced_representatives_df["archetype"].unique().tolist()):
        print("\nArchetype %s:" % int(archetype_id))
        archetype_source = action_df[action_df["archetype"] == archetype_id]
        present_scenarios = sorted(archetype_source["scenario"].astype(str).unique().tolist())
        missing_scenarios = [scenario for scenario in scenario_priority if scenario not in present_scenarios]
        print("  Missing scenarios in this cluster: %s" % ", ".join(missing_scenarios))

        subset = balanced_representatives_df[
            balanced_representatives_df["archetype"] == archetype_id
        ].sort_values("representative_rank")
        for _, row in subset.iterrows():
            selection_mode = str(row["selection_mode"])
            print(
                "  [%s] %s | %s | mode=%s | d=%.4f | %s" % (
                    int(row["representative_rank"]),
                    str(row["scenario"]),
                    str(row["context_condition"]),
                    selection_mode,
                    float(row["archetype_centroid_distance"]),
                    truncate_text_for_label(str(row["embedding_text"]), 140)
                )
            )

    return table_paths


if __name__ == "__main__":
    main()
