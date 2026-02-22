# Problem Set 2: Social Simulation Experiments

This repository runs the full social-simulation pipeline described in the main text for pset 2:
1. Generate personas with BFI-conditioned life stories
2. Simulate multi-agent interaction across three scenarios
3. Run analysis

Run these scripts from this directory:

1. Generate population:
   - `python3 generate_population.py`
2. Run simulations:
   - `python3 run_experiments.py`
3. Run analysis:
   - `python3 run_analysis.py`

`run_analysis.py` expects `simulation_manifest.json` under `output.base_dir` from the simulation step.

## Environment Setup

This project uses LiteLLM-backed model calls in all three stages. You will need provider credentials. By default we assume you will use Gemini, since we provided some credits for Google cloud. See the following configs to inspect and modify (or specify via the command line):
- `conf/agent/llm.yaml`
- `conf/analysis/default.yaml`
- `conf/population/default.yaml`

For Gemini defaults, set:
- `GEMINI_API_KEY`

Install dependencies:

```bash
conda create -n pset2 python=3.12 -y
conda activate pset2
pip install uv # we recommend uv, but you can also use pip instead
uv pip install -r requirements.txt
```

## Hydra Config

All scripts use `conf/config.yaml` with groups:
- `data`
- `population`
- `agent`
- `simulation`
- `analysis`
- `calibration`
- `output`

Default output directory is specified in:
- `conf/output/default.yaml`: `outputs/results`

Some useful overrides you might want to try:

```bash
# Faster/smaller smoke run
python run_experiments.py output.base_dir=outputs/test_run simulation.max_groups=3
python run_analysis.py output.base_dir=outputs/test_run

# Adjust cluster strategy
python run_analysis.py analysis.archetypes.selection_method=fixed analysis.archetypes.n_clusters=6
```

## Current Pipeline

### 1. Population generation (`generate_population.py`)

- Loads `data/surveys.json`
- Samples target traits, writes persona life stories, administers BFI via model calls, and computes trait scores
- Writes `data/population.json` by default

### 2. Simulation (`run_experiments.py`)

Runs all context conditions:
- `life_story_only`
- `bfi_only`
- `life_story_and_bfi`

Runs all scenarios:
- `public_goods`
- `negotiation`
- `collaboration`

Writes:
- `simulation_manifest.json`
- `simulation/step_records.jsonl`
- `simulation/dialogue_records.jsonl`
- `tables/simulation_steps.csv`
- `tables/dialogue_log.csv`

### 3. Analysis (`run_analysis.py`)

Builds action/step/persona panels, runs LLM-as-judge labels with cache, calibrates judged metrics, fits regressions, and writes a whole bunch of plots/tables/reports.

(Note the judge cache):
- `${output.base_dir}/judge_cache.jsonl`

## Output Artifacts

`summary.json` is an index of generated artifacts you can use to plan your report, iterate on your experiments, etc.

Typical outputs under `${output.base_dir}` (check if your run misses any of these):

- `summary.json`: master manifest of all produced artifacts plus run metadata, calibration assumptions, regression summary metrics, and archetype settings

- `reports/`: human-readable regression output files
  - `models_<outcome>.txt`: full model summaries for one dependent variable (`MODEL 1`, `MODEL 2`, `MODEL 3`) corresponding to agreeableness-only, BFI-only, and full specs
  - `stargazer_<outcome>.html`: formatted regression table (web-friendly) for a single dependent variable only
  - `stargazer_<outcome>.tex`: LaTeX version of the same Stargazer table for paper/report inclusion
  - `<outcome>` values include: `public_goods_action_share`, `negotiation_fairness`, `pooled_cooperation`, `pooled_reward`, and each judged outcome (`judged_fairness`, `judged_cooperative_intent`, `judged_strategic_constructiveness`, `judged_leadership`, `judged_toxicity`)

- `tables/`: machine-readable analysis data and summaries
  - `action_level.csv`: core panel at agent-step level with engineered outcomes, traits, scenario/context IDs, and text fields
  - `judge_action_level.csv`: compact agent-step table containing only judged labels and judge raw JSON
  - `step_level.csv`: step-level aggregates (welfare, reward dispersion, consensus variance)
  - `round_summary.csv`: round-level means and bootstrap CIs by scenario and context condition
  - `persona_summary.csv`: per-agent/per-condition behavioral summaries joined with trait covariates
  - `scenario_descriptives.csv`: scenario-condition descriptive averages used in top-line comparisons
  - `trait_outcome_correlations.csv`: trait-outcome correlation table used for trait relationship plots
  - `judge_calibration_table.csv`: raw vs bias-corrected judged metric rates with CI and calibration parameters
  - `condition_comparison.csv`: side-by-side condition comparison table for public goods and negotiation
  - `dialogue_samples.csv`: selected high-distance examples for qualitative inspection
  - `regression_coefficients.csv`: stacked coefficients for every fitted model with model/outcome/spec metadata
  - `public_goods_group_rankings_<context>.csv`: group-level ranking table by mean contribution within a context condition
  - `negotiation_group_rankings_<context>.csv`: group-level ranking table by mean fairness within a context condition
  - `archetype_distribution.csv`: archetype counts and percentages by context condition
  - `archetype_summary.csv`: per-archetype aggregate behavior summary plus dominant scenario share
  - `archetype_model_selection.csv`: silhouette model-selection diagnostics across candidate cluster counts
  - `archetype_representatives.csv`: closest-to-centroid examples per archetype (global nearest)
  - `archetype_representatives_balanced.csv`: closest examples per archetype with scenario-balanced selection first, then fallback nearest rows
  - `archetype_landscape_points.csv`: 2D t-SNE coordinates for each action embedding with archetype labels (for plotting/re-analysis)
  - `archetype_scenario_mix.csv`: scenario composition of each archetype (counts and shares)

- `plots/`: figures generated from the above tables
  - `public_goods_contribution_dynamics.pdf`: round dynamics of contribution share with bootstrap CI ribbons
  - `negotiation_fairness_dynamics.pdf`: round dynamics of fairness score with bootstrap CI ribbons
  - `welfare_and_inequality.pdf`: welfare and reward-SD trajectories by scenario and context condition
  - `cooperation_and_pressure_dynamics.pdf`: cooperation trajectories overlaid with scenario-specific pressure/event signals
  - `trait_<trait>_relationships.pdf`: scatter + fitted-line plots linking each trait (`O/C/E/A/N`) to contribution/fairness outcomes
  - `action_distributions.pdf`: marginal distributions of action shares for public goods and negotiation
  - `<scenario>_condition_comparison.pdf`: condition-level mean comparisons for action/cooperation in a given scenario
  - `judged_metrics_by_round.pdf`: bias-corrected judged metrics over rounds for each scenario and condition
  - `judged_metrics_bias_correction.pdf`: raw vs bias-corrected judged rates with uncertainty bars
  - `public_goods_group_ranked_means_<context>.pdf`: sorted group mean contribution bars by context
  - `negotiation_group_ranked_means_<context>.pdf`: sorted group mean fairness bars by context
  - `collaboration_consensus_convergence.pdf`: collaboration embedding-variance trajectory (lower means more consensus)
  - `archetype_distribution.pdf`: stacked archetype proportion chart by context condition
  - `archetype_landscape_tsne.pdf`: 2D embedding landscape colored by archetype with representative text annotations

## Codebase Map

- `generate_population.py`: persona generation + BFI scoring pipeline
- `run_experiments.py`: scenario rollouts and trace artifact generation
- `run_analysis.py`: panel construction, judge labeling/calibration, regression, plotting, archetypes
- `agents/`: LLM agent and prompt/memory integration
- `envs/`: public goods, negotiation, collaboration environments
- `eval/metrics.py`: welfare/variance/correlation and calibration math utilities
- `utils/`: IO and async parallel helpers
- `conf/`: Hydra config groups
- `data/`: survey schema and generated populations
