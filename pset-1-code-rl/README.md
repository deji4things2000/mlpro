# AI Agents @ Dartmouth College
## Problem Set 1, Part II: Code RL Post-Training

This project implements a GRPO-style RL post-training loop for a code-generating model using Tinker and a sandboxed Python execution environment. The goal is to train Qwen 3 4B Instruct (with LoRA) on the DeepCoder-Preview dataset to generate Python code that passes all provided tests.

## Dataset

DeepCoder-Preview combines problems from multiple sources (TACO Verified, PrimeIntellect SYNTHETIC-1, and LiveCodeBench). Each example includes a natural language description, test cases, optional starter code, and metadata.

## Reward

The environment computes a reward as:

```
r = α · r_format + r_correct
```

where `r_format` rewards correct code-block formatting and `r_correct` rewards passing tests.

## Sandbox configuration

The sandbox is compatible with LiveCodeBench utilities in `tinker_utils/lcb.py`. A base config is provided in `sandbox_config/local.yaml` (or `local.yaml` at the project root).

Start the sandbox with Docker:

```bash
docker run -d \
    -p 8080:8080 \
    -v ./sandbox_config/local.yaml:/root/sandbox/sandbox/configs/local.yaml \
    volcengine/sandbox-fusion:server-20250609
```

The environment uses the following defaults in `tinker_utils/env.py`:

```python
SANDBOX_URL = os.getenv("SANDBOX_URL", "http://localhost:8080/run_code")
SANDBOX_MAX_CONCURRENCY = int(os.getenv("SANDBOX_MAX_CONCURRENCY", "4"))
SANDBOX_CLIENT_TIMEOUT_SECONDS = int(os.getenv("SANDBOX_CLIENT_TIMEOUT_SECONDS", "6000"))
```

Override these via environment variables when launching `train.py`.

## Project structure (key modules)

- `tinker_utils/checkpoint.py`: checkpoint save/load (JSONL file `checkpoints.jsonl`).
- `tinker_utils/data.py`: prompt construction utilities.
- `tinker_utils/env.py`: `CodeEnv` (prompt rendering, code extraction, sandbox execution, reward).
- `tinker_utils/log.py`: multi-backend logging + setup helpers.
- `tinker_utils/lcb.py`: LiveCodeBench test utilities.
- `tinker_utils/renderers.py`: renderer factory, including `qwen3_instruct`.
- `tinker_utils/qwen.py`: Qwen-specific renderers.

## Training loop (train.py)

The training pipeline:

- Loads config via a `@chz.chz` `Config` class.
- Uses `Qwen/Qwen3-4B-Instruct-2507` and the `qwen3_instruct` renderer.
- Creates a Tinker `ServiceClient`, a LoRA `TrainingClient`, and a `SamplingClient`.
- Loads DeepCoder-Preview splits:
  - Train: `primeintellect`, `taco`, `lcbv5` (split `train`)
  - Test: `codeforces`, `lcbv5` (split `test`)
- Runs async GRPO updates:
  - Samples `group_size` completions per prompt.
  - Evaluates completions in `CodeEnv`.
  - Computes group-relative advantages `A_i = R_i - mean(R)`.
  - Skips degenerate groups (all rewards identical).
  - Trains with `tinker.types.Datum` objects.
- Logs metrics, runs periodic evals, and saves checkpoints.

## Requirements

Dependencies are in `requirements.txt` (e.g., `tinker`, `datasets`, `torch`, `transformers`, `aiohttp`, `chz`).

## Quickstart (local)

1. Start the sandbox:

```bash
docker run -d \
    -p 8080:8080 \
    -v ./sandbox_config/local.yaml:/root/sandbox/sandbox/configs/local.yaml \
    volcengine/sandbox-fusion:server-20250609
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your Tinker API key:

```bash
export TINKER_API_KEY="your_tinker_api_key_here"
```

4. (Optional) Smoke test the pipeline:

```bash
python test_pipeline.py
```

5. Run a short training test:

```bash
python train.py max_steps=5 batch_size=4 group_size=2
```

## 20-step test run

```bash
python train.py max_steps=20 batch_size=4 group_size=2
```

### Example results (20-step run)

Evaluation at steps 10 and 20 (50 test samples each):

- `eval_accuracy`: 0.0
- `eval_avg_reward`: 0.0
- `eval_format_errors`: 0.0

Checkpoints were saved at steps 10 and 20 and at completion. Logs and metrics were written to:

```
/Users/user_1/code-rl-logs/2026_02_09-16_32_51
```

## Notes

- Use `key=value` overrides for hyperparameters (e.g., `max_steps=5 batch_size=4 group_size=2`).
- If `max_steps=-1`, training runs until the dataset is exhausted.