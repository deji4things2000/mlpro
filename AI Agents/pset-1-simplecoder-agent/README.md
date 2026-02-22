# SimpleCoder (Problem Set 1, Part III)

SimpleCoder is a small CLI coding agent built for the Dartmouth “AI Agents” course. It uses a ReAct-style loop (model decides → tool call → observation → repeat) with optional planning and optional RAG over your local code.

The goal of this README is to help a new user install, run, and understand what the agent does at runtime (and how reproducible it is).

## Quickstart (new user, most reproducible)

1) Clone and enter the project:

```bash
git clone <your-repo-url>
cd mlpro/pset-1-simplecoder-agent
```

2) Create a virtualenv and install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

3) Configure a provider (choose one):

Gemini:

```bash
export GEMINI_API_KEY="..."
```

Dartmouth Chat (OpenAI-compatible):

```bash
export SIMPLECODER_API_BASE="https://chat.dartmouth.edu/api"
export DARTMOUTH_CHAT_API_KEY="..."
```

4) (Optional, recommended) Let SimpleCoder prompt you to pick a provider

If you want the built-in provider menu (Dartmouth Chat / Gemini / custom OpenAI-compatible), run interactive mode in a real terminal:

```bash
unset SIMPLECODER_NO_PROVIDER_PROMPT
python -m simplecoder.main --interactive --verbose
```

If you already set credentials and still want the menu, force it:

```bash
export SIMPLECODER_FORCE_PROVIDER_PROMPT=1
python -m simplecoder.main --interactive --verbose
```

5) Disable interactive prompts and auto-approve tool permissions (best for “reproduce my run”):

```bash
export SIMPLECODER_NO_PROVIDER_PROMPT=1
export SIMPLECODER_AUTO_APPROVE=1
```

6) Run a one-shot task:

```bash
python -m simplecoder.main --no-interactive --verbose \
  --model "gemini/gemini-3-flash-preview" \
  "create a hello.py file"
```

If you enabled RAG and want to force a re-index (fresh cache), delete `.simplecoder/`:

```bash
rm -rf .simplecoder/
```

## Usage

After installing, you can run SimpleCoder either as a module or via the console script:

```bash
# Always works
python -m simplecoder.main --interactive --verbose

# Console script (installed by the package)
simplecoder --interactive --verbose
```

Examples:

```bash
# Basic
simplecoder "create a hello.py file"

# Enable RAG
simplecoder --use-rag "what does the Agent class do?"

# Enable planning
simplecoder --use-planning "create a web server with routes for home and about"

# Non-interactive one-shot mode (recommended for reproduction)
python -m simplecoder.main --no-interactive "summarize the repo"
```

Key flags (see `--help` for the full list):

- `--model`: LLM model string (defaults to a Gemini model).
- `--max-iterations`: Max ReAct iterations per task.
- `--use-planning`: Generate a short plan and execute subtasks.
- `--use-rag`: Build/search a local embedding index over the workspace.
- `--rag-embedder`: Embedding model used by RAG.
- `--rag-index-pattern`: Glob for files to index (workspace-relative).

## Provider configuration

SimpleCoder supports:

- Gemini via `GEMINI_API_KEY`
- OpenAI-compatible endpoints (including Dartmouth Chat) via:
  - `SIMPLECODER_API_BASE` (or `OPENAI_API_BASE`)
  - `SIMPLECODER_API_KEY` / `DARTMOUTH_CHAT_API_KEY` / `OPENAI_API_KEY`

On startup (only in a real TTY), the agent may prompt you to pick a provider and paste a key.

- Disable the provider chooser: `SIMPLECODER_NO_PROVIDER_PROMPT=1`
- Force showing the provider chooser: `SIMPLECODER_FORCE_PROVIDER_PROMPT=1`

Important: if you set `SIMPLECODER_API_BASE`, Gemini model strings like `gemini/...` will be sent to that OpenAI-compatible endpoint and will fail with “model not found”.

## How it works (runtime logic)

This section describes the actual code path when you run the CLI.

### High-level flow

1) CLI parses flags in `simplecoder/main.py` and constructs `Agent(...)`.
2) The agent resolves provider settings (env vars and/or optional interactive prompt).
3) For each task:
   - Optional planning: create a short list of subtasks and execute them in order.
   - Optional RAG: ensure an embedding index exists, retrieve top code hits, and inject them into context.
   - ReAct loop: call the model, parse JSON output, run tools if requested, and iterate.

### ReAct loop details

The model is instructed to return **JSON only** in one of two forms:

- Tool call: `{"type":"tool","name":"<tool>","args":{...}}`
- Final answer: `{"type":"final","answer":"..."}`

The loop runs up to `--max-iterations`. Tool results are added back to the conversation as a system message.

### Tools and permissions

Tools live in `simplecoder/tools.py` and include:

- `list_files`, `read_file`, `search_files`
- `write_file`, `edit_file_replace`

All tool paths are **workspace-scoped**: the “workspace root” is the directory you run the CLI from (`pwd`). Paths that escape the workspace are rejected.

Tool execution is permission-gated by `simplecoder/permissions.py`:

- Reads are often auto-approved inside “safe directories”.
- Writes/edits typically require explicit confirmation.
- For demos/autograding you can bypass prompts with `SIMPLECODER_AUTO_APPROVE=1`.

### RAG (retrieval over local code)

When `--use-rag` is enabled:

- Files matching `--rag-index-pattern` are indexed.
- Python is chunked using the AST into function/class/module chunks.
- Embeddings are computed using `litellm.embedding(...)` and cached under `.simplecoder/`.
- For each task, the agent retrieves top matches and injects them as extra system context.

## Reproducibility

This project is **partially reproducible**.

Deterministic pieces:

- Filesystem tools behave deterministically given the same workspace state.
- RAG caching is deterministic for a fixed file set + mtimes (it reuses the cached vectors when unchanged).

Non-deterministic pieces:

- LLM responses can vary between runs even with the same prompt/model/temperature.
- External embedding/search behavior can drift if the provider updates models.

To maximize reproducibility:

```bash
export SIMPLECODER_NO_PROVIDER_PROMPT=1
export SIMPLECODER_AUTO_APPROVE=1
python -m simplecoder.main --no-interactive --verbose \
  --model "gemini/gemini-3-flash-preview" \
  "create a hello.py file"
```

Exact token-for-token reproduction is not guaranteed because LLMs are stochastic services.

## Project layout

- `simplecoder/main.py`: CLI entrypoint (provided by assignment; do not modify).
- `simplecoder/agent.py`: Core agent loop (provider selection, ReAct, optional planning/RAG).
- `simplecoder/tools.py`: Tool registry + filesystem tools (workspace-scoped).
- `simplecoder/permissions.py`: Permission model + prompting/auto-approve controls.
- `simplecoder/rag.py`: AST chunking + embedding index + cache.
- `simplecoder/context.py`: Context manager with compaction/summarization.
- `simplecoder/planner.py`: JSON plan generation and plan rendering.
- `.simplecoder/`: Local RAG cache (safe to delete).

## Troubleshooting

- "model not found" when using Gemini model strings:
  - Unset `SIMPLECODER_API_BASE` / `OPENAI_API_BASE` if you intend to use Gemini.
- Provider prompt appears when you don’t want it:
  - Set `SIMPLECODER_NO_PROVIDER_PROMPT=1`.
- Permission prompts block automation:
  - Set `SIMPLECODER_AUTO_APPROVE=1` (use with care).
