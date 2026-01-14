# AI Agents @ Dartmouth College
## Problem Set 1, Part III: SimpleCoder

In this assignment, we built **SimpleCoder**, a CLI coding agent that can help you write code, navigate codebases, and complete various software engineering tasks. This agent will use several useful concepts from the modern AI Agent development stack: tool use, Retrieval-Augmented Generation (RAG), context management, and task planning, etc.


## Overview

SimpleCoder is a ReAct-style agent that combines tool use, semantic code search, context management, and task planning.

## Getting Started

You are provided with:
- `simplecoder/main.py` - The CLI entry point (complete, do not modify)
- `pyproject.toml` - Package configuration (complete, do not modify)

You need to implement:
- Tool functions and schemas
- Semantic code RAG for code search
- Context management with compacting
- Task planning and decomposition
- Manage user permissions for file read/write access, etc. (must support a session-level persistence option)
- The main agent logic

### Set API Key

```bash
# You can pre-configure keys via environment variables.
# Gemini:
export GEMINI_API_KEY="your-key-here"
```

On startup, SimpleCoder will also guide you through selecting a provider (Dartmouth Chat / Gemini / custom OpenAI-compatible) and entering an API key with hidden input if no usable credentials are already configured.

Security note: avoid pasting API keys into terminals/chats; rotate any key you accidentally exposed.

### Install (recommended)

The most reliable way to run SimpleCoder is to install it into the Python environment you plan to use (this avoids PATH/venv confusion).

```bash
# From this folder (pset-1-simplecoder-agent/), install into your currently-active environment
python -m pip install -e .

# Or from anywhere:
# python -m pip install -e /path/to/pset-1-simplecoder-agent
```

After that, you can run either:

```bash
# Module form (always works)
python -m simplecoder.main --interactive --verbose

# Or the console script
simplecoder --interactive --verbose
```

### Dartmouth Chat setup (no CLI changes required)

Dartmouth Chat provides an OpenAI-compatible API at `https://chat.dartmouth.edu/api`. SimpleCoder will route requests to an OpenAI-compatible endpoint when `SIMPLECODER_API_BASE` (or `OPENAI_API_BASE`) is set.

Note: if `SIMPLECODER_API_BASE` is set, then Gemini model strings like `gemini/...` will be sent to the Dartmouth endpoint and will fail with “model not found”. To use Gemini, unset `SIMPLECODER_API_BASE`/`OPENAI_API_BASE` and set `GEMINI_API_KEY`, or select Gemini in the startup prompt.

```bash
# From pset-1-simplecoder-agent/
source scripts/simplecoder_init.sh

# Then run SimpleCoder with a Dartmouth model string
python -m simplecoder.main --model "$SIMPLECODER_MODEL" --verbose "What is deep learning?"
```

If you don’t want the prompt-based script, you can set env vars directly:

```bash
export SIMPLECODER_API_BASE="https://chat.dartmouth.edu/api"
export DARTMOUTH_CHAT_API_KEY="..."
python -m simplecoder.main --model "anthropic.claude-3-5-haiku-20241022" --verbose "Hello"
```

### Intended Usage


Your implementation should support the following task inputs:
```bash
# Basic usage
simplecoder "create a hello.py file"

# With RAG
simplecoder --use-rag "what does the Agent class do?"

# With planning
simplecoder --use-planning "create a web server with routes for home and about"

# Interactive mode
simplecoder --interactive

# Options
simplecoder --help
```

## Design (1 paragraph each)

- `simplecoder/agent.py`: Implements a ReAct loop where the model must output strict JSON (`tool` call or `final`). The agent executes tools via a registry, feeds back tool observations, optionally injects RAG snippets, and optionally creates/executes a short plan. For responsiveness, verbose mode streams model tokens and shows a spinner; failures return a structured message instead of crashing.

- `simplecoder/tools.py`: Defines a small tool schema layer (`Tool`, `ToolRegistry`) and implements the required filesystem tools: list/read/search/write/edit. All paths are workspace-scoped (prevents `..` escaping), and each tool call is wrapped by a permission request that can be accepted once per session.

- `simplecoder/rag.py`: Builds an embedding index over AST-derived Python chunks (functions/classes/modules), which improves retrieval granularity over naive fixed-size text chunks. Embeddings are computed via `litellm.embedding`, normalized for cosine similarity, and cached on disk keyed by workspace path + embedder model + glob pattern so repeated runs are fast.

- `simplecoder/context.py`: Tracks a running conversation token estimate, keeps the last `k` messages intact, and compacts older history by asking the model for a concise summary once a threshold is exceeded. This keeps long interactive sessions usable even when tasks span many steps.

- `simplecoder/planner.py`: Generates a short (3–7 step) checklist plan as JSON via the LLM, and provides a simple state machine (`pending` → `in_progress` → `completed/blocked`) to track incremental progress. Plans are rendered as Markdown so the CLI can display them cleanly.

- `simplecoder/permissions.py`: Enforces a conservative permission model for filesystem operations and other sensitive actions. It supports session-level persistence (“always allow” for the remainder of the session) and an environment escape hatch (`SIMPLECODER_AUTO_APPROVE=1`) for demos/autograding.
