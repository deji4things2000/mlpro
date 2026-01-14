from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from getpass import getpass
from typing import Any

import litellm
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from simplecoder.context import ContextManager
from simplecoder.permissions import create_permission_manager
from simplecoder.planner import Subtask, TaskPlanner
from simplecoder.rag import CodeRAG
from simplecoder.tools import ToolRegistry


console = Console()


@dataclass
class _RunConfig:
    model: str
    max_iterations: int
    verbose: bool
    use_planning: bool
    use_rag: bool
    rag_embedder: str
    rag_index_pattern: str
    api_base: str | None
    api_key: str | None
    custom_llm_provider: str | None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from model output."""
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class Agent:
    """SimpleCoder CLI agent.

    - ReAct-style loop: decide -> tool -> observe -> repeat.
    - Optional planning: decomposes task, tracks completion.
    - Optional RAG: AST-chunked embedding search with disk cache.
    - Permissions: asks before filesystem writes/edits.
    """

    def __init__(
        self,
        model: str,
        max_iterations: int = 10,
        verbose: bool = False,
        use_planning: bool = False,
        use_rag: bool = False,
        rag_embedder: str = "gemini/gemini-embedding-001",
        rag_index_pattern: str = "**/*.py",
    ):
        # Allow switching providers without modifying `simplecoder/main.py`.
        # Dartmouth Chat is OpenAI-compatible at: https://chat.dartmouth.edu/api
        api_base = os.environ.get("SIMPLECODER_API_BASE") or os.environ.get("OPENAI_API_BASE")
        api_key = (
            os.environ.get("SIMPLECODER_API_KEY")
            or os.environ.get("DARTMOUTH_CHAT_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        custom_llm_provider = os.environ.get("SIMPLECODER_LITELLM_PROVIDER")
        if custom_llm_provider is None and api_base:
            custom_llm_provider = "openai"

        model, api_base, api_key, custom_llm_provider = self._maybe_prompt_provider_setup(
            model=model,
            api_base=api_base,
            api_key=api_key,
            custom_llm_provider=custom_llm_provider,
        )

        self.cfg = _RunConfig(
            model=model,
            max_iterations=max_iterations,
            verbose=verbose,
            use_planning=use_planning,
            use_rag=use_rag,
            rag_embedder=rag_embedder,
            rag_index_pattern=rag_index_pattern,
            api_base=api_base,
            api_key=api_key,
            custom_llm_provider=custom_llm_provider,
        )

        self.permission_manager = create_permission_manager()
        self.tools = ToolRegistry(self.permission_manager)
        self.ctx = ContextManager(model=self.cfg.model, max_tokens=8000, keep_last_messages=12)
        self.planner = TaskPlanner(model=self.cfg.model)

        self.rag: CodeRAG | None = None
        if use_rag:
            self.rag = CodeRAG(embedder_model=rag_embedder, index_pattern=rag_index_pattern)

        self._plan: list[Subtask] | None = None

    def _maybe_prompt_provider_setup(
        self,
        *,
        model: str,
        api_base: str | None,
        api_key: str | None,
        custom_llm_provider: str | None,
    ) -> tuple[str, str | None, str | None, str | None]:
        """Interactive provider chooser.

        Runs on startup so the user can select a provider and paste a key once,
        then immediately start chatting.

        Safety: only prompts when no usable credentials are present, unless
        SIMPLECODER_FORCE_PROVIDER_PROMPT=1.
        """

        force = os.getenv("SIMPLECODER_FORCE_PROVIDER_PROMPT", "").lower() in {"1", "true", "yes"}
        no_prompt = os.getenv("SIMPLECODER_NO_PROVIDER_PROMPT", "").lower() in {"1", "true", "yes"}
        if no_prompt:
            return model, api_base, api_key, custom_llm_provider

        # Only prompt in interactive terminals.
        if not (sys.stdin is not None and sys.stdin.isatty()):
            return model, api_base, api_key, custom_llm_provider

        # If config already exists, skip unless forced.
        gemini_key = os.environ.get("GEMINI_API_KEY")
        has_openai_compat = bool(api_base and api_key)
        has_gemini = bool(gemini_key)
        if not force and (has_openai_compat or has_gemini):
            return model, api_base, api_key, custom_llm_provider

        console.print("\n[bold]Select an API provider:[/bold]")
        console.print("  1) Dartmouth Chat (recommended)")
        console.print("  2) Gemini")
        console.print("  3) OpenAI-compatible (custom base URL)")
        console.print("  4) Skip (I will set env vars myself)")

        choice = ""
        while choice not in {"1", "2", "3", "4"}:
            try:
                choice = input("> ").strip()
            except EOFError:
                return model, api_base, api_key, custom_llm_provider

        if choice == "4":
            console.print("[dim]Skipping provider setup.[/dim]")
            return model, api_base, api_key, custom_llm_provider

        if choice == "1":
            api_base = "https://chat.dartmouth.edu/api"
            custom_llm_provider = "openai"
            api_key = getpass("Dartmouth Chat API key (hidden): ").strip() or api_key
            if api_key:
                os.environ["DARTMOUTH_CHAT_API_KEY"] = api_key
                os.environ["SIMPLECODER_API_BASE"] = api_base

            # If the user didn't explicitly pick a model, override the gemini default.
            if model.startswith("gemini/"):
                model = os.environ.get("SIMPLECODER_MODEL", "anthropic.claude-3-5-haiku-20241022")

            console.print("[green]Welcome to SimpleCoder![/green] Using Dartmouth Chat.")
            return model, api_base, api_key, custom_llm_provider

        if choice == "2":
            # Gemini uses GEMINI_API_KEY and no api_base.
            key = getpass("GEMINI_API_KEY (hidden): ").strip() or gemini_key
            if key:
                os.environ["GEMINI_API_KEY"] = key

            api_base = None
            api_key = None
            custom_llm_provider = None
            if not model.startswith("gemini/"):
                model = "gemini/gemini-3-flash-preview"

            console.print("[green]Welcome to SimpleCoder![/green] Using Gemini.")
            return model, api_base, api_key, custom_llm_provider

        # choice == "3"
        base = input("OpenAI-compatible base URL (e.g. https://chat.dartmouth.edu/api): ").strip()
        if base:
            api_base = base
            os.environ["SIMPLECODER_API_BASE"] = base
        custom_llm_provider = "openai"
        api_key = getpass("API key (hidden): ").strip() or api_key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        if model.startswith("gemini/"):
            model = os.environ.get("SIMPLECODER_MODEL", "gpt-4o-mini")

        console.print("[green]Welcome to SimpleCoder![/green]")
        return model, api_base, api_key, custom_llm_provider

    def run(self, user_input: str) -> str:
        if self.cfg.use_planning and self._plan is None:
            self._plan = self._create_plan(user_input)

        if self.cfg.use_planning and self._plan:
            # Work through subtasks in order; each subtask gets its own ReAct run.
            outputs: list[str] = [TaskPlanner.render_plan_md(self._plan)]
            for st in self._plan:
                if st.status == "completed":
                    continue
                st.status = "in_progress"
                outputs.append(f"\n### Working on: {st.id}\n{st.description}")
                out = self._react(st.description, plan=self._plan)
                outputs.append(out)
                # Mark complete unless the model explicitly says it's blocked.
                st.status = "completed" if "BLOCKED" not in out else "blocked"
                outputs.insert(0, TaskPlanner.render_plan_md(self._plan))
            return "\n\n".join([o for o in outputs if o.strip()])

        return self._react(user_input, plan=self._plan)

    def _create_plan(self, task: str) -> list[Subtask]:
        if self.cfg.verbose:
            console.print("[cyan]Planning...[/cyan]")
        try:
            plan = self.planner.make_plan(task)
        except Exception as e:
            if self.cfg.verbose:
                console.print(f"[yellow]Planning failed, continuing without plan: {e}[/yellow]")
            return []
        if self.cfg.verbose and plan:
            console.print(self.planner.render_plan_md(plan))
        return plan

    def _progress(self, msg: str) -> None:
        if self.cfg.verbose:
            console.print(f"[dim]{msg}[/dim]")

    def _react(self, task: str, plan: list[Subtask] | None) -> str:
        system_prompt = self._system_prompt()

        self.ctx.add("system", system_prompt)
        self.ctx.add("user", task)

        if self.rag is not None:
            # Ensure index (cached)
            try:
                self.rag.ensure_index(progress_cb=self._progress)
                hits = self.rag.search(task, top_k=5)
                if hits:
                    rag_md = self._format_rag(hits)
                    self.ctx.add("system", f"Relevant code context (RAG):\n{rag_md}")
            except Exception as e:
                self._progress(f"RAG disabled due to error: {e}")

        tool_desc = json.dumps(self.tools.list_tools(), indent=2)
        self.ctx.add("system", f"Available tools (JSON):\n{tool_desc}")

        last_answer = ""

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            transient=True,
            console=console,
            disable=not self.cfg.verbose,
        ) as progress:
            task_id = progress.add_task("Thinking...", total=None)
            for i in range(self.cfg.max_iterations):
                progress.update(task_id, description=f"Iteration {i+1}/{self.cfg.max_iterations}")
                model_out = self._call_model(self.ctx.messages())
                last_answer = model_out

                parsed = _extract_json_object(model_out)
                if not parsed:
                    # Not structured; treat as final.
                    return self._wrap_final(plan, model_out)

                if parsed.get("type") == "final":
                    return self._wrap_final(plan, str(parsed.get("answer", "")).strip())

                if parsed.get("type") != "tool":
                    # Unknown shape; treat as final.
                    return self._wrap_final(plan, model_out)

                tool_name = str(parsed.get("name", ""))
                args = parsed.get("args", {})
                if not isinstance(args, dict):
                    args = {}

                self._progress(f"Tool call: {tool_name}({args})")
                try:
                    result = self.tools.execute(tool_name, args)
                except Exception as e:
                    result = {"error": str(e)}

                self.ctx.add("assistant", model_out)
                # Provider-agnostic observation message.
                # (OpenAI has a dedicated tool role + tool_call_id; Gemini uses a different schema.)
                # Storing tool results as text keeps the agent portable across providers.
                self.ctx.add(
                    "system",
                    "Tool result:\n" + json.dumps({"name": tool_name, "result": result}, indent=2),
                )

                # If the tool produced an error, encourage recovery.
                if isinstance(result, dict) and result.get("error"):
                    self.ctx.add(
                        "system",
                        "The previous tool call failed. Either try a different tool, fix args, or ask a clarifying question.",
                    )

        return self._wrap_final(plan, last_answer)

    def _call_model(self, messages: list[dict[str, str]]) -> str:
        extra: dict[str, Any] = {}
        if self.cfg.api_base:
            extra["api_base"] = self.cfg.api_base
        if self.cfg.api_key:
            extra["api_key"] = self.cfg.api_key
        if self.cfg.custom_llm_provider:
            extra["custom_llm_provider"] = self.cfg.custom_llm_provider

        # Streaming: print deltas in verbose mode, but still return full content.
        if self.cfg.verbose:
            try:
                chunks = []
                for part in litellm.completion(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=0.2,
                    stream=True,
                    **extra,
                ):
                    delta = part["choices"][0].get("delta", {}).get("content")
                    if delta:
                        chunks.append(delta)
                        console.print(delta, end="")
                console.print()  # newline after streamed output
                return "".join(chunks).strip()
            except Exception:
                # Fall back to non-stream
                pass

        try:
            resp = litellm.completion(model=self.cfg.model, messages=messages, temperature=0.2, **extra)
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            # Keep CLI responsive: return a structured final answer instead of crashing.
            msg = (
                "LLM call failed. Check your provider configuration.\n\n"
                f"- model: {self.cfg.model}\n"
                f"- SIMPLECODER_API_BASE / OPENAI_API_BASE: {self.cfg.api_base or '(not set)'}\n"
                "- For Dartmouth Chat: set SIMPLECODER_API_BASE=https://chat.dartmouth.edu/api and DARTMOUTH_CHAT_API_KEY\n"
                "- For Gemini: set GEMINI_API_KEY\n\n"
                f"Error: {type(e).__name__}: {e}"
            )
            return json.dumps({"type": "final", "answer": msg})

    def _system_prompt(self) -> str:
        return (
            "You are SimpleCoder, a CLI coding agent. "
            "Use a ReAct loop. When you need information or to change files, call a tool. "
            "You MUST respond with JSON only, no backticks, no extra text.\n\n"
            "Two allowed JSON shapes:\n"
            "1) Tool call: {\"type\":\"tool\", \"name\": <tool_name>, \"args\": {...}}\n"
            "2) Final answer: {\"type\":\"final\", \"answer\": <string>}\n\n"
            "Guidelines:\n"
            "- Prefer read/search tools before editing.\n"
            "- When writing/editing, be surgical and explain what changed.\n"
            "- If blocked, ask a single clear question in the final answer."
        )

    def _format_rag(self, hits: list[dict[str, Any]]) -> str:
        lines = []
        for h in hits:
            loc = f"{h['filepath']}:{h['start_line']}-{h['end_line']}"
            lines.append(f"- {h['kind']} {h['name']} @ {loc} (score={h['score']:.3f})")
            lines.append("```")
            lines.append(h["code"].strip())
            lines.append("```")
        return "\n".join(lines)

    def _wrap_final(self, plan: list[Subtask] | None, answer: str) -> str:
        if plan:
            return f"{TaskPlanner.render_plan_md(plan)}\n\n{answer}".strip()
        return answer.strip()