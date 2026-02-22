from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import litellm


Message = dict[str, str]


def estimate_tokens(text: str) -> int:
    # Dependency-free heuristic: ~4 chars/token in English-like text.
    # Good enough for deciding when to compact.
    return max(1, int(len(text) / 4))


@dataclass
class _StoredMessage:
    message: Message
    tokens: int


class ContextManager:
    """Tracks conversation and compacts it when it grows too large.

    - Keeps last `keep_last_messages` intact.
    - Summarizes older messages using the same LLM (when possible).
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 8000,
        keep_last_messages: int = 12,
        summarizer_model: str | None = None,
    ):
        self.model = model
        self.summarizer_model = summarizer_model or model
        self.max_tokens = max_tokens
        self.keep_last_messages = keep_last_messages
        self._history: list[_StoredMessage] = []

    def add(self, role: str, content: str) -> None:
        msg = {"role": role, "content": content}
        self._history.append(_StoredMessage(message=msg, tokens=estimate_tokens(content)))
        self._compact_if_needed()

    def extend(self, messages: list[Message]) -> None:
        for m in messages:
            self.add(m["role"], m["content"])

    def messages(self) -> list[Message]:
        return [m.message for m in self._history]

    def token_count(self) -> int:
        return sum(m.tokens for m in self._history)

    def _compact_if_needed(self) -> None:
        if self.token_count() <= self.max_tokens:
            return
        if len(self._history) <= self.keep_last_messages:
            return

        keep = self._history[-self.keep_last_messages :]
        old = self._history[: -self.keep_last_messages]
        summary = self._summarize(old)
        self._history = [_StoredMessage(message={"role": "system", "content": summary}, tokens=estimate_tokens(summary))] + keep

    def _summarize(self, items: list[_StoredMessage]) -> str:
        # If summarization fails, fall back to a compact heuristic.
        transcript = []
        for it in items:
            role = it.message["role"]
            content = it.message["content"].strip()
            transcript.append(f"{role}: {content}")

        prompt = """Summarize the prior conversation for a coding agent.

Keep:
- user goals and constraints
- decisions made
- files mentioned
- TODOs / open questions

Write a concise bullet summary.
""".strip()

        try:
            resp = litellm.completion(
                model=self.summarizer_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "\n\n".join(transcript)},
                ],
                temperature=0.2,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            # Heuristic fallback
            user_lines = [t for t in transcript if t.startswith("user:")]
            assistant_lines = [t for t in transcript if t.startswith("assistant:")]
            return (
                "Earlier conversation summary:\n"
                f"- User messages: {len(user_lines)}\n"
                f"- Assistant messages: {len(assistant_lines)}\n"
                + (f"- Recent user topics: {', '.join([l[6:][:80] for l in user_lines[-3:]])}\n" if user_lines else "")
            ).strip()