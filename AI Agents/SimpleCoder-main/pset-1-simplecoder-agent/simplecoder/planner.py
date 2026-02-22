from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import litellm


SubtaskStatus = Literal["pending", "in_progress", "completed", "blocked"]


@dataclass
class Subtask:
    id: str
    description: str
    status: SubtaskStatus = "pending"


class TaskPlanner:
    def __init__(self, model: str):
        self.model = model

    def make_plan(self, task_description: str) -> list[Subtask]:
        """Create a short, actionable plan (3-7 steps)."""
        system = (
            "You are a senior software agent. "
            "Return ONLY valid JSON with key 'steps'. "
            "Each step is {id, description}. Keep 3-7 steps, imperative, testable."
        )
        user = {
            "task": task_description,
            "constraints": [
                "Prefer small edits",
                "Use tools to inspect/write files",
                "Ask for clarification only if blocked",
            ],
        }
        resp = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)},
            ],
            temperature=0.2,
        )
        content = resp["choices"][0]["message"]["content"].strip()
        data = json.loads(content)
        steps = data.get("steps", [])
        out: list[Subtask] = []
        for i, s in enumerate(steps):
            sid = str(s.get("id") or f"step_{i+1}")
            desc = str(s.get("description") or "").strip()
            if not desc:
                continue
            out.append(Subtask(id=sid, description=desc, status="pending"))
        return out[:7]

    @staticmethod
    def render_plan_md(plan: list[Subtask]) -> str:
        if not plan:
            return ""
        status_icon = {
            "pending": "[ ]",
            "in_progress": "[~]",
            "completed": "[x]",
            "blocked": "[!]",
        }
        lines = ["### Plan"]
        for s in plan:
            lines.append(f"- {status_icon.get(s.status, '[ ]')} {s.id}: {s.description}")
        return "\n".join(lines)