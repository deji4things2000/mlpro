from typing import Any
from tinker_utils.lcb import fetch_live_code_bench_system_prompt


def build_question(example: dict[str, Any]) -> str | None:
    # Prefer preprocessed question if available.
    question = example.get("question") or example.get("prompt") or example.get("problem")
    if not isinstance(question, str) or not question.strip():
        return None
    starter_code = example.get("starter_code")
    if isinstance(starter_code, str) and starter_code.strip():
        return fetch_live_code_bench_system_prompt(question, starter_code)
    return fetch_live_code_bench_system_prompt(question)
