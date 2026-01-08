from __future__ import annotations

import os
from typing import Optional


def _call_openai(prompt: str, model: str, api_key: Optional[str]) -> Optional[str]:
    try:
        from openai import OpenAI
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            return None
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


def analyze_function(
    function_name: str,
    assembly: Optional[str] = None,
    decompiled: Optional[str] = None,
    provider: str = "local",
    model: str = "placeholder",
    api_key: Optional[str] = None,
) -> dict:
    """Analyze a function via chosen provider; fallback to local heuristics."""
    if provider == "openai":
        prompt = (
            f"Analyze the following function for potential security risks.\n"
            f"Name: {function_name}\n\n"
            f"Assembly:\n{assembly or '(none)'}\n\n"
            f"Decompiled:\n{decompiled or '(none)'}\n\n"
            "Return a brief summary, a one-line risk classification, and a suggested fix."
        )
        content = _call_openai(prompt, model=model, api_key=api_key)
        if content:
            # Simple parsing: expect three lines labelled Summary/Prediction/Fix
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            summary = next((l.split(":", 1)[1].strip() for l in lines if l.lower().startswith("summary:")), content)
            prediction = next((l.split(":", 1)[1].strip() for l in lines if l.lower().startswith("prediction:")), "UNKNOWN")
            fix = next((l.split(":", 1)[1].strip() for l in lines if l.lower().startswith("fix:")), "Review manually")
            return {"summary": summary, "prediction": prediction, "fix": fix}

    # Local heuristic fallback
    text = (assembly or "") + "\n" + (decompiled or "")
    lowered = text.lower()
    if "strcpy" in lowered or "memcpy" in lowered or "gets(" in lowered:
        return {
            "summary": "Potential unsafe string/memory use detected.",
            "prediction": "HIGH RISK FUNCTION",
            "fix": "Use bounded copies and validate sizes",
        }
    if "auth" in lowered or "login" in lowered:
        return {
            "summary": "Authentication-related code; review controls.",
            "prediction": "MEDIUM RISK FUNCTION",
            "fix": "Harden checks, sanitize inputs",
        }
    if function_name == "main":
        return {
            "summary": "Standard main function with typical control flow.",
            "prediction": "LOW RISK FUNCTION",
            "fix": "None required",
        }
    return {
        "summary": f"No specific analysis available for {function_name}.",
        "prediction": "UNKNOWN",
        "fix": "Review manually",
    }
