from __future__ import annotations

import os
from typing import Optional
import json


def _call_ollama(prompt: str, model: str) -> Optional[str]:
    """Call a local Ollama server for generation.
    Requires `ollama serve` running locally. Uses OLLAMA_HOST env if set.
    """
    try:
        import requests
        base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        url = base.rstrip("/") + "/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns { response: "..." }
        return data.get("response")
    except Exception:
        return None


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

    if provider == "ollama":
        prompt = (
            f"Analyze the following function for potential security risks.\n"
            f"Name: {function_name}\n\n"
            f"Assembly:\n{assembly or '(none)'}\n\n"
            f"Decompiled:\n{decompiled or '(none)'}\n\n"
            "Return three labeled lines: Summary:, Prediction:, Fix:."
        )
        content = _call_ollama(prompt, model=model)
        if content:
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


def translate_to_python(
    decompiled: str,
    provider: str = "local",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Translate decompiled C/C++ to Python using the chosen provider.
    Returns None if translation could not be performed by the provider.
    """
    if not decompiled.strip():
        return None

    if provider == "openai":
        prompt = (
            "You are a helpful assistant. Convert the following C/C++ decompiled code "
            "into clear, runnable Python where feasible, or faithful pseudocode when "
            "exact semantics are unknown. Prefer functions, clear variable names, and "
            "docstrings. Do not include commentary outside the code block.\n\n"
            f"C/C++ Decompiled:\n{decompiled}\n\n"
            "Python translation:"
        )
        content = _call_openai(prompt, model=model, api_key=api_key)
        return content

    if provider == "ollama":
        prompt = (
            "Convert the following decompiled C/C++ code into readable Python. "
            "If exact behavior is unclear, provide best-effort pseudocode in Python. "
            "Return only the Python code block.\n\n"
            f"C/C++ Decompiled:\n{decompiled}\n\n"
            "Python translation:"
        )
        content = _call_ollama(prompt, model=model)
        return content

    # Unknown/non-LLM providers should return None to allow heuristic fallback
    return None
