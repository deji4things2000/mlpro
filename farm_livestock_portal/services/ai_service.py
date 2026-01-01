# services/ai_service.py

import re
from datetime import date

HEALTH_STATUSES = {"Healthy", "Sick", "Recovering", "Under Treatment", "Injured", "Unknown"}

SYMPTOM_KEYWORDS = {
    "mastitis": "Sick",
    "lameness": "Sick",
    "pneumonia": "Sick",
    "fever": "Sick",
    "diarrhea": "Sick",
    "wound": "Injured",
    "injury": "Injured",
    "fracture": "Injured",
}

IMPROVE_KEYWORDS = {"improve", "recover", "better", "healing"}


def normalize(text: str) -> str:
    return (text or "").strip().lower()


def infer_health_status(
    diagnosis: str = None,
    treatment: str = None,
    severity: str = None,
    lab_result: str = None,
    notes: str = None,
    next_check: date = None,
    withdrawal_end: date = None,
) -> str:
    """Infer portal health status from health record fields.

    Heuristic mapping to keep it lightweight and offline. Replaceable with ML later.
    """
    d = normalize(diagnosis)
    t = normalize(treatment)
    s = normalize(severity)
    l = normalize(lab_result)
    n = normalize(notes)

    # Severe or explicit symptom keywords → Sick/Injured
    for kw, status in SYMPTOM_KEYWORDS.items():
        if kw in d or kw in n:
            return status

    if s == "severe":
        return "Sick"

    # Under Treatment if treatment present and moderate/mild
    if t:
        if s in {"moderate", "mild"}:
            return "Under Treatment"
        # If withdrawal period, still under treatment/recovering
        if withdrawal_end:
            return "Under Treatment"

    # Recovering if improvement notes or upcoming check scheduled
    if any(word in n for word in IMPROVE_KEYWORDS):
        return "Recovering"
    if next_check:
        return "Recovering"

    # Lab normal results hint Healthy
    if l and re.search(r"normal|negative|within range", l):
        return "Healthy"

    # Default
    return "Healthy" if not (d or t or l or n) else "Unknown"
