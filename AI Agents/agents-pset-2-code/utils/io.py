import logging
import time
import json
from pathlib import Path


def log_stage(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    logging.info(f"[{timestamp}] {message}")


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base_dir / path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
