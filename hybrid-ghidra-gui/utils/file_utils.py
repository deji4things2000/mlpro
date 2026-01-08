import os

DEFAULT_CONFIG = {
    "theme": "light",
    "ghidra_host": "localhost",
    "ghidra_port": 18001,
    "enable_llm": False,
}


def ensure_config_defaults(cfg: dict) -> dict:
    merged = DEFAULT_CONFIG.copy()
    merged.update(cfg or {})
    return merged


def is_binary_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
            return b"\x00" in chunk
    except Exception:
        return False


def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()
