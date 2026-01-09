import shutil
import subprocess
import tempfile
import os
from pathlib import Path
import json
import time


def is_retdec_available() -> bool:
    return bool(shutil.which("retdec-decompiler") or shutil.which("retdec-decompiler.py"))


def decompile_with_retdec(binary_path: str, timeout: int = 300) -> str:
    """Run RetDec to decompile the given binary and return C pseudocode.
    Requires RetDec to be installed and available on PATH.
    """
    tool = shutil.which("retdec-decompiler") or shutil.which("retdec-decompiler.py")
    if not tool:
        raise RuntimeError("RetDec not found on PATH. Install via 'brew install retdec' or from source.")
    bin_path = Path(binary_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"Binary not found: {binary_path}")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_c = Path(tmpdir) / "out.c"
        cmd = [tool, "-o", str(out_c), str(bin_path)]
        try:
            subprocess.run(cmd, check=True, timeout=timeout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"RetDec failed: {e}")
        except subprocess.TimeoutExpired:
            raise TimeoutError("RetDec decompilation timed out")
        if not out_c.exists():
            raise RuntimeError("RetDec did not produce output C file")
        return out_c.read_text(encoding="utf-8", errors="ignore")


def _cache_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    cdir = root / "cache"
    cdir.mkdir(exist_ok=True)
    return cdir


def _cache_key(binary_path: str) -> str:
    p = Path(binary_path)
    stat = p.stat()
    return f"{p.name}.{int(stat.st_mtime)}.{stat.st_size}.c"


def decompile_with_retdec_cached(binary_path: str, timeout: int = 300) -> str:
    """Cached wrapper: reuse previous C output when the file timestamp/size match."""
    cdir = _cache_dir()
    key = _cache_key(binary_path)
    out_file = cdir / key
    meta_file = cdir / (key + ".meta")
    if out_file.exists():
        try:
            return out_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass
    text = decompile_with_retdec(binary_path, timeout=timeout)
    try:
        out_file.write_text(text, encoding="utf-8")
        meta_file.write_text(json.dumps({"created": time.time()}))
    except Exception:
        pass
    return text
