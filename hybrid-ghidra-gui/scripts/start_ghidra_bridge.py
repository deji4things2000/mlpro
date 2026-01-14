import os
import sys
from pathlib import Path


def _load_config():
    """Load config.json if present; fall back to env vars."""
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config.json"
    cfg = {}
    if cfg_path.exists():
        try:
            import json
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    return cfg


def main():
    cfg = _load_config()
    gh = cfg.get("ghidra", {})

    use_bridge = bool(gh.get("use_bridge", False))
    host = str(gh.get("host", os.environ.get("GHIDRA_BRIDGE_HOST", "127.0.0.1")))
    port = int(gh.get("port", int(os.environ.get("GHIDRA_BRIDGE_PORT", "18001"))))
    install_dir = str(gh.get("install_dir", os.environ.get("GHIDRA_INSTALL_DIR", "/Applications/ghidra/ghidra_12.0_PUBLIC")))

    print(
        f"PyGhidra helper: install_dir={install_dir}, host={host}, port={port}, use_bridge={use_bridge}"
    )

    if not use_bridge:
        print("Bridge disabled by config. Nothing to do.")
        print("Tip: Set ghidra.use_bridge=true in config.json to enable.")
        sys.exit(0)

    # Optional: start Ghidra UI via PyGhidra to assist the user; this does NOT start the bridge server
    try:
        import pyghidra
        os.environ["GHIDRA_INSTALL_DIR"] = install_dir
        start_fn = getattr(pyghidra, "start", None) or getattr(pyghidra, "start_ghidra", None)
        if start_fn:
            start_fn()
    except Exception as e:
        print(f"NOTE: Could not start Ghidra via PyGhidra: {e}")
        print("Open Ghidra manually if needed and start the bridge inside that session.")

    # Client-only: attempt to connect and report status
    try:
        from ghidra_bridge import GhidraBridge
        GhidraBridge(connect_to_host=host, connect_to_port=port)
        print("CONNECTED: Attached to existing Ghidra Bridge.")
        return
    except Exception as e:
        print(f"DISCONNECTED: Could not connect to Ghidra Bridge on {host}:{port}: {e}")
        print("Start the Bridge Server inside Ghidra (Tools > Python > Start Bridge), then re-run or connect from the GUI.")
        sys.exit(1)


if __name__ == "__main__":
    main()
