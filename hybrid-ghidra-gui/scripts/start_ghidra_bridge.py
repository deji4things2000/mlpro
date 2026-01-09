import os
import sys


def main():
    host = os.environ.get("GHIDRA_BRIDGE_HOST", "127.0.0.1")
    port = int(os.environ.get("GHIDRA_BRIDGE_PORT", "18001"))
    install_dir = os.environ.get("GHIDRA_INSTALL_DIR", "/Applications/ghidra/ghidra_12.0_PUBLIC")

    print(
        f"Starting PyGhidra with install_dir={install_dir}, "
        f"host={host}, port={port}"
    )

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
        print("Start the Bridge Server inside Ghidra, then re-run or connect from the GUI.")
        sys.exit(1)


if __name__ == "__main__":
    main()
