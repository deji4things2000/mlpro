# Hybrid Ghidra GUI

A PyQt5-based GUI that integrates a Binary Explorer, Disassembly view, and LLM-assisted analysis with a Control Flow Graph (CFG) panel. The code is modular and designed to work cleanly with Ghidra 12 using a client-only bridge.

## Features
- Binary Explorer with functions list
- Disassembly and decompiled views with syntax highlighting
- LLM analysis summary/prediction/fix placeholders
- CFG builder stub and annotation stub via Ghidra

## Quick Start

### 1. Create a virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python main.py
```

## Configuration
- Edit `config.json` to set theme and integration settings.
- Important (macOS default): set `ghidra.install_dir` (e.g. `/Applications/ghidra/ghidra_12.0_PUBLIC`).
- Optional env vars used by scripts and the app:
```bash
export GHIDRA_INSTALL_DIR=/Applications/ghidra/ghidra_12.0_PUBLIC
export GHIDRA_BRIDGE_HOST=127.0.0.1
export GHIDRA_BRIDGE_PORT=18001
```

## Client-Only Ghidra Bridge (Ghidra 12 + PyGhidra)

This app connects to an existing bridge; it does not start or manage the bridge server.

1) Launch Ghidra
- Start Ghidra normally, or optionally use the helper script:
```bash
python hybrid-ghidra-gui/scripts/start_ghidra_bridge.py
```
This attempts to launch Ghidra via PyGhidra and then attaches as a client if a bridge is already running.

2) Start the Bridge inside Ghidra
- Enable the bridge server from within your Ghidra environment (extension or bundled script).
- The server should listen on the host/port in `config.json` or the env vars above.

3) Connect from the App
- Run the GUI:
```bash
python hybrid-ghidra-gui/main.py
```
- Use the Bridge menu to:
	- Connect to Bridge
	- Force Reconnect
	- Refresh Status

4) Verify
- In terminal:
```bash
lsof -i :18001 -sTCP:LISTEN -n -P
python -c 'from ghidra_bridge import GhidraBridge as GB; GB(connect_to_host="127.0.0.1", connect_to_port=18001); print("CONNECTED")'
```

## Behavior
- When connected and a program is open in Ghidra, the functions list populates from Ghidra.
- Selecting a function shows disassembly from Ghidra; decompiler output appears when available.
- If the bridge is unavailable, the app falls back to a local disassembly preview for supported binaries.

## Troubleshooting
- Confirm the install path exists:
```bash
ls -d /Applications/ghidra/ghidra_12.0_PUBLIC
```
- If connect fails, ensure the bridge server in Ghidra is running and matches `host`/`port`.
- Update `config.json` or env vars and try Bridge → Force Reconnect.

## Notes
- The app uses `ghidra-bridge` in client mode only.
- PyGhidra is used to assist in starting Ghidra; server lifecycle is owned by Ghidra.
- LLM features are placeholders; wire them to real services under `core/` when ready.
- Tested with PyQt5 5.15+.
