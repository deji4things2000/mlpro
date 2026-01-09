# Hybrid Ghidra GUI

A PyQt5-based GUI that integrates a Binary Explorer, Disassembly view, and LLM-assisted analysis with a Control Flow Graph (CFG) panel. The structure modularizes the original monolithic implementation.

## Features
- Binary Explorer with function list and risk filters
- Disassembly and decompiled code views with syntax highlighting
- LLM analysis summary/prediction/fix placeholders
- CFG builder stub and annotation stub for Ghidra

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
Edit `config.json` to adjust theme and integration settings.

## Notes
- Ghidra/LLM integrations are stubs; wire them to real services in `core/` when ready.
- Tested with PyQt5 5.15+.

## Ghidra Bridge Setup (Ghidra 12 via PyGhidra)

Enable live disassembly/decompilation and annotations via Ghidra Bridge:

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Configure Ghidra
- Set `ghidra.install_dir` in `config.json` (e.g., macOS: `/Applications/ghidra/ghidra_12.0_PUBLIC`).
- Optionally set env vars:
```bash
export GHIDRA_INSTALL_DIR=/Applications/ghidra/ghidra_12.0_PUBLIC
export GHIDRA_BRIDGE_HOST=127.0.0.1
export GHIDRA_BRIDGE_PORT=18001
```

3) Start the Bridge Server
```bash
python hybrid-ghidra-gui/scripts/start_ghidra_bridge.py
```
This launches Ghidra via PyGhidra and starts the bridge on the configured host/port.

4) Use the App
```bash
python hybrid-ghidra-gui/main.py
```
Set `ghidra.use_bridge` to `true` in `config.json` to enable integration.

Notes:
- If the bridge isn’t running, the app falls back to local disassembly preview for supported binaries.
- To change host/port, edit `config.json` or use env vars; the app picks them up at startup.
