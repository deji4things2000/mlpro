# Hybrid Ghidra GUI

A lightweight, hybrid GUI that stubs Ghidra integration and LLM-assisted analysis. Left: binary explorer, middle: disassembly view with syntax highlighting, right: analysis summary.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r hybrid-ghidra-gui/requirements.txt
python hybrid-ghidra-gui/main.py
```

## Configuration

Edit `hybrid-ghidra-gui/config.json`:
- `theme`: UI theme ("light"|"dark")
- `ghidra_host`/`ghidra_port`: placeholder for future ghidra_bridge
- `enable_llm`: enable advanced LLM analysis (stubbed)

## Notes
- Current `core/ghidra_bridge.py` uses a simple byte-dump pseudo-disassembly to keep things runnable without external tools.
- `core/llm_analyzer.py` provides heuristic text and CFG stats; wire up your own LLM provider as needed.
