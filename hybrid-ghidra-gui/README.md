# Hybrid Ghidra GUI

A lightweight, hybrid GUI that stubs Ghidra integration and LLM-assisted analysis.

Left: binary explorer, middle: disassembly view with syntax highlighting, right: analysis summary.

## Run the App

1) Create a virtual environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r hybrid-ghidra-gui/requirements.txt
```

2) Launch the GUI
```bash
python hybrid-ghidra-gui/main.py
```

When the window opens, click "Open Binary…" and select any file to view a pseudo-disassembly.

## Quick Demo: Generate and Open a Sample Binary

Generate a small x86-ish sample binary and then load it in the GUI:
```bash
source .venv/bin/activate
python hybrid-ghidra-gui/scripts/gen_sample_binary.py
python hybrid-ghidra-gui/main.py
```

In the app, choose `hybrid-ghidra-gui/assets/sample.bin` when prompted. The middle panel will render pseudo-disassembly; the right panel shows a short heuristic analysis.

## Configuration

Edit `hybrid-ghidra-gui/config.json`:
- `theme`: UI theme ("light"|"dark")
- `ghidra_host` / `ghidra_port`: placeholder for future ghidra_bridge
- `enable_llm`: enable advanced LLM analysis (stubbed)

## Notes
- Current `core/ghidra_bridge.py` uses a simple byte-dump pseudo-disassembly to keep things runnable without external tools.
- `core/llm_analyzer.py` provides heuristic text and CFG stats; wire up your own LLM provider as needed.
