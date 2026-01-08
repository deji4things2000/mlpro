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
