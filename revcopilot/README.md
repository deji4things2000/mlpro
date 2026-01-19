# RevCopilot

AI‑Powered Reverse Engineering Assistant

## Features
- 🧠 **AI‑Assisted Analysis**: GPT‑4 explains decompiled code
- ⚡ **Auto‑Solver**: Symbolic execution (angr) finds keys automatically
- 🎓 **Educational Mode**: Progressive hints for learning
- 🖥️ **Professional GUI**: VS Code‑like interface with disassembly viewer
- ☁️ **Cloud‑Ready**: Dockerized microservices architecture

## Quick Start

```bash
git clone https://github.com/yourusername/revcopilot.git
cd revcopilot
cp .env.example .env
# Edit .env with your OpenAI API key (optional)
docker-compose up
```

Open http://localhost:3000 and upload a binary.

## Modes

- Auto‑Solve: Fully automatic key extraction
- AI‑Assist: AI explains logic without spoiling
- Tutor Mode: Progressive hints for students

## Supported Binary Formats

- ELF (Linux)
- PE (Windows) – via Wine in emulation
- Mach‑O (macOS) – experimental

## Architecture

See docs/ARCHITECTURE.md

## License

MIT