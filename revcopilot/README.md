# RevCopilot

AI‑Powered Reverse Engineering Assistant

## Features
- 🧠 **AI‑Assisted Analysis**: GPT‑4 explains decompiled code
- ⚡ **Auto‑Solver**: Symbolic execution (angr) finds keys automatically
- 🎓 **Educational Mode**: Progressive hints for learning
- 🖥️ **Professional GUI**: VS Code‑like interface with disassembly viewer
- ☁️ **Cloud‑Ready**: Dockerized microservices architecture

## Quick Start (Docker - full stack)

Recommended: start the full stack with Docker Compose.

```bash
# from repository root
cd revcopilot
# If you have an example .env, copy it (only if you need to customize env vars)
# If .env.example is not present, the compose file will use defaults or environment values
# cp .env.example .env
docker compose up --build
```

Open http://localhost:3000 in your browser (Next.js frontend). The backend listens on port 8000 inside the stack.

## Start Backend Locally (no Docker)

If you prefer to run only the backend locally (useful for development or debugging), use a Python virtual environment and run `uvicorn` from the `backend` directory so imports resolve correctly:

```bash
# from repository root
cd revcopilot

# create a venv and install dependencies (one-time)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r backend/requirements.txt

# start the backend (run this inside the `backend` folder so `main.py` is importable)
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Alternatively, without activating the venv:
# ./ .venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Notes:
- Run `uvicorn main:app` from the `backend` directory (not from repo root) to avoid "ModuleNotFoundError: No module named 'backend'".
- If you prefer to run `uvicorn` from the repo root, set `PYTHONPATH` to include the repo, e.g.:

```bash
PYTHONPATH=. uvicorn revcopilot.backend.main:app --host 0.0.0.0 --port 8000
```

## Running Frontend Locally

The frontend is a Next.js app located at `revcopilot/frontend`. If `package.json` is missing but `package-lock.json` exists, you can reconstruct a minimal `package.json` from the lockfile and run:

```bash
cd revcopilot/frontend
npm ci
npm run dev
```

The frontend will attempt to reach the backend at `http://localhost:8000` by default; set `NEXT_PUBLIC_BACKEND_URL` to override.

## Troubleshooting

- If `uvicorn` is not found, ensure you installed `backend/requirements.txt` into the active venv.
- If you see `ModuleNotFoundError: No module named 'backend'`, start `uvicorn` from the `backend` directory or set `PYTHONPATH` as shown above.
- If Docker Compose fails because `.env.example` is missing, you can either create a `.env` with required values or run compose without it — the compose file may contain defaults.

## Docs

See `docs/ARCHITECTURE.md` and `docs/SETUP.md` for more details.

## License

MIT