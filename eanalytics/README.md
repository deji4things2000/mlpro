# OmniAnalytics Python Edition

A universal analytics platform built with Python, FastAPI, PostgreSQL, Redis, and a simple HTML/JS frontend.

## Features
- Modular FastAPI backend (API, connectivity, query engine, AI service, semantic layer)
- PostgreSQL, Redis, and multiple data source connectors
- AI/ML integration (OpenAI, LangChain, ChromaDB)
- Simple HTML/JS frontend (no build step required)
- Docker Compose for local database and cache
- One-step setup script

## Quick Start


```bash
# 1. Clone the repo and enter the directory
cd omnianalytics-python

# 2. Run the setup script
chmod +x setup.sh
./setup.sh

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Install dependencies (if needed)
pip install -r requirements.txt

# 5. Start the backend (from the project root):
PYTHONPATH=. uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# 6. Start the frontend (in a new terminal):
cd frontend
python3 -m http.server 8080
```

- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend: http://localhost:8080

## Manual Setup


1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start PostgreSQL and Redis (optional, uses SQLite if not available):
   ```bash
   docker-compose up -d
   ```
3. Run the backend (from the project root):
   ```bash
   PYTHONPATH=. uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
   ```
4. Serve the frontend (in a new terminal):
   ```bash
   cd frontend
   python3 -m http.server 8080
   ```
5. Open the frontend in your browser:
   - http://localhost:8080
## Troubleshooting

- If you see `ModuleNotFoundError: No module named 'backend'`, make sure you are running uvicorn from the omnianalytics-python directory and set `PYTHONPATH=.`
- If a port is already in use, kill the process using it (e.g., `lsof -i :8000` and `kill <PID>`).
- If you see missing dependencies, activate the virtual environment and run `pip install -r requirements.txt`.


## Directory Structure

```
omnianalytics-python/
├── requirements.txt
├── run.py
├── setup.sh
├── docker-compose.yml
├── .env
├── backend/
│   ├── api/
│   ├── connectivity/
│   ├── query_engine/
│   ├── ai_service/
│   └── semantic_layer/
├── frontend/
│   └── index.html
├── shared/
└── docker/
```
