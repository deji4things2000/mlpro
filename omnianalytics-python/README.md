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

# 4. Start the platform
python run.py
```

- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend: file://$(pwd)/frontend/index.html

## Manual Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start PostgreSQL and Redis (optional, uses SQLite if not available):
   ```bash
   docker-compose up -d
   ```
3. Run the backend:
   ```bash
   cd backend/api
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```
4. Open the frontend in your browser:
   ```bash
   open frontend/index.html  # macOS
   xdg-open frontend/index.html  # Linux
   # or double-click the file
   ```

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
