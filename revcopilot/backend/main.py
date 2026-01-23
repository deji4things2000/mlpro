from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os, uuid, json
from local_solver import LocalCrackmeSolver
import asyncio

app = FastAPI(title="RevCopilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In‑memory job store (replace with Redis in production)
jobs = {}

# Expected repo structure (for client-side comparison)
EXPECTED_STRUCTURE = [
    "revcopilot/",
    "revcopilot/.gitignore",
    "revcopilot/LICENSE",
    "revcopilot/README.md",
    "revcopilot/requirements.txt",
    "revcopilot/docker-compose.yml",
    "revcopilot/.env.example",
    "revcopilot/backend/",
    "revcopilot/backend/Dockerfile",
    "revcopilot/backend/requirements.txt",
    "revcopilot/backend/main.py",
    "revcopilot/backend/local_solver.py",
    "revcopilot/backend/ai_module.py",
    "revcopilot/backend/utils.py",
    "revcopilot/backend/models/__init__.py",
    "revcopilot/backend/models/analysis.py",
    "revcopilot/backend/api/__init__.py",
    "revcopilot/backend/api/endpoints.py",
    "revcopilot/backend/api/websockets.py",
    "revcopilot/frontend/",
    "revcopilot/frontend/Dockerfile",
    "revcopilot/frontend/package.json",
    "revcopilot/frontend/tsconfig.json",
    "revcopilot/frontend/next.config.js",
    "revcopilot/frontend/tailwind.config.js",
    "revcopilot/frontend/postcss.config.js",
    "revcopilot/frontend/public/favicon.ico",
    "revcopilot/frontend/public/logo.svg",
    "revcopilot/frontend/src/components/FileUpload.tsx",
    "revcopilot/frontend/src/components/CodeViewer.tsx",
    "revcopilot/frontend/src/components/ResultsPanel.tsx",
    "revcopilot/frontend/src/components/ControlFlowGraph.tsx",
    "revcopilot/frontend/src/components/Navbar.tsx",
    "revcopilot/frontend/src/components/ModeSelector.tsx",
    "revcopilot/frontend/src/pages/_app.tsx",
    "revcopilot/frontend/src/pages/index.tsx",
    "revcopilot/frontend/src/pages/results/[id].tsx",
    "revcopilot/frontend/src/styles/globals.css",
    "revcopilot/frontend/src/lib/api.ts",
    "revcopilot/frontend/src/lib/utils.ts",
    "revcopilot/frontend/.env.local.example",
    "revcopilot/ghidra_service/Dockerfile",
    "revcopilot/ghidra_service/server.py",
    "revcopilot/ghidra_service/requirements.txt",
    "revcopilot/docs/API.md",
    "revcopilot/docs/ARCHITECTURE.md",
    "revcopilot/docs/SETUP.md",
    "revcopilot/tests/test_solver.py",
    "revcopilot/tests/test_api.py",
    "revcopilot/tests/test_data/medium.bin",
]

@app.post("/analyze")
async def analyze_binary(
    file: UploadFile,
    mode: str = "auto",
    background_tasks: BackgroundTasks = None
):
    """Upload binary and start analysis."""
    # Save file
    file_id = str(uuid.uuid4())
    temp_path = f"/tmp/revcopilot_{file_id}.bin"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Initialize job
    jobs[file_id] = {
        "status": "processing",
        "mode": mode,
        "result": None,
        "error": None
    }
    
    # Process in background
    if background_tasks:
        background_tasks.add_task(process_analysis, file_id, temp_path, mode)
    else:
        asyncio.create_task(process_analysis(file_id, temp_path, mode))
    
    return JSONResponse({
        "job_id": file_id,
        "status": "started",
        "message": f"Analysis started in {mode} mode."
    })

async def process_analysis(file_id: str, path: str, mode: str):
    """Background analysis task."""
    try:
        if mode == "auto":
            solver = LocalCrackmeSolver(path)
            result = solver.solve()
            jobs[file_id]["result"] = result
        elif mode == "ai":
            # AI analysis stub
            jobs[file_id]["result"] = {"ai_insights": "AI analysis pending"}
        elif mode == "tutor":
            jobs[file_id]["result"] = {"hints": ["Hint 1: Check argv length"]}
        
        jobs[file_id]["status"] = "completed"
    except Exception as e:
        jobs[file_id]["status"] = "error"
        jobs[file_id]["error"] = str(e)
    finally:
        # Cleanup temp file
        try:
            os.unlink(path)
        except:
            pass

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Get analysis results."""
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    
    return JSONResponse(jobs[job_id])

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/structure")
async def structure():
    return {"expected": EXPECTED_STRUCTURE, "note": "This API does not access your file system."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)