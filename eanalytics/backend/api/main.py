from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import logging
from typing import List, Optional

from .config import settings
from .database import engine, Base, get_db
from .auth import get_current_user
from .routers import data_sources, queries, dashboards, visualizations, ai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    logger.info("Starting up OmniAnalytics API...")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Initialize connections
    from .database import init_redis
    await init_redis()
    
    yield
    
    logger.info("Shutting down OmniAnalytics API...")
    await engine.dispose()

# Create FastAPI app
app = FastAPI(
    title="OmniAnalytics API",
    description="Universal Analytics Platform API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS,
)

# Include routers
app.include_router(
    data_sources.router,
    prefix="/api/data-sources",
    tags=["Data Sources"],
    dependencies=[Depends(get_current_user)],
)

app.include_router(
    queries.router,
    prefix="/api/queries",
    tags=["Queries"],
    dependencies=[Depends(get_current_user)],
)

app.include_router(
    dashboards.router,
    prefix="/api/dashboards",
    tags=["Dashboards"],
    dependencies=[Depends(get_current_user)],
)

app.include_router(
    visualizations.router,
    prefix="/api/visualizations",
    tags=["Visualizations"],
    dependencies=[Depends(get_current_user)],
)

app.include_router(
    ai.router,
    prefix="/api/ai",
    tags=["AI"],
    dependencies=[Depends(get_current_user)],
)

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "omnianalytics-api",
        "version": "1.0.0",
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to OmniAnalytics API",
        "docs": "/docs" if settings.DEBUG else None,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )