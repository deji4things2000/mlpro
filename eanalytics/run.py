#!/usr/bin/env python3
import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def run_backend():
    """Start the FastAPI backend server."""
    print("🚀 Starting OmniAnalytics Backend...")
    
    # Set environment variables
    os.environ['DEBUG'] = 'True'
    os.environ['DATABASE_URL'] = 'postgresql+asyncpg://postgres:password@localhost:5432/omnianalytics'
    os.environ['REDIS_URL'] = 'redis://localhost:6379'
    
    # Start backend
    backend_dir = Path(__file__).parent / 'backend' / 'api'
    os.chdir(backend_dir)
    
    try:
        subprocess.run([
            'uvicorn', 'main:app',
            '--host', '127.0.0.1',
            '--port', '8000',
            '--reload'
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")

def run_frontend():
    """Open the frontend in browser."""
    print("🌐 Opening OmniAnalytics Frontend...")
    
    frontend_file = Path(__file__).parent / 'frontend' / 'index.html'
    
    if frontend_file.exists():
        webbrowser.open(f'file://{frontend_file.absolute()}')
        print("✅ Frontend opened in browser")
        print("📊 Open http://localhost:8000/docs for API documentation")
    else:
        print("❌ Frontend file not found")

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    try:
        import fastapi
        import uvicorn
        import asyncpg
        import redis
        print("✅ All dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Please run: pip install -r requirements.txt")
        sys.exit(1)

def setup_database():
    """Set up PostgreSQL and Redis."""
    print("🗄️  Setting up database...")
    
    # Check if Docker is running
    try:
        subprocess.run(['docker', '--version'], check=True, capture_output=True)
        print("✅ Docker is available")
        
        # Start PostgreSQL and Redis with Docker
        print("🐳 Starting PostgreSQL and Redis...")
        
        docker_compose = """
version: '3.8'
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: omnianalytics
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
"""
        
        # Write docker-compose.yml
        compose_file = Path(__file__).parent / 'docker-compose.yml'
        compose_file.write_text(docker_compose)
        
        # Start services
        subprocess.run(['docker-compose', 'up', '-d'], cwd=Path(__file__).parent)
        print("✅ Database services started")
        
    except Exception as e:
        print(f"⚠️  Could not start Docker services: {e}")
        print("📝 Using in-memory mode (data will not persist)")

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════╗
    ║      OmniAnalytics Platform              ║
    ║      Universal Analytics Solution        ║
    ╚══════════════════════════════════════════╝
    """)
    
    # Check dependencies
    check_dependencies()
    
    # Setup database
    setup_database()
    
    # Open frontend
    run_frontend()
    
    # Start backend
    run_backend()