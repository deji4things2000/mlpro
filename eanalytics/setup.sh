#!/bin/bash

echo "🚀 Setting up OmniAnalytics Platform..."

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p backend/api/uploads
mkdir -p logs

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "🔧 Creating .env file..."
    cat > .env << EOF
DEBUG=True
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/omnianalytics
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key-change-in-production
OPENAI_API_KEY=your-openai-api-key-here
EOF
fi

# Check if Docker is available for database
if command -v docker &> /dev/null; then
    echo "🐳 Starting database services with Docker..."
    docker-compose up -d
    echo "⏳ Waiting for databases to start..."
    sleep 5
else
    echo "⚠️  Docker not found. Using SQLite for demo (data won't persist)"
    echo "DATABASE_URL=sqlite+aiosqlite:///./omnianalytics.db" >> .env
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "1. Activate virtual environment: source .venv/bin/activate"
echo "2. Run backend: python run.py"
echo "3. Open frontend: open frontend/index.html (or double-click it)"
echo ""
echo "📊 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🌐 Frontend: file://$(pwd)/frontend/index.html"