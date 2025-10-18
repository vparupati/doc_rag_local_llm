#!/bin/bash

# RAG Application Startup Script

echo "Starting RAG Document Query System..."
echo "====================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/vectorstore data/chroma uploads logs

# Setup LLM services
echo "Setting up LLM services..."
python setup_llm.py

# Run tests
echo "Running tests..."
python test_app.py

# Start the application
echo "Starting application..."
echo "The application will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop the application"
echo ""

python main.py
