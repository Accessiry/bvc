#!/bin/bash

# BVC Vulnerability Detection System - Development Server Setup

echo "=== BVC Development Server Setup ==="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
cd web/api
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p ../assets/{data,models,datasets,images}

# Initialize database
echo "Initializing database..."
python3 training.py &
SERVER_PID=$!
sleep 3
kill $SERVER_PID

echo "=== Setup Complete ==="
echo ""
echo "To start the development server:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Start backend server: cd web/api && python3 training.py"
echo "3. Open web/index.html in your browser"
echo ""
echo "Server will be available at:"
echo "- Web Interface: Open web/index.html in browser"
echo "- API Endpoint: http://localhost:5000"
echo "- WebSocket: ws://localhost:5000"
echo ""
echo "For production deployment, consider using:"
echo "- gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 training:app"