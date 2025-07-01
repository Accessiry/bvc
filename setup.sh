#!/bin/bash

# BVC Vulnerability Detection System - Development Server Setup

echo "=== BVC Development Server Setup ==="

# Check if Python is installed and get version
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check Python version compatibility
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detected Python version: $PYTHON_VERSION"

# Check if Python version is supported
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "Python version is compatible (>= 3.8)"
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)"; then
        echo "Note: Using Python 3.12+ - eventlet has been replaced with threading for SocketIO"
    fi
else
    echo "Error: Python 3.8 or higher is required. Found: $PYTHON_VERSION"
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

# Try to install dependencies with better error handling
if ! pip install -r requirements.txt --timeout=120; then
    echo "Warning: Failed to install some dependencies. This might be due to:"
    echo "  - Network connectivity issues"
    echo "  - Python version compatibility problems"
    echo "  - Missing system dependencies"
    echo ""
    echo "You can try installing dependencies manually with:"
    echo "  pip install Flask Flask-CORS Flask-SocketIO SQLAlchemy pandas numpy Werkzeug"
    echo ""
    echo "For Python 3.12+, eventlet is not supported. The system will use threading mode."
    read -p "Continue setup anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

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
echo "Python version detected: $PYTHON_VERSION"
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)"; then
    echo "Note: Python 3.12+ detected - using threading mode for WebSocket backend"
fi
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
echo "- gunicorn --worker-class gthread --threads 4 --bind 0.0.0.0:5000 training:app"
echo "  (Note: Using gthread instead of eventlet for Python 3.12+ compatibility)"