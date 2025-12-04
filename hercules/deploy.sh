#!/bin/bash
# Hercules Deployment Script for WhisperSSTis
# Run this on the Hercules server to set up the GPU transcription service

set -e

echo "=== WhisperSSTis Hercules Server Setup ==="

# Configuration
PROJECT_DIR="${HOME}/Documents/Github/whisperSSTis"
VENV_DIR="${PROJECT_DIR}/venv"
SERVICE_PORT=8000

# Check if running on Hercules
if ! nvidia-smi &>/dev/null; then
    echo "Warning: NVIDIA GPU not detected. The server will run on CPU."
fi

# Create project directory if needed
mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"

# Clone or update repository
if [ -d ".git" ]; then
    echo "Updating existing repository..."
    git pull origin main
else
    echo "Cloning repository..."
    git clone https://github.com/Magnussmari/whisperSSTis.git .
fi

# Create virtual environment
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
fi

# Activate and install dependencies
echo "Installing dependencies..."
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart httpx scipy

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the server:"
echo "  cd ${PROJECT_DIR}"
echo "  source venv/bin/activate"
echo "  uvicorn hercules_server:app --host 0.0.0.0 --port ${SERVICE_PORT}"
echo ""
echo "Or use tmux for persistence:"
echo "  tmux new -s whisper"
echo "  cd ${PROJECT_DIR} && source venv/bin/activate"
echo "  uvicorn hercules_server:app --host 0.0.0.0 --port ${SERVICE_PORT}"
echo "  # Press Ctrl+B, then D to detach"
echo ""
echo "The server will be available at: http://172.26.12.7:${SERVICE_PORT}"
echo ""
