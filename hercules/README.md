# Hercules GPU Server Deployment

This directory contains files for deploying the WhisperSSTis transcription server on the Hercules GPU server.

## Server Specifications

- **GPUs**: NVIDIA RTX 5090 (32GB) + RTX 4090 (24GB)
- **Total VRAM**: ~56GB
- **IP Address**: 172.26.12.7
- **Service Port**: 8000

## Quick Start

### 1. SSH to Hercules

```bash
ssh hercules
# or
ssh magnus@172.26.12.7
```

### 2. Run the deployment script

```bash
cd ~/Documents/Github/whisperSSTis
bash hercules/deploy.sh
```

### 3. Start the server

```bash
cd ~/Documents/Github/whisperSSTis
source venv/bin/activate
uvicorn hercules_server:app --host 0.0.0.0 --port 8000
```

## Running as a Service (Optional)

To run the server as a systemd service that starts automatically:

```bash
# Copy service file
sudo cp hercules/whisper-server.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable whisper-server
sudo systemctl start whisper-server

# Check status
sudo systemctl status whisper-server
```

## Using tmux for Persistence

For development/testing, use tmux:

```bash
# Create new session
tmux new -s whisper

# Start server
cd ~/Documents/Github/whisperSSTis
source venv/bin/activate
uvicorn hercules_server:app --host 0.0.0.0 --port 8000

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -s whisper
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/transcribe` | POST | Transcribe audio file |
| `/llm` | POST | Ollama LLM completion |
| `/models` | GET | List Whisper models |
| `/ollama/models` | GET | List Ollama models |

### Example: Health Check

```bash
curl http://172.26.12.7:8000/health
```

### Example: Transcribe Audio

```bash
curl -X POST http://172.26.12.7:8000/transcribe \
  -F "file=@audio.wav" \
  -F "model_key=icelandic"
```

## Client Configuration

On your local machine, set these environment variables:

```bash
# In .env file
USE_HERCULES=true
HERCULES_URL=http://172.26.12.7:8000
```

Or toggle "Use Hercules (Remote GPU)" in the Streamlit sidebar.

## Requirements

The server needs these Python packages (installed via deploy.sh):

- fastapi
- uvicorn
- python-multipart
- httpx
- torch (with CUDA)
- transformers
- soundfile
- scipy
- numpy

## GPU Monitoring

```bash
# Check GPU status
nvidia-smi

# Watch GPU usage
watch -n 1 nvidia-smi
```

## Troubleshooting

### Connection refused
- Check if server is running: `systemctl status whisper-server`
- Check firewall: `sudo ufw status`
- Ensure port 8000 is open: `sudo ufw allow 8000`

### Out of memory
- The RTX 5090 (32GB) handles Whisper Large
- Check GPU memory: `nvidia-smi`

### Model loading slow
- First request loads the model (~1-2 minutes)
- Subsequent requests are fast (model cached)

## Ollama Setup

Ensure Ollama is running on Hercules:

```bash
# Check Ollama status
ollama list

# Pull models
ollama pull llama3
ollama pull mistral

# Ollama runs on port 11434 by default
```
