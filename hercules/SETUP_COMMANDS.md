# Hercules Server Setup Commands

Complete terminal commands to set up WhisperSSTis on Hercules.

---

## 1. Connect to Hercules

```bash
# From your local machine (must be on VPN)
ssh hercules
# or
ssh magnus@172.26.12.7
```

---

## 2. System Updates (Optional)

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Check NVIDIA drivers
nvidia-smi

# Install Python dev tools if needed
sudo apt install python3-pip python3-venv python3-dev -y
```

---

## 3. Clone Repository

```bash
# Create project directory
mkdir -p ~/Documents/Github
cd ~/Documents/Github

# Clone the repository
git clone https://github.com/Magnussmari/whisperSSTis.git
cd whisperSSTis

# Or if already cloned, update it
git pull origin main
```

---

## 4. Python Environment Setup

```bash
cd ~/Documents/Github/whisperSSTis

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

---

## 5. Install Dependencies

```bash
# Make sure venv is activated
source ~/Documents/Github/whisperSSTis/venv/bin/activate

# Install base requirements
pip install -r requirements.txt

# Install server-specific dependencies
pip install fastapi uvicorn python-multipart httpx scipy

# Install PyTorch with CUDA support (for RTX 5090/4090)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 6. Verify GPU Access

```bash
# Check CUDA is available in Python
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

# Should output:
# CUDA: True
# GPUs: 2

# Check GPU names
python3 -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Should output:
# GPU 0: NVIDIA GeForce RTX 5090
# GPU 1: NVIDIA GeForce RTX 4090
```

---

## 7. Set Up Ollama (for LLM)

```bash
# Check if Ollama is installed
ollama --version

# If not installed, install it
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
sudo systemctl enable ollama
sudo systemctl start ollama

# Pull required models
ollama pull llama3
ollama pull mistral

# Verify models
ollama list
```

---

## 8. Start the Server (Manual)

```bash
cd ~/Documents/Github/whisperSSTis
source venv/bin/activate

# Start the server
uvicorn hercules_server:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
uvicorn hercules_server:app --host 0.0.0.0 --port 8000 --reload
```

---

## 9. Start with tmux (Persistent)

```bash
# Create a new tmux session
tmux new -s whisper

# Inside tmux, start the server
cd ~/Documents/Github/whisperSSTis
source venv/bin/activate
uvicorn hercules_server:app --host 0.0.0.0 --port 8000

# Detach from tmux: Ctrl+B, then D
# The server keeps running in the background

# Reattach later
tmux attach -s whisper

# List sessions
tmux ls

# Kill session when done
tmux kill-session -s whisper
```

---

## 10. Install as Systemd Service (Production)

```bash
# Copy service file
sudo cp ~/Documents/Github/whisperSSTis/hercules/whisper-server.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable whisper-server

# Start the service
sudo systemctl start whisper-server

# Check status
sudo systemctl status whisper-server

# View logs
sudo journalctl -u whisper-server -f

# Stop/restart commands
sudo systemctl stop whisper-server
sudo systemctl restart whisper-server
```

---

## 11. Firewall Configuration

```bash
# Check firewall status
sudo ufw status

# Allow port 8000 for the API
sudo ufw allow 8000/tcp

# If using Ollama externally (optional)
sudo ufw allow 11434/tcp

# Verify
sudo ufw status numbered
```

---

## 12. Test the Server

```bash
# From Hercules
curl http://localhost:8000/health

# From your local machine
curl http://172.26.12.7:8000/health

# Or run the TUI test script locally
python hercules_test.py
```

---

## 13. Pre-load Whisper Models (Optional)

The first transcription request will download and load the model. To pre-load:

```bash
cd ~/Documents/Github/whisperSSTis
source venv/bin/activate

# Pre-download Icelandic model
python3 -c "
from transformers import WhisperProcessor, WhisperForConditionalGeneration
model_id = 'carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h'
print('Downloading model...')
WhisperProcessor.from_pretrained(model_id)
WhisperForConditionalGeneration.from_pretrained(model_id)
print('Done!')
"

# Pre-download English model
python3 -c "
from transformers import WhisperProcessor, WhisperForConditionalGeneration
model_id = 'openai/whisper-large-v3'
print('Downloading model...')
WhisperProcessor.from_pretrained(model_id)
WhisperForConditionalGeneration.from_pretrained(model_id)
print('Done!')
"
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Connect to Hercules | `ssh hercules` |
| Activate venv | `source ~/Documents/Github/whisperSSTis/venv/bin/activate` |
| Start server | `uvicorn hercules_server:app --host 0.0.0.0 --port 8000` |
| Check GPUs | `nvidia-smi` |
| Check server | `curl http://localhost:8000/health` |
| View logs | `sudo journalctl -u whisper-server -f` |
| Restart service | `sudo systemctl restart whisper-server` |

---

## Troubleshooting

### Server won't start
```bash
# Check if port is in use
sudo lsof -i :8000

# Kill process using port
sudo kill -9 $(sudo lsof -t -i :8000)
```

### CUDA out of memory
```bash
# Check GPU memory
nvidia-smi

# Clear CUDA cache (restart server)
sudo systemctl restart whisper-server
```

### Model download fails
```bash
# Set Hugging Face cache directory
export HF_HOME=~/.cache/huggingface

# Or use a different location with more space
export HF_HOME=/data/huggingface
```

### Ollama not responding
```bash
# Check Ollama status
sudo systemctl status ollama

# Restart Ollama
sudo systemctl restart ollama

# Check Ollama logs
sudo journalctl -u ollama -f
```
