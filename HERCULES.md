# WhisperSSTis Hercules Integration Guide

## Overview

The WhisperSSTis app can offload GPU-intensive tasks to Hercules, a remote server with powerful GPUs. This enables fast transcription and LLM processing without requiring local GPU resources.

---

## Architecture

```
┌─────────────────────┐         ┌─────────────────────────────────────┐
│   Local Mac (App)   │   HTTP  │         Hercules Server             │
│                     │ ──────► │                                     │
│  - Streamlit UI     │         │  - FastAPI (port 8000)              │
│  - Audio recording  │ ◄────── │  - Whisper models (GPU)             │
│  - File handling    │         │  - Ollama LLMs                      │
│                     │         │  - RTX 4090 (24GB VRAM)             │
└─────────────────────┘         └─────────────────────────────────────┘
```

---

## Server Details

| Property | Value |
|----------|-------|
| Host | `172.26.12.7` |
| Port | `8000` |
| Base URL | `http://172.26.12.7:8000` |
| API Docs | `http://172.26.12.7:8000/docs` |
| GPU | NVIDIA RTX 4090 (24GB) |

**Requirements:** Must be connected to UNAK VPN

---

## API Endpoints

### Health Check
```
GET /health
```
Returns server status, GPU info, loaded models, and Ollama status.

**Response:**
```json
{
  "status": "healthy",
  "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 4090", "memory_total_gb": 25.3}],
  "models_loaded": ["icelandic"],
  "ollama_status": "online"
}
```

---

### Transcribe Audio
```
POST /transcribe
Content-Type: multipart/form-data
```

**Parameters:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| file | File | required | Audio file (WAV recommended) |
| model | string | "icelandic" | Model: "icelandic" or "english" |
| long_audio | bool | false | Use chunked transcription |
| chunk_size | int | 30 | Chunk size in seconds |

**Response:**
```json
{
  "text": "Transcribed text here...",
  "model": "icelandic",
  "duration": 45.2,
  "chunks": ["[0:00:00 → 0:00:30] First chunk...", "..."]
}
```

---

### Transcribe to SRT
```
POST /transcribe/srt
Content-Type: multipart/form-data
```

Same parameters as `/transcribe`. Returns SRT subtitle format.

**Response:**
```json
{
  "srt": "1\n0:00:00,000 --> 0:00:30,000\nFirst subtitle...\n\n2\n...",
  "text": "Full transcription...",
  "model": "icelandic",
  "duration": 120.5
}
```

---

### Summarize/Process Text
```
POST /summarize
Content-Type: application/json
```

**Request Body:**
```json
{
  "text": "The transcript text to process...",
  "instruction": "Summarize this transcript concisely",
  "model": "mistral"  // optional, defaults to mistral
}
```

**Response:**
```json
{
  "result": "Summary of the transcript...",
  "model": "mistral"
}
```

---

### List Models
```
GET /models
```
Returns available Whisper models.

```
GET /ollama/models
```
Returns available Ollama LLM models.

---

## Python Integration Examples

### Using httpx (async)

```python
import httpx

HERCULES_URL = "http://172.26.12.7:8000"

async def check_hercules_health():
    """Check if Hercules server is available."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{HERCULES_URL}/health")
            return response.status_code == 200
    except Exception:
        return False

async def transcribe_on_hercules(audio_path: str, model: str = "icelandic") -> dict:
    """Send audio file to Hercules for transcription."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path, f, "audio/wav")}
            data = {"model": model}
            response = await client.post(
                f"{HERCULES_URL}/transcribe",
                files=files,
                data=data,
            )
            response.raise_for_status()
            return response.json()

async def summarize_on_hercules(text: str, instruction: str = "Summarize") -> str:
    """Send text to Hercules for LLM processing."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{HERCULES_URL}/summarize",
            json={"text": text, "instruction": instruction},
        )
        response.raise_for_status()
        return response.json()["result"]
```

### Using requests (sync)

```python
import requests

HERCULES_URL = "http://172.26.12.7:8000"

def check_hercules_health() -> bool:
    """Check if Hercules server is available."""
    try:
        response = requests.get(f"{HERCULES_URL}/health", timeout=10)
        return response.status_code == 200
    except Exception:
        return False

def transcribe_on_hercules(audio_path: str, model: str = "icelandic") -> dict:
    """Send audio file to Hercules for transcription."""
    with open(audio_path, "rb") as f:
        files = {"file": (audio_path, f, "audio/wav")}
        data = {"model": model}
        response = requests.post(
            f"{HERCULES_URL}/transcribe",
            files=files,
            data=data,
            timeout=300,
        )
        response.raise_for_status()
        return response.json()

def summarize_on_hercules(text: str, instruction: str = "Summarize") -> str:
    """Send text to Hercules for LLM processing."""
    response = requests.post(
        f"{HERCULES_URL}/summarize",
        json={"text": text, "instruction": instruction},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["result"]
```

---

## Integration Pattern for App

### Recommended Approach

Create a `hercules_client.py` module:

```python
"""Client for Hercules GPU server."""

import os
from typing import Optional
import httpx

HERCULES_URL = os.getenv("HERCULES_URL", "http://172.26.12.7:8000")
HERCULES_TIMEOUT = float(os.getenv("HERCULES_TIMEOUT", "300"))


class HerculesClient:
    """Client for interacting with Hercules GPU server."""
    
    def __init__(self, base_url: str = HERCULES_URL):
        self.base_url = base_url
        self._available: Optional[bool] = None
    
    async def is_available(self) -> bool:
        """Check if Hercules is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                self._available = response.status_code == 200
                return self._available
        except Exception:
            self._available = False
            return False
    
    async def get_status(self) -> dict:
        """Get detailed server status."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
    
    async def transcribe(
        self,
        audio_path: str,
        model: str = "icelandic",
        long_audio: bool = False,
        chunk_size: int = 30,
    ) -> dict:
        """Transcribe audio file on Hercules GPU."""
        async with httpx.AsyncClient(timeout=HERCULES_TIMEOUT) as client:
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
                data = {
                    "model": model,
                    "long_audio": str(long_audio).lower(),
                    "chunk_size": str(chunk_size),
                }
                response = await client.post(
                    f"{self.base_url}/transcribe",
                    files=files,
                    data=data,
                )
                response.raise_for_status()
                return response.json()
    
    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        model: str = "icelandic",
    ) -> dict:
        """Transcribe audio bytes on Hercules GPU."""
        async with httpx.AsyncClient(timeout=HERCULES_TIMEOUT) as client:
            files = {"file": (filename, audio_bytes, "audio/wav")}
            data = {"model": model}
            response = await client.post(
                f"{self.base_url}/transcribe",
                files=files,
                data=data,
            )
            response.raise_for_status()
            return response.json()
    
    async def summarize(
        self,
        text: str,
        instruction: str = "Summarize this transcript concisely",
        model: Optional[str] = None,
    ) -> str:
        """Process text with Ollama LLM on Hercules."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {"text": text, "instruction": instruction}
            if model:
                payload["model"] = model
            response = await client.post(
                f"{self.base_url}/summarize",
                json=payload,
            )
            response.raise_for_status()
            return response.json()["result"]
    
    async def get_srt(
        self,
        audio_path: str,
        model: str = "icelandic",
        chunk_size: int = 30,
    ) -> dict:
        """Transcribe audio and get SRT subtitles."""
        async with httpx.AsyncClient(timeout=HERCULES_TIMEOUT) as client:
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
                data = {"model": model, "chunk_size": str(chunk_size)}
                response = await client.post(
                    f"{self.base_url}/transcribe/srt",
                    files=files,
                    data=data,
                )
                response.raise_for_status()
                return response.json()


# Singleton instance
hercules = HerculesClient()
```

### Usage in Streamlit App

```python
import asyncio
from hercules_client import hercules

# Check availability
if asyncio.run(hercules.is_available()):
    st.success("🖥️ Hercules GPU server connected")
else:
    st.warning("⚠️ Hercules unavailable - using local processing")

# Transcribe
result = asyncio.run(hercules.transcribe(audio_file_path, model="icelandic"))
st.write(result["text"])

# Summarize
summary = asyncio.run(hercules.summarize(transcript, "Summarize in bullet points"))
st.write(summary)
```

---

## Configuration

### Environment Variables

Add to `.env`:
```
HERCULES_URL=http://172.26.12.7:8000
HERCULES_TIMEOUT=300
```

### Fallback Strategy

The app should gracefully fall back to local processing if Hercules is unavailable:

```python
async def transcribe_audio(audio_path: str, model: str = "icelandic") -> str:
    """Transcribe audio, preferring Hercules if available."""
    
    # Try Hercules first
    if await hercules.is_available():
        try:
            result = await hercules.transcribe(audio_path, model=model)
            return result["text"]
        except Exception as e:
            logger.warning(f"Hercules transcription failed: {e}, falling back to local")
    
    # Fall back to local processing
    return local_transcribe(audio_path, model=model)
```

---

## Testing

### Command Line Tests

```bash
# Health check
curl http://172.26.12.7:8000/health

# Transcribe a file
curl -X POST "http://172.26.12.7:8000/transcribe" \
  -F "file=@test_audio.wav" \
  -F "model=icelandic"

# Summarize text
curl -X POST "http://172.26.12.7:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your transcript here...", "instruction": "Summarize"}'
```

### Python Test Script

```python
#!/usr/bin/env python3
"""Test Hercules connection and endpoints."""

import asyncio
from hercules_client import hercules

async def main():
    print("Testing Hercules connection...")
    
    # Health check
    if await hercules.is_available():
        print("✅ Hercules is online")
        status = await hercules.get_status()
        print(f"   GPUs: {status['gpus']}")
        print(f"   Models: {status['models_loaded']}")
        print(f"   Ollama: {status['ollama_status']}")
    else:
        print("❌ Hercules is offline")
        return
    
    # Test summarization
    print("\nTesting summarization...")
    result = await hercules.summarize(
        "Þetta er prufa á íslensku texta.",
        "Translate to English"
    )
    print(f"   Result: {result}")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Troubleshooting

### Connection Refused
- Verify VPN is connected: `ping 172.26.12.7`
- Check server is running: `ssh hercules 'pgrep -f uvicorn'`
- Start server: `ssh hercules 'cd ~/Documents/Github/whisperSSTis && source venv/bin/activate && CUDA_VISIBLE_DEVICES=0 uvicorn hercules_server:app --host 0.0.0.0 --port 8000'`

### Timeout Errors
- Large files take longer - increase timeout
- Check GPU memory: `ssh hercules 'nvidia-smi'`
- Server may be processing another request

### Model Not Found
- Check available models: `curl http://172.26.12.7:8000/models`
- Models are: "icelandic", "english"

### Ollama Errors
- Check Ollama status: `ssh hercules 'systemctl status ollama'`
- Restart Ollama: `ssh hercules 'sudo systemctl restart ollama'`
- List models: `ssh hercules 'ollama list'`

---

## Server Management

### Start Server
```bash
ssh hercules 'cd ~/Documents/Github/whisperSSTis && source venv/bin/activate && CUDA_VISIBLE_DEVICES=0 nohup uvicorn hercules_server:app --host 0.0.0.0 --port 8000 > ~/whisper_server.log 2>&1 &'
```

### Stop Server
```bash
ssh hercules 'pkill -f "uvicorn hercules_server"'
```

### View Logs
```bash
ssh hercules 'tail -f ~/whisper_server.log'
```

### Check GPU Usage
```bash
ssh hercules 'nvidia-smi'
```

---

## Quick Reference

### Connection
| Item | Value |
|------|-------|
| SSH | `ssh hercules` or `ssh magnus@172.26.12.7` |
| API | `http://172.26.12.7:8000` |
| Docs | `http://172.26.12.7:8000/docs` |
| VPN Required | Yes (UNAK) |

### Common Commands

```bash
# Test connection
python hercules_test.py

# Quick health check
curl http://172.26.12.7:8000/health

# SSH to Hercules
ssh hercules

# Start server on Hercules
cd ~/Documents/Github/whisperSSTis && source venv/bin/activate
uvicorn hercules_server:app --host 0.0.0.0 --port 8000

# Start in tmux (persistent)
tmux new -s whisper
uvicorn hercules_server:app --host 0.0.0.0 --port 8000
# Ctrl+B, D to detach

# Reattach tmux
tmux attach -s whisper

# Check GPU status
nvidia-smi

# Check Ollama models
ollama list

# Pull LLM model
ollama pull ministral-3b:14b
```

### Environment Variables (.env)

```bash
USE_HERCULES=true
HERCULES_URL=http://172.26.12.7:8000
USE_HERCULES_LLM=true
OLLAMA_MODEL=ministral-3b:14b
```

### API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status + GPU info |
| `/transcribe` | POST | Transcribe audio file |
| `/transcribe/srt` | POST | Transcribe to SRT format |
| `/summarize` | POST | LLM text processing |
| `/models` | GET | List Whisper models |
| `/ollama/models` | GET | List Ollama models |

### Troubleshooting Checklist

1. **VPN connected?** `ping 172.26.12.7`
2. **Server running?** `curl http://172.26.12.7:8000/health`
3. **GPU available?** `ssh hercules nvidia-smi`
4. **Ollama running?** `ssh hercules 'systemctl status ollama'`
