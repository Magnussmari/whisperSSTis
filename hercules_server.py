"""FastAPI server for running Whisper and LLM on Hercules GPU server.

Deploy this on Hercules and run with:
    uvicorn hercules_server:app --host 0.0.0.0 --port 8000

Requirements on Hercules:
    pip install fastapi uvicorn python-multipart torch transformers numpy httpx
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

try:
    import httpx
except ImportError:
    httpx = None

try:
    import soundfile as sf
except ImportError:
    sf = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "icelandic": {
        "label": "Icelandic (fine-tuned)",
        "model_id": "carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h",
        "language_token": "<|is|>",
    },
    "english": {
        "label": "English (Whisper Large v3)",
        "model_id": "openai/whisper-large-v3",
        "language_token": "<|en|>",
    },
}

# Global model cache
models: dict = {}
processors: dict = {}

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def load_whisper_model(model_key: str):
    """Load a Whisper model if not already cached."""
    if model_key in models:
        return models[model_key], processors[model_key]

    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}")

    config = MODEL_CONFIGS[model_key]
    model_id = config["model_id"]

    logger.info(f"Loading model: {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use GPU 0 (RTX 5090) by default for Whisper
    if device == "cuda":
        torch.cuda.set_device(0)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    model.eval()

    models[model_key] = model
    processors[model_key] = processor

    logger.info(f"Model loaded on {device}")
    return model, processor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload default model on startup."""
    logger.info("Starting Hercules Whisper Server...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Preload Icelandic model
    try:
        load_whisper_model("icelandic")
    except Exception as e:
        logger.warning(f"Could not preload model: {e}")

    yield

    logger.info("Shutting down Hercules server...")


app = FastAPI(
    title="Hercules Whisper Server",
    description="GPU-accelerated Whisper transcription and LLM inference",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscriptionResponse(BaseModel):
    text: str
    model: str
    device: str


class LLMRequest(BaseModel):
    prompt: str
    system: Optional[str] = None
    model: str = "llama3"
    temperature: float = 0.3
    max_tokens: int = 400


class LLMResponse(BaseModel):
    text: str
    model: str


class HealthResponse(BaseModel):
    status: str
    cuda_available: bool
    gpus: list[str]
    loaded_models: list[str]
    ollama_available: bool


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server status and available resources."""
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            gpus.append(f"{name} ({memory:.1f}GB)")

    # Check Ollama
    ollama_ok = False
    if httpx:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=2.0)
                ollama_ok = r.status_code == 200
        except Exception:
            pass

    return HealthResponse(
        status="ok",
        cuda_available=torch.cuda.is_available(),
        gpus=gpus,
        loaded_models=list(models.keys()),
        ollama_available=ollama_ok,
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    model_key: str = Form("icelandic"),
    language_token: Optional[str] = Form(None),
):
    """Transcribe an audio file using Whisper."""
    if sf is None:
        raise HTTPException(500, "soundfile not installed on server")

    try:
        model, processor = load_whisper_model(model_key)
    except ValueError as e:
        raise HTTPException(400, str(e))

    config = MODEL_CONFIGS[model_key]
    lang_token = language_token or config["language_token"]

    # Read audio file
    try:
        audio_bytes = await file.read()
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import scipy.signal
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = scipy.signal.resample(audio_data, num_samples)
            sample_rate = 16000

        audio_data = audio_data.astype(np.float32)
    except Exception as e:
        raise HTTPException(400, f"Could not read audio file: {e}")

    # Transcribe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        features = processor(
            audio_data,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                features,
                task="transcribe",
                language=lang_token,
            )[0]

        text = processor.decode(predicted_ids, skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(500, f"Transcription failed: {e}")

    return TranscriptionResponse(
        text=text,
        model=config["model_id"],
        device=device,
    )


@app.post("/llm", response_model=LLMResponse)
async def llm_completion(request: LLMRequest):
    """Send a prompt to Ollama for completion."""
    if httpx is None:
        raise HTTPException(500, "httpx not installed on server")

    messages = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.prompt})

    payload = {
        "model": request.model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
        },
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException:
        raise HTTPException(504, "Ollama request timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"Ollama error: {e}")
    except Exception as e:
        raise HTTPException(500, f"LLM request failed: {e}")

    text = data.get("message", {}).get("content", "")
    return LLMResponse(text=text, model=request.model)


@app.get("/models")
async def list_models():
    """List available Whisper models."""
    return {
        "whisper_models": MODEL_CONFIGS,
        "loaded": list(models.keys()),
    }


@app.get("/ollama/models")
async def list_ollama_models():
    """List available Ollama models."""
    if httpx is None:
        raise HTTPException(500, "httpx not installed")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        raise HTTPException(502, f"Could not reach Ollama: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
