"""Transcription helpers built around the fine-tuned Whisper model.

Supports both local and remote (Hercules) transcription modes.
"""

from __future__ import annotations

import logging
import math
import os
import re
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from . import hercules_client

# Environment variable to control transcription mode
USE_HERCULES = os.getenv("USE_HERCULES", "false").lower() in ("true", "1", "yes")

MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
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

DEFAULT_MODEL_KEY = "icelandic"
logger = logging.getLogger(__name__)


def get_model_config(model_key: str = DEFAULT_MODEL_KEY) -> Dict[str, str]:
    """Return metadata for a supported Whisper model."""

    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key '{model_key}'. Available: {', '.join(MODEL_CONFIGS)}")
    return MODEL_CONFIGS[model_key]


def load_model(model_key: str = DEFAULT_MODEL_KEY):
    """Load the Whisper model/processor pair and place the model on the correct device."""

    config = get_model_config(model_key)
    model_name = config["model_id"]

    try:
        processor = WhisperProcessor.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        model.eval()
        return model, processor
    except Exception as exc:
        logger.error("Error loading model: %s", exc)
        raise


def _prepare_features(audio_data: np.ndarray, processor: WhisperProcessor, sample_rate: int, device: str):
    features = processor(
        audio_data,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    ).input_features.to(device)
    return features


def transcribe_audio(
    audio_data: np.ndarray,
    model,
    processor,
    sample_rate: int = 16000,
    language_token: str | None = None,
    use_hercules: Optional[bool] = None,
    model_key: str = DEFAULT_MODEL_KEY,
) -> str:
    """Transcribe a single audio array.

    Args:
        audio_data: Audio samples as numpy array
        model: Local Whisper model (ignored if using Hercules)
        processor: Local Whisper processor (ignored if using Hercules)
        sample_rate: Audio sample rate
        language_token: Language token override
        use_hercules: Force Hercules mode (None = use env var)
        model_key: Model key for Hercules transcription

    Returns:
        Transcribed text
    """
    # Determine whether to use Hercules
    remote = use_hercules if use_hercules is not None else USE_HERCULES

    if remote:
        return transcribe_audio_remote(
            audio_data,
            sample_rate=sample_rate,
            model_key=model_key,
            language_token=language_token,
        )

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_features = _prepare_features(audio_data, processor, sample_rate, device)

        if language_token is None:
            language_token = get_model_config()["language_token"]

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                task="transcribe",
                language=language_token,
            )[0]

        return processor.decode(predicted_ids, skip_special_tokens=True)
    except Exception as exc:
        logger.error("Error transcribing audio: %s", exc)
        raise


def transcribe_audio_remote(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    model_key: str = DEFAULT_MODEL_KEY,
    language_token: str | None = None,
) -> str:
    """Transcribe audio using Hercules remote server.

    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Audio sample rate
        model_key: Which Whisper model to use
        language_token: Language token override

    Returns:
        Transcribed text
    """
    try:
        client = hercules_client.get_client()
        return client.transcribe(
            audio_data,
            sample_rate=sample_rate,
            model_key=model_key,
            language_token=language_token,
        )
    except hercules_client.HerculesError as exc:
        logger.error("Hercules transcription error: %s", exc)
        raise
    except Exception as exc:
        logger.error("Remote transcription error: %s", exc)
        raise


def format_timestamp(seconds: float) -> str:
    """Convert seconds to a human-friendly timestamp (HH:MM:SS)."""

    return str(timedelta(seconds=int(seconds)))


def _timestamp_to_seconds(timestamp: str) -> float:
    parts = [int(p) for p in timestamp.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def _seconds_to_srt(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcribe_long_audio(
    audio_data: np.ndarray,
    model,
    processor,
    duration: float,
    chunk_size: int = 30,
    sample_rate: int = 16000,
    progress_callback: Callable[[int, int], None] | None = None,
    language_token: str | None = None,
    use_hercules: Optional[bool] = None,
    model_key: str = DEFAULT_MODEL_KEY,
) -> List[str]:
    """Transcribe long audio by breaking it into timestamped chunks.

    Args:
        audio_data: Audio samples as numpy array
        model: Local Whisper model (ignored if using Hercules)
        processor: Local Whisper processor (ignored if using Hercules)
        duration: Audio duration in seconds
        chunk_size: Size of each chunk in seconds
        sample_rate: Audio sample rate
        progress_callback: Optional callback for progress updates
        language_token: Language token override
        use_hercules: Force Hercules mode (None = use env var)
        model_key: Model key for remote transcription

    Returns:
        List of timestamped transcription strings
    """

    try:
        chunk_size = max(chunk_size, 1)
        chunk_samples = chunk_size * sample_rate
        total_samples = len(audio_data)
        if total_samples == 0:
            return []

        timestamps: List[str] = []
        num_chunks = int(math.ceil(total_samples / chunk_samples))

        for index, start in enumerate(range(0, total_samples, chunk_samples)):
            end = min(start + chunk_samples, total_samples)
            chunk = audio_data[start:end]
            chunk_transcription = transcribe_audio(
                chunk,
                model,
                processor,
                sample_rate,
                language_token=language_token,
                use_hercules=use_hercules,
                model_key=model_key,
            )

            start_time = format_timestamp(start / sample_rate)
            end_time = format_timestamp(end / sample_rate)
            timestamps.append(f"[{start_time} → {end_time}] {chunk_transcription}")

            if progress_callback:
                progress_callback(index + 1, num_chunks)

        return timestamps
    except Exception as exc:
        logger.error("Error transcribing long audio: %s", exc)
        raise


def create_srt(transcriptions: Sequence[str]) -> str:
    """Convert timestamped transcriptions to SRT text."""

    srt_lines = []
    for i, trans in enumerate(transcriptions, start=1):
        timestamp_match = re.match(r"\[(.*?) → (.*?)\] (.*)", trans)
        if not timestamp_match:
            continue

        start_time_str, end_time_str, text = timestamp_match.groups()
        start_time_seconds = _timestamp_to_seconds(start_time_str)
        end_time_seconds = _timestamp_to_seconds(end_time_str)

        srt_lines.extend(
            [
                str(i),
                f"{_seconds_to_srt(start_time_seconds)} --> {_seconds_to_srt(end_time_seconds)}",
                text,
                "",
            ]
        )
    return "\n".join(srt_lines)

