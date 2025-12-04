"""Audio capture and file loading utilities."""

from __future__ import annotations

import logging
import math
import os
import queue
import tempfile
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
from scipy import signal

logger = logging.getLogger(__name__)


def get_audio_devices() -> Dict[str, int]:
    """Return input-capable devices keyed by human-readable name."""

    devices = sd.query_devices()
    input_devices: Dict[str, int] = {}
    for idx, device in enumerate(devices):
        if not isinstance(device, dict):
            continue

        if device.get("max_input_channels", 0) <= 0:
            continue

        try:
            sd.check_input_settings(
                device=idx,
                channels=1,
                samplerate=16000,
                dtype=np.float32,
            )
        except sd.PortAudioError:
            # Device cannot be opened with the required settings.
            continue
        except Exception as exc:  # pragma: no cover - defensive logging for flaky drivers
            logger.debug("Skipping strict validation for %s: %s", device.get("name", idx), exc)

        device_name = device.get("name", f"Device {idx}")
        input_devices[f"{device_name} (ID: {idx})"] = idx
    return input_devices


@dataclass
class AudioStream:
    """Small helper to continuously read audio chunks from an input device."""

    device_id: Optional[int]
    samplerate: int = 16000
    chunk_size: int = 1024

    def __post_init__(self):
        self.stream: Optional[sd.InputStream] = None
        self.is_recording: bool = False
        self.target_samplerate = self.samplerate
        self.native_samplerate = self._detect_native_samplerate()
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def _detect_native_samplerate(self) -> int:
        try:
            info = sd.query_devices(device=self.device_id, kind="input")
            return int(info.get("default_samplerate", self.target_samplerate))
        except Exception:
            return self.target_samplerate

    def _audio_callback(self, indata, frames, time, status):  # pragma: no cover - real-time callback
        if status:
            logger.warning("Audio callback status: %s", status)
        chunk = np.array(indata).flatten()
        if self.native_samplerate != self.target_samplerate and len(chunk) > 0:
            target_len = int(len(chunk) * self.target_samplerate / self.native_samplerate)
            chunk = signal.resample(chunk, target_len)
        try:
            self._queue.put_nowait(chunk.astype(np.float32))
        except queue.Full:
            logger.debug("Audio queue full; dropping chunk")

    def start_stream(self):
        if self.stream is not None and self.is_recording:
            return

        self.stream = sd.InputStream(
            device=self.device_id,
            channels=1,
            samplerate=self.native_samplerate,
            dtype=np.float32,
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        )
        self.stream.start()
        self.is_recording = True

    def stop_stream(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None
        self.is_recording = False

    def get_audio_chunk(self) -> np.ndarray:
        if not self.is_recording:
            return np.zeros(self.chunk_size, dtype=np.float32)

        try:
            return self._queue.get(timeout=1)
        except queue.Empty:
            return np.zeros(self.chunk_size, dtype=np.float32)


def record_audio(duration: int, device_id: Optional[int] = None, samplerate: int = 16000, chunk_size: int = 1024) -> np.ndarray:
    """Record audio from the selected microphone using streamed chunks."""

    stream = AudioStream(device_id=device_id, samplerate=samplerate, chunk_size=chunk_size)
    target_samples = int(duration * samplerate)
    collected = []

    try:
        stream.start_stream()
        collected_samples = 0
        while collected_samples < target_samples:
            chunk = stream.get_audio_chunk()
            collected.append(chunk)
            collected_samples += len(chunk)
    finally:
        stream.stop_stream()

    audio_data = np.concatenate(collected) if collected else np.zeros(target_samples, dtype=np.float32)
    if len(audio_data) > target_samples:
        audio_data = audio_data[:target_samples]
    return audio_data.astype(np.float32)


def get_file_info(audio_data: np.ndarray, sample_rate: int) -> Dict[str, str]:
    """Return basic metadata for display purposes."""

    duration = len(audio_data) / sample_rate if sample_rate else 0
    channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]
    return {
        "Duration": str(timedelta(seconds=int(duration))),
        "Channels": channels,
        "Sample Rate": f"{sample_rate} Hz",
        "File Size": f"{audio_data.nbytes / (1024 * 1024):.2f} MB",
    }


def _convert_to_wav(temp_path: str, suffix: str) -> str:
    """Convert non-wav files to wav for consistent processing."""

    if suffix.lower() != ".m4a":
        return temp_path

    converted_path = f"{temp_path}.wav"
    audio_segment = AudioSegment.from_file(temp_path)
    audio_segment.export(converted_path, format="wav")
    return converted_path


def _read_audio_file(path: str) -> Tuple[np.ndarray, int]:
    audio_data, sr = sf.read(path)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    return audio_data.astype(np.float32), sr


def load_audio_file(uploaded_file, target_sr: int = 16000):
    """Load and normalize an uploaded audio file to the target sample rate."""

    tmp_path = None
    converted_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        converted_path = _convert_to_wav(tmp_path, Path(uploaded_file.name).suffix)
        audio_data, sr = _read_audio_file(converted_path)

        file_info = get_file_info(audio_data, sr)

        if sr != target_sr and sr > 0:
            target_len = int(len(audio_data) * target_sr / sr)
            audio_data = signal.resample(audio_data, target_len)

        duration = len(audio_data) / target_sr if target_sr else 0
        return audio_data.astype(np.float32), duration, file_info
    except Exception as exc:
        logger.error("Error loading audio file: %s", exc)
        raise
    finally:
        for path in (converted_path, tmp_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    logger.debug("Could not delete temp file: %s", path)

