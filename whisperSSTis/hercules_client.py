"""Client for communicating with Hercules GPU server.

Provides remote transcription and LLM capabilities via the Hercules FastAPI server.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import requests
except ImportError:
    requests = None

try:
    import soundfile as sf
except ImportError:
    sf = None

logger = logging.getLogger(__name__)

# Default Hercules server URL
DEFAULT_HERCULES_URL = os.getenv("HERCULES_URL", "http://172.26.12.7:8000")


@dataclass
class HerculesConfig:
    """Configuration for Hercules server connection."""

    base_url: str = field(default_factory=lambda: DEFAULT_HERCULES_URL)
    timeout: float = 120.0
    verify_ssl: bool = True


class HerculesError(Exception):
    """Raised when Hercules server communication fails."""
    pass


class HerculesClient:
    """Client for Hercules GPU server API."""

    def __init__(self, config: Optional[HerculesConfig] = None):
        if requests is None:
            raise RuntimeError("requests package is required for Hercules client")

        self.config = config or HerculesConfig()
        self._session = requests.Session()

    @property
    def base_url(self) -> str:
        return self.config.base_url.rstrip("/")

    def health_check(self) -> dict:
        """Check if Hercules server is available and healthy."""
        try:
            response = self._session.get(
                f"{self.base_url}/health",
                timeout=5.0,
                verify=self.config.verify_ssl,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise HerculesError(f"Cannot connect to Hercules at {self.base_url}")
        except requests.exceptions.Timeout:
            raise HerculesError("Hercules health check timed out")
        except Exception as e:
            raise HerculesError(f"Health check failed: {e}")

    def is_available(self) -> bool:
        """Return True if Hercules is reachable."""
        try:
            self.health_check()
            return True
        except HerculesError:
            return False

    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        model_key: str = "icelandic",
        language_token: Optional[str] = None,
    ) -> str:
        """Send audio to Hercules for transcription.

        Args:
            audio_data: Audio samples as numpy array (float32, mono)
            sample_rate: Sample rate of audio
            model_key: Model to use ("icelandic" or "english")
            language_token: Override language token

        Returns:
            Transcribed text
        """
        if sf is None:
            raise RuntimeError("soundfile package required for transcription")

        # Convert audio to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format="WAV")
        buffer.seek(0)

        files = {"file": ("audio.wav", buffer, "audio/wav")}
        data = {"model_key": model_key}
        if language_token:
            data["language_token"] = language_token

        try:
            response = self._session.post(
                f"{self.base_url}/transcribe",
                files=files,
                data=data,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )
            response.raise_for_status()
            result = response.json()
            return result["text"]
        except requests.exceptions.ConnectionError:
            raise HerculesError(f"Cannot connect to Hercules at {self.base_url}")
        except requests.exceptions.Timeout:
            raise HerculesError("Transcription request timed out")
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_detail = e.response.json().get("detail", "")
            except Exception:
                pass
            raise HerculesError(f"Transcription failed: {error_detail or e}")
        except Exception as e:
            raise HerculesError(f"Transcription error: {e}")

    def llm_complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: str = "ministral-3b:14b",
        temperature: float = 0.3,
        max_tokens: int = 400,
    ) -> str:
        """Send prompt to Ollama via Hercules.

        Args:
            prompt: User prompt
            system: System prompt
            model: Ollama model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        payload = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system

        try:
            response = self._session.post(
                f"{self.base_url}/llm",
                json=payload,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
            )
            response.raise_for_status()
            result = response.json()
            return result["text"]
        except requests.exceptions.ConnectionError:
            raise HerculesError(f"Cannot connect to Hercules at {self.base_url}")
        except requests.exceptions.Timeout:
            raise HerculesError("LLM request timed out")
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_detail = e.response.json().get("detail", "")
            except Exception:
                pass
            raise HerculesError(f"LLM request failed: {error_detail or e}")
        except Exception as e:
            raise HerculesError(f"LLM error: {e}")

    def list_whisper_models(self) -> dict:
        """Get available Whisper models."""
        try:
            response = self._session.get(
                f"{self.base_url}/models",
                timeout=10.0,
                verify=self.config.verify_ssl,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise HerculesError(f"Could not list models: {e}")

    def list_ollama_models(self) -> list:
        """Get available Ollama models on Hercules."""
        try:
            response = self._session.get(
                f"{self.base_url}/ollama/models",
                timeout=10.0,
                verify=self.config.verify_ssl,
            )
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            raise HerculesError(f"Could not list Ollama models: {e}")


# Module-level convenience functions
_default_client: Optional[HerculesClient] = None


def get_client(config: Optional[HerculesConfig] = None) -> HerculesClient:
    """Get or create the default Hercules client."""
    global _default_client
    if config or _default_client is None:
        _default_client = HerculesClient(config)
    return _default_client


def check_hercules() -> bool:
    """Quick check if Hercules is available."""
    try:
        return get_client().is_available()
    except Exception:
        return False
