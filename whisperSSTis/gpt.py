"""Lightweight GPT helper for working with transcript post-processing.

Supports both OpenAI API and Ollama via Hercules server.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from . import hercules_client

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore


DEFAULT_MODEL = os.getenv("GPT_MINI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Use Hercules for LLM by default if USE_HERCULES is set
USE_HERCULES_LLM = os.getenv("USE_HERCULES_LLM", os.getenv("USE_HERCULES", "false")).lower() in ("true", "1", "yes")


@dataclass
class GPTConfig:
    """Configuration for GPT API calls."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = DEFAULT_MODEL
    temperature: float = 0.3
    max_tokens: int = 400
    use_hercules: Optional[bool] = None  # None = use env var
    ollama_model: str = field(default_factory=lambda: DEFAULT_OLLAMA_MODEL)


def _build_client(config: GPTConfig) -> OpenAI:
    """Create an OpenAI client with environment fallbacks."""

    if OpenAI is None:
        raise RuntimeError("The 'openai' package is not installed. Please install dependencies.")

    api_key = config.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for GPT mini integration.")

    base_url = config.base_url or os.getenv("OPENAI_BASE_URL")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs)


SYSTEM_PROMPT = (
    "You are an Icelandic transcription assistant. Work only with the provided "
    "transcript text. Keep names and numbers intact. When asked to translate, "
    "preserve meaning and cultural context. Be concise unless the user requests details."
)


def run_on_transcript(transcript: str, instruction: str, config: Optional[GPTConfig] = None) -> str:
    """Send the transcript to GPT/Ollama with a user-provided instruction.

    Supports both OpenAI API and Ollama via Hercules server.
    Returns a human-friendly string so the UI can display it directly.
    """

    cfg = config or GPTConfig()

    # Determine whether to use Hercules
    use_hercules = cfg.use_hercules if cfg.use_hercules is not None else USE_HERCULES_LLM

    user_content = f"Instruction: {instruction}\n\nTranscript (may be Icelandic):\n{transcript}"

    if use_hercules:
        return run_on_transcript_hercules(
            user_content,
            system=SYSTEM_PROMPT,
            model=cfg.ollama_model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    # Use OpenAI API
    client = _build_client(cfg)

    try:
        response = client.chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content or ""
    except Exception as exc:  # pragma: no cover - network interactions
        logging.error("GPT mini request failed: %s", exc)
        raise


def run_on_transcript_hercules(
    prompt: str,
    system: Optional[str] = None,
    model: str = DEFAULT_OLLAMA_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 400,
) -> str:
    """Send prompt to Ollama via Hercules server.

    Args:
        prompt: User prompt with transcript
        system: System prompt
        model: Ollama model name
        temperature: Sampling temperature
        max_tokens: Max tokens to generate

    Returns:
        Generated text
    """
    try:
        client = hercules_client.get_client()
        return client.llm_complete(
            prompt=prompt,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except hercules_client.HerculesError as exc:
        logging.error("Hercules LLM request failed: %s", exc)
        raise
    except Exception as exc:
        logging.error("Remote LLM request failed: %s", exc)
        raise

