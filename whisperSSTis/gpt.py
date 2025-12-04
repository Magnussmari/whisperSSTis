"""Lightweight GPT helper for working with transcript post-processing."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore


DEFAULT_MODEL = os.getenv("GPT_MINI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


@dataclass
class GPTConfig:
    """Configuration for GPT API calls."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = DEFAULT_MODEL
    temperature: float = 0.3
    max_tokens: int = 400


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


def run_on_transcript(transcript: str, instruction: str, config: Optional[GPTConfig] = None) -> str:
    """Send the transcript to GPT mini with a user-provided instruction.

    Returns a human-friendly string so the UI can display it directly.
    """

    cfg = config or GPTConfig()
    client = _build_client(cfg)

    system_prompt = (
        "You are an Icelandic transcription assistant. Work only with the provided "
        "transcript text. Keep names and numbers intact. When asked to translate, "
        "preserve meaning and cultural context. Be concise unless the user requests details." 
    )

    try:
        response = client.chat.completions.create(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Instruction: {instruction}\n\n"
                        f"Transcript (may be Icelandic):\n{transcript}"
                    ),
                },
            ],
        )
        return response.choices[0].message.content or ""
    except Exception as exc:  # pragma: no cover - network interactions
        logging.error("GPT mini request failed: %s", exc)
        raise

