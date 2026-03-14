# CLAUDE.md — WhisperSST.is

## Identity
Local-first Icelandic/English speech-to-text app. Python 3.10+, Streamlit UI, PyTorch + HuggingFace Transformers 5.x, sounddevice audio capture. MIT licensed. No cloud processing (GPT post-processing is optional).

## Priority Order
correctness > privacy > clarity > maintainability > performance

## Quick Start
```bash
brew install portaudio ffmpeg       # macOS system deps
uv sync --all-extras                # Install everything via uv
uv run streamlit run app.py         # Launch on http://localhost:8501
```

## Commands
```bash
uv sync --all-extras              # Install all deps (lockfile-pinned)
uv run streamlit run app.py       # Run web app (port 8501)
uv run python launcher.py         # Run via Tkinter GUI launcher
uv run pytest                     # Run all tests (26 tests)
uv run python scripts/build.py    # PyInstaller packaging (from root)
```

## Structure
```
app.py                    # Streamlit web interface (main entry point)
launcher.py               # Tkinter GUI launcher wrapping Streamlit
whisperSSTis/             # Core Python package
  __init__.py             # Exports audio, transcribe, gpt
  audio.py                # AudioStream, recording, file loading, ffmpeg conversion
  transcribe.py           # Model loading, transcription, SRT generation
  gpt.py                  # Optional OpenAI GPT post-processing
tests/
  conftest.py             # Shared fixtures (silence_1s, mock_model, mock_processor)
  test_audio.py           # Audio module unit tests
  test_transcribe.py      # Transcribe module unit tests
  test_gpt.py             # GPT module unit tests
  demo/test_vedur.mp3     # Demo audio file for manual testing
scripts/
  build.py                # PyInstaller packaging (run from project root)
  run_whisperSST.sh/bat   # Distribution launcher scripts
  setup_dependencies.*    # End-user system dep installers
archive/                  # Superseded docs (agents.md, TODO.md, etc.)
docs/missions/            # Mission reports
.github/workflows/ci.yml # GitHub Actions CI (Python 3.10-3.12, pytest, pip-audit)
.streamlit/config.toml    # Streamlit config (max upload 1000MB)
pyproject.toml            # Package metadata, deps, pytest/ruff config
architecture.jsonld       # Machine-readable JSON-LD architecture graph
```

## Models
Two Whisper models in `transcribe.py:MODEL_CONFIGS`:
- `icelandic`: `carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h` (default)
- `english`: `openai/whisper-large-v3`

## Key Patterns
- **16kHz sample rate** — all audio resampled before Whisper inference
- **30-second chunk processing** — long audio split for memory efficiency
- **Session state** — model, processor, audio persisted in `st.session_state`
- **ffmpeg for conversion** — M4A/MP3/OGG converted via subprocess ffmpeg
- **GPT is optional** — gpt.py gracefully handles missing `openai` package
- **uv for packaging** — `uv sync` with lockfile for reproducible installs

## System Dependencies
- **PortAudio** — required by sounddevice for mic capture
- **FFmpeg** — required for non-WAV/FLAC audio file conversion

## Environment Variables
```
OPENAI_API_KEY      # Required only for GPT post-processing
GPT_MINI_MODEL      # Override GPT model (default: gpt-4o-mini)
OPENAI_BASE_URL     # Override OpenAI endpoint
```

## Anti-Patterns
1. **`unsafe_allow_html=True`** in `app.py` — static CSS only, never pass user input.

## Git Conventions
- Semantic prefixes: `fix:`, `feat:`, `refactor:`, `deps:`, `test:`, `docs:`, `ci:`
- Atomic commits, green-to-green

## Architecture Map
`architecture.jsonld` — 25 nodes. Update with `/update-architecture`.
