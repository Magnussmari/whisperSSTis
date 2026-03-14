# CLAUDE.md — WhisperSST.is

## Identity
Local-first Icelandic/English speech-to-text app. Python 3.10+, Streamlit UI, PyTorch + HuggingFace Whisper (4.44+), sounddevice audio capture. MIT licensed. No cloud processing (GPT post-processing is optional).

## Priority Order
correctness > privacy > clarity > maintainability > performance

## Commands
```bash
pip install -e ".[dev,gpt]"      # Install package in editable mode
streamlit run app.py              # Run web app directly (port 8501)
python launcher.py                # Run via Tkinter GUI launcher
pytest                            # Run all tests (25 tests)
pytest tests/test_audio.py        # Audio module tests (8)
pytest tests/test_transcribe.py   # Transcribe module tests (6)
pytest tests/test_gpt.py          # GPT module tests (11)
brew install portaudio            # macOS system dep
```

## Structure
```
app.py                    # Streamlit web interface (main entry point)
launcher.py               # Tkinter GUI launcher wrapping Streamlit
whisperSSTis/             # Core Python package
  __init__.py             # Exports audio, transcribe, gpt
  audio.py                # AudioStream class, recording, file loading, resampling
  transcribe.py           # Model loading, transcription, SRT generation
  gpt.py                  # Optional OpenAI GPT post-processing
tests/
  conftest.py             # Shared fixtures (silence_1s, mock_model, mock_processor)
  test_audio.py           # Audio module unit tests
  test_transcribe.py      # Transcribe module unit tests
  test_gpt.py             # GPT module unit tests
  demo/test_vedur.mp3     # Demo audio file for manual testing
.github/workflows/ci.yml # GitHub Actions CI (Python 3.10-3.12, pytest, pip-audit)
.streamlit/config.toml    # Streamlit config (max upload 1000MB, error-only logging)
pyproject.toml            # Package metadata, deps, pytest/ruff config
architecture.jsonld       # Machine-readable JSON-LD system architecture graph
assets/websitelogo.png    # App logo used in sidebar
build.py                  # PyInstaller packaging script
```

## Models
Two Whisper models configured in `transcribe.py:MODEL_CONFIGS`:
- `icelandic`: `carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h` (default)
- `english`: `openai/whisper-large-v3`

## Key Patterns
- **16kHz sample rate** — all audio resampled to this before Whisper inference
- **30-second chunk processing** — long audio split into chunks for memory efficiency
- **Session state** — model, processor, audio_data, transcriptions persisted in `st.session_state`
- **Device detection** — `audio.get_audio_devices()` validates 16kHz compatibility before listing
- **Temp file cleanup** — `load_audio_file()` uses try/finally to unlink temp files
- **GPT is optional** — gpt.py gracefully handles missing `openai` package
- **Editable install** — use `pip install -e ".[dev,gpt]"` for development

## Anti-Patterns (Evidence-Based)
1. **`unsafe_allow_html=True`** — used in `app.py` for CSS injection. Currently safe (static CSS only), but never pass user input through it.
2. **`pydub` uses deprecated `audioop`** — will break in Python 3.13+. Monitor for pydub update or replacement.

## Environment Variables
```
OPENAI_API_KEY      # Required only for GPT post-processing
GPT_MINI_MODEL      # Override GPT model (default: gpt-4o-mini)
OPENAI_BASE_URL     # Override OpenAI endpoint
OPENAI_MODEL        # Fallback model name
```

## Testing Conventions
- Framework: pytest + pytest-mock, configured in pyproject.toml
- Shared fixtures in `tests/conftest.py`
- All tests mock hardware (sounddevice, model loading)
- CI: GitHub Actions runs on Python 3.10, 3.11, 3.12

## Git Conventions
- Use semantic commit prefixes: `fix:`, `feat:`, `refactor:`, `deps:`, `test:`, `docs:`, `ci:`
- Atomic commits — one logical change per commit
- Green-to-green — every commit must leave tests passing

## Architecture Map
See `architecture.jsonld` in project root for machine-readable system graph (23 nodes).
Update with `/update-architecture` slash command.

## Do Not Touch
- `.streamlit/config.toml` — production Streamlit config
- `assets/websitelogo.png` — used by app sidebar
