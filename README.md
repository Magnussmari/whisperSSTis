# Norðlenski hreimurinn

**Local speech recognition for Icelandic and English — 100% private, runs on your machine.**

Norðlenski hreimurinn uses fine-tuned [Whisper](https://github.com/openai/whisper) models to transcribe audio locally. No audio ever leaves your computer. Supports microphone recording and file upload with export to TXT and SRT.

## Quick Start

```bash
# Install system dependencies
brew install portaudio ffmpeg        # macOS
# sudo apt install portaudio19-dev ffmpeg   # Ubuntu/Debian

# Clone and install
git clone https://github.com/Magnussmari/whisperSSTis.git
cd whisperSSTis
uv sync --all-extras

# Run
uv run streamlit run app.py
```

Opens at [http://localhost:8501](http://localhost:8501). First launch downloads the model (~4 GB).

## Features

- **Record or upload** — microphone recording with waveform visualization, or upload WAV/MP3/M4A/FLAC
- **Two language models** — Icelandic (fine-tuned) and English (Whisper Large v3), switchable in the sidebar
- **100% local** — all transcription runs on your machine. GPU accelerated when available, CPU fallback.
- **Timestamped export** — download transcripts as TXT or SRT subtitle files
- **AI assistant** (optional) — GPT post-processing for summarization, translation, cleanup. Requires `OPENAI_API_KEY`. Only sends text, never audio.
- **Bilingual UI** — Icelandic and English labels throughout

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- [PortAudio](http://www.portaudio.com/) — for microphone capture
- [FFmpeg](https://ffmpeg.org/) — for non-WAV audio conversion
- ~4 GB disk space per model
- CUDA GPU recommended (not required)

## Configuration

Copy `env.example` to `.env` for optional GPT features:

```bash
OPENAI_API_KEY=sk-your-key    # Required for AI assistant only
GPT_MINI_MODEL=gpt-4o-mini    # Optional: override model
```

## Project Structure

```
app.py                     Main Streamlit web application
launcher.py                Tkinter desktop launcher
whisperSSTis/              Core Python package
  audio.py                   Audio capture, file loading, ffmpeg conversion
  transcribe.py              Whisper model loading and inference
  gpt.py                     Optional GPT post-processing
tests/                     pytest test suite (26 tests)
scripts/                   Build and distribution scripts
archive/                   Superseded documentation
docs/missions/             Development mission reports
.github/workflows/ci.yml   GitHub Actions CI
pyproject.toml             Package config, dependencies, tool settings
uv.lock                    Reproducible dependency lockfile
architecture.jsonld        Machine-readable system architecture graph
```

## Development

```bash
# Install with all extras
uv sync --all-extras

# Run tests
uv run pytest

# Run the app
uv run streamlit run app.py

# Run via desktop launcher
uv run python launcher.py
```

### Testing

26 tests covering all three core modules (audio, transcribe, gpt). All tests mock hardware — no GPU or microphone needed in CI.

```bash
uv run pytest -v              # Full verbose suite
uv run pytest tests/test_gpt.py   # Single module
```

### CI

GitHub Actions runs on every push to `main` and on PRs:
- Python 3.10, 3.11, 3.12
- Full test suite
- `pip-audit` vulnerability scanning

## Technical Details

| Component | Details |
|-----------|---------|
| Frontend | Streamlit with custom CSS design system |
| Models | [Icelandic fine-tuned Whisper](https://huggingface.co/carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h), [Whisper Large v3](https://huggingface.co/openai/whisper-large-v3) |
| ML stack | PyTorch, HuggingFace Transformers 5.x |
| Audio | sounddevice, soundfile, scipy, subprocess ffmpeg |
| Package manager | [uv](https://docs.astral.sh/uv/) with lockfile |
| Sample rate | 16 kHz (Whisper requirement) |
| Chunk processing | Configurable 10–60 second segments |
| Max upload | 1 GB |

## Privacy & Security

- Audio is **never transmitted** over the network
- Models are downloaded once from Hugging Face, then cached locally
- GPT features are **opt-in** and only send text (not audio) to OpenAI
- Temporary files are cleaned up in `finally` blocks
- `.env` is excluded from version control via `.gitignore`
- `unsafe_allow_html` is used only for static CSS, never user input
- CI includes automated vulnerability scanning with `pip-audit`

## Credits

**Developer:** [Magnus Smari Smarason](https://smarason.is)

**Models:**
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Icelandic fine-tuned model](https://huggingface.co/carlosdanielhernandezmena/whisper-large-icelandic-10k-steps-1000h) by Carlos Daniel Hernandez Mena

**Built with:** [Streamlit](https://streamlit.io/) · [PyTorch](https://pytorch.org/) · [Hugging Face](https://huggingface.co/)

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Issues and pull requests welcome at [github.com/Magnussmari/whisperSSTis](https://github.com/Magnussmari/whisperSSTis/issues).
