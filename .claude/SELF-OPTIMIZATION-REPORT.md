# Self-Optimization & Architecture Mapping Report
**Project:** WhisperSST.is
**Date:** 2026-03-14
**Status:** COMPLETE

## Reconnaissance Summary

WhisperSST.is is a local-first Icelandic and English speech recognition application built on Python 3.8+ with a Streamlit web frontend. The core module (`whisperSSTis/`) contains three submodules: audio capture/processing (`audio.py`), Whisper model integration (`transcribe.py`), and optional GPT post-processing (`gpt.py`). A Tkinter launcher (`launcher.py`) wraps the Streamlit process for desktop distribution. The application uses a fine-tuned Whisper Large model for Icelandic and OpenAI's Whisper Large v3 for English, both running locally via PyTorch. Audio is captured through PortAudio/sounddevice, resampled to 16kHz, and processed in 30-second chunks. All transcription happens locally — the only network dependency is the optional GPT feature which sends text (never audio) to OpenAI's API. The project has a PyInstaller build system (`build.py`) for executable packaging, pytest-based unit tests, and a legacy monolithic predecessor (`sst_is_test.py`) that should be removed. There is no CI/CD pipeline, no Docker configuration, and no existing Claude Code infrastructure.

## Architecture Map

### Graph Statistics
| Metric | Count |
|--------|-------|
| Total nodes | 23 |
| Services | 7 |
| Data stores | 3 |
| Data flows | 4 |
| Infrastructure components | 1 |
| External dependencies | 6 |
| Environments | 1 |
| System (root) | 1 |
| Confidence: verified | 21 |
| Confidence: inferred | 1 |
| Confidence: uncertain | 0 |

### Architecture Map Location
`architecture.jsonld` (project root)

### Architecture Map Changes
| Node @id | Change Type | Description |
|----------|-------------|-------------|
| All 23 nodes | Added | First architecture map — no prior map existed |

### Key Data Flows

1. **Microphone Recording Flow** (`arch:flow/mic-to-transcription`): User clicks Record → AudioStream captures at native sample rate via PortAudio → resampled to 16kHz → concatenated numpy array → Whisper model inference → text displayed in Streamlit UI.

2. **File Upload Flow** (`arch:flow/upload-to-transcription`): User uploads audio → temp file created → M4A converted via pydub/FFmpeg if needed → soundfile reads to numpy → resampled to 16kHz → chunked into 30-second segments → each chunk transcribed with timestamps → results displayed with TXT/SRT export.

3. **GPT Post-Processing Flow** (`arch:flow/transcript-to-gpt`): Optional. User provides instruction + transcript text → sent to OpenAI chat completions API → response displayed in expandable panel. Only text transmitted, never audio.

4. **Model Download Flow** (`arch:flow/model-download`): First launch only. HuggingFace Hub downloads ~4GB model weights → cached at `~/.cache/huggingface/` → all subsequent launches load from local cache.

### Architecture Gaps & Unknowns

- **FFmpeg dependency** (`arch:external/ffmpeg`, confidence: inferred): Referenced via `pydub` and `ffmpeg-python` in requirements, but no explicit FFmpeg version pinning or availability check in the application. The `_convert_to_wav` function only handles M4A — other formats (MP3) are loaded directly by soundfile which may or may not use FFmpeg internally.
- **Hercules GPU server integration**: The `hercules-integration` remote branch suggests work on remote GPU inference, but nothing is merged to main. Could not assess architecture without checking that branch.
- **No health monitoring**: No health endpoints, uptime checks, or error reporting beyond console logging.

## Claude Code Configuration

### What Was Created or Modified
| File | Action | Description |
|------|--------|-------------|
| `CLAUDE.md` | Modified | Complete rewrite — 83 lines, structured with identity, commands, structure map, patterns, anti-patterns, env vars, testing conventions |
| `architecture.jsonld` | Created | JSON-LD architecture graph with 23 nodes covering all services, data stores, flows, and dependencies |
| `.claude/commands/test.md` | Created | Slash command to run test suite and report results |
| `.claude/commands/pre-commit.md` | Created | Pre-commit verification checklist (tests, imports, secrets, dead code, HTML safety) |
| `.claude/commands/new-feature.md` | Created | Feature scaffolding command following project patterns |
| `.claude/commands/update-architecture.md` | Created | Architecture re-audit and JSON-LD merge command |
| `.claude/settings.json` | Created | Permission denials for legacy file and .env protection |

### Anti-Patterns Discovered

1. **Dead code: `sst_is_test.py`** (555 lines)
   - Evidence: File contains inline versions of all functions now in `whisperSSTis/audio.py` and `whisperSSTis/transcribe.py`. Comparison of `load_model()`, `get_audio_devices()`, `record_audio()`, `transcribe_audio()`, `transcribe_long_audio()`, and `create_srt()` shows these are older, less-featured versions of the module code. The monolith also hardcodes the Icelandic model (`language="is"`) while the module supports configurable models.
   - Correct pattern: Delete `sst_is_test.py`. All functionality lives in `whisperSSTis/` package.

2. **Empty asset placeholders** (`assets/whisper-logo.png`, `assets/whisper-logo.svg` — both 0 bytes)
   - Evidence: `wc -c` confirms both files are 0 bytes. Neither is referenced in `app.py` (which uses `assets/websitelogo.png`).
   - Correct pattern: Remove empty files or replace with actual assets.

3. **Bare `except:` clause** (`launcher.py:21`)
   - Evidence: `except:` with `pass` on icon loading. Should be `except (tk.TclError, FileNotFoundError):`.
   - Correct pattern: Catch specific exceptions.

4. **Missing `.env` in `.gitignore`**
   - Evidence: `.gitignore` does not list `.env`. The `env.example` file exists, suggesting `.env` is expected but the actual `.env` could be accidentally committed with API keys.
   - Correct pattern: Add `.env` to `.gitignore`.

5. **No `pyproject.toml` or `setup.py`**
   - Evidence: Tests use `sys.path.insert(0, ...)` hack to import the module. This indicates the package is not installable.
   - Correct pattern: Add a minimal `pyproject.toml` to make the package pip-installable in editable mode.

6. **Terse, inconsistent commit messages**
   - Evidence: Git log shows "Update audio.py", "Update transcribe.py", "Alpha", "Testing", "Update". No conventional commit format.
   - Correct pattern: Use conventional commits (`feat:`, `fix:`, `refactor:`, etc.).

### Key Learnings for Future Sessions

1. **The module is `whisperSSTis/`, not the root files.** All core logic lives in `audio.py`, `transcribe.py`, and `gpt.py`. The root `app.py` is the Streamlit UI layer. Never confuse `sst_is_test.py` with the current codebase.

2. **Two model configs exist.** The `MODEL_CONFIGS` dict in `transcribe.py` defines both Icelandic and English models. The sidebar UI allows switching between them. Both use the same inference pipeline — only the model ID and language token differ.

3. **Audio resampling is critical.** Whisper requires 16kHz input. The `AudioStream` class handles real-time resampling from native device rates, and `load_audio_file()` handles file-based resampling. Always verify 16kHz when adding new audio paths.

4. **GPT module is truly optional.** The `openai` import is wrapped in try/except. The UI checks for `OPENAI_API_KEY` before enabling GPT features. This pattern should be preserved for any new external integrations.

5. **Tests are mock-heavy.** All tests mock hardware (sounddevice, model loading). There are no integration tests that run actual inference. The demo file `tests/demo/test_vedur.mp3` exists but isn't used in automated tests.

## Flags for Commander Review

1. **`.env` not in `.gitignore`** — Risk of committing `OPENAI_API_KEY` to the public GitHub repo. Recommend adding `.env` to `.gitignore` immediately.

2. **`sst_is_test.py` should be deleted** — 555 lines of dead code. All functionality has been refactored. Keeping it risks confusion and accidental use.

3. **Empty asset files** — `assets/whisper-logo.png` and `assets/whisper-logo.svg` are 0 bytes. Either populate them or remove them.

4. **`hercules-integration` branch** — Remote branch suggests GPU server work. If this is active, the architecture map should be updated to include remote inference topology.

5. **No CI/CD** — No automated testing on push. Given the test suite exists, adding a GitHub Actions workflow would catch regressions.

## Configuration Quick Reference

| Asset | Location | Purpose |
|-------|----------|---------|
| `CLAUDE.md` | Project root | Primary Claude Code context — stack, structure, patterns, anti-patterns |
| `architecture.jsonld` | Project root | Machine-readable JSON-LD architecture graph (23 nodes) |
| `/test` | `.claude/commands/test.md` | Run pytest suite and report results |
| `/pre-commit` | `.claude/commands/pre-commit.md` | Pre-commit verification (tests, secrets, imports, safety) |
| `/new-feature` | `.claude/commands/new-feature.md` | Feature scaffolding with project pattern guidance |
| `/update-architecture` | `.claude/commands/update-architecture.md` | Re-audit codebase and merge changes into JSON-LD |
| `.claude/settings.json` | `.claude/settings.json` | Denies edits to `sst_is_test.py` (legacy) and writes to `.env` (secrets) |

## Recommended Next Mission

1. **Add `.env` to `.gitignore`** — immediate security fix.
2. **Delete `sst_is_test.py`** — remove 555 lines of dead code.
3. **Add `pyproject.toml`** — enable editable install, remove `sys.path.insert` hack from tests.
4. **Add GitHub Actions CI** — run `pytest` on push to `main` and PRs. The test suite already works with mocks, so no GPU/hardware needed in CI.
5. **Clean up empty assets** — remove or replace `whisper-logo.png` and `whisper-logo.svg`.
