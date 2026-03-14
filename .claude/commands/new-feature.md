Scaffold a new feature for WhisperSST.is.

Feature description: $ARGUMENTS

Follow these steps:

1. **Analyze**: Determine which module(s) the feature belongs to:
   - Audio capture/processing → `whisperSSTis/audio.py`
   - Model/transcription → `whisperSSTis/transcribe.py`
   - GPT post-processing → `whisperSSTis/gpt.py`
   - UI/interaction → `app.py`
   - New module → `whisperSSTis/new_module.py` + update `__init__.py`

2. **Plan**: Present the implementation plan before writing code:
   - Which files will be modified
   - New functions/classes needed
   - Session state changes (if any)
   - Test coverage plan

3. **Wait for approval** before implementing.

4. **Implement**: Write the code following project patterns:
   - Type hints on function signatures
   - Docstrings on public functions
   - Logging via `logging.getLogger(__name__)`
   - Error handling with try/except and logger.error

5. **Test**: Add tests in `tests/` matching existing patterns (pytest + pytest-mock).

6. **Verify**: Run `pytest -v` to confirm nothing broke.