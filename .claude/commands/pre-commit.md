Pre-commit verification checklist. Run before committing changes.

Execute these checks in order:

1. **Tests**: Run `pytest -v` — all must pass
2. **Imports**: Verify no unused imports were added to changed files
3. **Temp files**: Confirm no `temp_recording.wav` or other temp files are staged
4. **Secrets**: Check that no API keys, `.env` files, or credentials are staged (`git diff --cached`)
5. **Dead code**: Verify changes don't reference `sst_is_test.py` (legacy file)
6. **HTML safety**: If `app.py` was modified, confirm `unsafe_allow_html=True` is only used with static CSS, never user input

Report results as a checklist with pass/fail for each item.