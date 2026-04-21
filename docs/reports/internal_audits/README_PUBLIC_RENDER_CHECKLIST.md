# README Public Render Checklist

- [x] Public raw README fetch succeeds.
- [x] README contains the quickstart install command: `python -m pip install -r requirements.txt`.
- [x] README contains the dry-run command: `python scripts/run_full_pipeline.py --dry-run`.
- [x] Windows PowerShell venv activation line is present as `./.venv/Scripts/Activate.ps1`.
- [x] README links to `docs/REPOSITORY_STRUCTURE_GUIDE.md`.
- [x] README links to `docs/PYTHON_CODE_NAVIGATION.md`.
- [x] README keeps canonical benchmark wording: `Transformer @ 60s`.
- [x] README keeps context separation between canonical benchmark, heldout replay, heldout synthetic zero-day-like evaluation, and extensions.
- [x] README does not claim AOI, human-like root-cause analysis, real-world zero-day robustness, or field edge deployment.

## Fixes During This Pass

No README command formatting changes were required during this pass. The public execution issue found was in config loading, not README rendering.

