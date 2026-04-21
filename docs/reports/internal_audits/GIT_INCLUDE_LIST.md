# Git Include List

This list names the files and folders that should be committed for a clean public DERGuardian release.

## Root Project Files

- `README.md` - GitHub front page.
- `LICENSE` - license text; confirm owner before release.
- `.gitignore` - protects generated data, checkpoints, zips, and root clutter.
- `requirements.txt` - pip dependency list.
- `environment.yml` - conda environment option.

## Source Code

- `common/` - shared config, OpenDSS, telemetry, timing, noise, metadata, and IO utilities.
- `opendss/` - IEEE 123-bus feeder, DER overlays, controls, monitors, and topology validation.
- `phase1/` - clean data generation, validation, and window construction.
- `phase1_models/` - detector feature construction, training, evaluation, ready packages, and metrics code.
- `phase2/` - scenario schemas, validation, injection compilation, cyber logs, attacked dataset generation, and validation.
- `phase3/` - zero-day-like evaluation, ablations, sweeps, error analysis, and final artifact assembly.
- `phase3_explanations/` - grounded explanation packet, prompt, schema, validation, and evaluator code.
- `scripts/` - reproducibility/audit/report-generation scripts, including detector, TTM, LoRA, XAI, deployment, and release cleanup scripts.

## Documentation And Reports

- `docs/methodology/` - diagram-safe methodology docs and slide text.
- `docs/reports/` - normalized paper-facing reports.
- `docs/figures/` - selected final figures.
- `GITHUB_READY_SUMMARY.md` - short GitHub summary.
- `PROFESSOR_READY_METHOD_SUMMARY.md` - professor-facing reading summary.
- `FINAL_PUBLISHABLE_REPO_DECISION.md` - release decision.
- `FINAL_REPO_PUBLISHABILITY_CHECKLIST.md` - final release checklist.
- `FINAL_REPO_PUBLISHABILITY_STATUS.csv` - final release status table.
- `MASTER_PROJECT_UNDERSTANDING_REPORT.md` - end-to-end master report.
- `MASTER_PROJECT_FILE_AUDIT.csv` - file-level release inventory.
- `MASTER_PROJECT_FINAL_VERIFICATION.md` - final verification statement.
- `MASTER_PROJECT_FINAL_STATUS.csv` - final master status table.
- `PROFESSOR_READING_GUIDE.md` - ordered professor reading path.

## Git-Friendly Evidence Artifacts

- `artifacts/benchmark/` - mirrored canonical benchmark CSV/manifest.
- `artifacts/replay/` - mirrored heldout replay CSV/manifest.
- `artifacts/zero_day_like/` - mirrored heldout synthetic and cross-context CSV/manifest.
- `artifacts/extensions/` - mirrored TTM and LoRA CSV/manifest.
- `artifacts/xai/` - mirrored XAI case audit/manifest.
- `artifacts/deployment/` - mirrored offline deployment benchmark/manifest.
- `REPO_ARTIFACT_MAP.csv` - maps source artifacts to publishable artifacts.
- `REPO_CONTEXT_INDEX.csv` - defines evaluation contexts.
- `EXPERIMENT_REGISTRY.csv` - model/window/context registry.
- `PUBLISHABLE_REPORT_INDEX.md` - report index.
- `REPORT_NAME_NORMALIZATION_MAP.csv` - normalizes old/new report names.

## Lightweight Project Support

- `configs/` - configuration notes.
- `data/sample_or_manifest_only/` - data-release policy notes.
- `src/README.md` - explains why root packages are preserved.
- `tests/` - regression and pipeline tests.
