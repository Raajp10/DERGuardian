# Git Release Priority Order

## Commit First

1. `README.md`, `LICENSE`, `.gitignore`, `requirements.txt`, `environment.yml`
2. Source code: `common/`, `opendss/`, `phase1/`, `phase1_models/`, `phase2/`, `phase3/`, `phase3_explanations/`, `scripts/`
3. Public docs: `docs/methodology/`, `docs/reports/`, `docs/figures/`
4. Evidence mirrors: `artifacts/`
5. Registries and release docs: `REPO_ARTIFACT_MAP.csv`, `REPO_CONTEXT_INDEX.csv`, `EXPERIMENT_REGISTRY.csv`, `PUBLISHABLE_REPORT_INDEX.md`, `MASTER_PROJECT_UNDERSTANDING_REPORT.md`, `MASTER_PROJECT_FILE_AUDIT.csv`, `MASTER_PROJECT_FINAL_VERIFICATION.md`, `MASTER_PROJECT_FINAL_STATUS.csv`

## Commit Next

- `tests/`
- `configs/`
- `data/sample_or_manifest_only/`
- `src/README.md`
- Professor/GitHub summaries and final publishability files.

## Optional

- Additional root-level legacy reports only if a reviewer explicitly asks for them and they are checked for claim safety.
- Extra figures only if moved under `docs/figures/` and indexed.

## Commit Only If Size/Privacy Allows

- Small sample data under `data/sample_or_manifest_only/`.
- Full generated datasets from `outputs/`.
- Model checkpoints or ready packages.
- Demo bundles.

Large artifacts are better attached to a release, stored in institutional storage, or represented by manifests.
