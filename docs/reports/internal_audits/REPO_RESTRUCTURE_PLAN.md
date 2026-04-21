# Repository Restructure Plan

The cleanup keeps original research artifacts in place and adds a publishable layer. This avoids breaking frozen canonical paths while giving GitHub/paper readers a clean entry point.

## Current Tree Audit

- The root contains many historical reports from the audit process; they are retained for traceability.
- Large generated folders such as `outputs/`, raw OpenDSS artifacts, demo zips, and runtime folders are not moved because existing scripts depend on those paths.
- Final publication-facing files are mirrored into `docs/` and `artifacts/` with context-specific grouping.
- Ambiguous legacy reports remain in place but are superseded by the normalized report index and experiment registry.

## Target Structure

- `README.md` - GitHub front page
- `docs/methodology/` - diagram and methodology documentation
- `docs/reports/` - normalized paper-facing reports
- `docs/figures/` - selected final figures
- `artifacts/` - lightweight mirrored CSV/MD artifacts by evaluation context
- `data/sample_or_manifest_only/` - data-access notes and manifests, not full raw data
- `scripts/` - reproducibility and audit scripts
- `tests/` - existing tests

## Preservation Rule

No canonical benchmark artifacts were moved or overwritten. The publishable folders mirror selected final artifacts only.

## Known Legacy Roots Kept

- `outputs/` remains the authoritative generated-artifact store.
- Root-level historical reports remain for traceability.
- Root-level Python packages remain because imports currently depend on them.

## Manual Cleanup Still Optional

- Large zip files and full generated outputs should normally stay out of Git.
- A future refactor may move root-level Python packages under `src/`, but that is intentionally not done here because it would require import rewrites.
