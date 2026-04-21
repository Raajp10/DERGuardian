# Root Cleanup Plan

## Goal

Keep the repository root as a clean GitHub front door and move audit/analysis clutter into `docs/reports/internal_audits/`.

## Root Files Kept

- `README.md`
- `LICENSE`
- `.gitignore`
- `requirements.txt`
- `environment.yml`
- `RELEASE_CHECKLIST.md`
- `FINAL_PUBLISHABLE_REPO_DECISION.md`
- `GITHUB_READY_SUMMARY.md`

## Reorganized Files

- Root-level audit reports, verification CSVs, old figures, generated summaries, draft reports, and local analysis artifacts were moved into `docs/reports/internal_audits/`.
- Phase 1 LLM/context helpers were moved from root into `phase1_models/context/`.
- Root-level model loader and token sequence helper were moved into `phase1_models/`.
- Configs now live under `configs/`.
- Sample/manifest-only data now lives under `data/sample_or_manifest_only/`.
- Public normalized reports remain under `docs/reports/` and selected evidence mirrors remain under `artifacts/`.

## Canonical Output Protection

No canonical generated outputs under `outputs/` were moved or overwritten. The canonical benchmark source remains `outputs/window_size_study/final_window_comparison.csv`, and the canonical winner remains `transformer @ 60s`.

## Remaining Manual Judgment

- Confirm license owner/copyright holder.
- Decide whether any data/model artifacts may be published as release assets.
