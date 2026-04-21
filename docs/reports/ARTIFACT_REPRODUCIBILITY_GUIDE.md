# Artifact Reproducibility Guide

## Primary Commands

```bash
python scripts/run_detector_window_zero_day_audit.py
python scripts/final_repo_publication_cleanup.py
```

## Source-of-Truth Files

- Canonical benchmark: `outputs/window_size_study/final_window_comparison.csv`
- Full benchmark comparison: `phase1_window_model_comparison_full.csv`
- Heldout synthetic detector matrix: `zero_day_model_window_results_full.csv`
- Cross-context summary: `FINAL_MODEL_CONTEXT_COMPARISON.csv`
- Phase 2 inventory: `phase2_scenario_master_inventory.csv`

## Large Artifacts

Full time-series outputs and model packages are large. For GitHub, use manifests and selected CSV/MD/PNG mirrors under `artifacts/` and `docs/` unless data-sharing approval allows full artifact publication.
