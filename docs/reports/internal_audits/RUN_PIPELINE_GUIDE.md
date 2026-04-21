# Run Pipeline Guide

Use `scripts/run_full_pipeline.py` as the single release-facing entrypoint.

## Default Run

```bash
python scripts/run_full_pipeline.py
```

The default behavior skips stages whose completion markers already exist. This protects the frozen canonical benchmark artifacts from accidental regeneration.

## Preview Commands

```bash
python scripts/run_full_pipeline.py --dry-run
```

## Run One Stage

```bash
python scripts/run_full_pipeline.py --stage phase3_detector_contexts
```

## Force Regeneration

```bash
python scripts/run_full_pipeline.py --force
```

Use `--force` only when you intentionally want to rerun generated artifacts. The publication claim remains that the canonical benchmark winner is `transformer @ 60s` from the frozen benchmark source.

## Stage Order

1. `phase1_clean_data`
2. `phase1_validate_clean`
3. `phase2_validate_scenarios`
4. `phase2_generate_attacked`
5. `phase2_validate_attacked`
6. `phase1_window_model_benchmark`
7. `phase3_detector_contexts`
8. `phase3_xai_reports`
9. `phase3_deployment_benchmark`
10. `publication_reports`
