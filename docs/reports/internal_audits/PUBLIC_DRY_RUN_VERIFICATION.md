# Public Dry-Run Verification

## Command

```bash
python scripts/run_full_pipeline.py --dry-run
```

## Result

- Dry-run status: pass
- Imports for the entrypoint: pass
- Relative path handling: pass
- Config file discovery: pass
- Missing sample-data assumptions: none
- Moved Phase 1 context-module imports: verified separately in the smoke test

## Observed Dry-Run Behavior

From a fresh public clone, the dry-run printed the complete ordered pipeline:

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

Because the clean clone contains publication mirror artifacts but not full generated `outputs/`, early data-generation stages were shown as `ready`, while report/artifact stages with committed mirrors were shown as `skip-existing`. This is expected for a public lightweight clone.

