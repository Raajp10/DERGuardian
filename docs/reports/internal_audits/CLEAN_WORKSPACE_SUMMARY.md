# Clean Workspace Summary

The workspace was pruned to keep the canonical project pipeline and the runnable demo/deployment prototype.

## Kept

- Core source: `common/`, `opendss/`, `phase1/`, `phase1_models/`, `phase2/`, `phase3/`, `phase3_explanations/`
- Top-level Phase 1 shared files:
  - `phase1_context_builder.py`
  - `phase1_fusion.py`
  - `phase1_llm_reasoner.py`
  - `phase1_model_loader.py`
  - `phase1_token_timeseries_model.py`
- Demo and runtime:
  - `demo/`
  - `deployment_runtime/`
- Reports/code helpers:
  - `reports/`
  - `scripts/`
  - `tests/`
  - `paper_figures/`
- Canonical outputs:
  - `outputs/clean/`
  - `outputs/attacked/`
  - `outputs/windows/`
  - `outputs/models_full_run/`
  - `outputs/phase1_ready_models/`
  - `outputs/phase3_zero_day/`
  - `outputs/deployment_runtime/`
  - `outputs/reports/model_full_run_artifacts/`
  - `outputs/reports/model_paper_artifacts/`
  - `outputs/reports/explanation_artifacts/`
  - `outputs/reports/phase3_zero_day_artifacts/`
  - `outputs/reports/deployment_runtime/`
- Helpful root docs:
  - `README.md`
  - `DEMO_RUNBOOK.md`
  - `DEMO_EXPLANATION_FOR_PROFESSOR.md`
  - `PHASE1_FILE_MAP.md`
  - `PHASE2_FILE_MAP.md`
  - `PHASE3_FILE_MAP.md`

## Removed

- Audit and inventory clutter:
  - `audit/`
  - old audit markdown/csv/txt inventories at repo root
- Benchmark sidecars and LLM package extras:
  - `phase2_llm_benchmark/`
  - `phase2_with_llm/`
- Paper packaging/bundle folders and archives:
  - `paper_figures_architecture/`
  - `paper_figure_suite/`
  - `paper_results_package/`
  - `paper_submission_bundle/`
  - `paper_upgrade/`
  - root zip/rar/package artifacts
- Demo reference-only extras:
  - LSTM demo package and LSTM model copies
  - standalone GRU reference copies outside `demo/models/gru_package/`
  - `demo/scenarios/`
- Legacy/noncanonical output folders:
  - `outputs/ablation/`
  - `outputs/cross_feeder_results/`
  - `outputs/dataset_phase2_expanded/`
  - `outputs/dataset_realistic/`
  - `outputs/feature_regime_cleanup_v1/`
  - `outputs/ieee_results_package/`
  - `outputs/labeled/`
  - `outputs/models/`
  - `outputs/paper_figures/`
  - `outputs/paper_tables/`
  - `outputs/stats/`
  - `outputs/topology_validation/`
  - `outputs/tsg_results_package/`
  - matching noncanonical report subfolders under `outputs/reports/`
- All `__pycache__/` folders and `.pyc` files

## Canonical First Commands

Professor demo:

```powershell
python demo/run_demo.py
```

Deployment replay prototype:

```powershell
python deployment_runtime/demo_replay.py --attacked --threshold-plus-autoencoder
```
