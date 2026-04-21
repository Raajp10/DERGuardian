# Phase 1 Upgraded Pipeline

This Phase 1 upgrade keeps the existing clean-data and merged-window pipeline intact and extends the canonical full-run with four additional layers:

1. a primary anomaly detector benchmark over aligned residual windows
2. structured per-window context summaries
3. an optional local prompt-style reasoning layer
4. a transparent fusion layer that combines detector, context, and token-model scores

The canonical artifact roots remain:

- reports: `outputs/reports/model_full_run_artifacts/`
- models: `outputs/models_full_run/`

## What Is New

- `phase1_context_builder.py`
  Builds JSONL and parquet context summaries for every aligned residual window. Each summary includes top deviating signals, mapped assets/components, environmental context, feeder context, voltage violations, PV availability-vs-dispatch consistency, and BESS SOC consistency.

- `phase1_llm_reasoner.py`
  Provides an optional local prompt-style reasoning baseline. It consumes the structured context summaries and emits:
  - likely anomaly family
  - human-readable explanation
  - confidence score
  - likely affected assets

- `phase1_token_timeseries_model.py`
  Exposes the tokenized time-series language-style baseline. This is the same experimental family represented internally as `llm_baseline`, but the implementation is now centralized in a dedicated module and documented clearly as a local tokenized temporal model rather than a foundation LLM.

- `phase1_fusion.py`
  Fits validation-grounded logistic score fusion for:
  - `detector_only`
  - `detector_plus_context`
  - `detector_plus_context_plus_token`

- `phase1_models/run_full_evaluation.py`
  Now generates the original model tables and plots plus:
  - context summaries
  - reasoning outputs
  - fusion ablation tables
  - per-mode fusion plots
  - a combined Phase 1 overview panel

## Canonical Training and Evaluation Commands

Rebuild the full canonical Phase 1 benchmark:

```powershell
python phase1_models/run_full_evaluation.py --project-root . --feature-counts 32,64,96,128 --seq-lens 4,8,12 --epochs 24 --patience 5
```

Export the markdown summary report:

```powershell
python phase1_models/export_model_report.py --project-root . --run-mode full
```

This writes the companion summary to `outputs/reports/model_full_run_artifacts/model_report_export.md`.
The canonical `model_report_full.md` is owned by `phase1_models/run_full_evaluation.py`.

Build only the structured context summaries:

```powershell
python phase1_context_builder.py --project-root . --top-k 8
```

Run only the optional local reasoning layer:

```powershell
python phase1_llm_reasoner.py --project-root .
```

Train only the tokenized time-series language-style baseline:

```powershell
python phase1_token_timeseries_model.py --project-root . --max-features 48 --seq-len 8 --epochs 6
```

Backward-compatible entry point for the token baseline:

```powershell
python phase1_models/train_llm_baseline.py --project-root . --max-features 48 --seq-len 8 --epochs 6
```

## Baseline Models Still Supported

- `threshold_baseline`
- `isolation_forest`
- `autoencoder`
- `gru`
- `lstm`
- `transformer`
- `llm_baseline` (paper-facing label: `Tokenized LLM-Style Baseline`)

## Saved Artifacts

Core detector artifacts:

- `outputs/models_full_run/<model_name>/model.pt` or `model.pkl`
- `outputs/models_full_run/<model_name>/metadata.pkl`
- `outputs/models_full_run/<model_name>/results.json`
- `outputs/models_full_run/<model_name>/predictions.parquet`

Standardized reload-ready packages:

- `outputs/phase1_ready_models/<model_name>/checkpoint.pt` or `checkpoint.pkl`
- `outputs/phase1_ready_models/<model_name>/preprocessing.pkl`
- `outputs/phase1_ready_models/<model_name>/feature_columns.json`
- `outputs/phase1_ready_models/<model_name>/thresholds.json`
- `outputs/phase1_ready_models/<model_name>/calibration.json`
- `outputs/phase1_ready_models/<model_name>/config.json`
- `outputs/phase1_ready_models/<model_name>/training_history.json`
- `outputs/phase1_ready_models/<model_name>/metrics.json`
- `outputs/phase1_ready_models/<model_name>/split_summary.json`
- `outputs/phase1_ready_models/<model_name>/predictions.parquet`
- `outputs/phase1_ready_models/<model_name>/model_manifest.json`
- `outputs/phase1_ready_models/fusion_<mode>/fusion_manifest.json`
- `outputs/phase1_ready_models/fusion_<mode>/fusion_config.json`
- `outputs/phase1_ready_models/fusion_<mode>/fusion_thresholds.json`
- `outputs/phase1_ready_models/fusion_<mode>/fusion_calibration.json`

Reload verification command:

```powershell
python phase1_model_loader.py --project-root . --verify-all
```

Phase 3 package-reuse command:

```powershell
python phase3/run_zero_day_evaluation.py --project-root . --reuse-ready-model autoencoder --holdout-scenarios scn_false_data_injection_pv60_voltage
```

New context and reasoning artifacts:

- `outputs/reports/model_full_run_artifacts/phase1_context_summaries.jsonl`
- `outputs/reports/model_full_run_artifacts/phase1_context_summaries.parquet`
- `outputs/reports/model_full_run_artifacts/phase1_context_reasoning_outputs.jsonl`
- `outputs/reports/model_full_run_artifacts/phase1_context_reasoning_outputs.parquet`
- `outputs/models_full_run/context_reasoner/prompt_template.txt`
- `outputs/models_full_run/context_reasoner/reasoner_metadata.json`

New fusion artifacts:

- `outputs/reports/model_full_run_artifacts/fusion_ablation_table.csv`
- `outputs/reports/model_full_run_artifacts/fusion_ablation_plot.png`
- `outputs/reports/model_full_run_artifacts/phase1_upgrade_overview_panel.png`
- `outputs/models_full_run/fusion/<mode>/results.json`
- `outputs/models_full_run/fusion/<mode>/predictions.parquet`
- `outputs/models_full_run/fusion/<mode>/validation_predictions.parquet`
- `outputs/reports/model_full_run_artifacts/fusion/<mode>/roc_curve.png`
- `outputs/reports/model_full_run_artifacts/fusion/<mode>/precision_recall_curve.png`
- `outputs/reports/model_full_run_artifacts/fusion/<mode>/confusion_matrix.png`
- `outputs/reports/model_full_run_artifacts/fusion/<mode>/anomaly_score_distribution.png`
- `outputs/reports/model_full_run_artifacts/fusion/<mode>/threshold_visualization.png`

## Paper-Quality Graphs Produced

Per-model graphs:

- train/validation loss
- anomaly score distribution
- ROC curve
- precision-recall curve
- confusion matrix
- threshold visualization

Combined graphs:

- `model_comparison_plot_full.png`
- `latency_by_scenario.png`
- `fusion_ablation_plot.png`
- `phase1_upgrade_overview_panel.png`

## Truthful Interpretation

- The primary detector remains the anomaly-detection backbone. The reasoning layer is optional and should be presented as contextual interpretation, not as the primary detector.
- The tokenized baseline is an experimental local sequence model using discretized feature tokens. It should not be described as GPT, Claude, Gemini, or as a foundation model.
- Fusion is validation-calibrated and reproducible, but its usefulness should still be established from the reported metrics rather than assumed.
