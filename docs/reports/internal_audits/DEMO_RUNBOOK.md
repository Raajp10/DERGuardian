# Demo Runbook

## Scope

This runbook is for the professor-facing demo in `demo/`. The runtime package in `deployment_runtime/` is a separate deployment prototype and is not the simplest script to show first.

## The One Script To Run

From the project root:

```powershell
python demo/run_demo.py
```

That is the canonical professor-demo entry point.

## What `demo/run_demo.py` Actually Does

1. Loads `demo/config.yaml`.
2. Calls `demo/utils/data_loader.py` to make sure demo assets exist.
3. Runs `demo/detector/run_detector.py`.
4. Runs `demo/xai/run_xai_v4.py`.
5. Runs `demo/utils/plotting.py`.
6. Prints a JSON summary of the output file locations.

## Minimum Files Needed For The Professor Demo

### Direct demo code

- `demo/run_demo.py`
- `demo/config.yaml`
- `demo/utils/data_loader.py`
- `demo/detector/run_detector.py`
- `demo/xai/run_xai_v4.py`
- `demo/utils/plotting.py`

### Demo-local data and model package

- `demo/data/sample_attack.csv`
- `demo/data/sample_clean.csv`
- `demo/models/gru_package/model_manifest.json`
- `demo/models/gru_package/config.json`
- `demo/models/gru_package/thresholds.json`
- `demo/models/gru_package/calibration.json`
- `demo/models/gru_package/feature_columns.json`
- `demo/models/gru_package/preprocessing.pkl`
- `demo/models/gru_package/checkpoint.pt`

### Repo-level dependencies the demo still reaches into

- `outputs/attacked/attack_labels.parquet`
- `outputs/attacked/cyber_events.parquet`
- `outputs/attacked/scenario_manifest.json`
- `outputs/reports/model_full_run_artifacts/feature_importance_table.csv`
- `outputs/models_full_run/` for plotting
- `outputs/reports/model_paper_artifacts/detection_latency_analysis.csv` for plotting
- `phase2_llm_benchmark/shared/llm_response/` only if the scenario JSONs need to be refreshed by bootstrap

## Inputs

### User input

- No arguments are required for the professor demo.

### File inputs consumed by the pipeline

- `demo/config.yaml`: scenario selection and source paths.
- `demo/data/sample_attack.csv`: attacked windows used for scoring and explanation.
- `demo/data/sample_clean.csv`: clean reference windows aligned to the attacked sample.
- `outputs/attacked/attack_labels.parquet`: labels used by `run_detector.py`.
- `outputs/attacked/cyber_events.parquet`: cyber evidence used by `run_xai_v4.py`.
- `outputs/attacked/scenario_manifest.json`: scenario metadata used by `run_xai_v4.py`.
- `outputs/reports/model_full_run_artifacts/feature_importance_table.csv`: ranked features used in explanation assembly.
- `demo/models/gru_package/*`: packaged Phase 1 detector assets.
- `outputs/models_full_run/*` and `outputs/reports/model_paper_artifacts/detection_latency_analysis.csv`: source metrics for the comparison plots.

## Outputs

The run writes or refreshes:

- `demo/outputs/predictions.csv`
- `demo/outputs/alerts.csv`
- `demo/outputs/detector_metadata.json`
- `demo/outputs/detector_results.json`
- `demo/outputs/explanations.json`
- `demo/results/xai_summary.csv`
- `demo/results/impact_results.png`
- `demo/results/false_alarm_comparison.png`
- `demo/results/latency_vs_f1.png`

## What To Show In A 2-Minute Demo

1. Run `python demo/run_demo.py`.
2. Open `demo/outputs/alerts.csv` and point out the flagged windows.
3. Open `demo/outputs/explanations.json` or `demo/results/xai_summary.csv` and show the grounded explanation result.
4. Open the three PNG files in `demo/results/` for detector comparison and latency context.

## What Not To Run First

- Do not start with `deployment_runtime/demo_replay.py` unless the professor specifically asks about deployment architecture.
- Do not start with any training or evaluation script from `phase1_models/`.
- Do not rely on the `demo/scenarios/*.json` files as runtime inputs; they are presentation/reference artifacts in the current flow.

## Important Caveats

- `demo/config.yaml` is named like YAML but is actually JSON syntax. It loads only because `data_loader.py` falls back to JSON parsing.
- The detector config mentions both GRU and LSTM, but `demo/detector/run_detector.py` only loads the GRU package.
- The demo is not fully self-contained. It still depends on repo-level `outputs/` artifacts for labels, cyber evidence, manifest data, feature importance, and plotting inputs.
- `demo/models/lstm_*` and `demo/models/lstm_package/*` are packaged references, not active requirements for the current run.
- `demo/models/gru_model.pt`, `demo/models/gru_metadata.pkl`, and `demo/models/gru_results.json` are also reference copies; the active loader uses `demo/models/gru_package/`.

## If The Demo Needs Cleanup Before Presentation

1. Rename `demo/config.yaml` to `demo/config.json`.
2. Remove the unused LSTM references from the demo config and README.
3. Make plotting optional if `outputs/models_full_run/` or latency artifacts are missing.
4. Either use `demo/scenarios/*.json` in the demo flow or move them to a `references/` folder.
5. Bundle the label/cyber/manifest dependencies into `demo/` so the professor demo is actually portable.

## Optional Secondary Demo

If you want to show the deployment concept after the main demo:

```powershell
python deployment_runtime/demo_replay.py --attacked --threshold-plus-autoencoder
```

That is a separate prototype showing edge, gateway, and control-center replay. It is not the simplest canonical professor-demo entry point.
