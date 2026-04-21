# Demo Explanation For Professor

## Bottom Line

The cleanest professor-facing demo is `python demo/run_demo.py`. It is a compact replay of the Phase 1 detector plus the Phase 3 explanation layer using a small sampled attacked scenario.

`deployment_runtime/` is a separate prototype that demonstrates how the research pipeline could be arranged as edge, gateway, and control-center components. It is useful as supporting evidence for deployment architecture, but it is not the simplest first demo.

## Canonical Professor Demo Flow

1. `demo/run_demo.py` loads `demo/config.yaml`.
2. `demo/utils/data_loader.py` ensures sample CSVs, model package files, and copied scenario JSONs exist.
3. `demo/detector/run_detector.py` scores the sampled attacked windows with the packaged GRU model.
4. `demo/xai/run_xai_v4.py` builds a grounded explanation packet for the highest-priority alert.
5. `demo/utils/plotting.py` creates three presentation plots from existing evaluation artifacts.

## Exact Files Needed For The Professor Demo

### Required and active

- `demo/run_demo.py`
- `demo/config.yaml`
- `demo/utils/data_loader.py`
- `demo/detector/run_detector.py`
- `demo/xai/run_xai_v4.py`
- `demo/utils/plotting.py`
- `demo/data/sample_attack.csv`
- `demo/data/sample_clean.csv`
- `demo/models/gru_package/*` except the historical/evaluation extras are optional for execution
- `outputs/attacked/attack_labels.parquet`
- `outputs/attacked/cyber_events.parquet`
- `outputs/attacked/scenario_manifest.json`
- `outputs/reports/model_full_run_artifacts/feature_importance_table.csv`
- `outputs/models_full_run/*` for plots
- `outputs/reports/model_paper_artifacts/detection_latency_analysis.csv` for plots

### Helpful but not strictly required for the current professor run

- `demo/scenarios/*.json`
- `demo/models/gru_model.pt`
- `demo/models/gru_metadata.pkl`
- `demo/models/gru_results.json`
- all LSTM copies in `demo/models/`

## What Is Just Packaged Artifact Or Reference

- `demo/scenarios/*.json`: copied LLM scenario bundles for context; not used by `run_demo.py`.
- `demo/models/lstm_*` and `demo/models/lstm_package/*`: showcase/reference copies; current detector does not load them.
- `demo/models/gru_model.pt`, `demo/models/gru_metadata.pkl`, `demo/models/gru_results.json`: reference copies outside the active package loader path.
- `demo/models/*/predictions.parquet`, `metrics.json`, `training_history.json`, `split_summary.json`: package documentation/evidence, not runtime-critical for the demo.
- `deployment_runtime/docs/*` and `deployment_runtime/examples/*`: reference material only.
- `deployment_runtime/__pycache__/*`: generated cache, safe to ignore.

## Inputs And Outputs

### Demo input files

| Path | Role |
|---|---|
| `demo/config.yaml` | Demo configuration and source-path map. |
| `demo/data/sample_attack.csv` | Attacked sampled window set. |
| `demo/data/sample_clean.csv` | Clean sampled window set used for residual alignment. |
| `outputs/attacked/attack_labels.parquet` | Ground-truth labels for the sampled scenario. |
| `outputs/attacked/cyber_events.parquet` | Cyber evidence for explanation grounding. |
| `outputs/attacked/scenario_manifest.json` | Scenario metadata used by XAI. |
| `outputs/reports/model_full_run_artifacts/feature_importance_table.csv` | Feature ranking for explanation packet assembly. |
| `demo/models/gru_package/*` | Packaged GRU detector assets. |
| `outputs/models_full_run/*` | Evaluation metrics and predictions used by plotting. |
| `outputs/reports/model_paper_artifacts/detection_latency_analysis.csv` | Latency data for the latency-vs-F1 figure. |

### Demo output files

| Path | Role |
|---|---|
| `demo/outputs/predictions.csv` | Per-window detector scores and predicted labels. |
| `demo/outputs/alerts.csv` | Alert-only subset, or highest-score row if no alert occurs. |
| `demo/outputs/detector_metadata.json` | Detector package metadata and run counts. |
| `demo/outputs/detector_results.json` | Temporary threshold-only JSON for the XAI wrapper. |
| `demo/outputs/explanations.json` | Grounded explanation packet plus explanation text and validation result. |
| `demo/results/xai_summary.csv` | One-row explanation summary for the selected alert. |
| `demo/results/impact_results.png` | Detector comparison figure. |
| `demo/results/false_alarm_comparison.png` | False-alarm comparison figure. |
| `demo/results/latency_vs_f1.png` | Latency/performance figure. |

## `demo/` Folder Inventory

### Folder: `demo/`

Purpose: self-contained presentation layer for the research demo.

| Path | Type | Size (bytes) | Status | What it does |
|---|---:|---:|---|---|
| `demo/config.yaml` | config | 913 | active but misleading | Main demo config. Uses JSON syntax despite `.yaml` extension. Names sample scenario, top-k explanation count, and source artifact paths. |
| `demo/README.md` | markdown | 1677 | active | Human-facing description of the packaged demo. Mostly accurate, but it understates the repo-level dependencies. |
| `demo/run_demo.py` | python | 1829 | canonical | Main professor-demo entry point. Calls bootstrap, detector, XAI, and plotting in sequence. |

### Folder: `demo/data/`

Purpose: small sampled windows so the demo runs quickly.

| Path | Type | Size (bytes) | Status | What it does |
|---|---:|---:|---|---|
| `demo/data/sample_attack.csv` | csv | 313482 | active/generated sample | 18 rows, 931 columns. Sampled attacked windows centered on `scn_pv60_curtailment_inconsistency`. Includes time columns, physical window aggregates, cyber count features, and attack label columns. |
| `demo/data/sample_clean.csv` | csv | 313311 | active/generated sample | Clean reference counterpart to the attacked sample. Used to build aligned residual features before scoring. |

Key column groups in the sampled CSVs:

- Identity/time: `window_start_utc`, `window_end_utc`, `window_seconds`, `scenario_id`, `run_id`, `split_id`
- Environment: `env_*`
- Feeder summary: `feeder_*`
- Bus voltage/angle blocks: `bus_*`
- Line flow blocks: `line_*`
- Aggregated load: `load_*`
- Regulators, capacitors, switches, breaker: `regulator_*`, `capacitor_*`, `switch_*`, `breaker_*`
- DER telemetry: `pv_*`, `bess_*`
- Derived residual features: `derived_*`
- Cyber window counts: `cyber_event_count_total`, `cyber_auth_count`, `cyber_command_count`, `cyber_attack_count`, protocol counts, auth failures
- Attack labels: `attack_present`, `attack_family`, `attack_severity`, `attack_affected_assets`, `scenario_window_id`

### Folder: `demo/detector/`

Purpose: lightweight wrapper around the packaged Phase 1 detector.

| Path | Type | Size (bytes) | Status | What it does |
|---|---:|---:|---|---|
| `demo/detector/run_detector.py` | python | 3655 | active | Loads sample attacked and clean windows, aligns residuals with `phase1_models.residual_dataset`, loads the GRU package via `phase1_model_loader`, writes predictions, alerts, and detector metadata. |

Important notes:

- It reads labels from `outputs/attacked/attack_labels.parquet`, so the demo is not fully isolated.
- It ignores the LSTM entry in `demo/config.yaml`; only GRU is used.

### Folder: `demo/models/`

Purpose: packaged detector artifacts copied in for presentation and quick loading.

#### Top-level model copies

| Path | Type | Size (bytes) | Status | What it does |
|---|---:|---:|---|---|
| `demo/models/gru_metadata.pkl` | pickle | 2949 | reference | Standalone GRU metadata copy. Not the path used by the active package loader. |
| `demo/models/gru_model.pt` | torch checkpoint | 78141 | reference | Standalone GRU weight file. Not directly used by `run_detector.py`. |
| `demo/models/gru_results.json` | json | 14910 | reference | Saved GRU evaluation summary. Useful for inspection, not required by the active demo path. |
| `demo/models/lstm_metadata.pkl` | pickle | 2950 | packaged reference | LSTM metadata copy. Current demo does not load it. |
| `demo/models/lstm_model.pt` | torch checkpoint | 103293 | packaged reference | LSTM weight file. Current demo does not load it. |
| `demo/models/lstm_results.json` | json | 14937 | packaged reference | LSTM evaluation summary. Current demo does not load it. |

#### Folder: `demo/models/gru_package/`

Purpose: actual active packaged detector used by the professor demo.

| Path | Type | Size (bytes) | Status | What it does |
|---|---:|---:|---|---|
| `demo/models/gru_package/calibration.json` | json | 190 | active | Saved score calibration settings. Loaded by `phase1_model_loader`. |
| `demo/models/gru_package/checkpoint.pt` | torch checkpoint | 78585 | active | GRU weights used for inference. |
| `demo/models/gru_package/config.json` | json | 333 | active | Model architecture and sequence settings. |
| `demo/models/gru_package/feature_columns.json` | json | 1298 | active | Ordered feature list expected by the model. |
| `demo/models/gru_package/metrics.json` | json | 464 | supporting | Saved package metrics; not needed for live inference. |
| `demo/models/gru_package/model_manifest.json` | json | 3960 | active | Central manifest naming all package files and metadata. |
| `demo/models/gru_package/predictions.parquet` | parquet | 7019 | supporting | Saved package predictions from original evaluation; not needed for the live professor demo. |
| `demo/models/gru_package/preprocessing.pkl` | pickle | 803 | active | Standardizer/discretizer package state. |
| `demo/models/gru_package/split_summary.json` | json | 2109 | supporting | Train/val/test counts and scenario counts. Good provenance, not runtime-critical. |
| `demo/models/gru_package/thresholds.json` | json | 100 | active | Primary decision threshold. |
| `demo/models/gru_package/training_history.json` | json | 457 | supporting | Training-curve summary; not used at demo runtime. |

Observed manifest/config facts:

- Model name: `gru`
- Family: `sequence_classifier`
- Feature mode: `aligned_residual`
- Sequence length: `4`
- Architecture: `input_dim=32`, `hidden_dim=64`, `num_layers=1`

#### Folder: `demo/models/lstm_package/`

Purpose: packaged secondary model kept as a reference/demo artifact.

| Path | Type | Size (bytes) | Status | What it does |
|---|---:|---:|---|---|
| `demo/models/lstm_package/calibration.json` | json | 191 | packaged reference | LSTM calibration. |
| `demo/models/lstm_package/checkpoint.pt` | torch checkpoint | 103673 | packaged reference | LSTM weights. |
| `demo/models/lstm_package/config.json` | json | 333 | packaged reference | LSTM configuration. |
| `demo/models/lstm_package/feature_columns.json` | json | 1298 | packaged reference | Feature order. |
| `demo/models/lstm_package/metrics.json` | json | 469 | packaged reference | Saved LSTM metrics. |
| `demo/models/lstm_package/model_manifest.json` | json | 3961 | packaged reference | LSTM package manifest. |
| `demo/models/lstm_package/predictions.parquet` | parquet | 7019 | packaged reference | Saved LSTM predictions. |
| `demo/models/lstm_package/preprocessing.pkl` | pickle | 803 | packaged reference | Saved preprocessing. |
| `demo/models/lstm_package/split_summary.json` | json | 2109 | packaged reference | Split counts. |
| `demo/models/lstm_package/thresholds.json` | json | 100 | packaged reference | LSTM threshold. |
| `demo/models/lstm_package/training_history.json` | json | 458 | packaged reference | Training history. |

Assessment: this entire LSTM package is not part of the current canonical professor-demo execution path.

### Folder: `demo/outputs/`

Purpose: latest generated detector and explanation outputs.

| Path | Type | Size (bytes) | Status | What it does |
|---|---:|---:|---|---|
| `demo/outputs/alerts.csv` | csv | 1995 | generated | 11 rows, 11 columns. Alert-only subset of predictions. |
| `demo/outputs/detector_metadata.json` | json | 1746 | generated | Package identity, threshold, feature columns, and row counts for the latest detector run. |
| `demo/outputs/detector_results.json` | json | 63 | generated/transient | Tiny threshold-only file created so the XAI wrapper can call the Phase 3 packet builder. |
| `demo/outputs/explanations.json` | json | 17555 | generated | Top-level keys: `packet`, `explanation`, `validation`, `mode`. Main explanation output for the demo. |
| `demo/outputs/predictions.csv` | csv | 2528 | generated | 15 rows, 11 columns. Columns: `window_start_utc`, `window_end_utc`, `scenario_id`, `attack_present`, `attack_family`, `attack_severity`, `attack_affected_assets`, `split_name`, `raw_score`, `score`, `predicted`. |

### Folder: `demo/results/`

Purpose: presentation-ready figures and compact explanation summary.

| Path | Type | Size (bytes) | Status | What it does |
|---|---:|---:|---|---|
| `demo/results/false_alarm_comparison.png` | png | 34871 | generated | Comparison figure built from existing evaluation artifacts. |
| `demo/results/impact_results.png` | png | 64714 | generated | Detector comparison summary plot. |
| `demo/results/latency_vs_f1.png` | png | 28950 | generated | Latency/performance tradeoff figure. |
| `demo/results/xai_summary.csv` | csv | 230 | generated | 1 row, 8 columns: `incident_id`, `suspected_attack_family`, `confidence_level`, `confidence_score`, `affected_assets`, `physical_evidence_count`, `cyber_evidence_count`, `validation_passed`. |

### Folder: `demo/scenarios/`

Purpose: copied LLM-generated scenario bundles for presentation context.

| Path | Type | Size (bytes) | Status | What it does |
|---|---:|---:|---|---|
| `demo/scenarios/chatgpt.json` | json | 9723 | reference | Copied scenario bundle. Top-level keys: `dataset_id`, `scenarios`. Not used by `run_demo.py`. |
| `demo/scenarios/claude.json` | json | 12720 | reference | Same role as above. |
| `demo/scenarios/gemini.json` | json | 11735 | reference | Same role as above. |
| `demo/scenarios/grok.json` | json | 11504 | reference | Same role as above. |

Important note: these are copied from `phase2_llm_benchmark/shared/llm_response/*/new_respnse.json` when present. The source filename typo is real and easy to miss.

### Folder: `demo/utils/`

Purpose: demo bootstrap and plot helper code.

| Path | Type | Size (bytes) | Status | What it does |
|---|---:|---:|---|---|
| `demo/utils/data_loader.py` | python | 6239 | active | Creates demo folders, samples attack/clean windows if needed, copies model artifacts/packages, copies LLM scenario JSONs, and adds repo root to `sys.path`. |
| `demo/utils/plotting.py` | python | 4909 | active | Reads evaluation artifacts from root `outputs/` and writes the three demo figures. |

### Folder: `demo/xai/`

Purpose: demo-only wrapper around the Phase 3 explanation pipeline.

| Path | Type | Size (bytes) | Status | What it does |
|---|---:|---:|---|---|
| `demo/xai/run_xai_v4.py` | python | 4045 | active | Selects the top alert, builds a grounded explanation packet through `phase3_explanations`, validates it, and writes `explanations.json` and `xai_summary.csv`. |

## `deployment_runtime/` Folder Inventory

### Folder: `deployment_runtime/`

Purpose: deployment-oriented replay prototype showing edge, gateway, and control-center behavior.

| Path | Type | Size (bytes) | Status | What it does |
|---|---:|---:|---|---|
| `deployment_runtime/__init__.py` | python | 484 | supporting | Package marker and lightweight package-level context. |
| `deployment_runtime/alert_packet_schema.json` | json schema | 2201 | active | JSON schema for edge alert packets. |
| `deployment_runtime/build_alert_packet.py` | python | 2511 | active | Builds and validates alert packets. |
| `deployment_runtime/config_runtime.json` | json config | 2679 | active | Runtime config: site IDs, output roots, gateway fusion window, demo default scenario, and deployed model profiles. |
| `deployment_runtime/control_center_runtime.py` | python | 12041 | active | Collects forwarded packets, writes central alert tables, and optionally attaches explanations. |
| `deployment_runtime/demo_replay.py` | python | 7073 | canonical for runtime prototype | Runs edge, gateway, control-center, and report export end to end. |
| `deployment_runtime/edge_runtime.py` | python | 19530 | active | Replays measured/cyber streams, builds rolling windows, scores deployed models, and emits alert packets plus local context. |
| `deployment_runtime/export_deployment_report.py` | python | 8194 | active | Produces a report-ready markdown summary and exported CSVs under `outputs/reports/deployment_runtime/`. |
| `deployment_runtime/gateway_runtime.py` | python | 7534 | active | Fuses/deduplicates edge packets and forwards site-level incidents. |
| `deployment_runtime/latency_budget.py` | python | 4897 | active | Summarizes latency by stage from runtime traces. |
| `deployment_runtime/load_deployed_models.py` | python | 13381 | active | Loads the threshold baseline and optional autoencoder into runtime-friendly wrappers. |
| `deployment_runtime/local_buffer.py` | python | 4715 | active | Maintains recent window and cyber context so alerts can link to local evidence. |
| `deployment_runtime/README.md` | markdown | 4899 | active | Good technical overview of what the deployment prototype implements. |
| `deployment_runtime/runtime_common.py` | python | 8595 | active | Shared config/path/timestamp/serialization helpers and runtime profile selection. |
| `deployment_runtime/stream_window_builder.py` | python | 9130 | active | Incrementally builds project-compatible rolling windows from ordered measured and cyber events. |

### Folder: `deployment_runtime/docs/`

Purpose: supporting narrative docs for the deployment prototype.

| Path | Type | Size (bytes) | Status | What it does |
|---|---:|---:|---|---|
| `deployment_runtime/docs/demo_usage.md` | markdown | 1412 | supporting | How to run the replay demo and what outputs to expect. |
| `deployment_runtime/docs/deployment_contract.md` | markdown | 1970 | supporting | Contract for alert packets and model artifact expectations. |
| `deployment_runtime/docs/runtime_architecture.md` | markdown | 2373 | supporting | Maps runtime modules to the paper’s architecture figure. |

### Folder: `deployment_runtime/examples/`

Purpose: illustrative reference artifacts, not live code paths.

| Path | Type | Size (bytes) | Status | What it does |
|---|---:|---:|---|---|
| `deployment_runtime/examples/example_alert_packet.json` | json | 1537 | reference | Example of a valid forwarded/edge alert packet shape. |
| `deployment_runtime/examples/example_runtime_config.json` | json | 1643 | reference | Minimal example runtime config. |

### Folder: `deployment_runtime/__pycache__/`

Purpose: generated bytecode cache; unnecessary for source understanding.

| Path | Type | Size (bytes) | Status | What it does |
|---|---:|---:|---|---|
| `deployment_runtime/__pycache__/build_alert_packet.cpython-314.pyc` | pyc | 3915 | generated | Python bytecode cache. |
| `deployment_runtime/__pycache__/control_center_runtime.cpython-314.pyc` | pyc | 16522 | generated | Python bytecode cache. |
| `deployment_runtime/__pycache__/demo_replay.cpython-314.pyc` | pyc | 9628 | generated | Python bytecode cache. |
| `deployment_runtime/__pycache__/edge_runtime.cpython-314.pyc` | pyc | 24503 | generated | Python bytecode cache. |
| `deployment_runtime/__pycache__/export_deployment_report.cpython-314.pyc` | pyc | 13040 | generated | Python bytecode cache. |
| `deployment_runtime/__pycache__/gateway_runtime.cpython-314.pyc` | pyc | 11705 | generated | Python bytecode cache. |
| `deployment_runtime/__pycache__/latency_budget.cpython-314.pyc` | pyc | 7532 | generated | Python bytecode cache. |
| `deployment_runtime/__pycache__/load_deployed_models.cpython-314.pyc` | pyc | 21710 | generated | Python bytecode cache. |
| `deployment_runtime/__pycache__/local_buffer.cpython-314.pyc` | pyc | 6594 | generated | Python bytecode cache. |
| `deployment_runtime/__pycache__/runtime_common.cpython-314.pyc` | pyc | 16649 | generated | Python bytecode cache. |
| `deployment_runtime/__pycache__/stream_window_builder.cpython-314.pyc` | pyc | 13346 | generated | Python bytecode cache. |
| `deployment_runtime/__pycache__/__init__.cpython-314.pyc` | pyc | 621 | generated | Python bytecode cache. |

## Which Script Should Be Run

### For the professor demo

```powershell
python demo/run_demo.py
```

### For the deployment prototype only

```powershell
python deployment_runtime/demo_replay.py --attacked --threshold-plus-autoencoder
```

Assessment: the second command is valuable for architecture discussion, but the first command is the canonical demonstration script.

## What Is Confusing, Misleading, Or Unnecessary

1. `demo/config.yaml` is not YAML.
2. `demo/config.yaml` mentions both GRU and LSTM, but only GRU is used.
3. `demo/scenarios/*.json` look important but are not consumed by `run_demo.py`.
4. `demo/models/lstm_*` and `demo/models/lstm_package/*` make the demo look multi-model even though the detector path is single-model.
5. `demo/utils/plotting.py` reaches into root `outputs/` directories, so the demo looks more self-contained than it is.
6. `demo/outputs/detector_results.json` is a tiny temporary bridge file with a grand-looking name.
7. The typo `new_respnse.json` is part of the active bootstrap logic and can easily break copying if “fixed” in one place only.
8. `deployment_runtime/` is easy to mistake for the main demo because it has stronger engineering structure and docs, but it is really a separate replay prototype.

## Suggested Cleanup So The Demo Is Understandable In 2 Minutes

1. Keep only the active professor path in `demo/`: GRU package, sample CSVs, detector, XAI, plotting, and outputs.
2. Move `demo/scenarios/` to `demo/references/` unless the professor will explicitly inspect cross-LLM scenario generation.
3. Move all LSTM copies out of `demo/models/` or label them clearly as optional comparison artifacts.
4. Rename `demo/config.yaml` to `demo/config.json`.
5. Add a preflight check in `demo/run_demo.py` that tells the user exactly which repo-level dependencies are missing.
6. Make `demo/utils/plotting.py` optional so the core detector-plus-explanation demo can run even without the full evaluation artifacts.
7. Add one short top-of-file comment in `demo/run_demo.py`: “professor demo entry point.”
8. Add one short top-of-file comment in `deployment_runtime/demo_replay.py`: “deployment prototype, not the main professor demo.”

## Conservative Conclusion

- Canonical professor demo: `demo/run_demo.py`
- Canonical deployment-architecture prototype: `deployment_runtime/demo_replay.py`
- Safe paper/demo references: `demo/README.md`, `deployment_runtime/README.md`, the generated `demo/results/*.png`, `demo/results/xai_summary.csv`, and the active source files listed above
- Files to ignore during a short demo: all `__pycache__`, all LSTM package copies, scenario JSON copies, and deployment docs/examples unless specifically asked
