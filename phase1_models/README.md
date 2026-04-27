# DERGuardian — Phase 1 (Data + Detector Benchmark)

**Version:** 1.0 (hardened)
**Target venue:** IEEE Smart Grid Conference
**Status:** Reviewer-ready after running the verification commands at the bottom of this README.

---

## 0. What is Phase 1?

DERGuardian is split into three phases. **Phase 1** is the foundation — everything else depends on it:

| Phase | Purpose | Status in this README |
|---|---|---|
| **Phase 1** | Generate physics-grounded clean data, build residual windows, train and benchmark anomaly detectors | **Documented here** |
| Phase 2 | Generate physically constrained attack scenarios | Documented separately |
| Phase 3 | Zero-day evaluation, explanation layer, deployment runtime | Documented separately |

Phase 1 produces **two outputs** that the paper relies on:

1. A residual-windowed dataset (`outputs/reports/model_full_run_artifacts/residual_windows_full_run.parquet`)
2. A trained detector benchmark with leave-one-family-out zero-day evaluation, multi-seed confidence intervals, per-family breakdown, and ablation tables

If anything in `paper_numbers.md` (produced at the end of a full run) doesn't match what's in the paper, the paper is wrong, not the run.

---

## 1. Folder Structure (canonical)

```
repo_root/
├── README.md                      ← THIS FILE
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── pipeline_config.yaml       ← data generation config
│   ├── data_config.yaml
│   ├── model_config.yaml          ← per-model hyperparameters
│   └── phase1_runner_config.yaml  ← master orchestration config
├── common/                        ← shared utilities (FROZEN)
│   ├── config.py
│   ├── dss_runner.py              ← OpenDSS wrapper (FROZEN)
│   ├── io_utils.py
│   ├── metadata_schema.py
│   ├── noise_models.py
│   ├── time_alignment.py
│   ├── units_and_channel_dictionary.py
│   └── weather_load_generators.py
├── phase1/                        ← data generation (FROZEN except CLI defaults)
│   ├── build_clean_dataset.py
│   ├── build_windows.py
│   ├── extract_channels.py
│   └── validate_clean_dataset.py
├── phase1_models/                 ← detector training/eval (HARDENED)
│   ├── checkpointing.py           ← NEW: per-epoch save + resume
│   ├── dataset_loader.py
│   ├── feature_builder.py
│   ├── metrics.py
│   ├── model_loader.py
│   ├── model_utils.py
│   ├── neural_models.py           ← FIXED: TokenBaseline pooling bug
│   ├── neural_training.py         ← MODIFIED: checkpointing hooks
│   ├── ready_package_utils.py
│   ├── residual_dataset.py
│   ├── thresholds.py
│   ├── run_full_evaluation.py     ← MODIFIED: new models, fixed splits
│   ├── run_hardened_phase1.py     ← NEW: master orchestrator
│   ├── zero_day_evaluation.py     ← NEW: leave-one-family-out
│   ├── aggregate_seeds.py         ← NEW: multi-seed stats
│   ├── per_family_metrics.py      ← NEW: per-anomaly-family table
│   ├── ablations.py               ← NEW: window/feature/seq_len sweeps
│   ├── reproducibility.py         ← NEW: SHA-256 manifest
│   ├── generate_paper_figures.py
│   ├── export_model_report.py
│   ├── train_threshold_baseline.py
│   ├── train_isolation_forest.py
│   ├── train_autoencoder.py
│   ├── train_gru.py
│   ├── train_lstm.py
│   ├── train_transformer.py
│   ├── train_llm_baseline.py
│   ├── train_lstm_autoencoder.py    ← NEW
│   ├── train_vae.py                 ← NEW
│   ├── train_ocsvm.py               ← NEW
│   ├── train_anomaly_transformer.py ← NEW
│   ├── models/                      ← NEW subfolder for added baselines
│   │   ├── __init__.py
│   │   ├── lstm_autoencoder.py
│   │   ├── vae.py
│   │   ├── ocsvm.py
│   │   └── anomaly_transformer.py
│   └── context/
│       ├── phase1_context_builder.py
│       ├── phase1_fusion.py
│       ├── phase1_llm_reasoner.py
│       └── structured_reasoning.py
├── tests/
│   └── test_phase1_hardened.py
└── outputs/
    ├── clean/                     ← clean simulation artifacts
    ├── attacked/                  ← Phase 2 attack scenarios
    ├── windows/                   ← merged windows
    ├── reports/                   ← all paper-facing outputs
    │   └── model_full_run_artifacts/
    │       ├── residual_windows_full_run.parquet
    │       ├── per_family_results.csv
    │       ├── multi_seed_summary.csv
    │       ├── ablation_*.csv
    │       └── paper_numbers.md   ← SOURCE OF TRUTH for the paper
    ├── models_full_run/           ← per-model checkpoints + predictions
    ├── zero_day_runs/             ← leave-one-family-out results
    └── phase1_hardened/           ← orchestrated outputs
```

---

## 2. Variable / Symbol Glossary

This is the canonical mapping between paper notation and code variables.

### 2.1 Time and windowing

| Paper symbol | Meaning | Code variable | File |
|---|---|---|---|
| $t$ | Continuous time index (seconds, UTC) | `timestamp_utc` | All raw data |
| $L$ | Sliding-window length (seconds) | `window_seconds` | `WindowConfig` |
| $\Delta t$ | Window step (seconds) | `step_seconds` | `WindowConfig` |
| $\mathbf{x}_t$ | System state at time $t$ | `truth_df.iloc[t]` | `truth_physical_timeseries.parquet` |
| $\tilde{\mathbf{x}}_t$ | Expected system state under benign operation | `clean_windows[t]` | `merged_windows_clean.parquet` |
| $\mathbf{r}_t = \mathbf{x}_t - \tilde{\mathbf{x}}_t$ | Residual at time $t$ | `delta__<signal>__<stat>` columns | `residual_windows_full_run.parquet` |
| $\mathbf{X}_i$ | $i$-th input window | window row | `merged_windows_*.parquet` |
| $\mathbf{R}_i$ / $\Delta\mathbf{X}_i$ | $i$-th residual window | residual row | `residual_windows_full_run.parquet` |

**Note on $\tilde{\mathbf{x}}_t$:** In Phase 1 we use a **paired clean-attacked simulation** to obtain $\tilde{\mathbf{x}}_t$ (i.e., re-running the same OpenDSS scenario without the attack). This is explicitly documented in Section IV-A of the paper. For deployment, $\tilde{\mathbf{x}}_t$ is replaced by an LSTM forecaster (Phase 3 deployment).

### 2.2 Detection

| Paper symbol | Meaning | Code variable |
|---|---|---|
| $f(\cdot)$ | Anomaly scoring function | `predict_classifier_scores`, `predict_autoencoder_errors`, etc. |
| $s_i$ | Anomaly score for window $i$ | `score` column in `predictions.parquet` |
| $\tau$ | Decision threshold | `threshold` in `thresholds.json` |
| $\hat{y}_i$ | Binary detection | `predicted` column |
| $y_i$ | Ground truth | `attack_present` column |

### 2.3 Splits and zero-day

| Paper symbol | Meaning | Code variable |
|---|---|---|
| $\mathcal{D}_{\text{train}}$ | Training set (benign + 5 of 6 anomaly families) | `splits["train"]` |
| $\mathcal{D}_{\text{val}}$ | Validation set | `splits["val"]` |
| $\mathcal{D}_{\text{test}}$ | Test set (held-out family) | `splits["test"]` |
| $\mathcal{S}_{\text{train}}$, $\mathcal{S}_{\text{test}}$ | Scenario family sets | filtered by `attack_family` column |

### 2.4 Anomaly families (canonical names — DO NOT change)

```python
ANOMALY_FAMILIES = [
    "false_data_injection",
    "command_delay",
    "command_suppression",
    "der_disconnect",
    "oscillatory_control",
    "coordinated_multi_asset",
]
```

These exact strings appear in `attack_family` columns throughout the codebase.

### 2.5 Evaluation metrics

| Metric | Code function | Where written |
|---|---|---|
| Precision | `metrics.compute_binary_metrics` → `precision` | `results.json` |
| Recall | same → `recall` | same |
| F1 | same → `f1` | same |
| ROC-AUC | same → `roc_auc` | same |
| Average Precision (AUPRC) | same → `average_precision` | same |
| Detection latency | `metrics.detection_latency_table` → `latency_seconds` | `latency.csv` |
| 95% Bootstrap CI | `aggregate_seeds.bootstrap_ci` | `multi_seed_summary.csv` |
| Paired t-test p-value | `aggregate_seeds.paired_ttest_vs_baseline` | `multi_seed_summary.csv` |

---

## 3. Models — Complete Inventory (11 detectors)

### 3.1 Group A — Statistical baselines (2)

#### 3.1.1 `threshold_baseline`
- **What it does:** Sums standardized residuals → flags if score > 99th percentile of validation benign scores.
- **Hyperparameters:** `threshold_percentile=0.99` (sweep over `{0.95, 0.99, 0.995}`)
- **File:** `phase1_models/train_threshold_baseline.py`
- **Why it's there:** Required reviewer baseline; non-parametric reference point.

#### 3.1.2 `isolation_forest`
- **What it does:** Random partitioning depth → anomaly score = $-\text{path length}$.
- **Hyperparameters:** `n_estimators=300`, `contamination=0.05`, `random_state=seed`
- **File:** `phase1_models/train_isolation_forest.py`
- **Library:** `sklearn.ensemble.IsolationForest`

### 3.2 Group B — Reconstruction baselines (3, was 1)

#### 3.2.1 `autoencoder` (existing)
- **Architecture:** MLP encoder–decoder $128 \rightarrow 32 \rightarrow 128$
- **Loss:** MSE; trained on benign-only.
- **Anomaly score:** Reconstruction MSE.

#### 3.2.2 `lstm_autoencoder` (NEW — see §6.1)
- **Architecture:** Encoder LSTM(64) → Linear(32) → Decoder LSTM(64) → Linear(input_dim)
- **Loss:** Sequence-level MSE.
- **Anomaly score:** Mean per-window reconstruction error.

#### 3.2.3 `vae` (NEW — see §6.2)
- **Architecture:** Encoder MLP → ($\mu, \log\sigma^2$) → reparameterize → Decoder MLP
- **Loss:** $\text{MSE}(\hat{x}, x) + \beta \cdot \text{KL}(q\|p)$ with $\beta=0.1$.
- **Anomaly score:** Reconstruction error + KL term.

### 3.3 Group C — Classical novelty detection (1, NEW)

#### 3.3.1 `ocsvm` (NEW — see §6.3)
- **Library:** `sklearn.svm.OneClassSVM`
- **Hyperparameters:** `kernel='rbf'`, `gamma='scale'`, `nu=0.05`
- **Anomaly score:** $-\text{decision\_function}(x)$

### 3.4 Group D — Sequence models (2, existing)

#### 3.4.1 `gru`
- **Architecture:** GRU(input → 64) → Linear(64 → 1) → Sigmoid
- **Sequence length:** 8 (sweep over `{4, 8, 12, 16}`)
- **Loss:** BCEWithLogitsLoss with `pos_weight = N_benign / N_anomaly`

#### 3.4.2 `lstm`
- Same as GRU but with LSTM layer.

### 3.5 Group E — Attention models (2, was 1)

#### 3.5.1 `transformer` (existing — `TinyTransformerClassifier`)
- **Architecture:** Linear(input → 64) → 2× TransformerEncoderLayer (4 heads, d_ff=128) → mean-pool → Linear(64 → 1)
- **Sequence length:** 8

#### 3.5.2 `anomaly_transformer` (NEW — see §6.4)
- **Architecture:** Reconstruction-style transformer with positional encoding; trained MSE on benign sequences.
- **Anomaly score:** Mean reconstruction error per window.
- **Reference:** Xu et al., "Anomaly Transformer" ICLR 2022.

### 3.6 Group F — Tokenized baseline (1, existing — bug fixed)

#### 3.6.1 `llm_baseline` (`TokenBaselineClassifier`)
- **What it does:** Discretizes residuals into 16 bins per feature, learns embeddings.
- **BUG FIX in this version:** Mean-pool is now sequence-only (was over both seq and feature axes — broken).
- **Architecture:** Embedding(16, 32) → reshape → Linear → mean-pool over time → Linear → 1

---

## 4. Hyperparameters — Single Source of Truth

`configs/model_config.yaml`:

```yaml
training:
  seeds: [1, 7, 42, 123, 1729]
  optimizer: adam
  learning_rate: 1.0e-3
  weight_decay: 1.0e-5
  batch_size: 64
  epochs: 24
  patience: 5
  min_delta: 1.0e-4

windowing:
  window_seconds: 60
  step_seconds: 60        # non-overlapping
  min_attack_overlap_fraction: 1.0  # full window must be attacked

split:
  train_fraction: 0.7
  val_fraction: 0.1
  test_fraction: 0.2
  buffer_windows: 2

threshold_calibration:
  default_percentile: 0.99
  sweep: [0.90, 0.95, 0.99, 0.995]

models:
  threshold_baseline:
    enabled: true

  isolation_forest:
    enabled: true
    n_estimators: 300
    contamination: 0.05

  ocsvm:
    enabled: true
    kernel: rbf
    gamma: scale
    nu: 0.05

  autoencoder:
    enabled: true
    hidden_dim: 128
    bottleneck_dim: 32

  vae:
    enabled: true
    hidden_dim: 128
    latent_dim: 32
    beta: 0.1

  lstm_autoencoder:
    enabled: true
    hidden_dim: 64
    latent_dim: 32
    num_layers: 1

  gru:
    enabled: true
    hidden_dim: 64
    seq_len: 8

  lstm:
    enabled: true
    hidden_dim: 64
    seq_len: 8

  transformer:
    enabled: true
    d_model: 64
    nhead: 4
    num_layers: 2
    seq_len: 8

  anomaly_transformer:
    enabled: true
    d_model: 64
    nhead: 4
    num_layers: 2
    seq_len: 8

  llm_baseline:
    enabled: true
    vocab_size: 16
    embed_dim: 32
    seq_len: 8

ablations:
  window_seconds_sweep: [30, 60, 120, 300]
  feature_count_sweep: [32, 64, 96, 128]
  seq_len_sweep: [4, 8, 12, 16]
  threshold_percentile_sweep: [0.90, 0.95, 0.99, 0.995]
```

---

## 5. Data Pipeline (Phase 1 → detector input)

### 5.1 Generation step

```bash
python -m phase1.build_clean_dataset --config configs/pipeline_config.yaml
```

Produces:
- `outputs/clean/truth_physical_timeseries.parquet` — full physics state, 1 Hz, single full-day cycle
- `outputs/clean/measured_physical_timeseries.parquet` — sensor-impaired (noise, missingness, latency, sparse coverage)
- `outputs/clean/cyber_events.parquet` — baseline cyber event stream
- `outputs/clean/input_environment_timeseries.parquet` — irradiance + temperature
- `outputs/clean/input_load_schedule.parquet`, `input_pv_schedule.parquet`, `input_bess_schedule.parquet`
- `outputs/clean/scenario_manifest.json` — run metadata + validation checks

### 5.2 IEEE 123-bus DER configuration

Configured in `configs/data_config.yaml`:

| Asset | Count | Rating | Locations |
|---|---|---|---|
| PV systems | 8 | 100–500 kW each | Buses 18, 35, 49, 65, 76, 86, 99, 113 |
| BESS | 4 | 250 kWh / 100 kW | Buses 47, 76, 86, 113 |
| Load | 85 | varies | All non-zero load buses |
| Regulators | 4 | — | Per IEEE 123 standard |
| Capacitors | 4 | — | Per IEEE 123 standard |

PV penetration: ~30% of peak load. Inverter mode: PQ + Volt-VAr (per IEEE 1547-2018).

Solar irradiance: synthetic clear-sky model with stochastic cloud cover (deterministic under seed).
Load profile: normalized residential + commercial composite (synthetic, scaled to match IEEE 123 nominal MW).

### 5.3 Window construction

```bash
python -m phase1.build_windows \
  --measured outputs/clean/measured_physical_timeseries.parquet \
  --cyber outputs/clean/cyber_events.parquet \
  --labels outputs/labeled/attack_labels_clean.csv \
  --output outputs/windows/merged_windows_clean.parquet \
  --window-seconds 60 --step-seconds 60 --min-attack-overlap-fraction 1.0
```

Each window contains, per signal:
- `<signal>__mean`, `<signal>__std`, `<signal>__min`, `<signal>__max`, `<signal>__last`

Plus cyber aggregates: `cyber_event_count_total`, `cyber_auth_count`, `cyber_command_count`, etc.
Plus labels: `attack_present`, `attack_family`, `attack_severity`, `attack_affected_assets`.

### 5.4 Residual construction

```bash
python -m phase1_models.run_full_evaluation --project-root . --feature-counts 128 --epochs 1 --patience 1
```

This step (executed once before any model trains) produces:

`outputs/reports/model_full_run_artifacts/residual_windows_full_run.parquet`

Each row has columns:
- `window_start_utc`, `window_end_utc`, `window_seconds`, `scenario_id`, `run_id`, `split_id`, `split_name`
- `attack_present`, `attack_family`, `attack_severity`, `attack_affected_assets`, `scenario_window_id`
- `delta__<signal>__<statistic>` for every numeric signal × statistic combination

Total residual columns: **920** (per the paper).
Total windows: **1,435** under the canonical 60s non-overlap configuration.

---

## 6. New Model Implementations

The four new models are in `phase1_models/models/`. Each follows the same contract:

```python
class NewModel(nn.Module):
    def __init__(self, ...): ...
    def forward(self, x): ...

# Plus a training entry-point in phase1_models/train_<name>.py:
def train_<name>_model(project_root, config, seed) -> dict: ...
```

See:
- `phase1_models/models/lstm_autoencoder.py`
- `phase1_models/models/vae.py`
- `phase1_models/models/ocsvm.py`
- `phase1_models/models/anomaly_transformer.py`

(Implementations are in this delivery package and ready to drop into your repo.)

---

## 7. Training Procedure (with auto-checkpointing)

Every training entry point uses `phase1_models/checkpointing.py`:

```python
from phase1_models.checkpointing import save_checkpoint, find_resume_checkpoint, CheckpointState

ckpt_dir = root / "outputs" / "models_full_run" / model_name / f"seed_{seed}" / "checkpoints"
resume_from = find_resume_checkpoint(ckpt_dir)
if resume_from:
    state = load_checkpoint(resume_from, model, optimizer)
    start_epoch = state.epoch + 1
else:
    start_epoch = 0

for epoch in range(start_epoch, total_epochs):
    train_loss = train_one_epoch(...)
    val_loss   = validate(...)
    is_best    = val_loss < best_val_loss
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        state=CheckpointState(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metric=-val_loss,  # higher = better
            best_metric=-best_val_loss,
            is_best=is_best,
            timestamp_utc=datetime.utcnow().isoformat(),
            seed=seed,
            model_name=model_name,
            config=config_dict,
        ),
        checkpoint_dir=ckpt_dir,
        keep_last_n=3,
    )
```

**Files written per (model, seed):**

```
outputs/models_full_run/<model_name>/seed_<n>/
├── checkpoints/
│   ├── epoch_0000.pt   (kept last 3)
│   ├── epoch_0001.pt
│   ├── ...
│   ├── latest.pt
│   ├── best.pt
│   ├── best_metadata.json
│   └── training_history.jsonl
├── results.json         (final metrics)
├── predictions.parquet
└── thresholds.json
```

Resume after crash:

```bash
python -m phase1_models.train_lstm --project-root . --resume --seed 42
```

---

## 8. Evaluation Protocol

### 8.1 Standard benchmark (single split, 5 seeds)

```bash
python -m phase1_models.run_hardened_phase1 \
  --config configs/phase1_runner_config.yaml \
  --mode benchmark \
  --seeds 1,7,42,123,1729
```

Writes:
- `outputs/phase1_hardened/reports/benchmark_per_seed.csv`
- `outputs/phase1_hardened/reports/benchmark_aggregated.csv` (mean ± std + 95% bootstrap CI)
- `outputs/phase1_hardened/reports/benchmark_significance.csv` (paired t-tests vs `threshold_baseline`)

### 8.2 True zero-day (leave-one-anomaly-family-out)

```bash
python -m phase1_models.zero_day_evaluation \
  --project-root . \
  --seeds 1,7,42,123,1729 \
  --epochs 24 \
  --patience 5
```

Loops over 6 held-out families × 5 seeds × 11 models = **330 training runs**.
Writes:
- `outputs/zero_day_runs/holdout_<family>/seed_<n>/<model>/results.json`
- `outputs/zero_day_runs/leave_one_family_out_summary.csv` (raw)
- `outputs/zero_day_runs/leave_one_family_out_aggregated.csv` (mean ± std per model × family)
- `outputs/zero_day_runs/zero_day_macro_summary.csv` (macro F1 across families)

### 8.3 Per-anomaly-family table

Aggregates from completed benchmark runs:

```bash
python -m phase1_models.per_family_metrics --project-root .
```

Writes: `outputs/phase1_hardened/reports/per_family_results.csv` and `.md`.

### 8.4 Ablations

```bash
python -m phase1_models.ablations \
  --project-root . \
  --sweeps window_size,feature_count,seq_len,threshold_percentile \
  --seeds 1,7,42
```

Writes:
- `outputs/phase1_hardened/reports/ablation_window_size.csv`
- `outputs/phase1_hardened/reports/ablation_feature_count.csv`
- `outputs/phase1_hardened/reports/ablation_seq_len.csv`
- `outputs/phase1_hardened/reports/ablation_threshold.csv`
- `outputs/phase1_hardened/reports/ablation_grid.png`

### 8.5 Paper numbers

After all of the above:

```bash
python -m phase1_models.export_paper_numbers --project-root .
```

Writes `outputs/phase1_hardened/reports/paper_numbers.md` — **the single source of truth** for what goes into the paper's Tables I–VI.

---

## 9. Reproducibility

Every run produces a manifest:

```bash
python -m phase1_models.reproducibility --project-root .
```

Writes `outputs/phase1_hardened/reports/MANIFEST.csv` containing for every artifact:
- relative path
- SHA-256 hash
- byte size
- modified timestamp
- git commit hash at write time

Plus `environment.json` with `pip freeze`, Python version, torch version, sklearn version, OS, CPU/GPU info.

Verify reproducibility:

```bash
python -m phase1_models.reproducibility --project-root . --verify
# Should print: "✓ All hashes match, all files present"
```

---

## 10. Quick Start (smoke test, ~5 minutes)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate clean data (~2 min)
python -m phase1.build_clean_dataset

# 3. Train ONE model with ONE seed for ONE epoch (smoke test)
python -m phase1_models.train_lstm \
  --project-root . \
  --epochs 1 \
  --seed 1 \
  --feature-counts 32

# 4. Verify outputs
ls outputs/models_full_run/lstm/seed_1/checkpoints/
# Expected: best.pt latest.pt epoch_0000.pt training_history.jsonl best_metadata.json
```

---

## 11. Full Run (paper-ready, ~12-24 hours on RTX 4090)

```bash
# 1. Generate clean data
python -m phase1.build_clean_dataset

# 2. Generate Phase 2 attack scenarios (separate phase)
python -m phase2.run_phase2 --config configs/pipeline_config.yaml

# 3. Build residuals
python -m phase1_models.run_full_evaluation \
  --project-root . \
  --feature-counts 32,64,96,128 \
  --seq-lens 4,8,12,16 \
  --epochs 24 \
  --patience 5

# 4. Multi-seed benchmark (all 11 models × 5 seeds)
python -m phase1_models.run_hardened_phase1 \
  --config configs/phase1_runner_config.yaml \
  --mode benchmark \
  --seeds 1,7,42,123,1729

# 5. Zero-day leave-one-family-out (6 holdouts × 5 seeds × 11 models)
python -m phase1_models.zero_day_evaluation \
  --project-root . \
  --seeds 1,7,42,123,1729

# 6. Ablations
python -m phase1_models.ablations --project-root . --seeds 1,7,42

# 7. Generate paper numbers
python -m phase1_models.export_paper_numbers --project-root .

# 8. Reproducibility manifest
python -m phase1_models.reproducibility --project-root .
```

After this, `outputs/phase1_hardened/reports/paper_numbers.md` contains every number that should appear in the paper.

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: residual_windows_full_run.parquet` | Phase 1+2 data not generated | Run `build_clean_dataset` and `phase2.run_phase2` first |
| OOM on RTX 4090 | Batch size too large | Reduce `batch_size` in `model_config.yaml` to 32 |
| Training plateaus at epoch 5 | Patience too aggressive | Increase `patience` to 10 |
| Different F1 between runs | Random seed not set | Use `--seed N` flag; verify `torch.backends.cudnn.deterministic=True` |
| `paper_numbers.md` empty | Some runs failed silently | Check `outputs/phase1_hardened/reports/run_log.txt` |
| `lstm_autoencoder` performs worse than `autoencoder` | Sequence too short | Increase `seq_len` to 16 |
| Low GPU utilization | DataLoader bottleneck | Set `num_workers=4` in DataLoader |

---

## 13. What's Different From The Original Phase 1

This is the consolidated change list relative to the version in your zip:

| Change | Reason |
|---|---|
| Added 4 new models (LSTM-AE, VAE, OCSVM, AnomalyTransformer) | Reviewer baselines |
| Per-epoch checkpointing with resume | Robustness to crashes |
| Fixed `TokenBaselineClassifier` pooling | Was destroying feature distinctness |
| Standardized split to 70/10/20 chronological | Was 3 different ratios across files |
| Window default changed to 60s non-overlap | Was 300s overlapping in CLI |
| Added `detector_emission_utc` to predictions | Honest latency measurement |
| Multi-seed runner with bootstrap CI + paired t-tests | No prior statistical rigor |
| Leave-one-family-out zero-day protocol | Was using same family in train+test |
| Per-anomaly-family results table | Paper claims 6 families but reported only aggregate |
| Ablations: window, feature count, seq_len, threshold | Paper claims sweeps but reported no results |
| `paper_numbers.md` as single source of truth | Paper numbers didn't match CSVs |
| Reproducibility manifest with hashes | No prior reproducibility tracking |
| Honest docstrings (no templated boilerplate) | Looked LLM-generated |

---

## 14. Citations to Add to the Paper (when adding these models)

```bibtex
@inproceedings{xu2022anomalytransformer,
  title={Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy},
  author={Xu, Jiehui and Wu, Haixu and Wang, Jianmin and Long, Mingsheng},
  booktitle={ICLR},
  year={2022}
}

@article{kingma2014vae,
  title={Auto-Encoding Variational Bayes},
  author={Kingma, Diederik P and Welling, Max},
  journal={ICLR},
  year={2014}
}

@inproceedings{malhotra2016lstmae,
  title={LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection},
  author={Malhotra, Pankaj and Ramakrishnan, Anusha and Anand, Gaurangi and Vig, Lovekesh and Agarwal, Puneet and Shroff, Gautam},
  booktitle={ICML Workshop on Anomaly Detection},
  year={2016}
}

@article{scholkopf2001ocsvm,
  title={Estimating the support of a high-dimensional distribution},
  author={Sch{\"o}lkopf, Bernhard and Platt, John C and Shawe-Taylor, John and Smola, Alex J and Williamson, Robert C},
  journal={Neural Computation},
  volume={13},
  number={7},
  pages={1443--1471},
  year={2001}
}
```

---

## 15. Methodology Diagram — Required Updates

The current `derguardian_methodology.png` has these issues that **must be fixed before submission**:

1. **Typo:** "Secnario" → "Scenario" (appears twice in right-side blue panel)
2. **Truncated text:** "Performance on Unseen Disturba-" → "Performance on Unseen Disturbances"
3. **Missing block:** "Multi-Model Benchmark (11 detectors)" — currently the diagram only shows generic "Deviation-Based Anomaly Detection"
4. **Missing block:** "Multi-Seed Statistical Analysis (5 seeds, 95% CI, paired t-tests)"
5. **Missing block:** "Per-Family Performance Breakdown"
6. **Missing block:** "Ablation Studies (window, features, seq_len, threshold)"
7. **Missing block:** "Reproducibility Manifest (SHA-256 hashes, git commit)"
8. The arrow from "Learn Normal System Behavior" to "Evaluate Detection Under Zero-Day" should be labeled "Leave-One-Family-Out Protocol"

A replacement SVG diagram is provided at `figures/derguardian_methodology_v2.svg`. Convert to PNG at 300 DPI for paper inclusion:

```bash
inkscape figures/derguardian_methodology_v2.svg --export-type=png --export-dpi=300 \
  -o figures/derguardian_methodology_v2.png
```

---

## 16. Acceptance Criteria (before paper submission)

- [ ] `paper_numbers.md` exists and is non-empty
- [ ] All 11 models trained successfully on at least 3 seeds
- [ ] Leave-one-family-out covers all 6 families
- [ ] Per-family table is filled in for all 11 models × 6 families
- [ ] At least 4 ablation tables exist
- [ ] `MANIFEST.csv` lists every paper-cited artifact with SHA-256
- [ ] `pytest tests/test_phase1_hardened.py` passes
- [ ] No file in `phase1_models/` contains the templated docstring `"Handle X within the Phase 1 detector modeling workflow."`
- [ ] Methodology diagram updated (no typos, no truncation, all blocks present)
- [ ] All paper Tables/Figures cite a row in `paper_numbers.md`

When all 10 boxes are checked, Phase 1 is **defensible**.

---

## 17. Contact / Authorship

Replace this section with author affiliations and contact info before submission.

---

*Last updated: April 2026. This README is the source of truth for Phase 1. If documentation in code comments contradicts this file, the README wins.*
