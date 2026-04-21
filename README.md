# DERGuardian

DERGuardian is a research pipeline for cyber-physical anomaly detection in distributed energy resource (DER) systems. It uses an IEEE 123-bus OpenDSS feeder with PV/BESS assets, schema-bound synthetic attack scenarios, residual/deviation features, detector benchmarking, heldout synthetic evaluation, explanation validation, and offline deployment benchmarking.

## Research Question

Can a simulation-grounded DER pipeline generate validated cyber-physical attack scenarios and evaluate anomaly detectors in a way that separates benchmark model selection from heldout synthetic transfer evidence?

## What DERGuardian Does

- Generates and preprocesses clean DER simulation/telemetry windows.
- Builds residual/deviation detector inputs without changing the frozen canonical benchmark outputs.
- Generates, validates, repairs, and audits synthetic cyber-physical attack scenarios.
- Evaluates detectors under four separated contexts: canonical benchmark, heldout replay, heldout synthetic zero-day-like evaluation, and extension branches.
- Adds conservative explanation and offline deployment evidence without claiming human-level reasoning or field deployment.

## Three-Phase Methodology

1. **Phase 1: Normal System Learning and Detector Benchmarking**
   - Builds clean DER simulation and measured telemetry.
   - Creates windowed residual/deviation features.
   - Benchmarks threshold, Isolation Forest, MLP autoencoder, GRU, LSTM, and Transformer detectors across 5s, 10s, 60s, and 300s windows.
   - Frozen canonical benchmark winner: **Transformer @ 60s** with benchmark F1 `0.8182`.

2. **Phase 2: Scenario and Attack Generation**
   - Uses schema-bound synthetic scenario JSON.
   - Validates scenario physics, safety, metadata, diversity, and difficulty.
   - Preserves accepted, rejected, and repaired scenario records.
   - Current master inventory rows: `99`.

3. **Phase 3: Detection and Intelligence Layer**
   - Keeps evaluation contexts separate: canonical benchmark, heldout replay, heldout synthetic zero-day-like evaluation, and extension experiments.
   - Reports XAI as grounded post-alert operator support, not human-like root-cause analysis.
   - Reports deployment as an offline lightweight benchmark, not field edge deployment.

## What "Zero-Day-Like" Means Here

Zero-day-like means heldout synthetic scenario evaluation on independently generated Phase 2 attack bundles. It does **not** mean real-world zero-day robustness. lstm @ 300s had the strongest mean heldout synthetic F1 (0.8099), reported separately from canonical model selection.

## Extension Branches

- **TTM**: real TinyTimeMixer extension benchmark at 60s only; extension-only, not canonical.
- **LoRA**: experimental tiny-LLM explanation/classification branch; weak evidence and not detector benchmark evidence.

## Repository Structure

- `README.md`, `LICENSE`, `.gitignore`, `requirements.txt`, `environment.yml` - root release surface.
- `docs/methodology/` - methodology and diagram-alignment docs.
- `docs/reports/` - publication-facing reports.
- `docs/reports/internal_audits/` - detailed audit trail, legacy reports, and final verification records.
- `docs/REPOSITORY_STRUCTURE_GUIDE.md` - guided map of folders, artifacts, and canonical-vs-extension boundaries.
- `docs/PYTHON_CODE_NAVIGATION.md` - developer-oriented map of the Python codebase.
- `docs/figures/` - selected final figures.
- `artifacts/` - lightweight mirrored evidence artifacts by context.
- `configs/` - release-facing pipeline, model, and data configs.
- `data/sample_or_manifest_only/` - tiny sample/manifest-only data for GitHub.
- `scripts/` - reproducibility and audit scripts.
- `outputs/` - local generated artifacts; large and usually not committed.

## Quickstart

```bash
python -m pip install -r requirements.txt
python scripts/run_full_pipeline.py
```

The full-pipeline entrypoint skips stages whose completion markers already exist, so it is safe as a release-facing reproduction command. To preview without executing, run:

```bash
python scripts/run_full_pipeline.py --dry-run
```

## Detailed Run Guide

### 1. Clone the Repository

```bash
git clone https://github.com/Raajp10/DERGuardian.git
cd DERGuardian
```

### 2. Create an Environment

Using `venv`:

```bash
python -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or use the environment file:

```bash
conda env create -f environment.yml
conda activate derguardian
```

### 3. Confirm the Pipeline Plan

```bash
python scripts/run_full_pipeline.py --dry-run
```

This prints each phase command without rerunning expensive stages.

### 4. Run the Pipeline

```bash
python scripts/run_full_pipeline.py
```

The orchestrator runs or verifies:

- Phase 1 clean data generation and detector benchmark preparation.
- Phase 2 scenario validation and attacked dataset generation.
- Phase 3 detector-context evaluation, XAI reporting, and offline deployment benchmark reporting.
- Final publication-layout verification.

The script is conservative by default: existing completion markers are respected so frozen canonical outputs are not overwritten during normal use.

### 5. Read the Main Results

Start with these files:

- `docs/reports/PROFESSOR_READY_METHOD_SUMMARY.md`
- `docs/reports/CANONICAL_VS_REPLAY_VS_ZERO_DAY.md`
- `docs/reports/phase1_window_model_comparison_full.md`
- `docs/reports/zero_day_model_window_results_full.md`
- `docs/reports/EXTENSION_BRANCHES.md`
- `docs/reports/PUBLICATION_SAFE_CLAIMS.md`
- `FINAL_PUBLISHABLE_REPO_DECISION.md`

### 6. Verify Release Readiness

```bash
python scripts/final_perfection_verify.py
```

The verification outputs are written to:

- `docs/reports/internal_audits/FINAL_PERFECTION_CHECKLIST.md`
- `docs/reports/internal_audits/FINAL_PERFECTION_STATUS.csv`

## Data and Artifact Policy

The public repository is designed to stay lightweight. Large generated outputs, full local datasets, runtime folders, and model checkpoint files are ignored by `.gitignore`. Small evidence mirrors and sample data are committed under:

- `artifacts/`
- `data/sample_or_manifest_only/`

The tiny sample parquet is only for structure inspection and smoke testing. It is not a replacement for the full generated experiment outputs.

## GitHub Push Workflow

For maintainers with write access:

```bash
git add .
git commit -m "Final DERGuardian public release (PhD-level clean)"
git remote add origin https://github.com/Raajp10/DERGuardian.git
git branch -M main
git push -u origin main
```

After the first push:

```bash
git add .
git commit -m "update"
git push
```

View results in:

- `docs/reports/CANONICAL_VS_REPLAY_VS_ZERO_DAY.md`
- `docs/reports/phase1_window_model_comparison_full.md`
- `docs/reports/zero_day_model_window_results_full.md`
- `docs/reports/EXTENSION_BRANCHES.md`
- `FINAL_PUBLISHABLE_REPO_DECISION.md`

For a full professor-facing evidence pass, start with:

- `GITHUB_READY_SUMMARY.md`
- `docs/reports/PROFESSOR_READY_METHOD_SUMMARY.md`
- `docs/REPOSITORY_STRUCTURE_GUIDE.md`
- `docs/PYTHON_CODE_NAVIGATION.md`
- `docs/reports/CANONICAL_VS_REPLAY_VS_ZERO_DAY.md`
- `docs/reports/internal_audits/FINAL_PERFECTION_CHECKLIST.md`

## Reproducibility Notes

- Original canonical artifacts are preserved in `outputs/window_size_study/`.
- Publication cleanup mirrors selected outputs; it does not move or rewrite canonical benchmark artifacts.
- Re-running detector-side audits may emit a non-blocking scikit-learn persistence warning for saved IsolationForest artifacts created under a different scikit-learn version.

## Limitations and Non-Claims

- No real-world zero-day robustness claim.
- No human-like root-cause analysis claim.
- No AOI detector metric claim.
- No field edge-deployment claim.
- No claim that TTM or LoRA replaces the canonical detector.
- The repo autoencoder is an MLP autoencoder; no LSTM autoencoder detector is implemented.

## Citation / Acknowledgement

This repository is a research prototype for DER cyber-physical anomaly detection. If cited before formal publication, cite the repository name, author/project team, and the specific commit or release used for evaluation.
