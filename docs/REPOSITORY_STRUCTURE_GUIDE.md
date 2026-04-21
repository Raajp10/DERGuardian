# Repository Structure Guide

This guide explains where the DERGuardian code, evidence artifacts, reports, and local/generated files live. It is written for a new reader who wants to understand the project without accidentally mixing canonical benchmark results with replay, heldout synthetic, or extension evidence.

## Read This First

1. `README.md`
2. `docs/methodology/METHODOLOGY_OVERVIEW.md`
3. `docs/reports/CANONICAL_VS_REPLAY_VS_ZERO_DAY.md`
4. `docs/REPOSITORY_STRUCTURE_GUIDE.md`
5. `docs/PYTHON_CODE_NAVIGATION.md`
6. `scripts/run_full_pipeline.py`

## Top-Level Files

- `README.md`: GitHub front page with project summary, quickstart, limitations, and result-reading path.
- `LICENSE`: MIT license for the public release.
- `.gitignore`: Keeps large generated outputs, caches, checkpoints, and local-only artifacts out of Git.
- `requirements.txt` and `environment.yml`: Python dependency entry points.
- `RELEASE_CHECKLIST.md`: Human release checklist.
- `FINAL_PUBLISHABLE_REPO_DECISION.md`: Final release decision and remaining manual judgment.
- `GITHUB_READY_SUMMARY.md`: Short GitHub/professor-facing summary.

## Source Code

- `common/`: Shared utilities for configuration, OpenDSS execution, metadata, IO, timing, units, weather/load generation, and sensor-noise simulation.
- `phase1/`: Phase 1 clean-data generation, channel extraction, window building, and clean-data validation.
- `phase1_models/`: Detector training, residual feature construction, model evaluation, ready-package loading, calibration, metrics, and report generation.
- `phase1_models/context/`: Context/fusion helpers used around normal-behavior summaries and explanation-oriented reasoning artifacts. These are support layers, not the canonical detector-selection mechanism.
- `phase2/`: Scenario schema validation, attack compilation, cyber log generation, attacked dataset generation, merge logic, and Phase 2 reporting.
- `phase3/`: Evaluation, ablation, latency, zero-day-like split/report helpers, and final Phase 3 artifacts.
- `phase3_explanations/`: Grounded explanation packet generation, family classification hints, rationale validation, schemas, prompts, and examples.
- `deployment_runtime/`: Offline runtime-demonstration code for alert packets, local buffering, stream windows, model loading, and reports. This supports offline deployment benchmarking only and is not field edge-deployment evidence.
- `scripts/`: Orchestration, audit, report-building, extension, and release verification scripts.
- `tests/`: Contract and smoke tests for the phase pipeline and ready packages.

## Configs

- `configs/pipeline_config.yaml`: Release-facing pipeline settings, including default paths and phase execution choices.
- `configs/model_config.yaml`: Detector and extension model settings, including window/model lists and benchmark-vs-extension labels.
- `configs/data_config.yaml`: Data/source-path conventions, sample-data policy, and output separation.

## Data

- `data/sample_or_manifest_only/`: Tiny sample/manifest-only data suitable for GitHub. The included parquet sample is for structure inspection and smoke testing, not full scientific reproduction.
- `outputs/`: Full generated local artifacts. This folder is ignored by Git because it can contain large datasets and frozen benchmark outputs. Canonical evidence remains local/source-of-truth there unless mirrored into `artifacts/`.

## Artifacts

The `artifacts/` folder contains lightweight evidence mirrors organized by evaluation context:

- `artifacts/benchmark/`: Canonical benchmark mirrors. The frozen benchmark winner remains `transformer @ 60s`.
- `artifacts/replay/`: Heldout replay evidence.
- `artifacts/zero_day_like/`: Heldout synthetic zero-day-like evaluation evidence. This is not real-world zero-day proof.
- `artifacts/extensions/`: Extension evidence such as TTM and LoRA. TTM stays extension-only; LoRA stays experimental/weak.
- `artifacts/xai/`: Grounded XAI audit evidence.
- `artifacts/deployment/`: Offline deployment benchmark evidence.

## Documentation

- `docs/methodology/`: Diagram-alignment, methodology, slide text, and safe box labels.
- `docs/reports/`: Publication-facing reports that should be read before the deeper audit archive.
- `docs/reports/internal_audits/`: Detailed verification/audit trail. This folder is intentionally verbose; it is for traceability, not the first reading path.
- `docs/figures/`: Selected publication-facing figures.

## Canonical vs Extension

- Canonical benchmark path: `phase1/`, `phase1_models/`, `outputs/window_size_study/final_window_comparison.csv`, and `artifacts/benchmark/`.
- Heldout replay path: replay reports and mirrors under `artifacts/replay/` and final comparison docs.
- Heldout synthetic zero-day-like path: Phase 2 generated bundles plus `artifacts/zero_day_like/`.
- Extension branches: TTM and LoRA reports under `artifacts/extensions/` and `docs/reports/EXTENSION_BRANCHES.md`.

## Where Each Phase Starts

- Phase 1 starts at `phase1/build_clean_dataset.py`, then `phase1/build_windows.py`, then detector workflows in `phase1_models/` and `scripts/run_window_size_study.py`.
- Phase 2 starts with `phase2/research_attack_scenarios.json`, `phase2/scenario_schema.json`, and `phase2/validate_scenarios.py`, then proceeds through `phase2/generate_attacked_dataset.py`.
- Phase 3 starts with frozen detector packages and evaluation scripts such as `scripts/run_detector_window_zero_day_audit.py`, `phase3/`, `phase3_explanations/`, and `scripts/run_deployment_benchmark.py`.

## Local/Generated Folders

These folders may exist locally but should be treated carefully:

- `outputs/`: Full generated experiment artifacts; ignored by Git.
- `phase2_llm_benchmark/`: Large local scenario-generation/runtime data; ignored by Git.
- `reports/`, `paper_figures/`, `demo/`, `DERGuardian_professor_demo/`: Legacy or generated presentation/demo outputs; ignored by Git unless mirrored into `docs/` or `artifacts/`.

## New Contributor Starting Point

Start with `README.md`, then read this guide and `docs/PYTHON_CODE_NAVIGATION.md`. To understand scientific claims, read `docs/reports/CANONICAL_VS_REPLAY_VS_ZERO_DAY.md` and `docs/reports/PUBLICATION_SAFE_CLAIMS.md` before opening detailed internal audits.

