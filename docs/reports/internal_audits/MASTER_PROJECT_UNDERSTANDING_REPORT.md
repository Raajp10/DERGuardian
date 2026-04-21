# Master Project Understanding Report

## A. Project Overview

DERGuardian is a research pipeline for cyber-physical anomaly detection in distributed energy resource (DER) systems. It combines an IEEE 123-bus OpenDSS simulation, DER assets, clean and attacked telemetry generation, residual/deviation features, detector benchmarking, structured synthetic attack scenarios, heldout replay, heldout synthetic zero-day-like evaluation, grounded explanation validation, and offline deployment benchmarking.

The problem addressed is not merely "can a detector flag anomalies." The project asks whether a simulation-grounded DER workflow can create scientifically auditable attack scenarios and evaluate detectors while keeping model selection, transfer/replay checks, zero-day-like synthetic evidence, and extension branches separate.

The research goal is to preserve a frozen canonical benchmark while adding honest evidence layers around scenario diversity, detector transfer, explanation support, and deployment feasibility. The contribution is a three-phase methodology with explicit artifact lineage and conservative publication claims.

## B. Canonical Evaluation Contexts

The repository uses four separated evaluation contexts:

- **Canonical benchmark**: frozen Phase 1 model-selection path. The source of truth is `outputs/window_size_study/final_window_comparison.csv`. The canonical winner remains `transformer @ 60s` with F1 `0.8182`, precision `0.6923`, recall `1.0000`, and mean detection latency `641.0` seconds.
- **Heldout replay**: frozen model packages replayed on heldout/repaired bundles. This evaluates transfer behavior and must not be relabeled as canonical benchmark selection.
- **Heldout synthetic zero-day-like evaluation**: generated Phase 2 attack bundles evaluated with frozen detector packages. It is bounded synthetic evidence only. lstm @ 300s had the strongest mean heldout synthetic F1 (0.8099) in that separate context.
- **Extension branches**: TTM detector extension, LoRA explanation/classification extension, XAI validation, and offline deployment benchmark. These are secondary evidence layers, not replacements for the canonical benchmark.

The normalized context index is `REPO_CONTEXT_INDEX.csv`; the integrated comparison is `FINAL_MODEL_CONTEXT_COMPARISON.csv` and its Git-friendly mirror `artifacts/zero_day_like/FINAL_MODEL_CONTEXT_COMPARISON.csv`.

## C. Phase 1 In Detail

### Purpose

Phase 1 creates clean DER behavior, transforms it into model-ready windows, trains detector candidates, evaluates them on the canonical benchmark split, and exports frozen ready packages. The canonical model-selection answer is fixed: `transformer @ 60s`.

### Data Generation Logic

The OpenDSS feeder and DER files are under:

- `opendss/Research_IEEE123_Master.dss`
- `opendss/DER_Assets.dss`
- `opendss/DER_Controls.dss`
- `opendss/Monitor_Config.dss`
- `opendss/ieee123_base/IEEE123Master.dss`
- `opendss/validate_ieee123_topology.py`

Shared simulation and telemetry infrastructure is in `common/dss_runner.py`, `common/config.py`, `common/weather_load_generators.py`, `common/noise_models.py`, `common/time_alignment.py`, `common/units_and_channel_dictionary.py`, and related `common/` utilities.

### OpenDSS / IEEE 123 Usage

The project uses the IEEE 123-bus feeder as the base network and overlays DER assets/control definitions. The exact asset definitions should be read from `opendss/DER_Assets.dss`; scenario/evaluation files mention assets such as `pv60`, `pv83`, `bess48`, and `bess108`, but the DSS file is the source of truth for placement/configuration.

### QSTS Role

The clean data path uses time-series OpenDSS simulation through the project runner infrastructure. The Phase 1 file map identifies `phase1/build_clean_dataset.py` as the canonical clean-data generation entry point and `common/dss_runner.py` as the OpenDSS execution support layer.

### Input Schedules And Environmental Drivers

Observed clean outputs include:

- `outputs/clean/input_environment_timeseries.parquet`
- `outputs/clean/input_environment_coarse.parquet`
- `outputs/clean/input_load_schedule.parquet`
- `outputs/clean/input_pv_schedule.parquet`
- `outputs/clean/input_bess_schedule.parquet`
- `outputs/clean/control_schedule.parquet`
- `outputs/clean/load_class_map.csv`
- `outputs/clean/measurement_impairments.csv`

The Phase 1 lineage audit records that environmental variables are simulated and present in window artifacts, but the selected canonical transformer feature subset uses zero environment residual columns.

### Preprocessing, Residuals, And Windowing

Important code:

- `phase1/build_clean_dataset.py`
- `phase1/validate_clean_dataset.py`
- `phase1/build_windows.py`
- `phase1_models/residual_dataset.py`
- `phase1_models/feature_builder.py`
- `phase1_models/run_full_evaluation.py`
- `phase1_model_loader.py`

Window contracts from the lineage audit:

- `5s`: 5 second window, 1 second step
- `10s`: 10 second window, 2 second step
- `60s`: 60 second window, 12 second step
- `300s`: 300 second window, 60 second step

Clean/attacked separation:

- Clean merged windows: `outputs/windows/merged_windows_clean.parquet`
- Attacked merged windows: `outputs/windows/merged_windows_attacked.parquet`
- Window-size-study residuals: `outputs/window_size_study/<window>/residual_windows.parquet`

Residual features are aligned clean-versus-attacked differences (`delta__*`). The canonical transformer uses aligned residual inputs with sequence length 12 and 64 selected features.

### Model Training Setup

Phase 1 trains/evaluates threshold baseline, Isolation Forest, MLP autoencoder, GRU, LSTM, and Transformer candidates. ARIMA is not implemented. The requested `lstm_ae` is not implemented; the repository's autoencoder is MLP-based.

Canonical benchmark outputs:

- `outputs/window_size_study/final_window_comparison.csv`
- `phase1_window_model_comparison_full.csv`
- `artifacts/benchmark/phase1_window_model_comparison_full.csv`
- `docs/reports/phase1_window_model_comparison_full.md`
- `docs/figures/phase1_window_comparison_5s_10s_60s_300s.png`
- `docs/figures/phase1_model_vs_window_heatmap.png`

The frozen ready package is `outputs/window_size_study/60s/ready_packages/transformer`; it is source-of-truth but should remain ignored or published separately as a release artifact because generated model packages are large/local artifacts.

## D. Phase 2 In Detail

### Scenario Authoring Logic

Phase 2 turns structured attack scenario definitions into injected physical/measurement changes, cyber logs, labels, manifests, and attacked windows.

Core scenario/config files:

- `phase2/research_attack_scenarios.json`
- `phase2/scenario_schema.json`
- `phase2/example_scenarios.json`
- `phase2/example_scenario_bundle/01_pv60_bias.json`
- `phase2/example_scenario_bundle/02_bess48_unauthorized_command.json`
- `phase2/example_scenario_bundle/03_bess108_command_suppression.json`

The canonical scenario bank to cite is `phase2/research_attack_scenarios.json`; example bundles are illustrative.

### LLM-Assisted Structured Generation

LLM-assisted scenario generation is treated as structured scenario authoring and heldout-generator evidence, not as proof of real-world zero-day robustness. The final inventory is `phase2_scenario_master_inventory.csv`, with `99` scenario rows, `88` accepted rows, `11` rejected rows, and `11` repaired rows. Generator sources include: `canonical_bundle, chatgpt, claude, gemini, grok, human_authored`.

### Schema Validation And Compile/Injection Logic

Important code:

- `phase2/validate_scenarios.py`
- `phase2/contracts.py`
- `phase2/compile_injections.py`
- `phase2/generate_attacked_dataset.py`
- `phase2/validate_attacked_dataset.py`
- `phase2/cyber_log_generator.py`
- `phase2/merge_physical_and_cyber.py`
- `phase2/reporting.py`

Physical and measured-layer actions are separated in generated artifacts:

- `outputs/attacked/compiled_physical_actions.json`
- `outputs/attacked/compiled_measurement_actions.json`
- `outputs/attacked/compiled_overrides.parquet`
- `outputs/attacked/compiled_manifest.json`

### Cyber Logs, Labels, And Attacked Outputs

Generated attacked outputs include:

- `outputs/attacked/truth_physical_timeseries.parquet`
- `outputs/attacked/measured_physical_timeseries.parquet`
- `outputs/attacked/cyber_events.parquet`
- `outputs/attacked/cyber_events.jsonl`
- `outputs/attacked/attack_labels.parquet`
- `outputs/attacked/scenario_manifest.json`
- `outputs/windows/merged_windows_attacked.parquet`

The Git-friendly final evidence mirrors are:

- `phase2_scenario_master_inventory.csv`
- `phase2_attack_family_distribution.csv`
- `phase2_asset_signal_coverage.csv`
- `phase2_difficulty_calibration.csv`
- `phase2_coverage_summary.md`
- `phase2_diversity_and_quality_report.md`
- `phase2_family_distribution.png`
- `phase2_asset_coverage_heatmap.png`
- `phase2_signal_coverage_heatmap.png`
- `phase2_difficulty_distribution.png`
- `phase2_generator_coverage_comparison.png`

Those Phase 2 coverage files remain root-level generated evidence and are currently ignored by `.gitignore`; publish the mirrored/selected docs or attach them as release artifacts if needed.

### Why This Is Zero-Day-Like, Not Real-World Zero-Day Proof

The heldout bundles are unseen synthetic generated scenarios. They test transfer to heldout generator/bundle distributions under the project simulator. They do not prove real-world zero-day robustness, field performance, or adversarial completeness.

## E. Phase 3 In Detail

### Training/Inference Separation

Canonical detector training and selection happens in Phase 1. Phase 3 reuses frozen detector packages for replay, heldout synthetic evaluation, XAI, and offline deployment. This separation prevents replay or heldout results from replacing the canonical model-selection result.

### Anomaly Scoring And Thresholding

Frozen ready packages include model scores and thresholds. Downstream code loads these packages through `phase1_model_loader.py` and audit/evaluation scripts such as `scripts/run_detector_window_zero_day_audit.py` and `scripts/run_heldout_phase3_pipeline.py`.

### Replay Evaluation

Heldout replay outputs include:

- `benchmark_vs_replay_metrics.csv`
- `multi_model_heldout_metrics.csv`
- `artifacts/replay/benchmark_vs_replay_metrics.csv`
- `artifacts/replay/multi_model_heldout_metrics.csv`
- `docs/reports/CANONICAL_VS_REPLAY_VS_ZERO_DAY.md`

Replay is frozen-package transfer evidence, not canonical model selection.

### Heldout Synthetic Evaluation

Heldout synthetic zero-day-like outputs include:

- `zero_day_model_window_results_full.csv`
- `zero_day_best_per_window_summary.csv`
- `zero_day_family_model_window_results.csv`
- `zero_day_family_difficulty_summary.md`
- `artifacts/zero_day_like/zero_day_model_window_results_full.csv`
- `docs/reports/zero_day_model_window_results_full.md`
- `docs/figures/zero_day_model_window_heatmap.png`

The 5s heldout synthetic sweep is explicitly blocked in `zero_day_model_window_results_full.csv` because reusable 5s replay residuals were unavailable and raw full-day 5s generation exceeded the CPU-only completion-pass budget.

### Explanation/XAI Layer

Important XAI files:

- `phase3_explanations/build_explanation_packet.py`
- `phase3_explanations/classify_attack_family.py`
- `phase3_explanations/generate_explanations_llm.py`
- `phase3_explanations/validate_explanations.py`
- `phase3_explanations/rationale_evaluator.py`
- `phase3_explanations/explanation_schema.json`
- `xai_case_level_audit.csv`
- `xai_error_taxonomy.md`
- `xai_qualitative_examples.md`
- `xai_final_validation_report.md`
- `artifacts/xai/xai_case_level_audit.csv`
- `docs/reports/xai_final_validation_report.md`

XAI summary: `155` audited cases, mean exact family match `0.7742`, mean asset accuracy `0.2677`, mean evidence-grounding overlap `0.2576`, and unsupported-claim count `8`. This supports grounded operator-facing assistance, not human-like root-cause analysis.

### LoRA / Tiny-LLM Extension

Important files:

- `scripts/run_phase3_lora_extension.py`
- `phase3_lora_dataset_manifest.json`
- `phase3_lora_training_config.yaml`
- `phase3_lora_results.csv`
- `phase3_lora_eval_report.md`
- `phase3_lora_model_card.md`
- `docs/reports/phase3_lora_eval_report.md`
- `artifacts/extensions/phase3_lora_results.csv`

LoRA is experimental/weak: on `test_generator_heldout`, family accuracy is `0.2340`, asset accuracy is `0.0000`, and grounding quality is `0.0000`.

### TTM Extension

Important files:

- `scripts/run_phase1_ttm_extension.py`
- `phase1_ttm_extension_config.yaml`
- `phase1_ttm_results.csv`
- `phase1_ttm_vs_all_models.csv`
- `phase1_ttm_window_comparison.csv`
- `phase1_ttm_eval_report.md`
- `docs/reports/phase1_ttm_eval_report.md`
- `artifacts/extensions/phase1_ttm_results.csv`

TTM extension is present at `60s` with F1 `0.7222` and remains extension-only.

### Deployment Benchmark

Important files:

- `scripts/run_deployment_benchmark.py`
- `deployment_benchmark_results.csv`
- `deployment_benchmark_report.md`
- `deployment_environment_manifest.md`
- `deployment_readiness_summary.md`
- `deployment_repro_command.sh`
- `artifacts/deployment/deployment_benchmark_results.csv`
- `docs/reports/deployment_benchmark_report.md`

Offline deployment benchmark contains `6` rows across profiles/models; it does not use real edge hardware.

This is an offline workstation/lightweight benchmark. It is not field deployment and not edge-hardware validation.

### Context Comparison Table

| context_name                             | context_family   | model_name         | window_label   | canonical_or_extension   |   precision |   recall |     f1 | notes                                                                |
|:-----------------------------------------|:-----------------|:-------------------|:---------------|:-------------------------|------------:|---------:|-------:|:---------------------------------------------------------------------|
| canonical_benchmark_test_split           | benchmark        | transformer        | 5s             | canonical                |      0.1013 |   0.9994 | 0.1839 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | gru                | 5s             | canonical                |      0.1013 |   0.9994 | 0.1839 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | autoencoder        | 5s             | canonical                |      0.0863 |   0.8194 | 0.1561 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | isolation_forest   | 5s             | canonical                |      0.1208 |   0.9508 | 0.2144 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | threshold_baseline | 5s             | canonical                |      0.0601 |   0.4938 | 0.1072 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | lstm               | 5s             | canonical                |      0.1011 |   0.9972 | 0.1835 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | transformer        | 10s            | canonical                |      0.0986 |   0.9921 | 0.1793 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | gru                | 10s            | canonical                |      0.0959 |   0.9616 | 0.1743 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | autoencoder        | 10s            | canonical                |      0.0861 |   0.8184 | 0.1559 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | isolation_forest   | 10s            | canonical                |      0.2029 |   0.9514 | 0.3345 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | threshold_baseline | 10s            | canonical                |      0.9669 |   0.5686 | 0.7161 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | lstm               | 10s            | canonical                |      0.0936 |   0.9952 | 0.1712 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | transformer        | 60s            | canonical                |      0.6923 |   1      | 0.8182 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | gru                | 60s            | canonical                |      0.775  |   0.8611 | 0.8158 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | autoencoder        | 60s            | canonical                |      0.0887 |   0.8228 | 0.1602 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | isolation_forest   | 60s            | canonical                |      0.2227 |   0.943  | 0.3603 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | threshold_baseline | 60s            | canonical                |      0.7899 |   0.5949 | 0.6787 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | ttm_extension      | 60s            | extension                |      0.6341 |   0.8387 | 0.7222 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | lstm               | 60s            | canonical                |      0.75   |   0.8333 | 0.7895 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | transformer        | 300s           | canonical                |      0.2308 |   0.75   | 0.3529 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | gru                | 300s           | canonical                |      0.2143 |   0.75   | 0.3333 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | autoencoder        | 300s           | canonical                |      0.0987 |   0.8333 | 0.1765 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | isolation_forest   | 300s           | canonical                |      0.1226 |   0.8889 | 0.2155 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | threshold_baseline | 300s           | canonical                |      0.2872 |   0.75   | 0.4154 | Frozen canonical benchmark context.                                  |
| canonical_benchmark_test_split           | benchmark        | lstm               | 300s           | canonical                |      0.3333 |   1      | 0.5    | Frozen canonical benchmark context.                                  |
| existing_frozen_candidate_heldout_replay | replay           | threshold_baseline | 10s            | replay                   |      0.4146 |   0.273  | 0.2512 | Existing improved replay context using frozen candidate subset only. |
| existing_frozen_candidate_heldout_replay | replay           | transformer        | 60s            | replay                   |      0.8256 |   0.5622 | 0.6613 | Existing improved replay context using frozen candidate subset only. |
| existing_frozen_candidate_heldout_replay | replay           | lstm               | 300s           | replay                   |      0.9463 |   0.7185 | 0.8099 | Existing improved replay context using frozen candidate subset only. |
| heldout_synthetic_zero_day_like          | zero_day_like    | transformer        | 10s            | canonical                |      0.0395 |   0.9645 | 0.0758 | New full heldout synthetic detector sweep from frozen packages.      |
| heldout_synthetic_zero_day_like          | zero_day_like    | gru                | 10s            | canonical                |      0.0411 |   0.9208 | 0.0784 | New full heldout synthetic detector sweep from frozen packages.      |

### Deployment Evidence Snapshot

| profile                   | model_name         | window_label   |   throughput_windows_per_sec |   model_package_size_mb |   replay_f1 |   benchmark_f1_reference |
|:--------------------------|:-------------------|:---------------|-----------------------------:|------------------------:|------------:|-------------------------:|
| workstation_cpu           | threshold_baseline | 10s            |                   188754     |                   0.246 |      0      |                   0.7161 |
| workstation_cpu           | transformer        | 60s            |                      613.244 |                   0.333 |      0.8844 |                   0.8182 |
| workstation_cpu           | lstm               | 300s           |                     1435.6   |                   0.122 |      0.94   |                   0.5    |
| constrained_single_thread | threshold_baseline | 10s            |                   204392     |                   0.246 |      0      |                   0.7161 |
| constrained_single_thread | transformer        | 60s            |                      610.302 |                   0.333 |      0.8844 |                   0.8182 |
| constrained_single_thread | lstm               | 300s           |                     1389.67  |                   0.122 |      0.94   |                   0.5    |

## F. Model Inventory

| model              | role                                                        | context                                                                        | status                  | canonical_or_extension   |
|:-------------------|:------------------------------------------------------------|:-------------------------------------------------------------------------------|:------------------------|:-------------------------|
| threshold_baseline | Simple residual/score baseline                              | canonical benchmark and heldout synthetic where available                      | implemented             | canonical comparator     |
| isolation_forest   | Classical unsupervised detector                             | canonical benchmark and heldout synthetic where available                      | implemented             | canonical comparator     |
| autoencoder        | MLP autoencoder reconstruction detector                     | canonical benchmark and heldout synthetic where available                      | implemented             | canonical comparator     |
| gru                | Sequence neural detector                                    | canonical benchmark and heldout synthetic where available                      | implemented             | canonical comparator     |
| lstm               | Sequence neural detector                                    | canonical benchmark and heldout synthetic where available                      | implemented             | canonical comparator     |
| transformer        | Sequence neural detector and frozen canonical winner at 60s | canonical benchmark winner; also replay/heldout synthetic evaluated separately | implemented             | canonical winner at 60s  |
| ttm_extension      | TinyTimeMixer forecast-error detector                       | 60s extension benchmark and heldout synthetic extension row                    | implemented at 60s only | extension-only           |
| lora_finetuned     | Tiny-LLM explanation/classification experiment              | explanation/family/asset/grounding support branch                              | implemented but weak    | experimental extension   |
| lstm_ae            | Requested name only; not the repo implementation            | reported as blocked/not implemented                                            | not implemented         | not canonical            |
| arima              | Requested classical time-series comparator                  | reported as not implemented                                                    | not implemented         | extension blocked        |

## G. Final Safe Claims

### Safe Now

- DERGuardian implements a three-phase simulation-grounded DER anomaly-detection pipeline.
- The frozen canonical benchmark winner is `transformer @ 60s`.
- The project includes validated Phase 2 scenarios, diversity/coverage/difficulty reporting, replay evidence, heldout synthetic zero-day-like evaluation, XAI validation, TTM extension evidence, LoRA extension evidence, and offline deployment benchmarking.
- XAI can be described as grounded post-alert operator-facing support.
- Deployment can be described as an offline lightweight deployment benchmark.

### Safe With Careful Wording

- "Zero-day-like" is safe only when defined as heldout synthetic generated attack evaluation.
- "LLM-assisted scenario generation" is safe only as structured scenario authoring and validation support.
- "TTM comparison" is safe only as an extension branch, mainly at 60s.
- "LoRA" is safe only as an experimental and weak explanation/classification branch.

### Not Safe

- Real-world zero-day robustness.
- Human-like root-cause analysis.
- AOI as an implemented detector metric.
- True edge/field deployment.
- Replacing the canonical winner with replay, heldout synthetic, TTM, or LoRA results.
- Claiming an LSTM autoencoder detector exists; the repo autoencoder is MLP-based.

### Future Work

- Public data/sample release decision.
- Optional package refactor under `src/`.
- More complete TTM windows if practical.
- Stronger LoRA or small-model explanation training if evidence improves.
- Real edge-hardware deployment testing if hardware becomes available.

## H. Git Release Guidance

Commit the clean source, docs, tests, manifests, and selected mirrored artifacts. Keep local generated outputs, zips, checkpoints, runtime caches, and full raw/generated time-series artifacts ignored unless publishing them as controlled release assets.

### Commit

- Root docs/configs: `README.md`, `LICENSE`, `.gitignore`, `requirements.txt`, `environment.yml`
- Source folders: `common/`, `opendss/`, `phase1/`, `phase1_models/`, `phase2/`, `phase3/`, `phase3_explanations/`, `scripts/`
- Public docs: `docs/methodology/`, `docs/reports/`, `docs/figures/`
- Git-friendly evidence: `artifacts/`
- Tests and lightweight manifests: `tests/`, `configs/`, `data/sample_or_manifest_only/`
- Final release docs: `FINAL_PUBLISHABLE_REPO_DECISION.md`, `FINAL_REPO_PUBLISHABILITY_CHECKLIST.md`, `FINAL_REPO_PUBLISHABILITY_STATUS.csv`, `GITHUB_READY_SUMMARY.md`, `PROFESSOR_READY_METHOD_SUMMARY.md`, `PUBLISHABLE_REPORT_INDEX.md`, `REPO_ARTIFACT_MAP.csv`, `REPO_CONTEXT_INDEX.csv`, `EXPERIMENT_REGISTRY.csv`, `MASTER_PROJECT_UNDERSTANDING_REPORT.md`

### Ignore Or Ship Separately

- `outputs/`
- `deployment_runtime/`
- `demo/`
- `DERGuardian_professor_demo/`
- `paper_figures/`
- `reports/`
- `phase2_llm_benchmark/`
- `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`
- `*.zip`, `*.tar`, `*.tar.gz`, `*.7z`
- `__pycache__/`, `.pytest_cache/`, `.ipynb_checkpoints/`, local environment folders
- Root-level generated reports/tables not explicitly whitelisted in `.gitignore`

Full generated data should be shipped as manifests, samples, or external release assets only if size/privacy policies allow.

## I. Final Reader Path

Read these 10 files first:

1. `README.md`
2. `docs/methodology/METHODOLOGY_OVERVIEW.md`
3. `FINAL_PHASE123_DIAGRAM_SPEC.md`
4. `docs/reports/CANONICAL_VS_REPLAY_VS_ZERO_DAY.md`
5. `docs/reports/phase1_window_model_comparison_full.md`
6. `docs/reports/zero_day_model_window_results_full.md`
7. `docs/reports/EXTENSION_BRANCHES.md`
8. `docs/reports/PUBLICATION_SAFE_CLAIMS.md`
9. `EXPERIMENT_REGISTRY.csv`
10. `MASTER_PROJECT_UNDERSTANDING_REPORT.md`

## Referenced File Check

No missing backtick file references were found in the final publication-facing docs.
