# Pipeline Alignment With Diagram

## Phase 1: Normal System Learning

- `Generate DER Grid Data` and `System Modeling IEEE 123 Bus + DER (PV, BESS)` are implemented through the clean OpenDSS feeder build and clean physical time-series outputs already present in `outputs/clean/`.
- `Normal Data Generation OpenDSS Simulation` is represented by the canonical clean truth and measured time-series files validated during the source audit.
- `Data Preprocessing Cleaning + Feature Engineering` is represented by canonical window building plus aligned residual construction in the Phase 1 benchmark pipeline.
- `Baseline Models / ML Models` is fully implemented by the completed benchmark suite: threshold baseline, isolation forest, autoencoder, GRU, LSTM, transformer, and tokenized LLM-style baseline.
- `LLM Analysis Context Learning of Normal Behavior` exists only in bounded benchmark form through the tokenized LLM-style baseline and downstream reasoning artifacts. It should not be overstated as a large generative reasoning layer for Phase 1 training.
- `Model Evaluation Reconstruction Loss, Threshold, Accuracy` is implemented and completed by the saved Phase 1 benchmark outputs and ready packages.

## Phase 2: Scenario & Attack Generation

- `System Card Creation` is implemented in the shared Phase 2 benchmark assets such as `system_card.json`, `validator_rules.json`, `family_taxonomy.json`, and `physical_constraints.json`.
- `LLM Scenario Generator` is implemented through the heldout generator JSON bundles plus the canonical attacked bundle workflow.
- `Scenario JSON Library` is present in the generator response folders and compiled Phase 2 accepted bundles.
- `Validation Layer Physics + Safety Constraints Check` is fully represented by the existing heldout validation summaries and rejected scenario reports.
- `Injection Engine` and `OpenDSS Execution Physical + Cyber Simulation` are already embodied in the existing canonical Phase 2 outputs reused here from `phase2_llm_benchmark/new_respnse_result/models/<generator>/datasets` and `outputs/attacked/`.
- `Attack Dataset Generation Time-Series with Labels` is complete for the canonical bundle and for the heldout bundles that passed validation.
- `Quality Check Plausibility + Diversity + Metadata` is partially complete through the compiled manifests, validation summaries, label summaries, and scenario difficulty reports. No extra step was missing for this paper-safe offline pipeline.

## Phase 3: Detection & Intelligence Layer

- `Unified Dataset Normal + Attack Data` is implemented by the 60s/12s windowed attacked datasets aligned against canonical clean windows.
- `Model Training ML Models + Tiny LLM (LoRA)` is only partially matched by the diagram. The canonical implemented pipeline uses classic ML, neural sequence models, and a tokenized LLM-style baseline. A true LoRA-trained tiny LLM is not part of the canonical paper-safe path and should not be claimed unless separately evidenced.
- `Anomaly Detection Prediction + Reconstruction Loss` is complete through the saved best-model package and heldout replay outputs.
- `Decision Layer Threshold + Context Reasoning` is complete in the sense of saved thresholds and post-alert reporting. It should still be described conservatively.
- `Human-like Explanation LLM Generates Root Cause Analysis` does not match the safe interpretation of the repository. The safer accurate wording is `grounded explanation layer` or `post-alert operator-facing explanation`.
- `Evaluation Metrics F1, Precision, Recall, AOI` is complete for the detector. AOI-like explanation metrics exist only through the heldout XAI grounding summaries and should be described separately from detection metrics.
- `Deployment Vision Lightweight Edge AI for DER` is only partially represented by the repository's deployment/demo assets. This run completed the offline pipeline and reporting layer, not a new on-device deployment benchmark.

## What Was Missing And Is Now Filled

- heldout bundle inventory and validation reporting
- frozen saved-model replay on heldout cross-generator bundles
- bundle-level and scenario-level heldout metrics
- final Phase 3 figures and publication-safe claim wording
- best-model heldout evaluation manifests

## What Is Still Not A Canonical Completed Claim

- real-world zero-day evidence
- LoRA-trained tiny LLM in the active benchmark path
- human-level root-cause analysis
- edge deployment timing claims from this run
