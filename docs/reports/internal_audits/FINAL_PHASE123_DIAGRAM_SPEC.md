# Final Phase 1-2-3 Diagram Specification

This specification rewrites the methodology diagram so it matches the actual repository evidence. It is intentionally conservative.

## Status Counts

| phase   | status                |   box_count |
|:--------|:----------------------|------------:|
| Phase 1 | fully implemented     |           7 |
| Phase 1 | partially implemented |           1 |
| Phase 2 | fully implemented     |           3 |
| Phase 3 | fully implemented     |           3 |
| Phase 3 | partially implemented |           4 |

## Final Box Text

| phase   | safe_box_text                                                                                   | status                | safe_wording                                                                                                                   |
|:--------|:------------------------------------------------------------------------------------------------|:----------------------|:-------------------------------------------------------------------------------------------------------------------------------|
| Phase 1 | Generate DER grid and clean simulation data                                                     | fully implemented     | IEEE 123-bus DER simulation produces clean physical and cyber baseline data.                                                   |
| Phase 1 | IEEE 123-bus feeder with PV/BESS assets                                                         | fully implemented     | The study uses an IEEE 123-bus feeder with simulated PV and BESS assets.                                                       |
| Phase 1 | Clean OpenDSS simulation and measured telemetry                                                 | fully implemented     | Clean physical and measured telemetry is generated and windowed for model inputs.                                              |
| Phase 1 | Windowing, cleaning, and feature engineering                                                    | fully implemented     | Raw time series are transformed into model-ready windows and aligned residual features.                                        |
| Phase 1 | Residual / deviation feature layer                                                              | fully implemented     | Detector inputs include aligned residual/deviation features between attacked and clean windows.                                |
| Phase 1 | Detector benchmark models: threshold, Isolation Forest, MLP autoencoder, GRU, LSTM, Transformer | fully implemented     | The canonical benchmark compares threshold, Isolation Forest, MLP autoencoder, GRU, LSTM, and Transformer detectors.           |
| Phase 1 | Tokenized sequence baseline and context artifacts                                               | partially implemented | Context artifacts and a tokenized baseline support analysis; do not claim an LLM learned normal behavior as the main detector. |
| Phase 1 | Benchmark evaluation: precision, recall, F1, latency, false positive rate                       | fully implemented     | Model selection uses precision, recall, F1, latency, and related detector metrics.                                             |
| Phase 2 | Schema-bound synthetic scenario generation                                                      | fully implemented     | Synthetic scenario JSON bundles are generated and validated under a bounded schema.                                            |
| Phase 2 | Physics, schema, and safety validation                                                          | fully implemented     | Scenarios are accepted, rejected, or repaired using schema and physics-aware validation.                                       |
| Phase 2 | Coverage, diversity, and difficulty audit                                                       | fully implemented     | Phase 2 includes coverage, diversity, metadata, and difficulty calibration audits.                                             |
| Phase 3 | Unified clean, attacked, and residual evaluation datasets                                       | fully implemented     | Phase 3 evaluates frozen detectors on canonical and heldout synthetic residual windows.                                        |
| Phase 3 | Model training: normal-only and supervised detector branches                                    | fully implemented     | Detector training includes supervised benchmark models and normal-only extension branches where applicable.                    |
| Phase 3 | Canonical detector training plus separate LoRA explanation extension                            | partially implemented | LoRA is experimental and separate from the canonical detector benchmark.                                                       |
| Phase 3 | Grounded operator-facing explanation support                                                    | partially implemented | Use grounded post-alert explanation support; do not claim human-like root-cause analysis.                                      |
| Phase 3 | Detector metrics plus separate explanation-grounding metrics                                    | partially implemented | Report precision, recall, F1, latency, false positive rate, and separate XAI grounding metrics. Do not claim AOI.              |
| Phase 3 | Context separation: benchmark, replay, heldout synthetic zero-day-like, extensions              | fully implemented     | Keep canonical benchmark, replay, heldout synthetic zero-day-like evaluation, and extensions separate.                         |
| Phase 3 | Offline lightweight deployment benchmark                                                        | partially implemented | Report offline workstation deployment benchmark only; do not claim field edge deployment.                                      |

## Required Context Boxes

- Residual / deviation feature layer
- Model training with supervised detectors and normal-only extension branches
- Evaluation context separation: canonical benchmark, heldout replay, heldout synthetic zero-day-like evaluation, and extension experiments
- Offline lightweight deployment benchmark

## Non-Claims

- No human-like root-cause analysis claim
- No AOI detector metric claim
- No real-world zero-day robustness claim
- No edge deployment claim without edge hardware
