# Final Phase 1-2-3 Slide Text

## Phase 1 Slide

DERGuardian first builds an IEEE 123-bus DER simulation with PV/BESS assets, measured telemetry, cyber events, and aligned residual/deviation windows. The canonical detector benchmark compares threshold, Isolation Forest, MLP autoencoder, GRU, LSTM, and Transformer models across 5s, 10s, 60s, and 300s windows. The frozen canonical benchmark winner is Transformer at 60 seconds.

## Phase 2 Slide

Phase 2 generates schema-bound synthetic attack scenarios, validates them with physics and safety constraints, compiles accepted scenarios into attacked time-series datasets, and audits coverage, diversity, metadata completeness, and difficulty. Rejected and repaired scenarios remain separate for traceability.

## Phase 3 Slide

Phase 3 evaluates frozen detector packages in separate contexts: canonical benchmark, heldout replay, heldout synthetic zero-day-like evaluation, and extension experiments. TTM is reported as a 60s extension branch. LoRA is an experimental explanation/classification branch. XAI is presented as grounded post-alert operator support, and deployment evidence is an offline lightweight benchmark.
