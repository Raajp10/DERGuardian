# Methodology Overview

DERGuardian is organized into three phases with explicit context separation.

## Phase 1

Phase 1 creates clean and measured DER time series, builds residual/deviation windows, and benchmarks detector models across multiple window sizes. The canonical selection source of truth remains `outputs/window_size_study/final_window_comparison.csv`; the selected canonical winner is Transformer @ 60s.

## Phase 2

Phase 2 creates and validates schema-bound synthetic attack scenarios. The final coverage package audits attack family, asset, signal, generator, repair, rejection, and difficulty coverage.

## Phase 3

Phase 3 evaluates frozen detector packages while keeping contexts separate: canonical benchmark, heldout replay, heldout synthetic zero-day-like evaluation, and extension experiments.

## Diagram Alignment

Use `docs/methodology/FINAL_PHASE123_DIAGRAM_SPEC.md` and `docs/methodology/FINAL_PHASE123_BOX_TEXT.csv` for slide-safe diagram labels.
