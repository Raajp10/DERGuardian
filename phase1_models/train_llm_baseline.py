"""Phase 1 detector training and evaluation support for DERGuardian.

This module implements train llm baseline logic for residual-window model training,
inference, packaging, metrics, or reporting. It supports the frozen benchmark
path and related audits while keeping benchmark selection separate from replay,
heldout synthetic zero-day-like, and extension contexts.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase1_models.token_timeseries_model import train_token_timeseries_language_style_baseline


def train_llm_style_baseline(
    project_root: str | Path | None = None,
    max_features: int = 48,
    seq_len: int = 8,
    epochs: int = 6,
    run_mode: str = "full",
    feature_counts: list[int] | None = None,
    seq_lens: list[int] | None = None,
    buffer_windows: int = 2,
    patience: int = 5,
) -> dict[str, object]:
    """Handle train llm style baseline within the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    return train_token_timeseries_language_style_baseline(
        project_root=project_root,
        max_features=max_features,
        seq_len=seq_len,
        epochs=epochs,
        run_mode=run_mode,
        feature_counts=feature_counts,
        seq_lens=seq_lens,
        buffer_windows=buffer_windows,
        patience=patience,
    )


def main() -> None:
    """Run the command-line entrypoint for the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Train the tokenized LLM-style time-series baseline.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--max-features", type=int, default=48)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--run-mode", default="full", choices=["full", "legacy-smoke"])
    parser.add_argument("--feature-counts", default="")
    parser.add_argument("--seq-lens", default="")
    parser.add_argument("--buffer-windows", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    feature_counts = [int(item) for item in args.feature_counts.split(",") if item.strip()] if args.feature_counts else None
    seq_lens = [int(item) for item in args.seq_lens.split(",") if item.strip()] if args.seq_lens else None
    result = train_llm_style_baseline(
        project_root=args.project_root,
        max_features=args.max_features,
        seq_len=args.seq_len,
        epochs=args.epochs,
        run_mode=args.run_mode,
        feature_counts=feature_counts,
        seq_lens=seq_lens,
        buffer_windows=args.buffer_windows,
        patience=args.patience,
    )
    print(f"Saved token baseline artifacts for {result['model_name']}")


if __name__ == "__main__":
    main()
