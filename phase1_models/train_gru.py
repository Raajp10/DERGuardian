"""Phase 1 detector training and evaluation support for DERGuardian.

This module implements train gru logic for residual-window model training,
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

from phase1_models.neural_models import GRUClassifier
from phase1_models.run_full_evaluation import prepare_full_run_data, run_sequence_model
from phase1_models.sequence_train_utils import train_sequence_classifier_model


def main() -> None:
    """Run the command-line entrypoint for the Phase 1 detector modeling workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Train the repaired GRU sequence classifier.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--max-features", type=int, default=96)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--run-mode", default="full", choices=["full", "legacy-smoke"])
    parser.add_argument("--feature-counts", default="")
    parser.add_argument("--seq-lens", default="")
    parser.add_argument("--buffer-windows", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    root = Path(args.project_root)
    if args.run_mode == "legacy-smoke":
        result = train_sequence_classifier_model(
            model_name="gru",
            model_ctor=lambda input_dim: GRUClassifier(input_dim=input_dim, hidden_dim=64),
            project_root=root,
            max_features=args.max_features,
            seq_len=args.seq_len,
            epochs=args.epochs,
        )
    else:
        feature_counts = [int(item) for item in args.feature_counts.split(",") if item.strip()] if args.feature_counts else [args.max_features]
        seq_lens = [int(item) for item in args.seq_lens.split(",") if item.strip()] if args.seq_lens else [args.seq_len]
        full_data = prepare_full_run_data(root, buffer_windows=args.buffer_windows)
        full_result = run_sequence_model(
            root=root,
            full_data=full_data,
            model_name="gru",
            model_ctor=lambda input_dim: GRUClassifier(input_dim=input_dim, hidden_dim=64),
            feature_counts=feature_counts,
            seq_lens=seq_lens,
            epochs=args.epochs,
            patience=args.patience,
            token_input=False,
            report_root=root / "outputs" / "reports" / "model_full_run_artifacts",
            model_root_name="models_full_run",
            artifact_root_name="model_full_run_artifacts",
        )
        result = {"model_name": "gru", "metrics": full_result["summary"], "info": full_result["summary"]}
    print(f"Saved GRU artifacts for {result['model_name']}")


if __name__ == "__main__":
    main()
