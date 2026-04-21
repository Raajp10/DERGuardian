"""Phase 2 scenario and attacked-dataset support for DERGuardian.

This module implements merge physical and cyber logic for schema-bound synthetic attack
scenarios, injection compilation, cyber logs, labels, validation, or reporting.
Generated scenarios are heldout synthetic evidence and are not claimed as
real-world zero-day proof.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import WindowConfig
from common.io_utils import read_dataframe, write_dataframe
from phase1.build_windows import build_merged_windows


def main() -> None:
    """Run the command-line entrypoint for the Phase 2 scenario and attacked-dataset workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Merge attacked physical and cyber layers into model-ready windows.")
    parser.add_argument("--measured", required=True)
    parser.add_argument("--cyber", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-seconds", type=int, default=300)
    parser.add_argument("--step-seconds", type=int, default=60)
    args = parser.parse_args()

    merged = build_merged_windows(
        measured_df=read_dataframe(args.measured),
        cyber_df=read_dataframe(args.cyber),
        labels_df=read_dataframe(args.labels),
        windows=WindowConfig(window_seconds=args.window_seconds, step_seconds=args.step_seconds),
    )
    suffix = Path(args.output).suffix.lower().lstrip(".") or "parquet"
    write_dataframe(merged, args.output, fmt=suffix)


if __name__ == "__main__":
    main()
