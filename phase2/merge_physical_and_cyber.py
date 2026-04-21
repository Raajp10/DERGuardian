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
