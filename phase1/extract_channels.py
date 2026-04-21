from __future__ import annotations

from pathlib import Path
import argparse
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.io_utils import read_dataframe, write_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract selected channels into long format.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pattern", default="^(feeder_|pv_|bess_|regulator_|capacitor_|switch_|env_)")
    args = parser.parse_args()

    df = read_dataframe(args.input)
    regex = re.compile(args.pattern)
    meta = [column for column in ["timestamp_utc", "scenario_id", "run_id", "split_id", "source_layer"] if column in df.columns]
    channels = [column for column in df.columns if regex.search(column)]
    long_df = df[meta + channels].melt(id_vars=meta, var_name="channel", value_name="value")
    suffix = Path(args.output).suffix.lower().lstrip(".") or "parquet"
    write_dataframe(long_df, args.output, fmt=suffix)


if __name__ == "__main__":
    main()
