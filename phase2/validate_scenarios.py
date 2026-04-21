"""Phase 2 scenario and attacked-dataset support for DERGuardian.

This module implements validate scenarios logic for schema-bound synthetic attack
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

import pandas as pd

from common.config import ProjectPaths, load_pipeline_config
from common.dss_runner import extract_inventory
from common.io_utils import read_dataframe, write_json
from phase2.compile_injections import build_asset_bounds, load_and_validate_scenarios
from phase2.contracts import CANONICAL_TARGET_COMPONENTS


def main() -> None:
    """Run the command-line entrypoint for the Phase 2 scenario and attacked-dataset workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Validate canonical Phase 2 scenario JSON against the executable schema, measured-layer channel inventory, and clean-dataset bounds.")
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--schema", default=str(ROOT / "phase2" / "scenario_schema.json"))
    parser.add_argument("--truth", default=str(ROOT / "outputs" / "clean" / "truth_physical_timeseries.parquet"))
    parser.add_argument("--measured", default=str(ROOT / "outputs" / "clean" / "measured_physical_timeseries.parquet"))
    parser.add_argument("--config", default=str(ROOT / "outputs" / "clean" / "config_snapshot.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    scenarios_path = Path(args.scenarios)
    schema_path = Path(args.schema)
    truth_path = Path(args.truth)
    measured_path = Path(args.measured)
    config_path = Path(args.config)
    output_path = Path(args.output)
    if not scenarios_path.is_absolute():
        scenarios_path = (ROOT / scenarios_path).resolve()
    if not schema_path.is_absolute():
        schema_path = (ROOT / schema_path).resolve()
    if not truth_path.is_absolute():
        truth_path = (ROOT / truth_path).resolve()
    if not measured_path.is_absolute():
        measured_path = (ROOT / measured_path).resolve()
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    if not output_path.is_absolute():
        output_path = (ROOT / output_path).resolve()

    config = load_pipeline_config(config_path)
    paths = ProjectPaths().ensure()
    inventory = extract_inventory(config, paths)
    available_assets = {asset.name for asset in inventory.pv_assets + inventory.bess_assets} | set(inventory.regulators) | set(inventory.capacitors) | set(inventory.switches)
    truth_df = read_dataframe(truth_path)
    measured_df = read_dataframe(measured_path)
    payload = load_and_validate_scenarios(
        scenario_source=scenarios_path,
        schema_file=schema_path,
        timeline=pd.to_datetime(truth_df["timestamp_utc"], utc=True),
        available_assets=available_assets,
        available_columns=set(measured_df.columns),
        asset_bounds=build_asset_bounds(config, inventory),
    )
    write_json(
        {
            "status": "valid",
            "scenario_count": len(payload["scenarios"]),
            "file": str(scenarios_path),
            "schema": str(schema_path),
            "canonical_target_components": list(CANONICAL_TARGET_COMPONENTS),
            "measured_column_count": len(measured_df.columns),
        },
        output_path,
    )


if __name__ == "__main__":
    main()
