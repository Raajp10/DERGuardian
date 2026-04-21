from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from phase1_models.run_full_evaluation import FullRunData, _split_indices, rank_features_by_effect_size
from phase3.experiment_utils import infer_candidate_columns, prepare_window_bundle


def available_scenarios(labels_df: pd.DataFrame) -> list[str]:
    if labels_df.empty or "scenario_id" not in labels_df.columns:
        return []
    return sorted(labels_df["scenario_id"].astype(str).unique().tolist())


def leave_one_scenario_out_split(
    feature_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    holdout_scenario: str,
    buffer_windows: int = 2,
    benign_train_fraction: float = 0.8,
    scenario_train_fraction: float = 0.75,
) -> pd.DataFrame:
    frame = feature_df.sort_values("window_start_utc").reset_index(drop=True).copy()
    frame["split_name"] = ""
    position_map = {int(idx): pos for pos, idx in enumerate(frame.index)}

    holdout_idx = frame.index[frame["scenario_window_id"] == holdout_scenario].tolist()
    frame.loc[holdout_idx, "split_name"] = "test"
    _assign_context(frame, holdout_idx, position_map, buffer_windows, split_name="test")

    for scenario_id in available_scenarios(labels_df):
        if scenario_id == holdout_scenario:
            continue
        scenario_idx = frame.index[frame["scenario_window_id"] == scenario_id].tolist()
        if not scenario_idx:
            continue
        train_idx, val_idx = _split_train_val_indices(scenario_idx, scenario_train_fraction)
        frame.loc[train_idx, "split_name"] = "train"
        frame.loc[val_idx, "split_name"] = "val"
        _assign_context(frame, train_idx, position_map, buffer_windows, split_name="train")
        _assign_context(frame, val_idx, position_map, buffer_windows, split_name="val")

    remaining_benign = frame.index[frame["split_name"] == ""].tolist()
    benign_train_idx, benign_val_idx = _split_train_val_indices(remaining_benign, benign_train_fraction)
    frame.loc[benign_train_idx, "split_name"] = "train"
    frame.loc[benign_val_idx, "split_name"] = "val"
    frame.loc[frame["split_name"] == "", "split_name"] = "train"
    frame["holdout_scenario"] = holdout_scenario
    return frame


def build_zero_day_full_run_data(
    feature_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    holdout_scenario: str,
    buffer_windows: int = 2,
    candidate_columns: list[str] | None = None,
    residual_artifact_path: Path | None = None,
) -> FullRunData:
    split_df = leave_one_scenario_out_split(
        feature_df=feature_df,
        labels_df=labels_df,
        holdout_scenario=holdout_scenario,
        buffer_windows=buffer_windows,
    )
    selected_columns = candidate_columns or infer_candidate_columns(feature_df, mode="residual")
    ranking = rank_features_by_effect_size(split_df[split_df["split_name"] == "train"], selected_columns)
    resolved_residual_path = residual_artifact_path or (
        ROOT / "outputs" / "reports" / "model_full_run_artifacts" / "residual_windows_full_run.parquet"
    )
    return FullRunData(
        residual_df=feature_df.copy(),
        labels_df=labels_df.copy(),
        split_df=split_df,
        feature_ranking=ranking,
        residual_artifact_path=resolved_residual_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect leave-one-scenario-out splits for the DER Phase 3 evaluation.")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--holdout-scenario", default="")
    parser.add_argument("--buffer-windows", type=int, default=2)
    args = parser.parse_args()

    root = Path(args.project_root)
    bundle = prepare_window_bundle(root)
    scenarios = available_scenarios(bundle.labels_df)
    holdout_scenario = args.holdout_scenario or (scenarios[0] if scenarios else "")
    if not holdout_scenario:
        raise SystemExit("No scenarios available for zero-day splitting.")
    full_data = build_zero_day_full_run_data(
        feature_df=bundle.residual_df,
        labels_df=bundle.labels_df,
        holdout_scenario=holdout_scenario,
        buffer_windows=args.buffer_windows,
    )
    summary = (
        full_data.split_df.groupby(["split_name", "attack_present"], observed=False)
        .size()
        .reset_index(name="window_count")
        .sort_values(["split_name", "attack_present"])
    )
    print(summary.to_string(index=False))


def _assign_context(
    frame: pd.DataFrame,
    scenario_indices: list[int],
    position_map: dict[int, int],
    buffer_windows: int,
    split_name: str,
) -> None:
    if not scenario_indices:
        return
    context_positions: list[int] = []
    for idx in scenario_indices:
        pos = position_map[int(idx)]
        context_positions.extend(range(max(0, pos - buffer_windows), min(len(frame), pos + buffer_windows + 1)))
    context_idx = [
        frame.index[pos]
        for pos in sorted(set(context_positions))
        if frame.loc[frame.index[pos], "attack_present"] == 0 and frame.loc[frame.index[pos], "split_name"] == ""
    ]
    if context_idx:
        frame.loc[context_idx, "split_name"] = split_name


def _split_train_val_indices(indices: list[int], train_fraction: float) -> tuple[list[int], list[int]]:
    if not indices:
        return [], []
    train_idx, val_idx, _ = _split_indices(indices, ratios=(train_fraction, 1.0 - train_fraction, 0.0))
    if not val_idx and len(indices) >= 2:
        val_idx = indices[-1:]
        train_idx = indices[:-1]
    return train_idx, val_idx
