"""Offline deployment-runtime support for DERGuardian.

This module implements edge runtime logic used by the workstation deployment
benchmark and runtime demonstration code. It describes lightweight offline
behavior only; it is not evidence of field edge deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
import argparse
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.time_alignment import reconstruct_nominal_timestamps
from deployment_runtime.build_alert_packet import build_alert_packet
from deployment_runtime.load_deployed_models import BaseDeployedModel, load_deployed_models
from deployment_runtime.local_buffer import LocalBuffer
from deployment_runtime.runtime_common import (
    ensure_dir,
    load_runtime_config,
    make_alert_id,
    normalize_timestamp,
    runtime_output_paths,
    select_model_profiles,
    timestamp_to_z,
    write_json,
)
from deployment_runtime.stream_window_builder import MEASUREMENT_META_COLUMNS, StreamingWindowBuilder


@dataclass(slots=True)
class EdgeReplaySummary:
    """Structured object used by the offline deployment-runtime workflow."""

    site_id: str
    dataset_kind: str
    model_mode: str
    edge_alert_count: int
    alert_packet_paths: list[str]
    edge_alert_log_path: str
    inference_trace_path: str
    run_manifest_path: str


def _resolve_model_mode(args: argparse.Namespace) -> str:
    if getattr(args, "autoencoder_only", False):
        return "autoencoder_only"
    if getattr(args, "threshold_plus_autoencoder", False):
        return "threshold_plus_autoencoder"
    return "threshold_only"


class EdgeRuntime:
    """Structured object used by the offline deployment-runtime workflow."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        model_mode: str = "threshold_only",
        site_id: str | None = None,
    ) -> None:
        self.config = load_runtime_config(config_path)
        self.model_mode = model_mode
        self.site_id = site_id or str(self.config.get("site_id", "der-site-demo"))
        self.runtime_paths = runtime_output_paths(self.config)
        self.edge_root = ensure_dir(self.runtime_paths["edge_root"] / self.site_id)
        self.alerts_dir = ensure_dir(self.edge_root / "alerts")
        self.context_dir = ensure_dir(self.edge_root / "local_buffer")
        self.models, self.model_warnings = load_deployed_models(config_path=config_path, model_mode=model_mode)
        self.profiles = {profile.runtime_name: profile for profile in select_model_profiles(self.config, model_mode=model_mode)}
        if not self.models:
            warning_suffix = f" Warnings: {'; '.join(self.model_warnings)}" if self.model_warnings else ""
            raise RuntimeError(f"No deployed models were loaded for mode `{model_mode}`.{warning_suffix}")
        self.local_buffer = LocalBuffer(
            site_id=self.site_id,
            output_dir=self.context_dir,
            window_history_limit=int(self.config.get("edge", {}).get("window_history_limit", 16)),
            cyber_history_limit=int(self.config.get("edge", {}).get("cyber_history_limit", 512)),
            cyber_lookback_seconds=int(self.config.get("edge", {}).get("cyber_lookback_seconds", 900)),
        )
        self.alert_on_state_change_only = bool(self.config.get("edge", {}).get("alert_on_state_change_only", True))
        self.model_last_alert_state: dict[str, bool] = {runtime_name: False for runtime_name in self.models}

    def run_replay(
        self,
        *,
        measured_path: str | Path,
        cyber_path: str | Path,
        dataset_kind: str,
        start_time_utc: str | None = None,
        end_time_utc: str | None = None,
        dataset_label: str | None = None,
    ) -> EdgeReplaySummary:
        measured, cyber = self._load_replay_frames(
            measured_path=measured_path,
            cyber_path=cyber_path,
            start_time_utc=start_time_utc,
            end_time_utc=end_time_utc,
        )
        builders = self._build_window_builders(measured)
        builder_model_map: dict[str, list[tuple[str, BaseDeployedModel]]] = {}
        for runtime_name, model in self.models.items():
            profile = self.profiles[runtime_name]
            builder_id = self._builder_id(profile.window_seconds, profile.step_seconds)
            builder_model_map.setdefault(builder_id, []).append((runtime_name, model))

        edge_alert_rows: list[dict[str, Any]] = []
        inference_trace_rows: list[dict[str, Any]] = []
        alert_packet_paths: list[str] = []
        replay_started = time.perf_counter()
        for event_type, row in self._iter_replay_events(measured, cyber):
            self.local_buffer.append_event(event_type, row)
            for builder_id, builder in builders.items():
                build_started = time.perf_counter()
                emissions = builder.process_event(event_type, row)
                build_elapsed_ms = (time.perf_counter() - build_started) * 1000.0
                if not emissions:
                    continue
                for emission in emissions:
                    self.local_buffer.append_window(builder_id, emission.window_record)
                    for runtime_name, model in builder_model_map.get(builder_id, []):
                        trace_row, packet_path = self._score_window_and_optionally_alert(
                            dataset_kind=dataset_kind,
                            dataset_label=dataset_label or dataset_kind,
                            runtime_name=runtime_name,
                            model=model,
                            emission=emission,
                            window_build_ms=build_elapsed_ms / max(len(emissions), 1),
                        )
                        inference_trace_rows.append(trace_row)
                        if packet_path is not None:
                            alert_packet_paths.append(str(packet_path))
                            edge_alert_rows.append(
                                {
                                    "alert_id": trace_row["alert_id"],
                                    "site_id": self.site_id,
                                    "dataset_kind": dataset_kind,
                                    "dataset_label": dataset_label or dataset_kind,
                                    "model_name": trace_row["model_name"],
                                    "model_profile": runtime_name,
                                    "score": trace_row["score"],
                                    "threshold": trace_row["threshold"],
                                    "severity": trace_row["severity"],
                                    "asset_id": trace_row["asset_id"],
                                    "affected_assets": trace_row["affected_assets"],
                                    "window_start_utc": trace_row["window_start_utc"],
                                    "window_end_utc": trace_row["window_end_utc"],
                                    "packet_path": str(packet_path),
                                    "local_context_path": trace_row["local_context_path"],
                                }
                            )

        edge_alert_log_path = self.edge_root / "edge_alert_log.csv"
        inference_trace_path = self.edge_root / "inference_trace.csv"
        pd.DataFrame(edge_alert_rows).to_csv(edge_alert_log_path, index=False)
        pd.DataFrame(inference_trace_rows).to_csv(inference_trace_path, index=False)
        manifest = {
            "site_id": self.site_id,
            "dataset_kind": dataset_kind,
            "dataset_label": dataset_label or dataset_kind,
            "model_mode": self.model_mode,
            "start_time_utc": timestamp_to_z(start_time_utc or measured["analysis_timestamp_utc"].min()),
            "end_time_utc": timestamp_to_z(end_time_utc or measured["analysis_timestamp_utc"].max()),
            "measured_rows": int(len(measured)),
            "cyber_rows": int(len(cyber)),
            "edge_alert_count": int(len(edge_alert_rows)),
            "alert_packet_paths": alert_packet_paths,
            "model_warnings": self.model_warnings,
            "runtime_seconds": round(time.perf_counter() - replay_started, 6),
        }
        run_manifest_path = write_json(manifest, self.edge_root / "run_manifest.json")
        return EdgeReplaySummary(
            site_id=self.site_id,
            dataset_kind=dataset_kind,
            model_mode=self.model_mode,
            edge_alert_count=len(edge_alert_rows),
            alert_packet_paths=alert_packet_paths,
            edge_alert_log_path=str(edge_alert_log_path),
            inference_trace_path=str(inference_trace_path),
            run_manifest_path=str(run_manifest_path),
        )

    def _score_window_and_optionally_alert(
        self,
        *,
        dataset_kind: str,
        dataset_label: str,
        runtime_name: str,
        model: BaseDeployedModel,
        emission,
        window_build_ms: float,
    ) -> tuple[dict[str, Any], Path | None]:
        prediction = model.predict(emission.window_record)
        should_emit_alert = bool(prediction.predicted)
        if self.alert_on_state_change_only:
            should_emit_alert = should_emit_alert and not self.model_last_alert_state.get(runtime_name, False)
        self.model_last_alert_state[runtime_name] = bool(prediction.predicted)
        packet_path: Path | None = None
        alert_id: str | None = None
        packet_build_ms = 0.0
        severity = "low"
        asset_id: str | None = None
        affected_assets: list[str] = []
        local_context_path: str | None = None
        if should_emit_alert:
            alert_id = make_alert_id(f"{self.site_id}-{runtime_name}", emission.window_end_utc)
            packet_started = time.perf_counter()
            context_start = time.perf_counter()
            highlighted_features = []
            for item in prediction.top_features:
                feature_name = str(item.get("feature"))
                highlighted_features.append(feature_name)
                if feature_name.startswith("delta__"):
                    highlighted_features.append(feature_name[len("delta__") :])
            local_context = self.local_buffer.capture_context(
                alert_id=alert_id,
                builder_id=emission.builder_id,
                window_start_utc=emission.window_start_utc,
                window_end_utc=emission.window_end_utc,
                highlighted_features=highlighted_features,
            )
            packet = build_alert_packet(
                site_id=self.site_id,
                packet_source=f"edge::{self.site_id}::{runtime_name}",
                model_name=model.profile.model_name,
                model_profile=runtime_name,
                score=prediction.score,
                threshold=prediction.threshold,
                modality=model.modality,
                window_start_utc=emission.window_start_utc,
                window_end_utc=emission.window_end_utc,
                top_features=prediction.top_features,
                local_context_path=local_context,
                alert_id=alert_id,
                metadata={
                    "dataset_kind": dataset_kind,
                    "dataset_label": dataset_label,
                    "deployment_role": model.profile.deployment_role,
                    "window_seconds": model.profile.window_seconds,
                    "step_seconds": model.profile.step_seconds,
                    "window_builder_id": emission.builder_id,
                    "measured_row_count": emission.measured_row_count,
                    "cyber_row_count": emission.cyber_row_count,
                    "raw_score": round(prediction.raw_score, 6),
                    "reference_found": prediction.reference_found,
                    "context_capture_ms": round((time.perf_counter() - context_start) * 1000.0, 6),
                },
            )
            packet_build_ms = (time.perf_counter() - packet_started) * 1000.0
            packet_path = self.alerts_dir / f"{alert_id}.json"
            write_json(packet, packet_path)
            severity = str(packet["severity"])
            asset_id = packet.get("asset_id")
            affected_assets = list(packet.get("affected_assets", []))
            local_context_path = str(local_context)
        trace_row = {
            "dataset_kind": dataset_kind,
            "dataset_label": dataset_label,
            "site_id": self.site_id,
            "model_name": model.profile.model_name,
            "model_profile": runtime_name,
            "builder_id": emission.builder_id,
            "window_start_utc": timestamp_to_z(emission.window_start_utc),
            "window_end_utc": timestamp_to_z(emission.window_end_utc),
            "score": round(prediction.score, 6),
            "threshold": round(prediction.threshold, 6),
            "predicted": int(prediction.predicted),
            "alert_emitted": int(should_emit_alert),
            "alert_id": alert_id,
            "severity": severity,
            "asset_id": asset_id,
            "affected_assets": "|".join(affected_assets),
            "reference_found": int(prediction.reference_found),
            "measured_row_count": emission.measured_row_count,
            "cyber_row_count": emission.cyber_row_count,
            "window_build_ms": round(window_build_ms, 6),
            "inference_ms": round(prediction.inference_ms, 6),
            "packet_build_ms": round(packet_build_ms, 6),
            "edge_total_ms": round(window_build_ms + prediction.inference_ms + packet_build_ms, 6),
            "local_context_path": local_context_path,
        }
        return trace_row, packet_path

    def _build_window_builders(self, measured: pd.DataFrame) -> dict[str, StreamingWindowBuilder]:
        numeric_columns = [
            column
            for column in measured.columns
            if column not in MEASUREMENT_META_COLUMNS and pd.api.types.is_numeric_dtype(measured[column])
        ]
        builders: dict[str, StreamingWindowBuilder] = {}
        anchor_lookup: dict[str, pd.Timestamp] = {}
        for runtime_name, model in self.models.items():
            profile = self.profiles[runtime_name]
            builder_id = self._builder_id(profile.window_seconds, profile.step_seconds)
            anchor_lookup.setdefault(builder_id, model.reference_store.frame["window_start_utc"].min())
        for profile in self.profiles.values():
            builder_id = self._builder_id(profile.window_seconds, profile.step_seconds)
            if builder_id not in builders:
                builders[builder_id] = StreamingWindowBuilder(
                    builder_id=builder_id,
                    window_seconds=profile.window_seconds,
                    step_seconds=profile.step_seconds,
                    numeric_columns=numeric_columns,
                    anchor_time_utc=anchor_lookup.get(builder_id),
                )
        return builders

    def _load_replay_frames(
        self,
        *,
        measured_path: str | Path,
        cyber_path: str | Path,
        start_time_utc: str | None = None,
        end_time_utc: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        measured = pd.read_parquet(measured_path) if str(measured_path).lower().endswith(".parquet") else pd.read_csv(measured_path)
        cyber = pd.read_parquet(cyber_path) if str(cyber_path).lower().endswith(".parquet") else pd.read_csv(cyber_path)
        measured["timestamp_utc"] = pd.to_datetime(measured["timestamp_utc"], utc=True)
        measured["analysis_timestamp_utc"] = reconstruct_nominal_timestamps(measured)
        cyber["timestamp_utc"] = pd.to_datetime(cyber["timestamp_utc"], utc=True)
        if start_time_utc:
            start = normalize_timestamp(start_time_utc)
            measured = measured[measured["analysis_timestamp_utc"] >= start].copy()
            cyber = cyber[cyber["timestamp_utc"] >= start].copy()
        if end_time_utc:
            end = normalize_timestamp(end_time_utc)
            measured = measured[measured["analysis_timestamp_utc"] <= end].copy()
            cyber = cyber[cyber["timestamp_utc"] <= end].copy()
        measured = measured.sort_values("analysis_timestamp_utc").reset_index(drop=True)
        cyber = cyber.sort_values("timestamp_utc").reset_index(drop=True)
        return measured, cyber

    def _iter_replay_events(self, measured: pd.DataFrame, cyber: pd.DataFrame) -> Iterator[tuple[str, dict[str, Any]]]:
        measured_columns = list(measured.columns)
        cyber_columns = list(cyber.columns)
        measured_iter = measured.itertuples(index=False, name=None)
        cyber_iter = cyber.itertuples(index=False, name=None)
        next_measured = self._tuple_to_dict(measured_columns, next(measured_iter, None))
        next_cyber = self._tuple_to_dict(cyber_columns, next(cyber_iter, None))
        while next_measured is not None or next_cyber is not None:
            measured_time = normalize_timestamp(next_measured["analysis_timestamp_utc"]) if next_measured is not None else None
            cyber_time = normalize_timestamp(next_cyber["timestamp_utc"]) if next_cyber is not None else None
            if next_cyber is None or (measured_time is not None and measured_time <= cyber_time):
                yield "measured", next_measured
                next_measured = self._tuple_to_dict(measured_columns, next(measured_iter, None))
            else:
                yield "cyber", next_cyber
                next_cyber = self._tuple_to_dict(cyber_columns, next(cyber_iter, None))

    @staticmethod
    def _tuple_to_dict(columns: list[str], row: tuple[Any, ...] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return dict(zip(columns, row))

    @staticmethod
    def _builder_id(window_seconds: int, step_seconds: int) -> str:
        return f"w{int(window_seconds)}_s{int(step_seconds)}"


def build_parser() -> argparse.ArgumentParser:
    """Build parser for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = argparse.ArgumentParser(description="Replay DER telemetry through the lightweight edge deployment runtime.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "config_runtime.json"))
    parser.add_argument("--measured", required=True)
    parser.add_argument("--cyber", required=True)
    parser.add_argument("--dataset-kind", default="attacked", choices=["clean", "attacked"])
    parser.add_argument("--dataset-label")
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    parser.add_argument("--site-id")
    parser.add_argument("--threshold-plus-autoencoder", action="store_true")
    parser.add_argument("--autoencoder-only", action="store_true")
    return parser


def main() -> None:
    """Run the command-line entrypoint for the offline deployment-runtime workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    parser = build_parser()
    args = parser.parse_args()
    runtime = EdgeRuntime(config_path=args.config, model_mode=_resolve_model_mode(args), site_id=args.site_id)
    summary = runtime.run_replay(
        measured_path=args.measured,
        cyber_path=args.cyber,
        dataset_kind=args.dataset_kind,
        start_time_utc=args.start_time,
        end_time_utc=args.end_time,
        dataset_label=args.dataset_label,
    )
    print(
        pd.Series(
            {
                "site_id": summary.site_id,
                "dataset_kind": summary.dataset_kind,
                "model_mode": summary.model_mode,
                "edge_alert_count": summary.edge_alert_count,
                "edge_alert_log_path": summary.edge_alert_log_path,
                "inference_trace_path": summary.inference_trace_path,
                "run_manifest_path": summary.run_manifest_path,
            }
        ).to_json()
    )


if __name__ == "__main__":
    main()
