"""Offline deployment-runtime support for DERGuardian.

This module implements stream window builder logic used by the workstation deployment
benchmark and runtime demonstration code. It describes lightweight offline
behavior only; it is not evidence of field edge deployment.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any
import math

import pandas as pd

from deployment_runtime.runtime_common import normalize_timestamp


MEASUREMENT_META_COLUMNS = {
    "timestamp_utc",
    "analysis_timestamp_utc",
    "scenario_id",
    "run_id",
    "split_id",
    "source_layer",
}


@dataclass(slots=True)
class WindowEmission:
    """Structured object used by the offline deployment-runtime workflow."""

    builder_id: str
    window_start_utc: pd.Timestamp
    window_end_utc: pd.Timestamp
    window_record: dict[str, Any]
    measured_row_count: int
    cyber_row_count: int


class StreamingWindowBuilder:
    """Incrementally emits project-compatible window aggregates from ordered events."""

    def __init__(
        self,
        builder_id: str,
        window_seconds: int,
        step_seconds: int,
        numeric_columns: list[str],
        anchor_time_utc: pd.Timestamp | None = None,
    ) -> None:
        self.builder_id = builder_id
        self.window_seconds = int(window_seconds)
        self.step_seconds = int(step_seconds)
        self.numeric_columns = list(numeric_columns)
        self.anchor_time_utc = normalize_timestamp(anchor_time_utc) if anchor_time_utc is not None else None
        self.measured_rows: deque[dict[str, Any]] = deque()
        self.cyber_rows: deque[dict[str, Any]] = deque()
        self.current_window_start: pd.Timestamp | None = None
        self.last_measured_time: pd.Timestamp | None = None
        self.last_event_time: pd.Timestamp | None = None

    def process_event(self, event_type: str, row: dict[str, Any]) -> list[WindowEmission]:
        if event_type not in {"measured", "cyber"}:
            raise ValueError(f"Unsupported event_type `{event_type}`.")
        if event_type == "measured":
            normalized = self._normalize_measured_row(row)
            event_time = normalized["analysis_timestamp_utc"]
            self.measured_rows.append(normalized)
            self.last_measured_time = event_time
            if self.current_window_start is None:
                self.current_window_start = self._initial_window_start(event_time)
        else:
            normalized = self._normalize_cyber_row(row)
            event_time = normalized["timestamp_utc"]
            self.cyber_rows.append(normalized)
        if self.last_event_time is not None and event_time < self.last_event_time:
            raise ValueError(
                f"Events must be ordered for builder `{self.builder_id}`. "
                f"Received {event_time.isoformat()} after {self.last_event_time.isoformat()}."
            )
        self.last_event_time = event_time
        return self._emit_ready_windows()

    def flush(self) -> list[WindowEmission]:
        return self._emit_ready_windows(force=True)

    def _emit_ready_windows(self, force: bool = False) -> list[WindowEmission]:
        if self.current_window_start is None or self.last_measured_time is None:
            return []
        width = pd.Timedelta(seconds=self.window_seconds)
        step = pd.Timedelta(seconds=self.step_seconds)
        emissions: list[WindowEmission] = []
        while self.current_window_start + width <= self.last_measured_time or force:
            window_end = self.current_window_start + width
            measured_frame = [
                row
                for row in self.measured_rows
                if self.current_window_start <= row["analysis_timestamp_utc"] < window_end
            ]
            if not measured_frame:
                if not force:
                    break
            else:
                cyber_frame = [
                    row
                    for row in self.cyber_rows
                    if self.current_window_start <= row["timestamp_utc"] < window_end
                ]
                record = self._aggregate_window(measured_frame, cyber_frame, self.current_window_start, window_end)
                emissions.append(
                    WindowEmission(
                        builder_id=self.builder_id,
                        window_start_utc=self.current_window_start,
                        window_end_utc=window_end,
                        window_record=record,
                        measured_row_count=len(measured_frame),
                        cyber_row_count=len(cyber_frame),
                    )
                )
            self.current_window_start = self.current_window_start + step
            self._prune_buffers()
            if force and self.current_window_start + width > self.last_measured_time:
                break
        return emissions

    def _aggregate_window(
        self,
        measured_rows: list[dict[str, Any]],
        cyber_rows: list[dict[str, Any]],
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> dict[str, Any]:
        frame = pd.DataFrame(measured_rows)
        record: dict[str, Any] = {
            "window_start_utc": window_start,
            "window_end_utc": window_end,
            "window_seconds": self.window_seconds,
            "scenario_id": str(frame["scenario_id"].iloc[0]) if "scenario_id" in frame.columns else "",
            "run_id": str(frame["run_id"].iloc[0]) if "run_id" in frame.columns else "",
            "split_id": str(frame["split_id"].iloc[0]) if "split_id" in frame.columns else "",
        }
        for column in self.numeric_columns:
            if column not in frame.columns:
                continue
            series = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            record[f"{column}__mean"] = float(series.mean())
            record[f"{column}__std"] = float(series.std(ddof=0))
            record[f"{column}__min"] = float(series.min())
            record[f"{column}__max"] = float(series.max())
            record[f"{column}__last"] = float(series.iloc[-1])

        cyber_frame = pd.DataFrame(cyber_rows) if cyber_rows else pd.DataFrame()
        if cyber_frame.empty:
            record["cyber_event_count_total"] = 0
            record["cyber_auth_count"] = 0
            record["cyber_command_count"] = 0
            record["cyber_config_count"] = 0
            record["cyber_telemetry_count"] = 0
            record["cyber_attack_count"] = 0
            record["cyber_auth_failures"] = 0
            record["cyber_protocol_modbus_count"] = 0
            record["cyber_protocol_dnp3_count"] = 0
            record["cyber_protocol_mqtt_count"] = 0
        else:
            event_types = cyber_frame.get("event_type", pd.Series(dtype="object")).fillna("")
            auth_result = cyber_frame.get("auth_result", pd.Series(dtype="object")).fillna("")
            protocols = cyber_frame.get("protocol", pd.Series(dtype="object")).fillna("").astype(str).str.lower()
            record["cyber_event_count_total"] = int(len(cyber_frame))
            record["cyber_auth_count"] = int((event_types == "authentication").sum())
            record["cyber_command_count"] = int((event_types == "command").sum())
            record["cyber_config_count"] = int((event_types == "configuration").sum())
            record["cyber_telemetry_count"] = int((event_types == "telemetry").sum())
            record["cyber_attack_count"] = int((event_types == "attack").sum())
            record["cyber_auth_failures"] = int(auth_result.isin(["fail", "failure"]).sum())
            record["cyber_protocol_modbus_count"] = int((protocols == "modbus").sum())
            record["cyber_protocol_dnp3_count"] = int((protocols == "dnp3").sum())
            record["cyber_protocol_mqtt_count"] = int((protocols == "mqtt").sum())
        return record

    def _prune_buffers(self) -> None:
        if self.current_window_start is None:
            return
        while self.measured_rows and self.measured_rows[0]["analysis_timestamp_utc"] < self.current_window_start:
            self.measured_rows.popleft()
        while self.cyber_rows and self.cyber_rows[0]["timestamp_utc"] < self.current_window_start:
            self.cyber_rows.popleft()

    def _normalize_measured_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        if "analysis_timestamp_utc" in normalized:
            normalized["analysis_timestamp_utc"] = normalize_timestamp(normalized["analysis_timestamp_utc"])
        else:
            normalized["analysis_timestamp_utc"] = normalize_timestamp(normalized["timestamp_utc"])
        normalized["timestamp_utc"] = normalize_timestamp(normalized["timestamp_utc"])
        return normalized

    def _normalize_cyber_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        normalized["timestamp_utc"] = normalize_timestamp(normalized["timestamp_utc"])
        return normalized

    def _initial_window_start(self, event_time: pd.Timestamp) -> pd.Timestamp:
        if self.anchor_time_utc is None:
            return event_time
        step = pd.Timedelta(seconds=self.step_seconds)
        delta_seconds = (event_time - self.anchor_time_utc).total_seconds()
        step_count = max(int(math.ceil(delta_seconds / self.step_seconds)), 0)
        return self.anchor_time_utc + step_count * step
