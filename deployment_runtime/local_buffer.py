"""Offline deployment-runtime support for DERGuardian.

This module implements local buffer logic used by the workstation deployment
benchmark and runtime demonstration code. It describes lightweight offline
behavior only; it is not evidence of field edge deployment.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd

from deployment_runtime.runtime_common import ensure_dir, normalize_timestamp, timestamp_to_z, write_json


CYBER_CONTEXT_COLUMNS = [
    "timestamp_utc",
    "event_type",
    "actor",
    "protocol",
    "resource",
    "command_type",
    "attack_flag",
    "attack_family",
    "result",
    "notes",
]


class LocalBuffer:
    """Small site-local historian window for alert context retrieval."""

    def __init__(
        self,
        site_id: str,
        output_dir: str | Path,
        window_history_limit: int = 16,
        cyber_history_limit: int = 512,
        cyber_lookback_seconds: int = 900,
    ) -> None:
        self.site_id = site_id
        self.output_dir = ensure_dir(output_dir)
        self.window_history_limit = int(window_history_limit)
        self.cyber_history_limit = int(cyber_history_limit)
        self.cyber_lookback_seconds = int(cyber_lookback_seconds)
        self.window_history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=self.window_history_limit))
        self.cold_store_path = ensure_dir(self.output_dir / "contexts")
        self.recent_cyber_events: deque[dict[str, Any]] = deque(maxlen=self.cyber_history_limit)

    def append_event(self, event_type: str, row: dict[str, Any]) -> None:
        if event_type != "cyber":
            return
        compact = {
            key: row.get(key)
            for key in CYBER_CONTEXT_COLUMNS
            if key in row
        }
        if "timestamp_utc" in compact:
            compact["timestamp_utc"] = normalize_timestamp(compact["timestamp_utc"])
        self.recent_cyber_events.append(compact)

    def append_window(self, builder_id: str, window_record: dict[str, Any]) -> None:
        compact = {
            "window_start_utc": normalize_timestamp(window_record["window_start_utc"]),
            "window_end_utc": normalize_timestamp(window_record["window_end_utc"]),
            "window_seconds": int(window_record["window_seconds"]),
        }
        for key, value in window_record.items():
            if key in compact or not key.endswith(("__mean", "__last", "__max", "__min", "__std")):
                continue
            compact[key] = value
        self.window_history[builder_id].append(compact)

    def capture_context(
        self,
        alert_id: str,
        builder_id: str,
        window_start_utc: pd.Timestamp | str,
        window_end_utc: pd.Timestamp | str,
        highlighted_features: list[str] | None = None,
    ) -> Path:
        start = normalize_timestamp(window_start_utc)
        end = normalize_timestamp(window_end_utc)
        selected_feature_columns = set(str(item) for item in (highlighted_features or []))
        matching_windows = []
        for row in self.window_history.get(builder_id, deque()):
            row_start = normalize_timestamp(row["window_start_utc"])
            row_end = normalize_timestamp(row["window_end_utc"])
            if row_end < start - pd.Timedelta(seconds=self.cyber_lookback_seconds):
                continue
            if row_start > end + pd.Timedelta(seconds=self.cyber_lookback_seconds):
                continue
            compact = {
                "window_start_utc": timestamp_to_z(row_start),
                "window_end_utc": timestamp_to_z(row_end),
                "window_seconds": row["window_seconds"],
            }
            for key, value in row.items():
                if key in compact:
                    continue
                if not selected_feature_columns or key in selected_feature_columns:
                    compact[key] = value
            matching_windows.append(compact)
        cyber_events = []
        lookback_start = start - pd.Timedelta(seconds=self.cyber_lookback_seconds)
        for row in self.recent_cyber_events:
            row_ts = normalize_timestamp(row["timestamp_utc"])
            if lookback_start <= row_ts <= end:
                event = dict(row)
                event["timestamp_utc"] = timestamp_to_z(row_ts)
                cyber_events.append(event)
        payload = {
            "alert_id": alert_id,
            "site_id": self.site_id,
            "builder_id": builder_id,
            "window_start_utc": timestamp_to_z(start),
            "window_end_utc": timestamp_to_z(end),
            "highlighted_features": sorted(selected_feature_columns),
            "window_history": matching_windows,
            "cyber_events": cyber_events,
        }
        output_path = self.cold_store_path / f"{alert_id}.json"
        write_json(payload, output_path)
        return output_path
