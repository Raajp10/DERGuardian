"""Shared utility support for DERGuardian.

This module provides io utils helpers used across the Phase 1 data
pipeline, Phase 2 scenario pipeline, and Phase 3 evaluation/reporting layers.
The functions here are infrastructure code: they prepare paths, metadata,
profiles, graphs, units, or time alignment without changing canonical detector
outputs or benchmark decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json
import re

import pandas as pd
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Handle ensure dir within the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def slugify(text: str) -> str:
    """Handle slugify within the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    sanitized = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return sanitized or "artifact"


def write_dataframe(df: pd.DataFrame, path: str | Path, fmt: str = "parquet") -> Path:
    """Write dataframe for the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(out_path, index=False)
    elif fmt == "parquet":
        df.to_parquet(out_path, index=False)
    else:
        raise ValueError(f"Unsupported dataframe format: {fmt}")
    return out_path


def read_dataframe(path: str | Path) -> pd.DataFrame:
    """Read dataframe for the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    src = Path(path)
    if src.suffix.lower() == ".csv":
        return pd.read_csv(src)
    if src.suffix.lower() == ".parquet":
        return pd.read_parquet(src)
    raise ValueError(f"Unsupported dataframe file: {src}")


def write_json(payload: Any, path: str | Path) -> Path:
    """Write json for the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    return out_path


def read_json(path: str | Path) -> Any:
    """Read json for the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write jsonl for the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=_json_default))
            handle.write("\n")
    return out_path


def list_relative_files(base: str | Path) -> list[str]:
    """Handle list relative files within the shared DERGuardian utility workflow.

        Arguments and returned values follow the explicit type hints and are used by the surrounding pipeline contracts.
        """

    root = Path(base)
    if not root.exists():
        return []
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
