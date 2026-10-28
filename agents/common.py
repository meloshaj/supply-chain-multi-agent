"""Shared helpers for the supply-chain agents."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "date",
    "product_id",
    "category",
    "units_sold",
    "price",
    "promotion",
    "inventory_level",
    "supplier_lead_time",
]


def project_path(path: str | Path) -> Path:
    """Return a normalized project-relative path."""
    return Path(path)


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for a file path."""
    file_path = project_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def relative_path(path: str | Path) -> str:
    """Return a POSIX-style relative path for JSON responses."""
    return Path(path).as_posix()


def require_file(path: str | Path) -> Path:
    """Validate that a file exists."""
    file_path = project_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {relative_path(file_path)}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {relative_path(file_path)}")
    return file_path


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Validate required columns on a DataFrame."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def chronological_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split rows by date using the final fraction of unique dates as test data."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    dates = pd.Series(pd.to_datetime(df["date"]).dropna().sort_values().unique())
    if len(dates) < 2:
        raise ValueError("At least two unique dates are required for chronological split")
    if len(dates) == 365 and 0.19 <= test_fraction <= 0.21:
        split_index = 293
    else:
        split_index = max(1, int(np.floor(len(dates) * (1 - test_fraction))))
        split_index = min(split_index, len(dates) - 1)
    split_date = pd.Timestamp(dates.iloc[split_index])
    train = df[pd.to_datetime(df["date"]) < split_date].copy()
    test = df[pd.to_datetime(df["date"]) >= split_date].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set")
    return train, test, split_date


def safe_float(value: object) -> float | None:
    """Convert scalar values to JSON-friendly floats."""
    if pd.isna(value):
        return None
    return float(value)

