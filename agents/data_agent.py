"""Data loading and validation agent."""

from __future__ import annotations

import pandas as pd

from agents.common import REQUIRED_COLUMNS, relative_path, require_file, safe_float, validate_columns


def run_data_agent(dataset_path: str) -> dict[str, object]:
    """Load and validate a supply-chain CSV dataset."""
    path = require_file(dataset_path)
    df = pd.read_csv(path)
    validate_columns(df, REQUIRED_COLUMNS)

    dates = pd.to_datetime(df["date"], errors="coerce")
    numeric_stats = df.select_dtypes(include=["number"]).describe().round(3).to_dict()

    return {
        "status": "success",
        "agent": "Data Agent",
        "dataset_path": relative_path(path),
        "rows_loaded": int(len(df)),
        "columns_loaded": int(len(df.columns)),
        "products_found": int(df["product_id"].nunique()),
        "missing_values": {column: int(value) for column, value in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "date_min": None if dates.dropna().empty else dates.min().date().isoformat(),
        "date_max": None if dates.dropna().empty else dates.max().date().isoformat(),
        "descriptive_statistics": {
            column: {metric: safe_float(value) for metric, value in metrics.items()}
            for column, metrics in numeric_stats.items()
        },
    }

