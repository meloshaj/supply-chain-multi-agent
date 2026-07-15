"""Preprocessing and feature engineering agent."""

from __future__ import annotations

import pandas as pd

from agents.common import REQUIRED_COLUMNS, chronological_split, ensure_parent, relative_path, require_file, validate_columns


PROCESSED_PATH = "data/processed/processed_supply_chain_data.csv"
TRAIN_PATH = "data/processed/train_supply_chain_data.csv"
TEST_PATH = "data/processed/test_supply_chain_data.csv"
FEATURES_CREATED = [
    "day_of_week",
    "month",
    "week_of_year",
    "is_weekend",
    "inventory_lag_1",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7",
]


def _fill_numeric_missing(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = df.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        df[column] = df.groupby("product_id")[column].transform(lambda s: s.fillna(s.median()))
        df[column] = df[column].fillna(df[column].median())
    return df


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    demand = df.groupby("product_id", group_keys=False)["units_sold"]
    inventory = df.groupby("product_id", group_keys=False)["inventory_level"]
    prior_demand = demand.shift(1)
    prior_inventory = inventory.shift(1)

    df["inventory_lag_1"] = prior_inventory.fillna(0)
    df["lag_1"] = prior_demand.fillna(0)
    df["lag_7"] = demand.shift(7)
    df["rolling_mean_7"] = prior_demand.groupby(df["product_id"]).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    df["rolling_std_7"] = prior_demand.groupby(df["product_id"]).rolling(7, min_periods=2).std().reset_index(level=0, drop=True)

    prior_expanding_mean = prior_demand.groupby(df["product_id"]).expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    df["lag_7"] = df["lag_7"].fillna(prior_expanding_mean).fillna(0)
    df["rolling_mean_7"] = df["rolling_mean_7"].fillna(prior_expanding_mean).fillna(0)
    df["rolling_std_7"] = df["rolling_std_7"].fillna(0)
    return df


def run_preprocessing_agent(dataset_path: str, test_fraction: float = 0.2) -> dict[str, object]:
    """Clean raw data, create time-series features, and save train/test splits."""
    path = require_file(dataset_path)
    df = pd.read_csv(path)
    validate_columns(df, REQUIRED_COLUMNS)
    rows_before = len(df)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates().copy()
    df = _fill_numeric_missing(df)
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    df = _engineer_features(df)
    df["date"] = df["date"].dt.date.astype(str)

    train, test, _ = chronological_split(df, test_fraction)

    processed_path = ensure_parent(PROCESSED_PATH)
    train_path = ensure_parent(TRAIN_PATH)
    test_path = ensure_parent(TEST_PATH)
    df.to_csv(processed_path, index=False)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    train_dates = pd.to_datetime(train["date"])
    test_dates = pd.to_datetime(test["date"])
    return {
        "status": "success",
        "agent": "Preprocessing Agent",
        "processed_path": relative_path(processed_path),
        "training_path": relative_path(train_path),
        "testing_path": relative_path(test_path),
        "rows_before": int(rows_before),
        "rows_after": int(len(df)),
        "training_rows": int(len(train)),
        "testing_rows": int(len(test)),
        "training_start": train_dates.min().date().isoformat(),
        "training_end": train_dates.max().date().isoformat(),
        "testing_start": test_dates.min().date().isoformat(),
        "testing_end": test_dates.max().date().isoformat(),
        "features_created": FEATURES_CREATED,
    }
