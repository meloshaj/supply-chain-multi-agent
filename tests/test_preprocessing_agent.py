from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents.preprocessing_agent import run_preprocessing_agent
from generate_dataset import generate_supply_chain_dataset


def test_duplicate_handling_and_feature_creation(isolated_project: Path) -> None:
    path = Path(generate_supply_chain_dataset())
    df = pd.read_csv(path)
    df = pd.concat([df, df.head(3)], ignore_index=True)
    df.to_csv(path, index=False)
    result = run_preprocessing_agent(path.as_posix(), test_fraction=0.2)
    processed = pd.read_csv(result["processed_path"])
    assert result["rows_before"] == 4383
    assert result["rows_after"] == 4380
    assert processed.duplicated().sum() == 0
    for feature in result["features_created"]:
        assert feature in processed.columns


def test_chronological_split(isolated_project: Path) -> None:
    path = generate_supply_chain_dataset()
    result = run_preprocessing_agent(path, test_fraction=0.2)
    assert result["training_end"] < result["testing_start"]
    assert result["training_rows"] > result["testing_rows"]


def test_no_future_leakage_in_lag_features(isolated_project: Path) -> None:
    path = generate_supply_chain_dataset()
    result = run_preprocessing_agent(path, test_fraction=0.2)
    processed = pd.read_csv(result["processed_path"]).sort_values(["product_id", "date"])
    sample = processed.groupby("product_id").nth(10).reset_index()
    for _, row in sample.iterrows():
        product_rows = processed[processed["product_id"] == row["product_id"]].reset_index(drop=True)
        idx = product_rows[product_rows["date"] == row["date"]].index[0]
        assert row["lag_1"] == product_rows.loc[idx - 1, "units_sold"]
        assert row["lag_7"] == product_rows.loc[idx - 7, "units_sold"]


def test_future_target_change_does_not_change_earlier_features(isolated_project: Path) -> None:
    path = Path(generate_supply_chain_dataset())
    original = pd.read_csv(path)
    run_preprocessing_agent(path.as_posix(), test_fraction=0.2)
    before = pd.read_csv("data/processed/processed_supply_chain_data.csv").sort_values(["product_id", "date"])

    future_idx = original[original["product_id"] == "SKU-001"].tail(1).index[0]
    original.loc[future_idx, "units_sold"] = original.loc[future_idx, "units_sold"] + 10_000
    original.to_csv(path, index=False)
    run_preprocessing_agent(path.as_posix(), test_fraction=0.2)
    after = pd.read_csv("data/processed/processed_supply_chain_data.csv").sort_values(["product_id", "date"])

    feature_columns = ["lag_1", "lag_7", "rolling_mean_7", "rolling_std_7"]
    earlier = (before["product_id"] == "SKU-001") & (before["date"] < before.loc[future_idx, "date"])
    pd.testing.assert_frame_equal(
        before.loc[earlier, feature_columns].reset_index(drop=True),
        after.loc[earlier, feature_columns].reset_index(drop=True),
    )


def test_inventory_lag_uses_prior_inventory_not_current_day(isolated_project: Path) -> None:
    path = generate_supply_chain_dataset()
    result = run_preprocessing_agent(path, test_fraction=0.2)
    processed = pd.read_csv(result["processed_path"]).sort_values(["product_id", "date"]).reset_index(drop=True)
    product_rows = processed[processed["product_id"] == "SKU-001"].reset_index(drop=True)
    assert product_rows.loc[0, "inventory_lag_1"] == 0
    assert product_rows.loc[1, "inventory_lag_1"] == product_rows.loc[0, "inventory_level"]
    assert not (product_rows["inventory_lag_1"] == product_rows["inventory_level"]).all()
