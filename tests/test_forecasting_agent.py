from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agents.forecasting_agent import FEATURES, _safe_mape, _wape, run_forecasting_agent
from agents.preprocessing_agent import run_preprocessing_agent
from generate_dataset import generate_supply_chain_dataset


def test_forecasting_output_and_non_negative_predictions(isolated_project: Path) -> None:
    raw_path = generate_supply_chain_dataset()
    pre = run_preprocessing_agent(raw_path, test_fraction=0.2)
    result = run_forecasting_agent(pre["processed_path"], test_fraction=0.2)
    assert Path(result["model_path"]).exists()
    assert Path(result["forecast_results_path"]).exists()
    forecasts = pd.read_csv(result["forecast_results_path"])
    assert len(forecasts) > 0
    assert (forecasts["random_forest_prediction"] >= 0).all()
    assert result["random_forest_mae"] >= 0
    assert "baseline_wape" in result
    assert "random_forest_wape" in result
    assert len(forecasts["baseline_prediction"]) == len(forecasts["random_forest_prediction"])
    assert forecasts["baseline_prediction"].equals(forecasts["lag_7"].astype(float))


def test_forecasting_feature_list_excludes_target_derived_flags() -> None:
    assert "inventory_lag_1" in FEATURES
    assert "inventory_level" not in FEATURES
    assert "stockout_flag" not in FEATURES
    assert "overstock_flag" not in FEATURES


def test_safe_mape_excludes_zero_actuals_and_wape_is_percentage() -> None:
    actual = pd.Series([0, 10, 20])
    predicted = np.array([100, 8, 25])
    assert round(_safe_mape(actual, predicted), 4) == 22.5
    assert round(_wape(actual, predicted), 4) == 356.6667
