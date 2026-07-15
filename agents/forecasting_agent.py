"""Demand forecasting agent."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from agents.common import chronological_split, ensure_parent, relative_path, require_file


MODEL_PATH = "models/demand_forecasting_model.joblib"
FORECAST_RESULTS_PATH = "outputs/tables/forecast_results.csv"
CATEGORICAL_FEATURES = ["product_id", "category"]
NUMERIC_FEATURES = [
    "inventory_lag_1",
    "supplier_lead_time",
    "promotion",
    "price",
    "day_of_week",
    "month",
    "week_of_year",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _rmse(y_true: pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_mape(y_true: pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """MAPE percentage excluding rows where actual demand is zero."""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    mask = y_true_array != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true_array[mask] - y_pred_array[mask]) / y_true_array[mask])) * 100)


def _wape(y_true: pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    denominator = np.abs(y_true_array).sum()
    if denominator == 0:
        return 0.0
    return float(np.abs(y_true_array - np.asarray(y_pred, dtype=float)).sum() / denominator * 100)


def _improvement(baseline: float, model: float) -> float:
    if baseline == 0:
        return 0.0
    return float((baseline - model) / baseline * 100)


def run_forecasting_agent(processed_dataset_path: str, test_fraction: float = 0.2) -> dict[str, object]:
    """Train a Random Forest forecaster and compare it with a lag-7 baseline."""
    path = require_file(processed_dataset_path)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    missing_features = [feature for feature in FEATURES + ["units_sold"] if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required forecasting columns: {', '.join(missing_features)}")

    train, test, _ = chronological_split(df, test_fraction)
    X_train = train[FEATURES]
    y_train = train["units_sold"]
    X_test = test[FEATURES]
    y_test = test["units_sold"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    rf_predictions = np.maximum(model.predict(X_test), 0)
    baseline_predictions = np.maximum(test["lag_7"].to_numpy(dtype=float), 0)

    baseline_mae = float(mean_absolute_error(y_test, baseline_predictions))
    rf_mae = float(mean_absolute_error(y_test, rf_predictions))
    baseline_rmse = _rmse(y_test, baseline_predictions)
    rf_rmse = _rmse(y_test, rf_predictions)

    results = test.copy()
    results["actual_units_sold"] = y_test.to_numpy()
    results["baseline_prediction"] = baseline_predictions
    results["random_forest_prediction"] = rf_predictions
    results["forecast_error"] = results["actual_units_sold"] - results["random_forest_prediction"]
    results["absolute_error"] = results["forecast_error"].abs()
    results["date"] = pd.to_datetime(results["date"]).dt.date.astype(str)

    model_path = ensure_parent(MODEL_PATH)
    forecast_path = ensure_parent(FORECAST_RESULTS_PATH)
    joblib.dump(model, model_path)
    results.to_csv(forecast_path, index=False)

    return {
        "status": "success",
        "agent": "Forecasting Agent",
        "baseline_mae": round(baseline_mae, 4),
        "baseline_rmse": round(baseline_rmse, 4),
        "baseline_mape": round(_safe_mape(y_test, baseline_predictions), 4),
        "baseline_wape": round(_wape(y_test, baseline_predictions), 4),
        "random_forest_mae": round(rf_mae, 4),
        "random_forest_rmse": round(rf_rmse, 4),
        "random_forest_mape": round(_safe_mape(y_test, rf_predictions), 4),
        "random_forest_wape": round(_wape(y_test, rf_predictions), 4),
        "mae_improvement_percent": round(_improvement(baseline_mae, rf_mae), 4),
        "rmse_improvement_percent": round(_improvement(baseline_rmse, rf_rmse), 4),
        "model_path": relative_path(model_path),
        "forecast_results_path": relative_path(forecast_path),
    }
