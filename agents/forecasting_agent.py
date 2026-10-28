"""Demand forecasting agent with feature ablation and promotion-stratified evaluation."""

from __future__ import annotations

from pathlib import Path
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
ABLATION_RESULTS_PATH = "outputs/tables/ablation_results.csv"
PRODUCT_METRICS_PATH = "outputs/tables/product_test_metrics.csv"

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


def _rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray | pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_mape(y_true: pd.Series | np.ndarray, y_pred: np.ndarray | pd.Series) -> float:
    """MAPE percentage excluding rows where actual demand is zero."""
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    mask = y_true_array != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true_array[mask] - y_pred_array[mask]) / y_true_array[mask])) * 100)


def _wape(y_true: pd.Series | np.ndarray, y_pred: np.ndarray | pd.Series) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    denominator = np.abs(y_true_array).sum()
    if denominator == 0:
        return 0.0
    return float(np.abs(y_true_array - np.asarray(y_pred, dtype=float)).sum() / denominator * 100)


def _improvement(baseline: float, model: float) -> float:
    if baseline == 0:
        return 0.0
    return float((baseline - model) / baseline * 100)


def _build_rf_pipeline(numeric_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", numeric_features),
        ]
    )
    return Pipeline(
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


def run_forecasting_agent(processed_dataset_path: str, test_fraction: float = 0.2) -> dict[str, object]:
    """Train a Random Forest forecaster, run 9-configuration ablation suite, and compute stratified metrics."""
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

    # 1. Full Model
    full_model = _build_rf_pipeline(NUMERIC_FEATURES)
    full_model.fit(X_train, y_train)
    rf_predictions = np.maximum(full_model.predict(X_test), 0)
    baseline_predictions = np.maximum(test["lag_7"].to_numpy(dtype=float), 0)

    baseline_mae = float(mean_absolute_error(y_test, baseline_predictions))
    rf_mae = float(mean_absolute_error(y_test, rf_predictions))
    baseline_rmse = _rmse(y_test, baseline_predictions)
    rf_rmse = _rmse(y_test, rf_predictions)
    baseline_mape = _safe_mape(y_test, baseline_predictions)
    rf_mape = _safe_mape(y_test, rf_predictions)
    baseline_wape = _wape(y_test, baseline_predictions)
    rf_wape = _wape(y_test, rf_predictions)

    results = test.copy()
    results["actual_units_sold"] = y_test.to_numpy()
    results["baseline_prediction"] = baseline_predictions
    results["random_forest_prediction"] = rf_predictions
    results["forecast_error"] = results["actual_units_sold"] - results["random_forest_prediction"]
    results["absolute_error"] = results["forecast_error"].abs()
    results["date"] = pd.to_datetime(results["date"]).dt.date.astype(str)

    model_path = ensure_parent(MODEL_PATH)
    forecast_path = ensure_parent(FORECAST_RESULTS_PATH)
    ablation_path = ensure_parent(ABLATION_RESULTS_PATH)
    product_metrics_path = ensure_parent(PRODUCT_METRICS_PATH)

    joblib.dump(full_model, model_path)
    results.to_csv(forecast_path, index=False)

    # 2. 9-Configuration Feature Ablation Suite (Table 5)
    calendar_features = ["day_of_week", "month", "week_of_year", "is_weekend"]
    lag_features = ["lag_1", "lag_7"]
    rolling_features = ["rolling_mean_7", "rolling_std_7"]
    commercial_features = ["price", "promotion"]

    ablation_configs = {
        "Full model (all features)": NUMERIC_FEATURES,
        "Without promotion": [f for f in NUMERIC_FEATURES if f != "promotion"],
        "Without price": [f for f in NUMERIC_FEATURES if f != "price"],
        "Without calendar features": [f for f in NUMERIC_FEATURES if f not in calendar_features],
        "Without lag features": [f for f in NUMERIC_FEATURES if f not in lag_features],
        "Without rolling statistics": [f for f in NUMERIC_FEATURES if f not in rolling_features],
        "Without prior-day inventory": [f for f in NUMERIC_FEATURES if f != "inventory_lag_1"],
        "Without supplier lead time": [f for f in NUMERIC_FEATURES if f != "supplier_lead_time"],
        "Minimal configuration (calendar + commercial only)": calendar_features + commercial_features,
    }

    ablation_rows = []
    for cfg_name, num_feats in ablation_configs.items():
        if cfg_name == "Full model (all features)":
            m_mae, m_rmse, m_mape, m_wape = rf_mae, rf_rmse, rf_mape, rf_wape
        else:
            abl_model = _build_rf_pipeline(num_feats)
            abl_model.fit(train[CATEGORICAL_FEATURES + num_feats], y_train)
            pred = np.maximum(abl_model.predict(test[CATEGORICAL_FEATURES + num_feats]), 0)
            m_mae = float(mean_absolute_error(y_test, pred))
            m_rmse = _rmse(y_test, pred)
            m_mape = _safe_mape(y_test, pred)
            m_wape = _wape(y_test, pred)

        delta_mae = ((m_mae - rf_mae) / rf_mae * 100) if rf_mae > 0 else 0.0
        ablation_rows.append({
            "configuration": cfg_name,
            "mae": round(m_mae, 4),
            "rmse": round(m_rmse, 4),
            "mape": round(m_mape, 4),
            "wape": round(m_wape, 4),
            "delta_mae_percent": round(delta_mae, 4),
        })

    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(ablation_path, index=False)

    # 3. Per-Product Test Metrics & Stratified Analysis
    product_metrics_rows = []
    y_test_arr = y_test.to_numpy()
    promo_mask_all = (test["promotion"] == 1).to_numpy()

    for pid in test["product_id"].unique():
        p_mask = (test["product_id"] == pid).to_numpy()
        sub_y = y_test_arr[p_mask]
        sub_rf = rf_predictions[p_mask]
        sub_base = baseline_predictions[p_mask]
        sub_promo = promo_mask_all[p_mask]

        p_vol = int(sub_y.sum())
        p_mean = float(sub_y.mean())
        p_std = float(sub_y.std(ddof=1)) if len(sub_y) > 1 else 0.0
        p_cv = float(p_std / p_mean) if p_mean > 0 else 0.0

        p_base_mae = float(mean_absolute_error(sub_y, sub_base))
        p_rf_mae = float(mean_absolute_error(sub_y, sub_rf))
        p_base_wape = _wape(sub_y, sub_base)
        p_rf_wape = _wape(sub_y, sub_rf)

        # Promo breakdown
        has_promo = sub_promo.sum() > 0
        has_non_promo = (~sub_promo).sum() > 0
        p_base_promo_mae = float(mean_absolute_error(sub_y[sub_promo], sub_base[sub_promo])) if has_promo else None
        p_rf_promo_mae = float(mean_absolute_error(sub_y[sub_promo], sub_rf[sub_promo])) if has_promo else None
        p_base_non_promo_mae = float(mean_absolute_error(sub_y[~sub_promo], sub_base[~sub_promo])) if has_non_promo else None
        p_rf_non_promo_mae = float(mean_absolute_error(sub_y[~sub_promo], sub_rf[~sub_promo])) if has_non_promo else None

        dept = str(test[test["product_id"] == pid]["category"].iloc[0])
        product_metrics_rows.append({
            "product_id": pid,
            "category": dept,
            "test_volume": p_vol,
            "daily_mean": round(p_mean, 2),
            "cv": round(p_cv, 4),
            "baseline_mae": round(p_base_mae, 4),
            "rf_mae": round(p_rf_mae, 4),
            "baseline_wape": round(p_base_wape, 4),
            "rf_wape": round(p_rf_wape, 4),
            "promo_days": int(sub_promo.sum()),
            "baseline_promo_mae": round(p_base_promo_mae, 4) if p_base_promo_mae is not None else None,
            "rf_promo_mae": round(p_rf_promo_mae, 4) if p_rf_promo_mae is not None else None,
            "baseline_non_promo_mae": round(p_base_non_promo_mae, 4) if p_base_non_promo_mae is not None else None,
            "rf_non_promo_mae": round(p_rf_non_promo_mae, 4) if p_rf_non_promo_mae is not None else None,
        })

    df_prod_metrics = pd.DataFrame(product_metrics_rows).sort_values("test_volume", ascending=False)
    df_prod_metrics.to_csv(product_metrics_path, index=False)

    # Panel-wide promo stratification
    panel_base_promo_mae = float(mean_absolute_error(y_test_arr[promo_mask_all], baseline_predictions[promo_mask_all]))
    panel_rf_promo_mae = float(mean_absolute_error(y_test_arr[promo_mask_all], rf_predictions[promo_mask_all]))
    panel_base_non_promo_mae = float(mean_absolute_error(y_test_arr[~promo_mask_all], baseline_predictions[~promo_mask_all]))
    panel_rf_non_promo_mae = float(mean_absolute_error(y_test_arr[~promo_mask_all], rf_predictions[~promo_mask_all]))

    return {
        "status": "success",
        "agent": "Forecasting Agent",
        "baseline_mae": round(baseline_mae, 4),
        "baseline_rmse": round(baseline_rmse, 4),
        "baseline_mape": round(baseline_mape, 4),
        "baseline_wape": round(baseline_wape, 4),
        "random_forest_mae": round(rf_mae, 4),
        "random_forest_rmse": round(rf_rmse, 4),
        "random_forest_mape": round(rf_mape, 4),
        "random_forest_wape": round(rf_wape, 4),
        "mae_improvement_percent": round(_improvement(baseline_mae, rf_mae), 4),
        "rmse_improvement_percent": round(_improvement(baseline_rmse, rf_rmse), 4),
        "model_path": relative_path(model_path),
        "forecast_results_path": relative_path(forecast_path),
        "ablation_results_path": relative_path(ablation_path),
        "product_metrics_path": relative_path(product_metrics_path),
        "ablation_suite": ablation_rows,
        "per_product_metrics": product_metrics_rows,
        "promotion_stratification": {
            "panel_baseline_promo_mae": round(panel_base_promo_mae, 4),
            "panel_rf_promo_mae": round(panel_rf_promo_mae, 4),
            "panel_baseline_non_promo_mae": round(panel_base_non_promo_mae, 4),
            "panel_rf_non_promo_mae": round(panel_rf_non_promo_mae, 4),
            "case_c_household_1_418": next((p for p in product_metrics_rows if p["product_id"] == "HOUSEHOLD_1_418"), None),
            "case_a_foods_3_586": next((p for p in product_metrics_rows if p["product_id"] == "FOODS_3_586"), None),
            "case_b_hobbies_1_155": next((p for p in product_metrics_rows if p["product_id"] == "HOBBIES_1_155"), None),
        },
    }
