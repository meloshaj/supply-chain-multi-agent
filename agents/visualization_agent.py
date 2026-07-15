"""Visualization agent for report-ready PNG charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from agents.common import relative_path, require_file


FIGURES_DIR = Path("outputs/figures")


def _save(fig: plt.Figure, filename: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return relative_path(path)


def run_visualization_agent(
    processed_dataset_path: str,
    forecast_results_path: str,
    inventory_results_path: str,
) -> dict[str, object]:
    """Generate report-ready Matplotlib visualizations."""
    processed_path = require_file(processed_dataset_path)
    forecast_path = require_file(forecast_results_path)
    inventory_path = require_file(inventory_results_path)

    processed = pd.read_csv(processed_path)
    forecasts = pd.read_csv(forecast_path)
    inventory = pd.read_csv(inventory_path)
    processed["date"] = pd.to_datetime(processed["date"], errors="coerce")
    forecasts["date"] = pd.to_datetime(forecasts["date"], errors="coerce")

    figure_paths: list[str] = []

    daily = processed.groupby("date", as_index=False)["units_sold"].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily["date"], daily["units_sold"], color="#1f77b4", linewidth=1.8, label="Units sold")
    ax.set_title("Total Demand Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold")
    ax.legend()
    figure_paths.append(_save(fig, "demand_over_time.png"))

    actual_pred = forecasts.groupby("date", as_index=False)[["actual_units_sold", "random_forest_prediction"]].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual_pred["date"], actual_pred["actual_units_sold"], label="Actual", linewidth=1.8)
    ax.plot(actual_pred["date"], actual_pred["random_forest_prediction"], label="Random Forest", linewidth=1.8)
    ax.set_title("Actual Versus Predicted Demand")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units")
    ax.legend()
    figure_paths.append(_save(fig, "actual_vs_predicted_demand.png"))

    mae_baseline = (forecasts["actual_units_sold"] - forecasts["baseline_prediction"]).abs().mean()
    mae_rf = (forecasts["actual_units_sold"] - forecasts["random_forest_prediction"]).abs().mean()
    rmse_baseline = ((forecasts["actual_units_sold"] - forecasts["baseline_prediction"]) ** 2).mean() ** 0.5
    rmse_rf = ((forecasts["actual_units_sold"] - forecasts["random_forest_prediction"]) ** 2).mean() ** 0.5
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = pd.DataFrame({"Baseline": [mae_baseline, rmse_baseline], "Random Forest": [mae_rf, rmse_rf]}, index=["MAE", "RMSE"])
    metrics.plot(kind="bar", ax=ax, color=["#8c564b", "#2ca02c"])
    ax.set_title("Baseline Versus Random Forest Metrics")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Error")
    ax.legend()
    figure_paths.append(_save(fig, "baseline_vs_random_forest_metrics.png"))

    product_demand = processed.groupby("product_id", as_index=False)["units_sold"].sum().sort_values("units_sold", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(product_demand["product_id"], product_demand["units_sold"], color="#17becf")
    ax.set_title("Demand By Product")
    ax.set_xlabel("Product")
    ax.set_ylabel("Units sold")
    ax.tick_params(axis="x", rotation=45)
    figure_paths.append(_save(fig, "demand_by_product.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(forecasts["forecast_error"], bins=30, color="#9467bd", edgecolor="white")
    ax.set_title("Forecast Error Distribution")
    ax.set_xlabel("Actual minus predicted")
    ax.set_ylabel("Count")
    figure_paths.append(_save(fig, "forecast_error_distribution.png"))

    risk_counts = inventory["risk_status"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(risk_counts.index, risk_counts.values, color="#d62728")
    ax.set_title("Inventory Risk Category Counts")
    ax.set_xlabel("Risk category")
    ax.set_ylabel("Products")
    ax.tick_params(axis="x", rotation=25)
    figure_paths.append(_save(fig, "inventory_risk_category_counts.png"))

    compare = inventory.sort_values("product_id")
    x = range(len(compare))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - 0.2 for i in x], compare["inventory_level"], width=0.4, label="Current inventory", color="#7f7f7f")
    ax.bar([i + 0.2 for i in x], compare["reorder_point"], width=0.4, label="Reorder point", color="#ff7f0e")
    ax.set_xticks(list(x), compare["product_id"], rotation=45)
    ax.set_title("Current Inventory Versus Reorder Point")
    ax.set_xlabel("Product")
    ax.set_ylabel("Units")
    ax.legend()
    figure_paths.append(_save(fig, "current_inventory_vs_reorder_point.png"))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(compare["product_id"], compare["recommended_replenishment_quantity"], color="#2ca02c")
    ax.set_title("Recommended Replenishment Quantity By Product")
    ax.set_xlabel("Product")
    ax.set_ylabel("Units")
    ax.tick_params(axis="x", rotation=45)
    figure_paths.append(_save(fig, "recommended_replenishment_quantity_by_product.png"))

    return {
        "status": "success",
        "agent": "Visualization Agent",
        "figures_generated": len(figure_paths),
        "figure_paths": figure_paths,
    }
