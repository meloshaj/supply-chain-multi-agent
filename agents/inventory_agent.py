"""Inventory risk analysis agent."""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.common import ensure_parent, relative_path, require_file


INVENTORY_RESULTS_PATH = "outputs/tables/inventory_risks.csv"


def classify_inventory_risk(inventory_level: float, reorder_point: float) -> str:
    """Classify stock risk using inventory-to-reorder-point ratio."""
    if reorder_point <= 0:
        return "Healthy Inventory"
    ratio = inventory_level / reorder_point
    if ratio < 0.5:
        return "High Stockout Risk"
    if ratio < 0.9:
        return "Medium Stockout Risk"
    if ratio <= 1.5:
        return "Healthy Inventory"
    if ratio <= 2.5:
        return "Potential Overstock"
    return "Severe Overstock"


def run_inventory_agent(forecast_results_path: str, service_level_z: float = 1.65) -> dict[str, object]:
    """Calculate reorder points and inventory risk from latest forecasts."""
    if service_level_z < 0:
        raise ValueError("service_level_z must be non-negative")
    path = require_file(forecast_results_path)
    df = pd.read_csv(path)
    required = ["date", "product_id", "random_forest_prediction", "supplier_lead_time", "rolling_std_7", "inventory_level"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required inventory columns: {', '.join(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    latest = df.sort_values(["product_id", "date"]).groupby("product_id", as_index=False).tail(1).copy()
    latest["predicted_daily_demand"] = latest["random_forest_prediction"].clip(lower=0)
    latest["lead_time_demand"] = latest["predicted_daily_demand"] * latest["supplier_lead_time"]
    latest["safety_stock"] = service_level_z * latest["rolling_std_7"].fillna(0).clip(lower=0) * np.sqrt(latest["supplier_lead_time"])
    latest["reorder_point"] = latest["lead_time_demand"] + latest["safety_stock"]
    latest["recommended_replenishment_quantity"] = (latest["reorder_point"] - latest["inventory_level"]).clip(lower=0)
    latest["risk_status"] = latest.apply(lambda row: classify_inventory_risk(row["inventory_level"], row["reorder_point"]), axis=1)
    latest["inventory_to_reorder_ratio"] = np.where(
        latest["reorder_point"] > 0,
        latest["inventory_level"] / latest["reorder_point"],
        np.inf,
    )

    output_columns = [
        "product_id",
        "category",
        "date",
        "inventory_level",
        "supplier_lead_time",
        "predicted_daily_demand",
        "lead_time_demand",
        "rolling_std_7",
        "safety_stock",
        "reorder_point",
        "recommended_replenishment_quantity",
        "inventory_to_reorder_ratio",
        "risk_status",
    ]
    output = latest[output_columns].copy()
    for column in output.select_dtypes(include=["number"]).columns:
        output[column] = output[column].round(4)
    output["date"] = pd.to_datetime(output["date"]).dt.date.astype(str)

    output_path = ensure_parent(INVENTORY_RESULTS_PATH)
    output.to_csv(output_path, index=False)
    counts = output["risk_status"].value_counts().to_dict()
    return {
        "status": "success",
        "agent": "Inventory Analysis Agent",
        "products_analyzed": int(len(output)),
        "high_stockout_risk": int(counts.get("High Stockout Risk", 0)),
        "medium_stockout_risk": int(counts.get("Medium Stockout Risk", 0)),
        "healthy_inventory": int(counts.get("Healthy Inventory", 0)),
        "potential_overstock": int(counts.get("Potential Overstock", 0)),
        "severe_overstock": int(counts.get("Severe Overstock", 0)),
        "inventory_results_path": relative_path(output_path),
    }
