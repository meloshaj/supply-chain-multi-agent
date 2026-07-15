"""Deterministic recommendation agent."""

from __future__ import annotations

import pandas as pd

from agents.common import ensure_parent, relative_path, require_file


RECOMMENDATION_PATH = "outputs/recommendations/inventory_recommendations.csv"


def _recommend(row: pd.Series) -> tuple[str, str]:
    risk = row["risk_status"]
    if risk == "High Stockout Risk":
        action = "Urgent reorder and review supplier lead time"
        priority = "High"
    elif risk == "Medium Stockout Risk":
        action = "Schedule normal replenishment and review safety-stock policy"
        priority = "Medium"
    elif risk == "Potential Overstock":
        action = "Reduce the next purchase order and consider a targeted promotion"
        priority = "Medium"
    elif risk == "Severe Overstock":
        action = "Suspend immediate replenishment and consider a targeted promotion"
        priority = "High"
    else:
        action = "Maintain current inventory strategy"
        priority = "Low"
    return action, priority


def run_recommendation_agent(inventory_results_path: str) -> dict[str, object]:
    """Generate one deterministic operational recommendation per product."""
    path = require_file(inventory_results_path)
    df = pd.read_csv(path)
    required = [
        "product_id",
        "risk_status",
        "inventory_level",
        "predicted_daily_demand",
        "reorder_point",
        "recommended_replenishment_quantity",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required recommendation columns: {', '.join(missing)}")

    recommendations = df.copy()
    recs = recommendations.apply(_recommend, axis=1)
    recommendations["recommendation"] = [item[0] for item in recs]
    recommendations["priority"] = [item[1] for item in recs]
    output = recommendations[
        [
            "product_id",
            "risk_status",
            "inventory_level",
            "predicted_daily_demand",
            "reorder_point",
            "recommended_replenishment_quantity",
            "recommendation",
            "priority",
        ]
    ].copy()

    output_path = ensure_parent(RECOMMENDATION_PATH)
    output.to_csv(output_path, index=False)
    urgent = output["priority"].eq("High").sum()
    return {
        "status": "success",
        "agent": "Recommendation Agent",
        "recommendations_generated": int(len(output)),
        "urgent_actions": int(urgent),
        "recommendation_path": relative_path(output_path),
    }
