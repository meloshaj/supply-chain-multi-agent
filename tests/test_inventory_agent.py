from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents.inventory_agent import classify_inventory_risk, run_inventory_agent


def test_inventory_calculations_and_risk_classification(isolated_project: Path) -> None:
    forecast_path = Path("outputs/tables/forecast_results.csv")
    pd.DataFrame(
        [
            {
                "date": "2025-12-31",
                "product_id": "SKU-001",
                "category": "Grocery",
                "random_forest_prediction": 20,
                "supplier_lead_time": 5,
                "rolling_std_7": 4,
                "inventory_level": 30,
            },
            {
                "date": "2025-12-31",
                "product_id": "SKU-002",
                "category": "Home",
                "random_forest_prediction": 10,
                "supplier_lead_time": 3,
                "rolling_std_7": 2,
                "inventory_level": 500,
            },
        ]
    ).to_csv(forecast_path, index=False)
    result = run_inventory_agent(forecast_path.as_posix(), service_level_z=1.65)
    risks = pd.read_csv(result["inventory_results_path"])
    assert result["products_analyzed"] == 2
    assert (risks["recommended_replenishment_quantity"] >= 0).all()
    sku_001 = risks[risks["product_id"] == "SKU-001"].iloc[0]
    expected_safety_stock = 1.65 * 4 * (5**0.5)
    assert round(sku_001["lead_time_demand"], 4) == 100
    assert round(sku_001["safety_stock"], 4) == round(expected_safety_stock, 4)
    assert round(sku_001["reorder_point"], 4) == round(100 + expected_safety_stock, 4)
    assert round(sku_001["recommended_replenishment_quantity"], 4) == round(100 + expected_safety_stock - 30, 4)
    assert risks.loc[risks["product_id"] == "SKU-001", "risk_status"].iloc[0] == "High Stockout Risk"
    assert risks.loc[risks["product_id"] == "SKU-002", "risk_status"].iloc[0] == "Severe Overstock"


def test_risk_classification_thresholds() -> None:
    assert classify_inventory_risk(0, 0) == "Healthy Inventory"
    assert classify_inventory_risk(40, 100) == "High Stockout Risk"
    assert classify_inventory_risk(50, 100) == "Medium Stockout Risk"
    assert classify_inventory_risk(70, 100) == "Medium Stockout Risk"
    assert classify_inventory_risk(90, 100) == "Healthy Inventory"
    assert classify_inventory_risk(120, 100) == "Healthy Inventory"
    assert classify_inventory_risk(150, 100) == "Healthy Inventory"
    assert classify_inventory_risk(151, 100) == "Potential Overstock"
    assert classify_inventory_risk(200, 100) == "Potential Overstock"
    assert classify_inventory_risk(250, 100) == "Potential Overstock"
    assert classify_inventory_risk(251, 100) == "Severe Overstock"
    assert classify_inventory_risk(300, 100) == "Severe Overstock"
