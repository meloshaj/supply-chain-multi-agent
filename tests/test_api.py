from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app import app
from generate_dataset import generate_supply_chain_dataset


def test_api_health_endpoint(isolated_project: Path) -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "supply-chain-multi-agent-api"}


def test_full_pipeline_endpoint(isolated_project: Path) -> None:
    generate_supply_chain_dataset()
    client = TestClient(app)
    response = client.post(
        "/pipeline/run",
        json={"dataset_path": "data/raw/supply_chain_data.csv", "test_fraction": 0.2, "service_level_z": 1.65},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["recommendation_agent"]["recommendations_generated"] == 12
    assert Path("outputs/tables/forecast_results.csv").exists()
    assert Path("outputs/tables/inventory_risks.csv").exists()
    assert Path("outputs/recommendations/inventory_recommendations.csv").exists()
    assert len(list(Path("outputs/figures").glob("*.png"))) == 8

