"""FastAPI application for the supply-chain multi-agent system."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.data_agent import run_data_agent
from agents.forecasting_agent import run_forecasting_agent
from agents.inventory_agent import run_inventory_agent
from agents.preprocessing_agent import run_preprocessing_agent
from agents.recommendation_agent import run_recommendation_agent
from agents.visualization_agent import run_visualization_agent


app = FastAPI(title="Supply Chain Multi-Agent API", version="1.0.0")


class DataRequest(BaseModel):
    dataset_path: str = "data/raw/supply_chain_data.csv"


class PreprocessRequest(BaseModel):
    dataset_path: str = "data/raw/supply_chain_data.csv"
    test_fraction: float = Field(default=0.2, gt=0, lt=1)


class ForecastRequest(BaseModel):
    processed_dataset_path: str = "data/processed/processed_supply_chain_data.csv"
    test_fraction: float = Field(default=0.2, gt=0, lt=1)


class InventoryRequest(BaseModel):
    forecast_results_path: str = "outputs/tables/forecast_results.csv"
    service_level_z: float = Field(default=1.65, ge=0)


class VisualizationRequest(BaseModel):
    processed_dataset_path: str = "data/processed/processed_supply_chain_data.csv"
    forecast_results_path: str = "outputs/tables/forecast_results.csv"
    inventory_results_path: str = "outputs/tables/inventory_risks.csv"


class RecommendationRequest(BaseModel):
    inventory_results_path: str = "outputs/tables/inventory_risks.csv"


class PipelineRequest(BaseModel):
    dataset_path: str = "data/raw/supply_chain_data.csv"
    test_fraction: float = Field(default=0.2, gt=0, lt=1)
    service_level_z: float = Field(default=1.65, ge=0)


def _agent_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health."""
    return {"status": "ok", "service": "supply-chain-multi-agent-api"}


@app.post("/agents/data")
def data_agent(request: DataRequest) -> dict[str, object]:
    try:
        return run_data_agent(request.dataset_path)
    except Exception as exc:
        raise _agent_error(exc) from exc


@app.post("/agents/preprocess")
def preprocessing_agent(request: PreprocessRequest) -> dict[str, object]:
    try:
        return run_preprocessing_agent(request.dataset_path, request.test_fraction)
    except Exception as exc:
        raise _agent_error(exc) from exc


@app.post("/agents/forecast")
def forecasting_agent(request: ForecastRequest) -> dict[str, object]:
    try:
        return run_forecasting_agent(request.processed_dataset_path, request.test_fraction)
    except Exception as exc:
        raise _agent_error(exc) from exc


@app.post("/agents/inventory")
def inventory_agent(request: InventoryRequest) -> dict[str, object]:
    try:
        return run_inventory_agent(request.forecast_results_path, request.service_level_z)
    except Exception as exc:
        raise _agent_error(exc) from exc


@app.post("/agents/visualize")
def visualization_agent(request: VisualizationRequest) -> dict[str, object]:
    try:
        return run_visualization_agent(
            request.processed_dataset_path,
            request.forecast_results_path,
            request.inventory_results_path,
        )
    except Exception as exc:
        raise _agent_error(exc) from exc


@app.post("/agents/recommend")
def recommendation_agent(request: RecommendationRequest) -> dict[str, object]:
    try:
        return run_recommendation_agent(request.inventory_results_path)
    except Exception as exc:
        raise _agent_error(exc) from exc


@app.post("/pipeline/run")
def run_pipeline(request: PipelineRequest) -> dict[str, object]:
    """Run the complete agent pipeline in n8n-compatible order."""
    response: dict[str, object] = {"status": "success", "pipeline": "Supply Chain Demand Forecasting"}
    steps = [
        ("data_agent", lambda: run_data_agent(request.dataset_path)),
        ("preprocessing_agent", lambda: run_preprocessing_agent(request.dataset_path, request.test_fraction)),
        ("forecasting_agent", lambda: run_forecasting_agent("data/processed/processed_supply_chain_data.csv", request.test_fraction)),
        ("inventory_agent", lambda: run_inventory_agent("outputs/tables/forecast_results.csv", request.service_level_z)),
        (
            "visualization_agent",
            lambda: run_visualization_agent(
                "data/processed/processed_supply_chain_data.csv",
                "outputs/tables/forecast_results.csv",
                "outputs/tables/inventory_risks.csv",
            ),
        ),
        ("recommendation_agent", lambda: run_recommendation_agent("outputs/tables/inventory_risks.csv")),
    ]
    for name, step in steps:
        try:
            response[name] = step()
        except Exception as exc:
            response["status"] = "failed"
            response["failed_agent"] = name
            response["error"] = str(exc)
            raise HTTPException(status_code=400, detail=response) from exc
    return response

