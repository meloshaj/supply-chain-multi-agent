# Supply Chain Multi-Agent Demand Forecasting

This project is a complete local multi-agent system for retail supply-chain demand forecasting and inventory optimization. It uses synthetic data for demonstration, deterministic Python agents for every processing stage, and FastAPI endpoints that can be called sequentially from n8n.

No LangChain, LLM, cloud service, or n8n container modification is required.

## Architecture

```text
Synthetic CSV
  -> Data Agent
  -> Preprocessing Agent
  -> Forecasting Agent
  -> Inventory Analysis Agent
  -> Visualization Agent
  -> Recommendation Agent
  -> CSV outputs, PNG figures, JSON API responses
```

## Agents

- Data Agent: loads CSV files, validates required columns, and reports dataset quality.
- Preprocessing Agent: cleans dates, removes duplicates, fills numeric gaps, creates lag and rolling time-series features, and writes train/test splits.
- Forecasting Agent: compares a lag-7 baseline with a RandomForestRegressor pipeline using one-hot encoded categorical features.
- Inventory Analysis Agent: calculates lead-time demand, safety stock, reorder points, recommended replenishment quantities, and risk categories.
- Visualization Agent: creates report-ready Matplotlib PNG charts under `outputs/figures/`.
- Recommendation Agent: generates deterministic operational recommendations from inventory risk rules.

## Folder Structure

```text
supply-chain-multi-agent/
|-- app.py
|-- generate_dataset.py
|-- requirements.txt
|-- README.md
|-- agents/
|   |-- data_agent.py
|   |-- preprocessing_agent.py
|   |-- forecasting_agent.py
|   |-- inventory_agent.py
|   |-- visualization_agent.py
|   |-- recommendation_agent.py
|-- data/
|   |-- raw/
|   |-- processed/
|-- models/
|-- outputs/
|   |-- figures/
|   |-- tables/
|   |-- recommendations/
|-- tests/
```

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Generate the synthetic dataset:

```powershell
python generate_dataset.py
```

Start FastAPI:

```powershell
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

n8n URL, if running locally in Docker:

```text
http://localhost:5678
```

When n8n runs inside Docker, `localhost` points to the n8n container, not the Windows host. Use this host address from n8n HTTP Request nodes:

```text
http://host.docker.internal:8000
```

## API Endpoints

Health:

```http
GET /health
```

Data Agent:

```http
POST /agents/data
Content-Type: application/json

{
  "dataset_path": "data/raw/supply_chain_data.csv"
}
```

Preprocessing Agent:

```http
POST /agents/preprocess
Content-Type: application/json

{
  "dataset_path": "data/raw/supply_chain_data.csv",
  "test_fraction": 0.2
}
```

Forecasting Agent:

```http
POST /agents/forecast
Content-Type: application/json

{
  "processed_dataset_path": "data/processed/processed_supply_chain_data.csv",
  "test_fraction": 0.2
}
```

Inventory Analysis Agent:

```http
POST /agents/inventory
Content-Type: application/json

{
  "forecast_results_path": "outputs/tables/forecast_results.csv",
  "service_level_z": 1.65
}
```

Visualization Agent:

```http
POST /agents/visualize
Content-Type: application/json

{
  "processed_dataset_path": "data/processed/processed_supply_chain_data.csv",
  "forecast_results_path": "outputs/tables/forecast_results.csv",
  "inventory_results_path": "outputs/tables/inventory_risks.csv"
}
```

Recommendation Agent:

```http
POST /agents/recommend
Content-Type: application/json

{
  "inventory_results_path": "outputs/tables/inventory_risks.csv"
}
```

Full pipeline:

```http
POST /pipeline/run
Content-Type: application/json

{
  "dataset_path": "data/raw/supply_chain_data.csv",
  "test_fraction": 0.2,
  "service_level_z": 1.65
}
```

## n8n Workflow Setup

Recommended workflow:

```text
Manual Trigger
-> Edit Fields: Project Configuration
-> HTTP Request: Health Check
-> HTTP Request: Data Agent
-> HTTP Request: Preprocessing Agent
-> HTTP Request: Forecasting Agent
-> HTTP Request: Inventory Agent
-> HTTP Request: Visualization Agent
-> HTTP Request: Recommendation Agent
```

Use these values in `Edit Fields: Project Configuration`:

```json
{
  "base_url": "http://host.docker.internal:8000",
  "dataset_path": "data/raw/supply_chain_data.csv",
  "processed_dataset_path": "data/processed/processed_supply_chain_data.csv",
  "forecast_results_path": "outputs/tables/forecast_results.csv",
  "inventory_results_path": "outputs/tables/inventory_risks.csv",
  "test_fraction": 0.2,
  "service_level_z": 1.65
}
```

HTTP Request node details:

| Node | Method | URL | JSON body |
| --- | --- | --- | --- |
| Health Check | GET | `{{$json.base_url}}/health` | none |
| Data Agent | POST | `{{$json.base_url}}/agents/data` | `{"dataset_path":"{{$json.dataset_path}}"}` |
| Preprocessing Agent | POST | `{{$json.base_url}}/agents/preprocess` | `{"dataset_path":"{{$json.dataset_path}}","test_fraction":{{$json.test_fraction}}}` |
| Forecasting Agent | POST | `{{$json.base_url}}/agents/forecast` | `{"processed_dataset_path":"{{$json.processed_dataset_path}}","test_fraction":{{$json.test_fraction}}}` |
| Inventory Agent | POST | `{{$json.base_url}}/agents/inventory` | `{"forecast_results_path":"{{$json.forecast_results_path}}","service_level_z":{{$json.service_level_z}}}` |
| Visualization Agent | POST | `{{$json.base_url}}/agents/visualize` | `{"processed_dataset_path":"{{$json.processed_dataset_path}}","forecast_results_path":"{{$json.forecast_results_path}}","inventory_results_path":"{{$json.inventory_results_path}}"}` |
| Recommendation Agent | POST | `{{$json.base_url}}/agents/recommend` | `{"inventory_results_path":"{{$json.inventory_results_path}}"}` |

Simpler alternative:

```text
Manual Trigger
-> Edit Fields
-> HTTP Request: POST /pipeline/run
```

Pipeline HTTP Request:

- Method: `POST`
- URL: `{{$json.base_url}}/pipeline/run`
- JSON body:

```json
{
  "dataset_path": "data/raw/supply_chain_data.csv",
  "test_fraction": 0.2,
  "service_level_z": 1.65
}
```

## Outputs

- `data/raw/supply_chain_data.csv`: synthetic source data.
- `data/processed/processed_supply_chain_data.csv`: cleaned and feature-engineered data.
- `data/processed/train_supply_chain_data.csv`: chronological training split.
- `data/processed/test_supply_chain_data.csv`: chronological testing split.
- `models/demand_forecasting_model.joblib`: trained scikit-learn pipeline.
- `outputs/tables/forecast_results.csv`: actual demand, baseline predictions, Random Forest predictions, and errors.
- `outputs/tables/inventory_risks.csv`: reorder points, recommended replenishment quantities, and risk categories.
- `outputs/recommendations/inventory_recommendations.csv`: one recommendation per product.
- `outputs/figures/*.png`: report-ready charts.

## Tests

```powershell
pytest
```

The tests cover missing files, missing columns, duplicate handling, chronological splitting, lag feature leakage checks, forecasting outputs, non-negative predictions, inventory calculations, risk classification, API health, and the full pipeline endpoint.

## Limitations

- The dataset is synthetic and intended for demonstration, not production planning.
- The model is deterministic but intentionally simple enough to run locally.
- Inventory rules are transparent business heuristics and should be calibrated before real operational use.
- n8n orchestrates HTTP calls only; all forecasting and calculations run inside the local Python API.
