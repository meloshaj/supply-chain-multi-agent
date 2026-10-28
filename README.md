# Supply Chain Multi-Agent Demand Forecasting & Inventory Optimization

This repository implements a modular, deterministic multi-agent decision-support pipeline for retail demand forecasting and inventory replenishment. It evaluates demand-inventory alignment using the **Walmart M5 retail benchmark** (Store `CA_1`, 12 SKUs across 6 merchandise categories, 365 daily observations = 4,380 panel rows).

The system coordinates six specialized agent services via **FastAPI** REST endpoints orchestrated live through a local **n8n** container.

No LangChain, external LLM API keys, cloud services, or custom n8n container modifications are required.

---

## Architecture Overview

```text
Walmart M5 Panel (4,380 rows) / Synthetic Benchmark
  │
  ▼
[ 1. Data Agent ] ──────────────► Schema validation & dataset profiling
  │
  ▼
[ 2. Preprocessing Agent ] ─────► Temporal feature engineering & 80/20 chronological split
  │
  ▼
[ 3. Forecasting Agent ] ───────► Random Forest vs. Seasonal-Naive Baseline + 9-Config Ablation
  │
  ▼
[ 4. Inventory Analysis Agent ] ► Lead-time demand, safety stock (Z=1.65), reorder points & risk
  │
  ▼
[ 5. Visualization Agent ] ─────► 8 pipeline charts (outputs/figures/) + 6 300 DPI figures (figures/)
  │
  ▼
[ 6. Recommendation Agent ] ────► Deterministic replenishment actions & priority assignments
  │
  ▼
Persisted Artifacts: CSV Tables, PNG Figures, Trained Joblib Model, JSON Payloads
```

---

## Agents & Core Responsibilities

1. **Data Agent (`/agents/data`):**
   - Validates CSV schema, non-null constraints, and data types (`date`, `product_id`, `category`, `actual_units_sold`, `inventory_level`, `price`, `promotion`, `supplier_lead_time`).
   - Generates summary statistics and data quality profiles.

2. **Preprocessing Agent (`/agents/preprocess`):**
   - Parses temporal dates, deduplicates records, and fills numeric gaps.
   - Computes calendar features (`day_of_week`, `month`, `week_of_year`, `is_weekend`), autoregressive lags (`lag_1`, `lag_7`), and rolling window statistics (`rolling_mean_7`, `rolling_std_7`).
   - Performs strict chronological train/test splits (80% train: 293 days / 3,516 rows; 20% test: 72 days / 864 rows).

3. **Forecasting Agent (`/agents/forecast`):**
   - Fits a `scikit-learn` Pipeline with one-hot categorical encoding and `RandomForestRegressor` (200 estimators).
   - Benchmarks predictions against a 7-day seasonal-naive baseline (`baseline_prediction = lag_7`).
   - Computes MAE, RMSE, safe MAPE, and WAPE across all 864 test observations.
   - Executes a comprehensive **9-configuration feature ablation sweep** (Table 5) and per-product evaluation.
   - Serializes the trained model artifact to `models/demand_forecasting_model.joblib`.

4. **Inventory Analysis Agent (`/agents/inventory`):**
   - Computes expected lead-time demand: $\mu_{\text{LTD}} = \hat{d} \times L$.
   - Computes lead-time demand standard deviation: $\sigma_{\text{LTD}} = \sqrt{L} \times \sigma_d$.
   - Calculates safety stock ($SS = Z \times \sigma_{\text{LTD}}$ at $Z=1.65$ for 95% service level) and reorder points ($ROP = \mu_{\text{LTD}} + SS$).
   - Determines recommended replenishment quantity and categorizes inventory status (*Severe Overstock*, *Potential Overstock*, *Healthy Inventory*, *Medium Stockout Risk*, *High Stockout Risk*).

5. **Visualization Agent (`/agents/visualize`):**
   - Generates 8 standard pipeline PNG charts saved to `outputs/figures/`.
   - Generates 6 high-resolution (300 DPI) publication-grade vector figures saved to `figures/` (error comparisons, ablation MAE, feature importance via joblib model inspection, and 3 distinct empirical case studies).

6. **Recommendation Agent (`/agents/recommend`):**
   - Evaluates inventory positions against heuristic operational rules.
   - Produces structured procurement decisions (e.g. urgent reorder, reduce purchase order, suspend replenishment / trigger promotion) with execution priority levels.

---

## Folder Structure

```text
supply-chain-multi-agent/
│── app.py                                   # FastAPI REST service
│── generate_dataset.py                      # Synthetic dataset generator for unit tests
│── requirements.txt                         # Python environment dependencies
│── results_summary.md                       # Comprehensive empirical benchmark findings
│── README.md                                # Project documentation
│
├── agents/                                  # Modular multi-agent Python services
│   ├── common.py                            # Shared schemas, validations & chronological splitting
│   ├── data_agent.py                        # Data profiling & validation
│   ├── preprocessing_agent.py               # Feature engineering & train/test splitting
│   ├── forecasting_agent.py                 # RF & baseline modeling, ablation suite
│   ├── inventory_agent.py                   # Safety stock, ROP & risk classification
│   ├── visualization_agent.py               # 180 DPI pipeline charts & 300 DPI publication figures
│   └── recommendation_agent.py              # Procurement & inventory recommendations
│
├── data/
│   ├── raw/
│   │   └── supply_chain_data.csv            # Standardized 12-SKU Walmart M5 panel (4,380 rows)
│   └── processed/
│       ├── processed_supply_chain_data.csv  # Cleaned dataset with engineered features
│       ├── train_supply_chain_data.csv      # First 293 days (3,516 rows)
│       └── test_supply_chain_data.csv       # Final 72 days (864 rows)
│
├── figures/                                 # 300 DPI publication figures
│   ├── fig1_error_metrics.png               # Baseline vs. RF metric comparisons
│   ├── fig2_ablation_mae.png                # 9-model feature ablation MAE
│   ├── fig3_feature_importance.png          # Random Forest feature importance ranking
│   ├── fig4_case_a_high_volume.png          # High-volume staple (FOODS_3_586)
│   ├── fig5_case_b_volatile.png             # Intermittent volatility (HOBBIES_1_155)
│   └── fig6_case_c_promotion.png            # Promotional demand spike (HOUSEHOLD_1_418)
│
├── models/
│   └── demand_forecasting_model.joblib      # Serialized scikit-learn model pipeline
│
├── n8n/
│   └── Supply Chain Multi-Agent Pipeline.json # Full 13-node n8n orchestration workflow
│
├── outputs/                                 # Live pipeline run outputs
│   ├── figures/                             # 8 standard operational PNG plots
│   ├── recommendations/
│   │   └── inventory_recommendations.csv    # Final SKU-level action items
│   └── tables/
│       ├── forecast_results.csv             # Day-by-day test forecasts & residuals
│       ├── inventory_risks.csv              # Safety stocks, ROP & risk levels
│       ├── product_test_metrics.csv         # Product-level MAE, RMSE, MAPE, WAPE
│       └── ablation_results.csv             # Full 9-configuration ablation matrix
│
├── scripts/
│   └── extract_m5_panel.py                  # Standalone M5 extraction ETL script
│
└── tests/                                   # Pytest test suite (15 passing tests)
    ├── conftest.py
    ├── test_api.py
    ├── test_data_agent.py
    ├── test_forecasting_agent.py
    ├── test_inventory_agent.py
    └── test_preprocessing_agent.py
```

---

## Installation & Setup

### 1. Python Environment Setup

Using Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Start the FastAPI Service

Launch the backend daemon on port `8000`:

```powershell
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- Interactive Swagger Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### 3. Deploy & Run n8n Container

Deploy the official n8n Docker container:

```bash
docker run -d --name n8n-local -p 5678:5678 -v n8n_data:/home/node/.n8n n8nio/n8n:latest
```

Open the n8n web interface at [http://localhost:5678](http://localhost:5678), import `n8n/Supply Chain Multi-Agent Pipeline.json`, and click **Test Workflow**.

> **Note on Docker Networking:** When n8n runs inside Docker, `localhost` points to the container itself. The workflow connects to your host machine's FastAPI server using:
> `http://host.docker.internal:8000`

---

## API Endpoints Reference

| Method | Endpoint | Description | Sample Payload |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service health status | None |
| `POST` | `/agents/data` | Validate & profile raw dataset | `{"dataset_path": "data/raw/supply_chain_data.csv"}` |
| `POST` | `/agents/preprocess` | Clean, lag, and split data (80/20) | `{"dataset_path": "data/raw/supply_chain_data.csv", "test_fraction": 0.2}` |
| `POST` | `/agents/forecast` | Train RF, benchmark, ablation sweep | `{"processed_dataset_path": "data/processed/processed_supply_chain_data.csv", "test_fraction": 0.2}` |
| `POST` | `/agents/inventory` | Compute lead-time demand, SS, ROP | `{"forecast_results_path": "outputs/tables/forecast_results.csv", "service_level_z": 1.65}` |
| `POST` | `/agents/visualize` | Render PNGs (180 & 300 DPI) | `{"processed_dataset_path": "data/processed/processed_supply_chain_data.csv", "forecast_results_path": "outputs/tables/forecast_results.csv", "inventory_results_path": "outputs/tables/inventory_risks.csv"}` |
| `POST` | `/agents/recommend` | Generate operational inventory actions | `{"inventory_results_path": "outputs/tables/inventory_risks.csv"}` |
| `POST` | `/pipeline/run` | Execute complete pipeline end-to-end | `{"dataset_path": "data/raw/supply_chain_data.csv", "test_fraction": 0.2, "service_level_z": 1.65}` |

---

## Empirical Benchmark Highlights (Walmart M5)

Evaluated across the 12-product, 72-day test holdout window ($N = 864$ observations):

* **Overall Forecast Improvement:**
  * **MAE:** 5.47 (Baseline) $\rightarrow$ **4.13** (Random Forest) — **24.51% reduction**
  * **RMSE:** 8.23 (Baseline) $\rightarrow$ **5.95** (Random Forest) — **27.71% reduction**
  * **MAPE:** 76.04% (Baseline) $\rightarrow$ **57.47%** (Random Forest) — **24.41% reduction**
  * **WAPE:** 43.98% (Baseline) $\rightarrow$ **33.20%** (Random Forest) — **24.51% reduction**
* **Ablation Key Insight:**
  * Removing prior-day inventory from the forecasting feature set slightly *improves* test MAE (-0.94%), confirming that lagged on-hand inventory is predictive noise for consumer demand generation, while remaining mathematically essential for downstream safety-stock and replenishment decision rules.
* **Three Empirical Case Studies:**
  * **Case Study A (`FOODS_3_586`):** High-volume staple with predictable velocity (WAPE 17.32%), but substantial stock buffer that requires high safety-stock allocation.
  * **Case Study B (`HOBBIES_1_155`):** Intermittent, volatile item ($\sigma / \mu = 0.62$, WAPE 49.19%) where seasonal-naive baseline severely lags underlying shifts.
  * **Case Study C (`HOUSEHOLD_1_418`):** Promotional demand spike where baseline error inflates by +70.4% during promotions (MAE 5.77 $\rightarrow$ 9.83), whereas Random Forest conditions on price and promo flags to maintain stable 5.97 MAE.

For complete empirical tables, ablation sweeps, and case study breakdowns, see [results_summary.md](results_summary.md).

---

## Running the Automated Test Suite

The test suite exercises data validation, chronological splitting, data leakage prevention, model training, non-negative predictions, inventory formulas, risk logic, and FastAPI REST endpoints:

```powershell
pytest
```

All 15 tests run against isolated temporary projects using dynamic synthetic datasets generated on the fly via [generate_dataset.py](generate_dataset.py).
