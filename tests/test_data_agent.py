from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agents.data_agent import run_data_agent
from generate_dataset import generate_supply_chain_dataset


def test_missing_dataset_file(isolated_project: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_data_agent("data/raw/missing.csv")


def test_missing_required_columns(isolated_project: Path) -> None:
    path = Path("data/raw/bad.csv")
    pd.DataFrame({"date": ["2025-01-01"], "product_id": ["SKU-001"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        run_data_agent(path.as_posix())


def test_data_agent_loads_generated_dataset(isolated_project: Path) -> None:
    path = generate_supply_chain_dataset()
    result = run_data_agent(path)
    assert result["rows_loaded"] >= 3650
    assert result["products_found"] == 12
    assert result["duplicate_rows"] == 0

