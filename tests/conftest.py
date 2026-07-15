from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def isolated_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    for folder in [
        "data/raw",
        "data/processed",
        "models",
        "outputs/figures",
        "outputs/tables",
        "outputs/recommendations",
    ]:
        Path(folder).mkdir(parents=True, exist_ok=True)
    return tmp_path

