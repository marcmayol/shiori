"""Test de leakage sobre el dataset construido (DoD Fase 3).

Verifica que ningún task_id aparece en train y test a la vez. Se salta si el
dataset no está construido (CI no lo genera; la mecánica se prueba en
test_splits.py::test_end_to_end_no_leakage).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "data" / "dataset"


def _task_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            ids.add(str(json.loads(line)["task_id"]))
    return ids


def test_built_dataset_has_no_cross_split_leakage() -> None:
    train = DATASET / "train.jsonl"
    test = DATASET / "test.jsonl"
    if not (train.exists() and test.exists()):
        pytest.skip("dataset no construido (correr scripts/build_dataset.py)")
    train_ids = _task_ids(train)
    test_ids = _task_ids(test)
    assert train_ids, "train vacío"
    assert test_ids, "test vacío"
    assert train_ids.isdisjoint(test_ids), "leakage: task_id en train y test"
