"""Tests de la augmentación y el recálculo de etiqueta por pool."""

from __future__ import annotations

import random

from routerpolicy.dataset.augment import (
    BaseTask,
    augment_task,
    difficulty_from_sufficiency,
    label_for_pool,
    mode_difficulty,
)
from routerpolicy.registry.synthetic import SyntheticModel
from routerpolicy.schema.core import Locality, Mode, ModelSpec

POOL_IDS = ["cheap", "mid", "capable"]


def _sm(model_id: str, capability: int, cost: float, tools: bool = False) -> SyntheticModel:
    return SyntheticModel(
        spec=ModelSpec(
            id=model_id,
            tags=("code",),
            context_window=8192,
            cost=cost,
            locality=Locality.LOCAL,
            supports_tools=tools,
        ),
        capability=capability,
    )


def test_difficulty_from_sufficiency() -> None:
    assert difficulty_from_sufficiency("cheap", POOL_IDS) == 1
    assert difficulty_from_sufficiency("mid", POOL_IDS) == 2
    assert difficulty_from_sufficiency("capable", POOL_IDS) == 3
    assert difficulty_from_sufficiency(None, POOL_IDS) == 4  # ninguno suficiente


def test_mode_difficulty() -> None:
    assert mode_difficulty(Mode.DIRECT) == 1
    assert mode_difficulty(Mode.PLAN) == 3


def test_label_picks_cheapest_sufficient() -> None:
    task = BaseTask("t", "src", "p", Mode.DIRECT, difficulty=2, requires_tools=False)
    pool = [_sm("weak", 1, 2.0), _sm("ok-cheap", 2, 5.0), _sm("ok-pricey", 3, 9.0)]
    dec = label_for_pool(task, pool)
    assert dec.mode is Mode.DIRECT
    assert dec.model_id == "ok-cheap"  # capacidad>=2 y el más barato de los que bastan


def test_label_none_sufficient_falls_back_to_most_capable() -> None:
    task = BaseTask("t", "src", "p", Mode.DIRECT, difficulty=4, requires_tools=False)
    pool = [_sm("a", 1, 2.0), _sm("b", 2, 3.0)]  # ninguno llega a 4
    dec = label_for_pool(task, pool)
    assert dec.model_id == "b"  # el más capaz


def test_label_requires_tools() -> None:
    task = BaseTask("t", "src", "p", Mode.TOOL_CALL, difficulty=1, requires_tools=True)
    # el más barato suficiente NO soporta tools; hay uno más caro que sí
    pool = [_sm("cheap-notools", 2, 1.0, tools=False), _sm("dear-tools", 2, 5.0, tools=True)]
    dec = label_for_pool(task, pool)
    assert dec.mode is Mode.TOOL_CALL
    assert dec.model_id == "dear-tools"  # el único que soporta tools


def test_augment_task_produces_factor_examples() -> None:
    task = BaseTask("t", "code", "prompt text", Mode.DIRECT, difficulty=2, requires_tools=False)
    examples = augment_task(task, random.Random(0), factor=6)
    assert len(examples) == 6
    # cada ejemplo referencia un registro válido y la decisión apunta a un id del pool
    for ex in examples:
        assert ex.registry.contains(ex.decision.model_id)
        assert ex.decision.mode is Mode.DIRECT
        assert ex.task_id == "t"


def test_augment_respects_signature_filter() -> None:
    task = BaseTask("t", "code", "p", Mode.DIRECT, difficulty=1, requires_tools=False)
    # solo pools de tamaño par
    examples = augment_task(
        task, random.Random(1), factor=5, allow_signature=lambda sig: sig[0] % 2 == 0
    )
    assert len(examples) == 5
    assert all(ex.n_models % 2 == 0 for ex in examples)
