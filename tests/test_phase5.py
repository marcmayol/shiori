"""Tests de los análisis de Fase 5 (baselines, confusión, suficiencia)."""

from __future__ import annotations

import random

from routerpolicy.evaluation.analysis import (
    INVALID,
    capability_from_tags,
    is_sufficient,
    mode_confusion,
)
from routerpolicy.evaluation.baselines import cascade_baseline
from routerpolicy.registry.synthetic import generate_pool
from routerpolicy.schema.core import Locality, Mode, ModelSpec, Registry, RoutingDecision


def test_capability_from_tags_inverts_encoding() -> None:
    # coincide con el encoding de generate_pool
    for cap in range(5):
        pool = generate_pool(random.Random(5), capabilities=[cap])
        assert capability_from_tags(pool[0].spec.tags) == cap


def test_cascade_baseline_picks_cheapest() -> None:
    reg = Registry(
        models=(
            _m("a", 5.0),
            _m("b", 2.0),
            _m("c", 9.0),
        )
    )
    dec = cascade_baseline(reg)
    assert dec.model_id == "b"
    assert dec.mode is Mode.DIRECT


def _m(model_id: str, cost: float, tags: tuple[str, ...] = ("code",)) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        tags=tags,
        context_window=8192,
        cost=cost,
        locality=Locality.LOCAL,
        supports_tools=False,
    )


def test_mode_confusion_counts() -> None:
    golds = [
        RoutingDecision(mode=Mode.PLAN, model_id="x"),
        RoutingDecision(mode=Mode.PLAN, model_id="y"),
        RoutingDecision(mode=Mode.DIRECT, model_id="z"),
    ]
    preds = [
        RoutingDecision(mode=Mode.DIRECT, model_id="x"),  # PLAN -> DIRECT
        None,  # PLAN -> INVALID
        RoutingDecision(mode=Mode.DIRECT, model_id="z"),  # DIRECT -> DIRECT
    ]
    matrix = mode_confusion(golds, preds)
    assert matrix[Mode.PLAN][Mode.DIRECT.value] == 1
    assert matrix[Mode.PLAN][INVALID] == 1
    assert matrix[Mode.DIRECT][Mode.DIRECT.value] == 1


def test_is_sufficient() -> None:
    # capacidad recuperada de tags vs dificultad
    strong = _m("s", 9.0, tags=("code", "reasoning", "planning"))  # cap 3
    weak = _m("w", 1.0, tags=("code",))  # cap 1
    assert is_sufficient(strong, difficulty=3)
    assert not is_sufficient(weak, difficulty=3)
    assert is_sufficient(weak, difficulty=1)
    assert not is_sufficient(None, difficulty=1)
