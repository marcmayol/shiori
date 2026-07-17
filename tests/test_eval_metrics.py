"""Tests de las métricas de evaluación (Fase 5)."""

from __future__ import annotations

import pytest

from routerpolicy.evaluation.metrics import compute_metrics
from routerpolicy.schema.core import Mode, RoutingDecision


def _d(mode: Mode, model_id: str) -> RoutingDecision:
    return RoutingDecision(mode=mode, model_id=model_id)


def test_all_correct() -> None:
    gold = [_d(Mode.DIRECT, "a"), _d(Mode.PLAN, "b")]
    m = compute_metrics(gold, list(gold))
    assert m.exact_match == 1.0
    assert m.mode_accuracy == 1.0
    assert m.model_id_accuracy == 1.0
    assert m.invalid_rate == 0.0


def test_partial_and_invalid() -> None:
    gold = [_d(Mode.DIRECT, "a"), _d(Mode.PLAN, "b"), _d(Mode.TOOL_CALL, "c")]
    pred = [
        _d(Mode.DIRECT, "a"),  # exacto
        _d(Mode.PLAN, "z"),  # modo ok, id mal
        None,  # inválido
    ]
    m = compute_metrics(gold, pred)
    assert m.n == 3
    assert m.exact_match == pytest.approx(1 / 3)
    assert m.mode_accuracy == pytest.approx(2 / 3)
    assert m.model_id_accuracy == pytest.approx(1 / 3)
    assert m.invalid_rate == pytest.approx(1 / 3)
    assert m.mode_accuracy_by_class[Mode.DIRECT] == 1.0
    assert m.mode_accuracy_by_class[Mode.TOOL_CALL] == 0.0  # era inválido


def test_length_and_empty_validation() -> None:
    with pytest.raises(ValueError, match="misma longitud"):
        compute_metrics([_d(Mode.DIRECT, "a")], [])
    with pytest.raises(ValueError, match="no hay ejemplos"):
        compute_metrics([], [])
