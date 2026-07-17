"""Tests del validador leniente de la salida JSON."""

from __future__ import annotations

import pytest

from routerpolicy.schema.core import Mode, Registry
from routerpolicy.schema.validation import (
    DecisionValidationError,
    is_valid_decision_json,
    validate_decision_json,
)


def test_valid_decision_parses(golden_registry: Registry) -> None:
    dec = validate_decision_json('{"mode": "DIRECT", "model_id": "local-mini"}', golden_registry)
    assert dec.mode is Mode.DIRECT
    assert dec.model_id == "local-mini"


def test_valid_tolerates_whitespace(golden_registry: Registry) -> None:
    assert is_valid_decision_json('{"mode":"PLAN","model_id":"api-large"}', golden_registry)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"mode": "DIRECT"}',  # falta model_id
        '{"mode": "DIRECT", "model_id": "local-mini", "extra": 1}',  # campo extra
        '{"mode": "WRONG", "model_id": "local-mini"}',  # modo inválido
        '{"mode": "DIRECT", "model_id": "ghost"}',  # id no en registro
        '{"mode": "DIRECT", "model_id": 3}',  # tipo incorrecto
    ],
)
def test_invalid_decisions_rejected(text: str, golden_registry: Registry) -> None:
    assert not is_valid_decision_json(text, golden_registry)
    with pytest.raises(DecisionValidationError):
        validate_decision_json(text, golden_registry)
