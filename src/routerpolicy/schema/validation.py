"""Validadores del JSON de salida de la política.

Versión LENIENTE del parseo (tolera variaciones de espaciado), usada en la
evaluación SIN constraint para medir la tasa de salida inválida. El decoding
constreñido (inference/constraints) es la versión estricta y canónica.
"""

from __future__ import annotations

import json
from typing import Any

from routerpolicy.schema.core import Mode, Registry, RoutingDecision


class DecisionValidationError(ValueError):
    """La salida no es una decisión de routing válida para el registro dado."""


def validate_decision_json(text: str, registry: Registry) -> RoutingDecision:
    """Parsea y valida una salida contra el registro de su propio ejemplo.

    Comprueba: JSON de objeto bien formado, `mode` en el enum, `model_id`
    presente en el registro, y que no haya campos extra. Lanza
    `DecisionValidationError` con un mensaje claro si algo falla.
    """
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DecisionValidationError(f"JSON malformado: {exc}") from exc

    if not isinstance(parsed, dict):
        raise DecisionValidationError("la salida debe ser un objeto JSON")

    keys = set(parsed.keys())
    expected = {"mode", "model_id"}
    if keys != expected:
        raise DecisionValidationError(
            f"claves inesperadas: {sorted(keys)} (se esperaban {sorted(expected)})"
        )

    mode_raw = parsed["mode"]
    if not isinstance(mode_raw, str) or mode_raw not in Mode.__members__:
        valid = ", ".join(Mode.__members__)
        raise DecisionValidationError(f"mode inválido: {mode_raw!r} (válidos: {valid})")

    model_id = parsed["model_id"]
    if not isinstance(model_id, str):
        raise DecisionValidationError(f"model_id debe ser string: {model_id!r}")
    if not registry.contains(model_id):
        raise DecisionValidationError(
            f"model_id {model_id!r} no está en el registro {list(registry.model_ids)}"
        )

    return RoutingDecision(mode=Mode[mode_raw], model_id=model_id)


def is_valid_decision_json(text: str, registry: Registry) -> bool:
    """Variante booleana de `validate_decision_json`."""
    try:
        validate_decision_json(text, registry)
    except DecisionValidationError:
        return False
    return True
