"""Métricas de routing (Fase 5), puras y testeables.

Operan sobre listas de decisiones predichas vs oro, más la salida cruda para la
tasa de inválidos SIN constraint.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from routerpolicy.schema.core import Mode, RoutingDecision


@dataclass(frozen=True)
class EvalMetrics:
    """Resultado de la evaluación de un conjunto."""

    n: int
    exact_match: float  # {mode, model_id} exactos
    mode_accuracy: float
    model_id_accuracy: float
    invalid_rate: float  # fracción de salidas no parseables/ inválidas
    mode_accuracy_by_class: dict[Mode, float] = field(default_factory=dict)


def compute_metrics(
    gold: Sequence[RoutingDecision],
    predicted: Sequence[RoutingDecision | None],
) -> EvalMetrics:
    """Calcula métricas. `predicted[i] is None` = salida inválida (no parseable)."""
    if len(gold) != len(predicted):
        raise ValueError("gold y predicted deben tener la misma longitud")
    n = len(gold)
    if n == 0:
        raise ValueError("no hay ejemplos que evaluar")

    exact = 0
    mode_ok = 0
    model_ok = 0
    invalid = 0
    per_class_total: dict[Mode, int] = {}
    per_class_ok: dict[Mode, int] = {}

    for g, p in zip(gold, predicted, strict=True):
        per_class_total[g.mode] = per_class_total.get(g.mode, 0) + 1
        if p is None:
            invalid += 1
            continue
        if p.mode == g.mode:
            mode_ok += 1
            per_class_ok[g.mode] = per_class_ok.get(g.mode, 0) + 1
        if p.model_id == g.model_id:
            model_ok += 1
        if p.mode == g.mode and p.model_id == g.model_id:
            exact += 1

    by_class = {
        mode: per_class_ok.get(mode, 0) / total
        for mode, total in per_class_total.items()
        if total > 0
    }
    return EvalMetrics(
        n=n,
        exact_match=exact / n,
        mode_accuracy=mode_ok / n,
        model_id_accuracy=model_ok / n,
        invalid_rate=invalid / n,
        mode_accuracy_by_class=by_class,
    )
