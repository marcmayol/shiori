"""Calibración de la decisión de modo por logprobs (prior a PLAN).

Bajo el constraint, la decisión de modo es el argmax del logprob por modo. Un
PRIOR λ hacia PLAN suma λ al score de PLAN antes del argmax: subirlo aumenta el
recall de PLAN a costa de falsos positivos (DIRECT/TOOL -> PLAN) y de coste
medio (PLAN enruta al modelo más capaz). Todo puro y testeable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from routerpolicy.schema.core import Mode, RoutingDecision

# Candidatos puntuados de un ejemplo: [(decisión, logprob), ...]
ScoredCandidates = Sequence[tuple[RoutingDecision, float]]


def decision_at_prior(scored: ScoredCandidates, plan_prior: float) -> RoutingDecision:
    """Decisión (modo, model_id) tras sumar `plan_prior` al score de PLAN."""
    best: dict[Mode, tuple[float, str]] = {}
    for dec, score in scored:
        if dec.mode not in best or score > best[dec.mode][0]:
            best[dec.mode] = (score, dec.model_id)
    adjusted = {m: s + (plan_prior if m is Mode.PLAN else 0.0) for m, (s, _) in best.items()}
    chosen_mode = max(adjusted, key=lambda m: adjusted[m])
    return RoutingDecision(mode=chosen_mode, model_id=best[chosen_mode][1])


@dataclass(frozen=True)
class CalibrationPoint:
    """Un punto del barrido de prior."""

    plan_prior: float
    plan_recall: float
    direct_recall: float
    tool_recall: float
    mean_cost: float
    mode_accuracy: float


@dataclass(frozen=True)
class CalibExample:
    """Un ejemplo para calibrar: candidatos puntuados, modo oro, costes del pool."""

    scored: ScoredCandidates
    gold_mode: Mode
    cost_by_model_id: dict[str, float]


def _recall(golds: list[Mode], preds: list[Mode], target: Mode) -> float:
    total = sum(1 for g in golds if g is target)
    if total == 0:
        return float("nan")
    hit = sum(1 for g, p in zip(golds, preds, strict=True) if g is target and p is target)
    return hit / total


def sweep_plan_prior(
    examples: Sequence[CalibExample], priors: Sequence[float]
) -> list[CalibrationPoint]:
    """Evalúa cada prior: recall por clase, coste medio y exactitud de modo."""
    golds = [ex.gold_mode for ex in examples]
    out: list[CalibrationPoint] = []
    for prior in priors:
        decisions = [decision_at_prior(ex.scored, prior) for ex in examples]
        preds = [d.mode for d in decisions]
        costs = [
            ex.cost_by_model_id.get(d.model_id, 0.0)
            for ex, d in zip(examples, decisions, strict=True)
        ]
        mode_acc = sum(1 for g, p in zip(golds, preds, strict=True) if g is p) / len(golds)
        out.append(
            CalibrationPoint(
                plan_prior=prior,
                plan_recall=_recall(golds, preds, Mode.PLAN),
                direct_recall=_recall(golds, preds, Mode.DIRECT),
                tool_recall=_recall(golds, preds, Mode.TOOL_CALL),
                mean_cost=sum(costs) / len(costs),
                mode_accuracy=mode_acc,
            )
        )
    return out
