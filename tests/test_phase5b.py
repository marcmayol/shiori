"""Tests de calibración (prior a PLAN) y simulación económica (Fase 5b)."""

from __future__ import annotations

import pytest

from routerpolicy.evaluation.calibration import (
    CalibExample,
    decision_at_prior,
    sweep_plan_prior,
)
from routerpolicy.evaluation.economic import (
    cascade_cost_pass,
    router_cost_pass,
    simulate_cascade,
    simulate_router,
)
from routerpolicy.schema.core import Mode, RoutingDecision


def _sc(mode: Mode, mid: str, score: float) -> tuple[RoutingDecision, float]:
    return (RoutingDecision(mode=mode, model_id=mid), score)


def test_decision_at_prior_shifts_to_plan() -> None:
    scored = [
        _sc(Mode.DIRECT, "cheap", -0.5),
        _sc(Mode.PLAN, "capable", -1.0),
        _sc(Mode.TOOL_CALL, "cheap", -2.0),
    ]
    assert decision_at_prior(scored, 0.0).mode is Mode.DIRECT  # sin prior gana DIRECT
    # un prior >= 0.5 a PLAN lo pone por encima de DIRECT
    d = decision_at_prior(scored, 0.6)
    assert d.mode is Mode.PLAN
    assert d.model_id == "capable"


def test_sweep_increases_plan_recall() -> None:
    # dos ejemplos gold PLAN donde DIRECT gana por poco
    scored = [_sc(Mode.DIRECT, "cheap", -0.5), _sc(Mode.PLAN, "capable", -0.7)]
    ex = CalibExample(
        scored=scored, gold_mode=Mode.PLAN, cost_by_model_id={"cheap": 1, "capable": 6}
    )
    points = sweep_plan_prior([ex, ex], priors=[0.0, 0.5])
    assert points[0].plan_recall == 0.0  # sin prior, se predice DIRECT
    assert points[1].plan_recall == 1.0  # con prior 0.5, se predice PLAN
    assert points[1].mean_cost > points[0].mean_cost  # PLAN enruta al caro


def test_router_cost_pass() -> None:
    costs = [1.0, 3.0, 6.0]  # rangos 1,2,3
    # apuesta correcta (rango 2 para dificultad 2): resuelve, coste 3
    assert router_cost_pass(2, 2, costs, capable_rank=3) == (3.0, True)
    # apuesta baja (rango 1, dificultad 2): falla, escala al capaz -> 1+6, resuelve
    assert router_cost_pass(1, 2, costs, capable_rank=3) == (7.0, True)
    # apuesta al capaz y dificultad 4 (nadie basta): coste 6, no resuelve
    assert router_cost_pass(3, 4, costs, capable_rank=3) == (6.0, False)


def test_cascade_cost_pass() -> None:
    costs = [1.0, 3.0, 6.0]
    assert cascade_cost_pass(1, costs) == (1.0, True)  # rango1 basta
    assert cascade_cost_pass(2, costs) == (4.0, True)  # 1+3
    assert cascade_cost_pass(4, costs) == (10.0, False)  # prueba todos, nadie basta


def test_simulate_router_vs_cascade() -> None:
    costs = [1.0, 3.0, 6.0]
    # tareas fáciles (dificultad 1): router que apuesta bien (rango1) es barato
    bets = [(1, 1), (1, 1)]
    r = simulate_router(bets, costs)
    c = simulate_cascade([1, 1], costs)
    assert r.pass_rate == 1.0 and c.pass_rate == 1.0
    assert r.mean_cost == pytest.approx(1.0)
    assert c.mean_cost == pytest.approx(1.0)
