"""Simulación económica: router (con escalado) vs cascada pura.

Reutiliza la verificación de la Fase 2 (dificultad = rango del modelo mínimo
suficiente en el pool real). Un modelo de rango r resuelve la tarea si r >= d.

- Router: apuesta primero por la elección de la política; si NO resuelve, escala
  al modelo más capaz. Coste = coste(apuesta) [+ coste(capaz) si escala].
- Cascada pura: prueba del más barato al más capaz hasta que uno resuelve. Coste
  = suma de los costes probados.

pass-rate = fracción de tareas que acaban resueltas; coste medio por tarea.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def router_cost_pass(
    bet_rank: int, difficulty: int, costs: Sequence[float], capable_rank: int
) -> tuple[float, bool]:
    """Coste y si resuelve, para el router con escalado al capaz."""
    if bet_rank >= difficulty:
        return costs[bet_rank - 1], True
    if bet_rank == capable_rank:  # ya apostó por el capaz y falló
        return costs[bet_rank - 1], False
    cost = costs[bet_rank - 1] + costs[capable_rank - 1]
    return cost, capable_rank >= difficulty


def cascade_cost_pass(difficulty: int, costs: Sequence[float]) -> tuple[float, bool]:
    """Coste y si resuelve, para la cascada pura (barato -> capaz)."""
    total = 0.0
    for rank in range(1, len(costs) + 1):
        total += costs[rank - 1]
        if rank >= difficulty:
            return total, True
    return total, False  # ningún modelo del pool basta


@dataclass(frozen=True)
class EconomyResult:
    n: int
    mean_cost: float
    pass_rate: float


def simulate_router(
    bets_and_difficulty: Sequence[tuple[int, int]], costs: Sequence[float]
) -> EconomyResult:
    capable_rank = len(costs)
    total_cost = 0.0
    passed = 0
    for bet_rank, d in bets_and_difficulty:
        c, ok = router_cost_pass(bet_rank, d, costs, capable_rank)
        total_cost += c
        passed += ok
    n = len(bets_and_difficulty)
    return EconomyResult(n=n, mean_cost=total_cost / n, pass_rate=passed / n)


def simulate_cascade(difficulties: Sequence[int], costs: Sequence[float]) -> EconomyResult:
    total_cost = 0.0
    passed = 0
    for d in difficulties:
        c, ok = cascade_cost_pass(d, costs)
        total_cost += c
        passed += ok
    n = len(difficulties)
    return EconomyResult(n=n, mean_cost=total_cost / n, pass_rate=passed / n)
