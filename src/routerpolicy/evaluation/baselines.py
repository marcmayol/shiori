"""Baselines de Fase 5 (mismo test set), como decisiones {mode, model_id}.

- cascade_baseline: cascada pura sin política — siempre el más barato del pool
  (modo DIRECT por defecto). Es el suelo de coste (regret ~0) pero ignora la
  suficiencia y el modo.
- Los baselines basados en modelo (base zero-shot, API zero-shot) se ejecutan en
  el script con el mismo prompt y decoding; aquí solo el heurístico puro.
"""

from __future__ import annotations

from routerpolicy.schema.core import Mode, Registry, RoutingDecision


def cascade_baseline(registry: Registry) -> RoutingDecision:
    """Elige siempre el modelo más barato del pool (modo DIRECT)."""
    cheapest = min(registry.models, key=lambda m: (m.cost, m.id))
    return RoutingDecision(mode=Mode.DIRECT, model_id=cheapest.id)
