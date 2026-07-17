"""Proyección y contabilidad de coste del etiquetado.

Dos caminos:
- `cost_of_results`: coste REAL medido a partir de los intentos ya ejecutados
  (tokens registrados por la cascada) — se usa tras correr la muestra.
- `project_cost`: proyección a priori desde una muestra de tareas y supuestos
  de escalado, para la estimación obligatoria de la sección 3 ANTES del run.

Los modelos locales tienen coste 0; solo el tier de API cuesta.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from routerpolicy.harness.cascade import CascadeResult


@dataclass(frozen=True)
class PriceSpec:
    """Precio de un modelo en USD por millón de tokens."""

    model_id: str
    input_per_mtok: float
    output_per_mtok: float
    is_local: bool = False

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        if self.is_local:
            return 0.0
        return (
            prompt_tokens / 1_000_000 * self.input_per_mtok
            + completion_tokens / 1_000_000 * self.output_per_mtok
        )


class PricingTable:
    """Tabla de precios indexada por model_id."""

    def __init__(self, specs: Iterable[PriceSpec]) -> None:
        self._by_id = {s.model_id: s for s in specs}

    def get(self, model_id: str) -> PriceSpec | None:
        return self._by_id.get(model_id)

    def cost_of(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        spec = self._by_id.get(model_id)
        if spec is None:
            raise KeyError(f"sin precio para el modelo {model_id!r}")
        return spec.cost(prompt_tokens, completion_tokens)


@dataclass(frozen=True)
class CostReport:
    """Resumen de coste (medido o proyectado)."""

    total_usd: float
    n_tasks: int
    per_model_usd: dict[str, float] = field(default_factory=dict)
    per_model_calls: dict[str, int] = field(default_factory=dict)

    def per_task_usd(self) -> float:
        return self.total_usd / self.n_tasks if self.n_tasks else 0.0


def cost_of_results(results: Iterable[CascadeResult], pricing: PricingTable) -> CostReport:
    """Coste REAL medido desde los intentos ejecutados por la cascada."""
    per_model_usd: dict[str, float] = {}
    per_model_calls: dict[str, int] = {}
    n = 0
    for res in results:
        n += 1
        for att in res.attempts:
            c = pricing.cost_of(att.model_id, att.prompt_tokens, att.completion_tokens)
            per_model_usd[att.model_id] = per_model_usd.get(att.model_id, 0.0) + c
            per_model_calls[att.model_id] = per_model_calls.get(att.model_id, 0) + 1
    total = sum(per_model_usd.values())
    return CostReport(
        total_usd=total,
        n_tasks=n,
        per_model_usd=per_model_usd,
        per_model_calls=per_model_calls,
    )


def project_cost(
    n_tasks: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    tiers: Sequence[PriceSpec],
    fail_rates: Sequence[float],
) -> CostReport:
    """Proyecta el coste de la cascada bajo supuestos de escalado.

    `tiers` ordenado de más barato a más capaz. `fail_rates[k]` es la fracción
    de tareas que el tier k NO resuelve (y por tanto escala al siguiente). La
    probabilidad de que un tier se ejecute es el producto de los fallos previos.
    """
    if len(fail_rates) != len(tiers):
        raise ValueError("fail_rates y tiers deben tener la misma longitud")
    if not all(0.0 <= r <= 1.0 for r in fail_rates):
        raise ValueError("fail_rates deben estar en [0, 1]")

    per_model_usd: dict[str, float] = {}
    per_model_calls: dict[str, int] = {}
    reach = 1.0  # el primer tier siempre se ejecuta
    for k, tier in enumerate(tiers):
        expected_calls = reach * n_tasks
        cost_per_call = tier.cost(avg_input_tokens, avg_output_tokens)
        per_model_usd[tier.model_id] = expected_calls * cost_per_call
        per_model_calls[tier.model_id] = round(expected_calls)
        reach *= fail_rates[k]  # solo escalan las que este tier no resuelve
    total = sum(per_model_usd.values())
    return CostReport(
        total_usd=total,
        n_tasks=n_tasks,
        per_model_usd=per_model_usd,
        per_model_calls=per_model_calls,
    )
