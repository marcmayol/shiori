"""Estimación de coste del etiquetado (gate obligatorio de la sección 3).

Proyecta el coste del run completo bajo varios escenarios de escalado, a partir
de una tabla de precios y del tamaño medio de prompt/salida. Cuando el pool real
esté decidido y haya muestra ejecutada, se sustituyen los supuestos por medidas.

Uso:
    uv run python scripts/estimate_cost.py --n-tasks 8000 --avg-input 220 \
        --avg-output 256 --pricing configs/pricing.example.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from routerpolicy.harness.cost import CostReport, PriceSpec, project_cost
from routerpolicy.harness.pricing_io import load_tiers

# Escenarios de escalado: fracción de tareas que cada tier NO resuelve. El último
# tier "falla" (queda sin resolver) pero su coste se incurre igual. Con 3 tiers:
# [small, medium, api]. Ej. [0.6, 0.5, 1.0] => 60% escala a medium, y de esas el
# 50% escala a api => 30% de las tareas llegan a la API.
SCENARIOS: dict[str, list[float]] = {
    "optimista": [0.40, 0.30, 1.0],  # 12% llega a la API
    "esperado": [0.60, 0.50, 1.0],  # 30% llega a la API
    "pesimista": [0.80, 0.70, 1.0],  # 56% llega a la API
}


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def _adjust_fail_rates(base: list[float], n_tiers: int) -> list[float]:
    """Ajusta la longitud de fail_rates al número de tiers (último = 1.0)."""
    if n_tiers <= 0:
        return []
    rates = base[: n_tiers - 1]
    while len(rates) < n_tiers - 1:
        rates.append(base[-2] if len(base) >= 2 else 0.5)
    rates.append(1.0)
    return rates


def _print_report(name: str, tiers: list[PriceSpec], report: CostReport) -> None:
    print(f"\n== escenario: {name} ==")
    print(f"  tareas: {report.n_tasks}")
    for tier in tiers:
        calls = report.per_model_calls.get(tier.model_id, 0)
        usd = report.per_model_usd.get(tier.model_id, 0.0)
        loc = "local" if tier.is_local else "api"
        print(f"  {tier.model_id:<20} [{loc:>5}]  llamadas≈{calls:>7}  USD≈{usd:>10.2f}")
    print(f"  TOTAL≈ ${report.total_usd:,.2f}   (${report.per_task_usd():.4f}/tarea)")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="Estimación de coste del etiquetado")
    parser.add_argument("--n-tasks", type=int, default=8000)
    parser.add_argument("--avg-input", type=int, default=220, help="tokens de entrada/prompt")
    parser.add_argument("--avg-output", type=int, default=256, help="tokens de salida")
    parser.add_argument(
        "--pricing",
        type=Path,
        default=Path("configs/pricing.example.yaml"),
    )
    args = parser.parse_args(argv)

    tiers = load_tiers(args.pricing)
    print(f"Tabla de precios: {args.pricing}")
    print("(!) Precios ILUSTRATIVOS hasta confirmar el pool real en el gate.")
    for name, base in SCENARIOS.items():
        fail_rates = _adjust_fail_rates(base, len(tiers))
        report = project_cost(
            n_tasks=args.n_tasks,
            avg_input_tokens=args.avg_input,
            avg_output_tokens=args.avg_output,
            tiers=tiers,
            fail_rates=fail_rates,
        )
        _print_report(name, tiers, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
