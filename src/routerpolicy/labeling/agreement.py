"""Control de ruido: acuerdo entre dos anotaciones (doble anotación 10%).

Reporta acuerdo bruto, kappa de Cohen y acuerdo por clase, para decidir si la
rúbrica del juez es fiable antes de escalar el etiquetado.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from routerpolicy.schema.core import Mode


@dataclass(frozen=True)
class AgreementReport:
    """Resultado del análisis de acuerdo entre dos anotadores."""

    n: int
    raw_agreement: float  # fracción de ítems en los que coinciden
    cohen_kappa: float  # acuerdo corregido por azar
    per_class_agreement: dict[Mode, float]  # acuerdo sobre ítems de cada clase (por a)


def _cohen_kappa(a: Sequence[Mode], b: Sequence[Mode]) -> float:
    n = len(a)
    agree = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    po = agree / n
    count_a = Counter(a)
    count_b = Counter(b)
    pe = sum((count_a.get(m, 0) / n) * (count_b.get(m, 0) / n) for m in Mode)
    if pe >= 1.0:
        # sin varianza esperada (todo una clase y coinciden): acuerdo perfecto.
        return 1.0
    return (po - pe) / (1.0 - pe)


def agreement_report(a: Sequence[Mode], b: Sequence[Mode]) -> AgreementReport:
    """Calcula el reporte de acuerdo entre dos secuencias de anotaciones."""
    if len(a) != len(b):
        raise ValueError("las dos anotaciones deben tener la misma longitud")
    if not a:
        raise ValueError("no hay anotaciones que comparar")

    n = len(a)
    agree = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    raw = agree / n

    per_class: dict[Mode, float] = {}
    for m in Mode:
        idx = [i for i, x in enumerate(a) if x == m]
        if idx:
            hits = sum(1 for i in idx if a[i] == b[i])
            per_class[m] = hits / len(idx)

    return AgreementReport(
        n=n,
        raw_agreement=raw,
        cohen_kappa=_cohen_kappa(a, b),
        per_class_agreement=per_class,
    )
