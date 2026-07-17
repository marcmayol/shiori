"""Análisis de Fase 5: matriz de confusión, capacidad recuperada, suficiencia.

Funciones puras usadas por el informe comparativo.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from routerpolicy.schema.core import Mode, ModelSpec, RoutingDecision

INVALID = "INVALID"


def capability_from_tags(tags: Sequence[str]) -> int:
    """Recupera la capacidad oculta (0..4) desde los tags del render.

    Inverso del encoding de registry/synthetic: expert->4, planning->3,
    reasoning->2, code->1, resto->0.
    """
    t = set(tags)
    if "expert" in t:
        return 4
    if "planning" in t:
        return 3
    if "reasoning" in t:
        return 2
    if "code" in t:
        return 1
    return 0


def mode_confusion(
    golds: Sequence[RoutingDecision], preds: Sequence[RoutingDecision | None]
) -> dict[Mode, Counter[str]]:
    """Matriz de confusión de modo: gold -> Counter de modo predicho (o INVALID)."""
    matrix: dict[Mode, Counter[str]] = {m: Counter() for m in Mode}
    for gold, pred in zip(golds, preds, strict=True):
        key = pred.mode.value if pred is not None else INVALID
        matrix[gold.mode][key] += 1
    return matrix


def format_confusion(matrix: dict[Mode, Counter[str]]) -> str:
    """Render de la matriz de confusión de modo."""
    cols = [m.value for m in Mode] + [INVALID]
    header = "  gold\\pred   " + "".join(f"{c:>10}" for c in cols)
    lines = [header]
    for gold in Mode:
        row = matrix[gold]
        cells = "".join(f"{row.get(c, 0):>10}" for c in cols)
        lines.append(f"  {gold.value:<11}{cells}")
    return "\n".join(lines)


def is_sufficient(chosen: ModelSpec | None, difficulty: int) -> bool:
    """True si la capacidad (recuperada de tags) del modelo elegido basta.

    La dificultad viene de la cascada real (Fase 2); la suficiencia es
    capacidad_elegida >= dificultad, la misma relación con la que se etiquetó.
    """
    if chosen is None:
        return False
    return capability_from_tags(chosen.tags) >= difficulty
