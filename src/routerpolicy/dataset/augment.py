"""Augmentación: recalcula la etiqueta {mode, model_id} por pool sintético.

Cada tarea base tiene una DIFICULTAD (0..4) y si REQUIERE tools. La etiqueta por
pool es el modelo más barato cuya capacidad basta (y soporta tools si hace
falta); si ninguno basta, el más capaz del pool (regla de Fase 3).

Dificultad:
- Código: del resultado de la cascada (rank del modelo mínimo suficiente en el
  pool de generación real; "ninguno" = máxima).
- Modo: DIRECT=1, TOOL_CALL=1 (+tools), PLAN=3.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from routerpolicy.registry.synthetic import (
    SyntheticModel,
    generate_pool,
    pool_signature,
)
from routerpolicy.schema.core import Mode, Registry, RoutingDecision

# Firma estructural de un pool (tamaño, capacidades, patrón de tools).
PoolSignature = tuple[int, tuple[int, ...], tuple[int, ...]]

# Dificultad por modo de las tareas del eje de modo.
_MODE_DIFFICULTY = {Mode.DIRECT: 1, Mode.TOOL_CALL: 1, Mode.PLAN: 3}


@dataclass(frozen=True)
class BaseTask:
    """Tarea base lista para augmentar (etiqueta relativa al pool)."""

    task_id: str
    source: str
    prompt: str
    mode: Mode
    difficulty: int  # 0..MAX_CAPABILITY
    requires_tools: bool


def difficulty_from_sufficiency(
    sufficient_model_id: str | None, ordered_pool_ids: Sequence[str]
) -> int:
    """Dificultad de una tarea de código según el modelo mínimo suficiente.

    rank 0 (más barato) -> dificultad 1 ... ; "ninguno suficiente" -> len+1.
    """
    if sufficient_model_id is None:
        return len(ordered_pool_ids) + 1
    return ordered_pool_ids.index(sufficient_model_id) + 1


def mode_difficulty(mode: Mode) -> int:
    return _MODE_DIFFICULTY[mode]


def label_for_pool(task: BaseTask, pool: list[SyntheticModel]) -> RoutingDecision:
    """Etiqueta {mode, model_id} de la tarea para ESE pool sintético.

    model_id = el más barato con capacidad>=dificultad (y tools si se requieren);
    si ninguno cumple, el más capaz del pool.
    """
    solvers = [
        sm
        for sm in pool
        if sm.capability >= task.difficulty and (sm.spec.supports_tools or not task.requires_tools)
    ]
    if solvers:
        chosen = min(solvers, key=lambda sm: (sm.spec.cost, sm.capability, sm.spec.id))
    else:
        # ninguno suficiente -> el más capaz (prioriza tools si la tarea lo pide,
        # luego el más barato) para el fallback determinista.
        chosen = max(
            pool,
            key=lambda sm: (
                sm.capability,
                int(sm.spec.supports_tools) if task.requires_tools else 0,
                -sm.spec.cost,
            ),
        )
    return RoutingDecision(mode=task.mode, model_id=chosen.spec.id)


@dataclass(frozen=True)
class AugmentedExample:
    """Un ejemplo augmentado: registro renderizable + decisión + metadatos."""

    task_id: str
    source: str
    prompt: str
    registry: Registry
    decision: RoutingDecision
    difficulty: int
    n_models: int
    signature: PoolSignature  # firma estructural del pool (para leakage)


def augment_task(
    task: BaseTask,
    rng: random.Random,
    factor: int,
    allow_signature: Callable[[PoolSignature], bool] | None = None,
) -> list[AugmentedExample]:
    """Genera `factor` ejemplos de la tarea con pools sintéticos distintos.

    `allow_signature`: si se pasa un callable firma->bool, solo se aceptan pools
    cuya firma estructural lo cumpla (para reservar composiciones al test).
    """
    out: list[AugmentedExample] = []
    attempts = 0
    while len(out) < factor and attempts < factor * 20:
        attempts += 1
        pool = generate_pool(rng)
        signature = pool_signature(pool)
        if allow_signature is not None and not allow_signature(signature):
            continue
        decision = label_for_pool(task, pool)
        registry = Registry(models=tuple(sm.spec for sm in pool))
        out.append(
            AugmentedExample(
                task_id=task.task_id,
                source=task.source,
                prompt=task.prompt,
                registry=registry,
                decision=decision,
                difficulty=task.difficulty,
                n_models=len(pool),
                signature=signature,
            )
        )
    return out
