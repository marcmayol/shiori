"""Splits estratificados + test congelado con composiciones de pool no vistas.

- Las TAREAS se reparten train/test estratificando por (modo, dificultad); una
  tarea entera va a un split (nunca a los dos) -> sin leakage de tarea.
- Las FIRMAS de pool se particionan por hash: el test augmenta solo con firmas
  reservadas, nunca presentes en train -> composiciones no vistas.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass

from routerpolicy.dataset.augment import AugmentedExample, BaseTask, PoolSignature

TEST_SIGNATURE_BUCKETS = 1  # de cada N buckets, 1 se reserva al test
SIGNATURE_TOTAL_BUCKETS = 10


def is_test_signature(
    signature: PoolSignature,
    reserved: int = TEST_SIGNATURE_BUCKETS,
    total: int = SIGNATURE_TOTAL_BUCKETS,
) -> bool:
    """True si la firma pertenece al espacio reservado al test (determinista)."""
    digest = hashlib.blake2b(repr(signature).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % total < reserved


def stratified_split(
    tasks: Sequence[BaseTask], test_frac: float, rng: random.Random
) -> tuple[list[BaseTask], list[BaseTask]]:
    """Reparte tareas en (train, test) estratificando por (modo, dificultad)."""
    groups: dict[tuple[str, int], list[BaseTask]] = {}
    for task in tasks:
        groups.setdefault((task.mode.value, task.difficulty), []).append(task)

    train: list[BaseTask] = []
    test: list[BaseTask] = []
    for _, members in sorted(groups.items()):
        shuffled = list(members)
        rng.shuffle(shuffled)
        n_test = round(len(shuffled) * test_frac)
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])
    return train, test


@dataclass(frozen=True)
class LeakageReport:
    task_overlap: int  # task_ids presentes en train y test
    signature_overlap: int  # firmas de pool en train y test
    clean: bool


def check_leakage(
    train: Sequence[AugmentedExample], test: Sequence[AugmentedExample]
) -> LeakageReport:
    """Verifica que no hay leakage de tarea ni de composición de pool."""
    train_tasks = {ex.task_id for ex in train}
    test_tasks = {ex.task_id for ex in test}
    task_overlap = len(train_tasks & test_tasks)

    train_sigs = {ex.signature for ex in train}
    test_sigs = {ex.signature for ex in test}
    signature_overlap = len(train_sigs & test_sigs)

    return LeakageReport(
        task_overlap=task_overlap,
        signature_overlap=signature_overlap,
        clean=task_overlap == 0 and signature_overlap == 0,
    )
