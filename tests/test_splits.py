"""Tests de splits estratificados y test de leakage."""

from __future__ import annotations

import random

from routerpolicy.dataset.augment import BaseTask, augment_task
from routerpolicy.dataset.splits import (
    check_leakage,
    is_test_signature,
    stratified_split,
)
from routerpolicy.schema.core import Mode


def _task(i: int, mode: Mode, difficulty: int) -> BaseTask:
    return BaseTask(
        task_id=f"t{i}",
        source="src",
        prompt=f"prompt number {i} with some words to shingle",
        mode=mode,
        difficulty=difficulty,
        requires_tools=(mode is Mode.TOOL_CALL),
    )


def test_is_test_signature_deterministic_and_partitions() -> None:
    sig = (3, (0, 1, 2), (0, 0, 1))
    assert is_test_signature(sig) == is_test_signature(sig)
    # con muchos ejemplos, ~10% caen en test
    sigs = [(n, tuple(range(n)), (0,) * n) for n in range(2, 9)]
    reserved = sum(is_test_signature(s) for s in sigs)
    assert 0 <= reserved <= len(sigs)


def test_stratified_split_no_task_overlap_and_proportions() -> None:
    tasks = [_task(i, Mode.DIRECT, 1) for i in range(50)]
    tasks += [_task(100 + i, Mode.PLAN, 3) for i in range(50)]
    train, test = stratified_split(tasks, test_frac=0.2, rng=random.Random(0))
    train_ids = {t.task_id for t in train}
    test_ids = {t.task_id for t in test}
    assert train_ids.isdisjoint(test_ids)
    assert len(train) + len(test) == 100
    # cada estrato aporta ~20% al test
    assert sum(1 for t in test if t.mode is Mode.DIRECT) == 10
    assert sum(1 for t in test if t.mode is Mode.PLAN) == 10


def test_check_leakage_detects_task_and_signature_overlap() -> None:
    task = _task(1, Mode.DIRECT, 2)
    same = augment_task(task, random.Random(1), factor=3)  # misma tarea en ambos
    report = check_leakage(same, same)
    assert report.task_overlap == 1
    assert not report.clean


def test_end_to_end_no_leakage() -> None:
    tasks = [_task(i, Mode.DIRECT, (i % 4) + 1) for i in range(40)]
    train_tasks, test_tasks = stratified_split(tasks, test_frac=0.25, rng=random.Random(2))

    rng = random.Random(3)
    train_ex = []
    for t in train_tasks:
        train_ex += augment_task(
            t, rng, factor=4, allow_signature=lambda s: not is_test_signature(s)
        )
    test_ex = []
    for t in test_tasks:
        test_ex += augment_task(t, rng, factor=4, allow_signature=is_test_signature)

    report = check_leakage(train_ex, test_ex)
    assert report.task_overlap == 0
    assert report.signature_overlap == 0
    assert report.clean
