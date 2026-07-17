"""Tests del dedup near-duplicate."""

from __future__ import annotations

from routerpolicy.dataset.augment import BaseTask
from routerpolicy.dataset.dedup import dedup_tasks
from routerpolicy.schema.core import Mode


def _t(task_id: str, prompt: str) -> BaseTask:
    return BaseTask(task_id, "src", prompt, Mode.DIRECT, difficulty=1, requires_tools=False)


def test_removes_exact_duplicates() -> None:
    tasks = [_t("a", "hello world this is a test"), _t("b", "hello world this is a test")]
    kept, report = dedup_tasks(tasks)
    assert report.kept == 1
    assert report.removed == 1
    assert kept[0].task_id == "a"  # mantiene la primera


def test_removes_near_duplicates() -> None:
    tasks = [
        _t("a", "write a function that adds two integer numbers together"),
        _t("b", "write a function that adds two integer numbers together now"),
        _t("c", "compute the eigenvalues of a large sparse matrix efficiently"),
    ]
    kept, _report = dedup_tasks(tasks, threshold=0.6)
    kept_ids = {t.task_id for t in kept}
    assert "c" in kept_ids  # distinta, se mantiene
    assert len(kept) == 2  # a y b colapsan


def test_distinct_tasks_all_kept() -> None:
    tasks = [
        _t("a", "the quick brown fox jumps over the lazy dog"),
        _t("b", "lorem ipsum dolor sit amet consectetur adipiscing"),
        _t("c", "machine learning models require careful evaluation and testing"),
    ]
    _kept, report = dedup_tasks(tasks)
    assert report.kept == 3
    assert report.removed == 0
