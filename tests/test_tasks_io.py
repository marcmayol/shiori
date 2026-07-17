"""Tests de la E/S canónica JSONL de tareas."""

from __future__ import annotations

from pathlib import Path

from routerpolicy.dataset.tasks_io import load_jsonl, write_jsonl
from routerpolicy.harness.tasks import CodeTask, TaskSource


def _task(i: int) -> CodeTask:
    return CodeTask(
        task_id=f"t{i}",
        source=TaskSource.MBPP_PLUS,
        prompt=f"prompt {i}",
        entry_point="f",
        test_code="assert f() == 1",
    )


def test_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    tasks = [_task(i) for i in range(3)]
    assert write_jsonl(path, tasks) == 3
    loaded = load_jsonl(path, CodeTask)
    assert loaded == tasks


def test_limit_applies(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    write_jsonl(path, [_task(i) for i in range(10)])
    assert len(load_jsonl(path, CodeTask, limit=4)) == 4


def test_blank_lines_ignored(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    write_jsonl(path, [_task(0)])
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert len(load_jsonl(path, CodeTask)) == 1
