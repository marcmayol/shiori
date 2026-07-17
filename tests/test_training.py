"""Tests de config/data/prepare de entrenamiento (sin GPU)."""

from __future__ import annotations

from pathlib import Path

from routerpolicy.training.config import load_train_config
from routerpolicy.training.data import gold_completion, load_rows, prompt_messages
from routerpolicy.training.prepare import IGNORE_INDEX, merge_system_into_user

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_gemma_config() -> None:
    cfg = load_train_config(REPO_ROOT / "configs" / "train_gemma_270m.yaml")
    assert cfg.step_name == "gemma-3-270m"
    assert cfg.model_id == "google/gemma-3-270m"
    assert cfg.bf16 is True
    assert cfg.save_total_limit == 2


def _row() -> dict[str, object]:
    return {
        "task_id": "t1",
        "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "registry + task"},
            {"role": "assistant", "content": '{"mode": "DIRECT", "model_id": "x"}'},
        ],
    }


def test_load_rows_and_accessors(tmp_path: Path) -> None:
    import json

    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    rows = load_rows(path)
    assert len(rows) == 1
    assert gold_completion(rows[0]) == '{"mode": "DIRECT", "model_id": "x"}'
    assert [m["role"] for m in prompt_messages(rows[0])] == ["system", "user"]


def test_merge_system_into_user() -> None:
    merged = merge_system_into_user(_row()["messages"])  # type: ignore[arg-type]
    assert [m["role"] for m in merged] == ["user", "assistant"]
    assert merged[0]["content"].startswith("SYS\n\n")
    assert "registry + task" in merged[0]["content"]


def test_ignore_index_value() -> None:
    assert IGNORE_INDEX == -100
