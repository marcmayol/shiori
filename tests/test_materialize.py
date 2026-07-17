"""Tests de la unión labels+prompts (materialize)."""

from __future__ import annotations

from routerpolicy.dataset.materialize import build_base_tasks
from routerpolicy.harness.tasks import TaskSource
from routerpolicy.labeling.records import ModeRecord, SufficiencyRecord
from routerpolicy.schema.core import Mode, Provenance

POOL_IDS = ["cheap", "mid", "capable"]


def test_build_base_tasks_joins_and_maps() -> None:
    suff = [
        SufficiencyRecord(
            task_id="c1", source=TaskSource.MBPP_PLUS, sufficient_model_id="mid", passed_by={}
        ),
        SufficiencyRecord(
            task_id="c2", source=TaskSource.HUMANEVAL_PLUS, sufficient_model_id=None, passed_by={}
        ),
    ]
    modes = [
        ModeRecord(
            task_id="t1", source=TaskSource.HERMES, mode=Mode.TOOL_CALL, mode_source=Provenance.RULE
        ),
        ModeRecord(
            task_id="d1", source=TaskSource.DOLLY, mode=Mode.PLAN, mode_source=Provenance.JUDGE
        ),
    ]
    index = {"c1": "code one", "c2": "code two", "t1": "use a tool", "d1": "make a plan"}
    tasks = build_base_tasks(suff, modes, index, POOL_IDS)

    by_id = {t.task_id: t for t in tasks}
    assert by_id["c1"].mode is Mode.DIRECT and by_id["c1"].difficulty == 2
    assert by_id["c2"].difficulty == 4  # ninguno suficiente
    assert by_id["t1"].mode is Mode.TOOL_CALL and by_id["t1"].requires_tools
    assert by_id["d1"].mode is Mode.PLAN and by_id["d1"].difficulty == 3


def test_missing_prompt_is_skipped() -> None:
    suff = [
        SufficiencyRecord(
            task_id="ghost", source=TaskSource.MBPP_PLUS, sufficient_model_id="cheap", passed_by={}
        )
    ]
    assert build_base_tasks(suff, [], {}, POOL_IDS) == []
