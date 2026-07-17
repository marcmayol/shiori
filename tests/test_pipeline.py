"""Tests del orquestador de etiquetado: reanudación, modo y reporte."""

from __future__ import annotations

from pathlib import Path

from routerpolicy.dataset.tasks_io import load_jsonl
from routerpolicy.harness.cascade import build_code_prompt
from routerpolicy.harness.runner import FakeRunner
from routerpolicy.harness.tasks import ChatTask, CodeTask, TaskSource, ToolTask
from routerpolicy.labeling.judge import FakeJudge
from routerpolicy.labeling.pipeline import (
    NONE_BUCKET,
    build_report,
    done_task_ids,
    label_mode,
    run_mode_labeling,
    run_sufficiency_labeling,
)
from routerpolicy.labeling.records import ModeRecord, SufficiencyRecord
from routerpolicy.schema.core import Mode, Provenance


def _code(i: int) -> CodeTask:
    return CodeTask(
        task_id=f"code-{i}",
        source=TaskSource.MBPP_PLUS,
        prompt=f"Write add{i}(a,b) returning a+b.",
        entry_point=f"add{i}",
        test_code=f"assert add{i}(2,3) == 5",
    )


def _good(i: int) -> str:
    return f"```python\ndef add{i}(a, b):\n    return a + b\n```"


def _runner_for(tasks: list[CodeTask], model_id: str) -> FakeRunner:
    responder = {build_code_prompt(t): _good(int(t.task_id.split("-")[1])) for t in tasks}
    return FakeRunner(model_id, responder=responder)


def test_sufficiency_labeling_writes_records(tmp_path: Path) -> None:
    tasks = [_code(i) for i in range(3)]
    out = tmp_path / "suff.jsonl"
    n = run_sufficiency_labeling(tasks, [_runner_for(tasks, "cheap")], out)
    assert n == 3
    records = load_jsonl(out, SufficiencyRecord)
    assert len(records) == 3
    assert all(r.sufficient_model_id == "cheap" for r in records)
    assert all(r.provenance is Provenance.VERIFIED for r in records)


def test_resume_launch_interrupt_relaunch(tmp_path: Path) -> None:
    # Ciclo exigido por el plan: lanzar -> interrumpir -> relanzar.
    tasks = [_code(i) for i in range(5)]
    out = tmp_path / "suff.jsonl"

    # 1) "lanzar" e "interrumpir": procesa solo las 2 primeras.
    first_runner = _runner_for(tasks, "cheap")
    n1 = run_sufficiency_labeling(tasks[:2], [first_runner], out)
    assert n1 == 2
    assert done_task_ids(out) == {"code-0", "code-1"}

    # 2) "relanzar" sobre TODAS: debe saltar las 2 ya hechas y procesar 3.
    fresh_runner = _runner_for(tasks, "cheap")
    n2 = run_sufficiency_labeling(tasks, [fresh_runner], out)
    assert n2 == 3
    # el runner del relanzamiento NO re-ejecutó las tareas ya hechas
    assert len(fresh_runner.calls) == 3

    records = load_jsonl(out, SufficiencyRecord)
    assert {r.task_id for r in records} == {f"code-{i}" for i in range(5)}
    assert len(records) == 5  # sin duplicados


def test_label_mode_rule_and_judge() -> None:
    tool = ToolTask(task_id="t", source=TaskSource.XLAM, prompt="do it", tool_names=("f",))
    assert label_mode(tool, FakeJudge({})).mode is Mode.TOOL_CALL
    assert label_mode(tool, FakeJudge({})).mode_source is Provenance.RULE

    factual = ChatTask(task_id="c", source=TaskSource.WILDCHAT, prompt="Who wrote Hamlet?")
    rec = label_mode(factual, FakeJudge({}))
    assert rec.mode is Mode.DIRECT and rec.mode_source is Provenance.RULE

    planning = ChatTask(
        task_id="p", source=TaskSource.WILDCHAT, prompt="Build a scalable pipeline for me."
    )
    rec2 = label_mode(planning, FakeJudge({"Build a scalable pipeline for me.": Mode.PLAN}))
    assert rec2.mode is Mode.PLAN and rec2.mode_source is Provenance.JUDGE


class _FailingJudge:
    """Juez que siempre falla (para probar la resiliencia del run de modo)."""

    def judge(self, task_prompt: str) -> Mode:
        raise RuntimeError("judge boom")


def test_mode_labeling_skips_judge_failures(tmp_path: Path) -> None:
    # una tarea factual (regla -> DIRECT) y otra que necesita juez (falla).
    tasks: list[ToolTask | ChatTask] = [
        ChatTask(task_id="factual", source=TaskSource.DOLLY, prompt="Who wrote Hamlet?"),
        ChatTask(task_id="hard", source=TaskSource.DOLLY, prompt="Compare and contrast X and Y."),
    ]
    out = tmp_path / "mode.jsonl"
    # la regla resuelve la factual; la otra va al juez, que falla -> se salta.
    n = run_mode_labeling(tasks, _FailingJudge(), out)
    assert n == 1
    records = load_jsonl(out, ModeRecord)
    assert [r.task_id for r in records] == ["factual"]


def test_mode_labeling_resumes(tmp_path: Path) -> None:
    tasks: list[ToolTask | ChatTask] = [
        ToolTask(task_id=f"t{i}", source=TaskSource.XLAM, prompt="x", tool_names=("f",))
        for i in range(3)
    ]
    out = tmp_path / "mode.jsonl"
    assert run_mode_labeling(tasks[:1], FakeJudge({}), out) == 1
    assert run_mode_labeling(tasks, FakeJudge({}), out) == 2  # salta la primera
    assert len(load_jsonl(out, ModeRecord)) == 3


def test_build_report_distributions() -> None:
    suff = [
        SufficiencyRecord(
            task_id="a", source=TaskSource.MBPP_PLUS, sufficient_model_id="cheap", passed_by={}
        ),
        SufficiencyRecord(
            task_id="b", source=TaskSource.MBPP_PLUS, sufficient_model_id="cheap", passed_by={}
        ),
        SufficiencyRecord(
            task_id="c", source=TaskSource.MBPP_PLUS, sufficient_model_id=None, passed_by={}
        ),
    ]
    modes = [
        ModeRecord(
            task_id="m1", source=TaskSource.XLAM, mode=Mode.TOOL_CALL, mode_source=Provenance.RULE
        ),
        ModeRecord(
            task_id="m2", source=TaskSource.WILDCHAT, mode=Mode.PLAN, mode_source=Provenance.JUDGE
        ),
    ]
    report = build_report(suff, modes)
    assert report.n_code == 3 and report.n_mode == 2
    assert report.by_sufficient_model == {"cheap": 2, NONE_BUCKET: 1}
    assert report.by_mode == {Mode.TOOL_CALL: 1, Mode.PLAN: 1}
    assert report.by_mode_source == {Provenance.RULE: 1, Provenance.JUDGE: 1}
