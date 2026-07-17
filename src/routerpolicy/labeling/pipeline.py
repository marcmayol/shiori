"""Orquestación del etiquetado con checkpointing incremental y reanudación.

Todo run largo debe guardar progreso por tarea y reanudarse solo desde donde se
quedó (el equipo puede reiniciarse). Los registros se ANEXAN a un JSONL; al
reanudar se saltan las tareas ya presentes.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from routerpolicy.harness.cascade import run_cascade
from routerpolicy.harness.runner import ModelRunner
from routerpolicy.harness.tasks import ChatTask, CodeTask, ToolTask
from routerpolicy.harness.verify import DEFAULT_TIMEOUT_S
from routerpolicy.labeling.judge import ModeJudge
from routerpolicy.labeling.mode_rules import (
    rule_mode_for_chat_task,
    rule_mode_for_tool_task,
)
from routerpolicy.labeling.records import ModeRecord, SufficiencyRecord
from routerpolicy.schema.core import Mode, Provenance


def done_task_ids(path: Path) -> set[str]:
    """Lee los task_id ya presentes en un JSONL de registros (para reanudar)."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        ids.add(str(json.loads(line)["task_id"]))
    return ids


def _append_record(path: Path, record: SufficiencyRecord | ModeRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")
        fh.flush()


def label_mode(task: ToolTask | ChatTask, judge: ModeJudge) -> ModeRecord:
    """Etiqueta el modo: regla para casos obvios, juez para el resto."""
    if isinstance(task, ToolTask):
        return ModeRecord(
            task_id=task.task_id,
            source=task.source,
            mode=rule_mode_for_tool_task(task),
            mode_source=Provenance.RULE,
        )
    ruled = rule_mode_for_chat_task(task)
    if ruled is not None:
        return ModeRecord(
            task_id=task.task_id,
            source=task.source,
            mode=ruled,
            mode_source=Provenance.RULE,
        )
    return ModeRecord(
        task_id=task.task_id,
        source=task.source,
        mode=judge.judge(task.prompt),
        mode_source=Provenance.JUDGE,
    )


def run_sufficiency_labeling(
    tasks: Sequence[CodeTask],
    runners: list[ModelRunner],
    out_path: Path,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    resume: bool = True,
) -> int:
    """Etiqueta suficiencia por cascada, anexando por tarea. Devuelve nº nuevos.

    Al reanudar (`resume=True`) salta las tareas ya presentes en `out_path`, sin
    re-ejecutar sus modelos.
    """
    skip = done_task_ids(out_path) if resume else set()
    processed = 0
    for task in tasks:
        if task.task_id in skip:
            continue
        result = run_cascade(task, runners, timeout_s=timeout_s)
        passed_by = {att.model_id: att.passed for att in result.attempts}
        record = SufficiencyRecord(
            task_id=task.task_id,
            source=task.source,
            sufficient_model_id=result.sufficient_model_id,
            passed_by=passed_by,
        )
        _append_record(out_path, record)
        processed += 1
    return processed


def run_mode_labeling(
    tasks: Sequence[ToolTask | ChatTask],
    judge: ModeJudge,
    out_path: Path,
    resume: bool = True,
) -> int:
    """Etiqueta modo por tarea, anexando incrementalmente. Devuelve nº nuevos.

    Resiliente: si el juez falla en una tarea (tras sus reintentos), se salta y
    el run continúa; se reintentará en la próxima reanudación.
    """
    skip = done_task_ids(out_path) if resume else set()
    processed = 0
    skipped = 0
    for task in tasks:
        if task.task_id in skip:
            continue
        try:
            record = label_mode(task, judge)
        except Exception as exc:  # el juez agotó reintentos: saltar, no crashear
            skipped += 1
            print(f"  [skip] {task.task_id}: {exc}", file=sys.stderr, flush=True)
            continue
        _append_record(out_path, record)
        processed += 1
    if skipped:
        print(f"  ({skipped} tareas saltadas por fallo del juez)", file=sys.stderr, flush=True)
    return processed


@dataclass(frozen=True)
class LabelingReport:
    """Distribución de las etiquetas (DoD de la Fase 2)."""

    n_code: int
    n_mode: int
    by_sufficient_model: dict[str, int] = field(default_factory=dict)
    by_mode: dict[Mode, int] = field(default_factory=dict)
    by_mode_source: dict[Provenance, int] = field(default_factory=dict)


NONE_BUCKET = "(ninguno suficiente)"


def build_report(
    sufficiency: Sequence[SufficiencyRecord],
    modes: Sequence[ModeRecord],
) -> LabelingReport:
    """Agrega distribución por modelo-suficiente y por modo."""
    by_model: dict[str, int] = {}
    for srec in sufficiency:
        key = srec.sufficient_model_id or NONE_BUCKET
        by_model[key] = by_model.get(key, 0) + 1

    by_mode: dict[Mode, int] = {}
    by_source: dict[Provenance, int] = {}
    for mrec in modes:
        by_mode[mrec.mode] = by_mode.get(mrec.mode, 0) + 1
        by_source[mrec.mode_source] = by_source.get(mrec.mode_source, 0) + 1

    return LabelingReport(
        n_code=len(sufficiency),
        n_mode=len(modes),
        by_sufficient_model=by_model,
        by_mode=by_mode,
        by_mode_source=by_source,
    )
