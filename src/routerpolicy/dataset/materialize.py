"""Materializa BaseTask uniendo labels (Fase 2) con los prompts de las fuentes.

Los registros de Fase 2 guardan task_id + etiqueta, no el texto. Aquí se re-ingiere
cada fuente para construir un índice task_id->prompt y se unen con los labels. La
lógica de unión es pura y testeable; la ingesta (I/O) se usa solo en el script.
"""

from __future__ import annotations

from collections.abc import Sequence

from routerpolicy.dataset.augment import BaseTask, difficulty_from_sufficiency, mode_difficulty
from routerpolicy.labeling.records import ModeRecord, SufficiencyRecord
from routerpolicy.schema.core import Mode


def base_task_from_sufficiency(
    rec: SufficiencyRecord, prompt: str, ordered_pool_ids: Sequence[str]
) -> BaseTask:
    """Tarea de código -> BaseTask (modo DIRECT, dificultad de la cascada)."""
    return BaseTask(
        task_id=rec.task_id,
        source=rec.source.value,
        prompt=prompt,
        mode=Mode.DIRECT,
        difficulty=difficulty_from_sufficiency(rec.sufficient_model_id, ordered_pool_ids),
        requires_tools=False,
    )


def base_task_from_mode(rec: ModeRecord, prompt: str) -> BaseTask:
    """Tarea de modo -> BaseTask (dificultad por modo; tools si TOOL_CALL)."""
    return BaseTask(
        task_id=rec.task_id,
        source=rec.source.value,
        prompt=prompt,
        mode=rec.mode,
        difficulty=mode_difficulty(rec.mode),
        requires_tools=rec.mode is Mode.TOOL_CALL,
    )


def build_base_tasks(
    sufficiency: Sequence[SufficiencyRecord],
    modes: Sequence[ModeRecord],
    prompt_index: dict[str, str],
    ordered_pool_ids: Sequence[str],
) -> list[BaseTask]:
    """Une labels + prompts. Salta los task_id sin prompt en el índice."""
    out: list[BaseTask] = []
    for srec in sufficiency:
        prompt = prompt_index.get(srec.task_id)
        if prompt:
            out.append(base_task_from_sufficiency(srec, prompt, ordered_pool_ids))
    for mrec in modes:
        prompt = prompt_index.get(mrec.task_id)
        if prompt:
            out.append(base_task_from_mode(mrec, prompt))
    return out


def build_prompt_index(sample: int | None = None) -> dict[str, str]:
    """Re-ingiere las fuentes y devuelve task_id->prompt (I/O, extra `label`)."""
    from routerpolicy.dataset.sources import (
        ingest_dolly,
        ingest_hermes,
        ingest_humaneval_plus,
        ingest_mbpp_plus,
    )

    index: dict[str, str] = {}
    for task in ingest_humaneval_plus():
        index[task.task_id] = task.prompt
    for task in ingest_mbpp_plus():
        index[task.task_id] = task.prompt
    for tool in ingest_hermes(limit=sample):
        index[tool.task_id] = tool.prompt
    for chat in ingest_dolly(limit=sample):
        index[chat.task_id] = chat.prompt
    return index
