"""Registros de etiquetado con procedencia (persistibles en JSONL).

Dos ejes:
- SufficiencyRecord: para tareas de código, qué modelo del pool de generación es
  el mínimo suficiente (VERIFIED) o None si ninguno pasó.
- ModeRecord: para tareas de modo, el modo etiquetado y cómo (regla/juez).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from routerpolicy.harness.tasks import TaskSource
from routerpolicy.schema.core import Mode, Provenance


class SufficiencyRecord(BaseModel):
    """Suficiencia verificada de una tarea de código sobre el pool de generación."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    source: TaskSource
    sufficient_model_id: str | None  # None => ningún modelo del pool pasó
    passed_by: dict[str, bool]  # model_id -> pasó la verificación
    provenance: Provenance = Provenance.VERIFIED


class ModeRecord(BaseModel):
    """Modo etiquetado de una tarea (regla o juez)."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    source: TaskSource
    mode: Mode
    mode_source: Provenance
