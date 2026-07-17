"""Tipos de tarea de las fuentes (por eje del plan).

- Suficiencia (código verificable): CodeTask con tests mecánicos.
- Modo TOOL_CALL: ToolTask con herramientas declaradas.
- Modo DIRECT/PLAN: ChatTask (instrucción general).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TaskSource(StrEnum):
    """Origen de la tarea (para procedencia y licencias)."""

    MBPP_PLUS = "mbpp_plus"
    HUMANEVAL_PLUS = "humaneval_plus"
    BIGCODEBENCH = "bigcodebench"
    XLAM = "xlam_function_calling"
    BFCL = "bfcl"
    WILDCHAT = "wildchat"
    LMSYS = "lmsys_chat_1m"


class CodeTask(BaseModel):
    """Tarea de código con verificación mecánica (eje suficiencia)."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(min_length=1)
    source: TaskSource
    prompt: str = Field(min_length=1)  # enunciado + firma
    entry_point: str = Field(min_length=1)  # función que se debe implementar
    test_code: str = Field(min_length=1)  # python que verifica entry_point


class ToolTask(BaseModel):
    """Tarea con herramientas declaradas (ejemplo natural de TOOL_CALL)."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(min_length=1)
    source: TaskSource
    prompt: str = Field(min_length=1)
    tool_names: tuple[str, ...] = Field(min_length=1)  # herramientas disponibles


class ChatTask(BaseModel):
    """Instrucción general (para etiquetar DIRECT vs PLAN)."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(min_length=1)
    source: TaskSource
    prompt: str = Field(min_length=1)
