"""LLM juez para el modo, con rúbrica fija y salida JSON.

Se usa solo donde no hay verificación mecánica (DIRECT vs PLAN, sobre todo). El
juez llama a un modelo capaz (vía ModelRunner) con una rúbrica congelada y
parsea un JSON `{"mode": ...}`. FakeJudge permite testear sin llamadas.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from routerpolicy.harness.runner import ModelRunner
from routerpolicy.schema.core import Mode

# Rúbrica FIJA (parte del contrato de etiquetado; cambiarla invalida el acuerdo).
JUDGE_RUBRIC = (
    "You classify a task by the execution mode it requires. Definitions:\n"
    "- DIRECT: answerable in a single response from knowledge; no tools, no "
    "multi-step planning.\n"
    "- TOOL_CALL: needs calling external tools/APIs in a short loop, but no "
    "deep planning.\n"
    "- PLAN: needs decomposition, planning or reflection across multiple steps "
    "before acting.\n"
    'Reply with ONLY a JSON object: {"mode": "DIRECT"|"TOOL_CALL"|"PLAN"}.'
)


class JudgeError(ValueError):
    """El juez devolvió una salida no parseable o inválida."""


def build_judge_prompt(task_prompt: str) -> str:
    """Prompt del juez: rúbrica + la tarea a clasificar."""
    return f"{JUDGE_RUBRIC}\n\nTask:\n{task_prompt}\n\nJSON:"


_JSON_OBJ = re.compile(r"\{.*?\}", re.DOTALL)


def parse_judge_output(text: str) -> Mode:
    """Extrae el modo del JSON del juez; lanza JudgeError si no es válido."""
    match = _JSON_OBJ.search(text)
    if match is None:
        raise JudgeError(f"sin objeto JSON en la salida del juez: {text!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeError(f"JSON malformado del juez: {exc}") from exc
    mode_raw = data.get("mode") if isinstance(data, dict) else None
    if not isinstance(mode_raw, str) or mode_raw not in Mode.__members__:
        raise JudgeError(f"mode inválido del juez: {mode_raw!r}")
    return Mode[mode_raw]


class ModeJudge(Protocol):
    """Contrato del juez de modo."""

    def judge(self, task_prompt: str) -> Mode: ...


class LlmModeJudge:
    """Juez respaldado por un ModelRunner (modelo capaz vía API/local).

    Reintenta si el modelo no devuelve JSON parseable (un modelo local puede
    fallar el formato de vez en cuando).
    """

    def __init__(self, runner: ModelRunner, max_attempts: int = 3) -> None:
        self._runner = runner
        self._max_attempts = max_attempts

    def judge(self, task_prompt: str) -> Mode:
        prompt = build_judge_prompt(task_prompt)
        last_error: Exception | None = None
        for _ in range(self._max_attempts):
            try:
                return parse_judge_output(self._runner.complete(prompt).text)
            except Exception as exc:
                last_error = exc
        raise JudgeError(f"juez falló tras {self._max_attempts} intentos: {last_error}")


class FakeJudge:
    """Juez determinista para tests: mapea prompt->Mode."""

    def __init__(self, responder: dict[str, Mode], default: Mode = Mode.DIRECT) -> None:
        self._responder = responder
        self._default = default

    def judge(self, task_prompt: str) -> Mode:
        return self._responder.get(task_prompt, self._default)
