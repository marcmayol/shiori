"""Cascada offline: etiqueta el modelo mínimo suficiente ejecutando y verificando.

Se ejecuta cada tarea de código empezando por el modelo más barato; se verifica
con los tests; la etiqueta de suficiencia es el PRIMER modelo que pasa. Si
ninguno pasa, no hay modelo suficiente en el pool (el consumidor decide el
fallback; en el dataset será "el más capaz", ver Fase 3).

Este es el mismo mecanismo try→verify→next que en producción, usado aquí offline
para generar las etiquetas.
"""

from __future__ import annotations

from dataclasses import dataclass

from routerpolicy.harness.runner import Completion, ModelRunner, extract_code
from routerpolicy.harness.tasks import CodeTask
from routerpolicy.harness.verify import DEFAULT_TIMEOUT_S, verify_code

_CODE_INSTRUCTION = (
    "Implement the following in Python. Output only the function definition(s), no explanation.\n\n"
)


def build_code_prompt(task: CodeTask) -> str:
    """Prompt determinista que se envía a los modelos del pool de generación."""
    return f"{_CODE_INSTRUCTION}{task.prompt}"


@dataclass(frozen=True)
class CascadeAttempt:
    """Un intento de un modelo del pool sobre una tarea."""

    model_id: str
    passed: bool
    error: str | None
    prompt_tokens: int
    completion_tokens: int
    errored: bool = False  # True si falló la INFRA (timeout/red), no la verificación


@dataclass(frozen=True)
class CascadeResult:
    """Resultado de la cascada para una tarea de código."""

    task_id: str
    sufficient_model_id: str | None  # None si ningún modelo del pool pasó
    attempts: tuple[CascadeAttempt, ...]

    @property
    def any_sufficient(self) -> bool:
        return self.sufficient_model_id is not None


def _complete_with_retry(
    runner: ModelRunner, prompt: str, max_attempts: int
) -> tuple[Completion | None, str | None]:
    """Intenta la completion con reintentos. Devuelve (completion|None, error).

    Un timeout/red suele resolverse al reintentar (el modelo ya quedó cargado en
    VRAM tras el primer intento fallido).
    """
    last_error: str | None = None
    for _ in range(max_attempts):
        try:
            return runner.complete(prompt), None
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return None, last_error


def run_cascade(
    task: CodeTask,
    runners: list[ModelRunner],
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_attempts: int = 2,
) -> CascadeResult:
    """Ejecuta la cascada sobre una tarea.

    `runners` DEBE venir ordenado de más barato a más capaz. Se detiene en el
    primer modelo que pasa la verificación (no ejecuta los más caros). Si un
    modelo falla de INFRA (timeout/red) tras los reintentos, se marca `errored`
    y se escala al siguiente en vez de abortar el run.
    """
    prompt = build_code_prompt(task)
    attempts: list[CascadeAttempt] = []
    sufficient: str | None = None
    for runner in runners:
        completion, infra_error = _complete_with_retry(runner, prompt, max_attempts)
        if completion is None:
            attempts.append(
                CascadeAttempt(
                    model_id=runner.model_id,
                    passed=False,
                    error=infra_error,
                    prompt_tokens=0,
                    completion_tokens=0,
                    errored=True,
                )
            )
            continue  # escala al siguiente tier
        candidate = extract_code(completion.text)
        result = verify_code(candidate, task.test_code, timeout_s=timeout_s)
        attempts.append(
            CascadeAttempt(
                model_id=runner.model_id,
                passed=result.passed,
                error=result.error,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
            )
        )
        if result.passed:
            sufficient = runner.model_id
            break
    return CascadeResult(
        task_id=task.task_id,
        sufficient_model_id=sufficient,
        attempts=tuple(attempts),
    )
