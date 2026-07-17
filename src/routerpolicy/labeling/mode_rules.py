"""Reglas deterministas para los casos obvios de modo (sección Fase 2).

- Tarea con herramientas declaradas -> TOOL_CALL.
- Pregunta factual corta -> DIRECT.
El resto devuelve None: lo resuelve el LLM juez (labeling/judge).
"""

from __future__ import annotations

from routerpolicy.harness.tasks import ChatTask, ToolTask
from routerpolicy.schema.core import Mode

# Señales de que la tarea requiere planificar/reflexionar (no es DIRECT).
_PLAN_KEYWORDS = frozenset(
    {
        "plan",
        "design",
        "architecture",
        "architect",
        "strategy",
        "roadmap",
        "step-by-step",
        "step by step",
        "multi-step",
        "break down",
        "outline",
        "compare and",
        "trade-off",
        "tradeoff",
        "pros and cons",
    }
)

# Longitud máxima (en palabras) para considerar una pregunta "corta".
_SHORT_FACTUAL_MAX_WORDS = 12


def _has_plan_signal(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in _PLAN_KEYWORDS)


def is_short_factual(prompt: str) -> bool:
    """True si el prompt es una pregunta corta y sin señales de planificación."""
    stripped = prompt.strip()
    if not stripped.endswith("?"):
        return False
    if _has_plan_signal(stripped):
        return False
    return len(stripped.split()) <= _SHORT_FACTUAL_MAX_WORDS


def rule_mode_for_tool_task(task: ToolTask) -> Mode:
    """Herramientas declaradas -> TOOL_CALL (caso obvio)."""
    return Mode.TOOL_CALL


def rule_mode_for_chat_task(task: ChatTask) -> Mode | None:
    """DIRECT si es pregunta factual corta; None si necesita el juez."""
    if is_short_factual(task.prompt):
        return Mode.DIRECT
    return None
