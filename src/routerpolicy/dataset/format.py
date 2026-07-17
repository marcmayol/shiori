"""Formato de ejemplo de entrenamiento/inferencia (chat de 2-3 turnos).

system fijo y corto · user = registro renderizado + tarea · assistant = JSON
mínimo canónico. Es el MISMO formato en entrenamiento e inferencia (sección 2).
"""

from __future__ import annotations

from routerpolicy.registry.render import render_registry_prompt
from routerpolicy.schema.core import Registry, RoutingDecision

# System prompt FIJO (parte del contrato). Corto para no gastar presupuesto ni
# capacidad. En inglés por estabilidad y economía de tokens; la tarea puede ir
# en cualquier idioma.
SYSTEM_PROMPT = (
    "You are a model-routing policy. Given the available models and a task, "
    'reply with a single JSON object: {"mode": <DIRECT|TOOL_CALL|PLAN>, '
    '"model_id": <one id from the list>}. Pick the smallest sufficient model. '
    "Output only the JSON."
)

ChatMessage = dict[str, str]


def build_user_message(registry: Registry, task: str) -> str:
    """Contenido del turno de usuario: registro renderizado + la tarea."""
    return f"{render_registry_prompt(registry)}\n\nTask:\n{task}"


def build_chat_example(
    registry: Registry,
    task: str,
    decision: RoutingDecision | None = None,
) -> list[ChatMessage]:
    """Construye la lista de mensajes del ejemplo.

    Si `decision` se pasa (entrenamiento/etiqueta conocida), se añade el turno
    del assistant con el JSON canónico. Si no (inferencia), se omite.
    """
    messages: list[ChatMessage] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(registry, task)},
    ]
    if decision is not None:
        messages.append({"role": "assistant", "content": decision.to_canonical_json()})
    return messages
