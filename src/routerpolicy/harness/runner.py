"""Runner de modelos: protocolo común + fake para tests + utilidades.

Todos los backends (Ollama local, API) implementan el mismo protocolo, así el
resto del harness (cascada, coste, juez) es agnóstico al modelo concreto.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from routerpolicy.tokens import estimate_tokens


@dataclass(frozen=True)
class Completion:
    """Resultado de una completion, con contabilidad de tokens.

    `prompt_tokens`/`completion_tokens` pueden venir del backend o estimarse.
    """

    model_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int

    @classmethod
    def estimated(cls, model_id: str, prompt: str, text: str) -> Completion:
        """Crea una Completion estimando el conteo de tokens desde el texto."""
        return cls(
            model_id=model_id,
            text=text,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(text),
        )


@runtime_checkable
class ModelRunner(Protocol):
    """Contrato mínimo de un modelo ejecutable en el harness offline."""

    @property
    def model_id(self) -> str: ...

    def complete(self, prompt: str) -> Completion: ...


class FakeRunner:
    """Runner determinista para tests: mapea prompt→texto, o función.

    Registra las llamadas para poder aseverar cuántas veces se invocó cada tier.
    """

    def __init__(
        self,
        model_id: str,
        responder: dict[str, str] | None = None,
        default: str = "",
    ) -> None:
        self._model_id = model_id
        self._responder = responder or {}
        self._default = default
        self.calls: list[str] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete(self, prompt: str) -> Completion:
        self.calls.append(prompt)
        text = self._responder.get(prompt, self._default)
        return Completion.estimated(self._model_id, prompt, text)


_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Extrae el bloque de código de una completion.

    Si hay vallas markdown ```...``` devuelve el primer bloque; si no, el texto
    tal cual (asumiendo que ya es código). Recorta espacios envolventes.
    """
    match = _FENCE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()
