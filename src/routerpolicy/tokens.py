"""Estimación ligera de tokens, sin dependencias pesadas.

En Fase 1 no cargamos el tokenizer real (Gemma/Qwen) para mantener CI offline y
ligera. Usamos una heurística de ~4 caracteres por token (regla estándar). Como
el render es compacto y rígido, el margen respecto al presupuesto de 1000 tokens
es enorme, así que el error de la heurística es irrelevante para el gate. Fase 2+
puede sustituirla por un conteo con el tokenizer real cuando esté disponible.
"""

from __future__ import annotations

import math

CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Estimación conservadora del número de tokens de `text`."""
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN)
