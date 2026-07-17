"""Decoding constreñido generado a partir del registro de cada ejemplo.

Convierte la generación en clasificación efectiva (sección 2): la salida queda
limitada al enum de modos y a los `model_id` presentes en el registro concreto.

- `build_decision_regex`: patrón que admite EXACTAMENTE la forma canónica válida
  (mode ∈ enum, model_id ∈ registro). Es el artefacto testeable de Fase 1.
- `build_json_schema`: JSON schema dinámico para los structured outputs de Ollama.

La gramática GBNF de llama.cpp se genera en la Fase 6 (empaquetado), donde hay
un test de integración real contra el runtime; aquí la referencia es el regex.
"""

from __future__ import annotations

import re

from routerpolicy.schema.core import Mode, Registry


def build_decision_regex(registry: Registry) -> re.Pattern[str]:
    """Regex que solo admite la forma canónica `to_canonical_json` válida.

    Coincide con `json.dumps({"mode": ..., "model_id": ...})` (espacios tras
    `:` y `,`), con `mode` en el enum y `model_id` en el registro.
    """
    modes = "|".join(re.escape(m.value) for m in Mode)
    ids = "|".join(re.escape(mid) for mid in registry.model_ids)
    pattern = rf'^\{{"mode": "(?:{modes})", "model_id": "(?:{ids})"\}}$'
    return re.compile(pattern)


def build_json_schema(registry: Registry) -> dict[str, object]:
    """JSON schema dinámico (Ollama structured outputs) para este registro."""
    return {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": [m.value for m in Mode]},
            "model_id": {"type": "string", "enum": list(registry.model_ids)},
        },
        "required": ["mode", "model_id"],
        "additionalProperties": False,
    }
