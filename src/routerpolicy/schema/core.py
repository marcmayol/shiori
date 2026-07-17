"""Tipos núcleo del contrato de Shiori.

Estos tipos son el contrato FIJO del producto (sección 2 del plan): el formato
de entrada (registro de modelos) y de salida (`{mode, model_id}`) no se augmenta
ni se reinterpreta, se congela. Cambiar algo aquí obliga a subir SCHEMA_VERSION.
"""

from __future__ import annotations

import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Versión del esquema de datos/etiquetas. Cualquier cambio incompatible en los
# tipos de este módulo o en el formato de salida debe incrementarla.
SCHEMA_VERSION = "1.0"


class Mode(StrEnum):
    """Modo de ejecución que predice la política."""

    DIRECT = "DIRECT"  # conocimiento embebido, una sola completion
    TOOL_CALL = "TOOL_CALL"  # bucle de herramientas corto
    PLAN = "PLAN"  # requiere planificar; se delega a un modelo capaz


class Locality(StrEnum):
    """Dónde corre el modelo."""

    LOCAL = "local"
    API = "api"


class Provenance(StrEnum):
    """Cómo se obtuvo una etiqueta (procedencia)."""

    VERIFIED = "verified"  # ejecución real + verificación mecánica (tests)
    JUDGE = "judge"  # LLM juez con rúbrica
    RULE = "rule"  # regla determinista


class ModelSpec(BaseModel):
    """Especificación de un modelo dentro de un registro/pool.

    Campos mínimos del contrato (una línea por modelo al renderizar).
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    id: str = Field(min_length=1)
    tags: tuple[str, ...] = Field(min_length=1)  # capacidades: code, reasoning, ...
    context_window: int = Field(gt=0)
    cost: float = Field(ge=0)  # coste relativo (adimensional)
    locality: Locality
    supports_tools: bool


class Registry(BaseModel):
    """Pool de modelos que se le presenta a la política en un prompt.

    El ORDEN de `models` es significativo: se augmenta en la Fase 3 pero el
    render es determinista respecto a este orden.
    """

    model_config = ConfigDict(frozen=True)

    models: tuple[ModelSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> Registry:
        ids = [m.id for m in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("los model_id del registro deben ser únicos")
        return self

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(m.id for m in self.models)

    def contains(self, model_id: str) -> bool:
        return model_id in self.model_ids

    def get(self, model_id: str) -> ModelSpec | None:
        for m in self.models:
            if m.id == model_id:
                return m
        return None


class RoutingDecision(BaseModel):
    """La salida de la política: modo + modelo elegido."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    mode: Mode
    model_id: str = Field(min_length=1)

    def to_canonical_json(self) -> str:
        """Serialización canónica y estable usada en entrenamiento e inferencia.

        El orden de claves es fijo (mode, model_id). Esta es exactamente la
        forma que el decoding constreñido debe admitir.
        """
        return json.dumps(
            {"mode": self.mode.value, "model_id": self.model_id},
            ensure_ascii=True,
        )


class Label(BaseModel):
    """Etiqueta de un ejemplo con su procedencia completa.

    La etiqueta es función del registro presentado ("el mínimo suficiente en
    ESTE pool"); al augmentar registros se recalcula (Fase 3).
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    mode: Mode
    model_id: str = Field(min_length=1)
    mode_source: Provenance  # cómo se decidió el modo
    sufficiency_source: Provenance  # cómo se decidió que model_id basta
    scores: dict[str, float] = Field(default_factory=dict)  # scores de la corrida
    schema_version: str = SCHEMA_VERSION

    @property
    def decision(self) -> RoutingDecision:
        return RoutingDecision(mode=self.mode, model_id=self.model_id)
