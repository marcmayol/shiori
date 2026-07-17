"""Tests de los tipos núcleo del contrato."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from routerpolicy.schema.core import (
    SCHEMA_VERSION,
    Label,
    Locality,
    Mode,
    ModelSpec,
    Provenance,
    Registry,
    RoutingDecision,
)


def test_registry_ids_and_lookup(golden_registry: Registry) -> None:
    reg = golden_registry
    assert reg.model_ids == ("local-mini", "local-med", "api-large")
    assert reg.contains("local-med")
    assert not reg.contains("nope")
    spec = reg.get("api-large")
    assert spec is not None
    assert spec.locality is Locality.API
    assert reg.get("nope") is None


def test_registry_rejects_duplicate_ids() -> None:
    dup = ModelSpec(
        id="x",
        tags=("code",),
        context_window=1024,
        cost=1,
        locality=Locality.LOCAL,
        supports_tools=False,
    )
    with pytest.raises(ValidationError):
        Registry(models=(dup, dup))


def test_registry_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        Registry(models=())


def test_modelspec_requires_tags_and_positive_ctx() -> None:
    with pytest.raises(ValidationError):
        ModelSpec(
            id="x",
            tags=(),
            context_window=1024,
            cost=1,
            locality=Locality.LOCAL,
            supports_tools=False,
        )
    with pytest.raises(ValidationError):
        ModelSpec(
            id="x",
            tags=("code",),
            context_window=0,
            cost=1,
            locality=Locality.LOCAL,
            supports_tools=False,
        )


def test_routing_decision_canonical_json() -> None:
    dec = RoutingDecision(mode=Mode.DIRECT, model_id="local-mini")
    assert dec.to_canonical_json() == '{"mode": "DIRECT", "model_id": "local-mini"}'


def test_label_provenance_and_decision() -> None:
    label = Label(
        mode=Mode.PLAN,
        model_id="api-large",
        mode_source=Provenance.JUDGE,
        sufficiency_source=Provenance.VERIFIED,
        scores={"pass_rate": 1.0},
    )
    assert label.schema_version == SCHEMA_VERSION
    assert label.decision == RoutingDecision(mode=Mode.PLAN, model_id="api-large")
    assert label.scores["pass_rate"] == 1.0
