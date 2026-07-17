"""Tests del decoding constreñido (regex + JSON schema) desde el registro."""

from __future__ import annotations

from routerpolicy.inference.constraints import build_decision_regex, build_json_schema
from routerpolicy.schema.core import Mode, Registry, RoutingDecision


def test_regex_admits_all_valid_canonical_decisions(golden_registry: Registry) -> None:
    regex = build_decision_regex(golden_registry)
    for mode in Mode:
        for model_id in golden_registry.model_ids:
            text = RoutingDecision(mode=mode, model_id=model_id).to_canonical_json()
            assert regex.match(text), text


def test_regex_rejects_invalid(golden_registry: Registry) -> None:
    regex = build_decision_regex(golden_registry)
    invalid = [
        '{"mode": "WRONG", "model_id": "local-mini"}',  # modo fuera del enum
        '{"mode": "DIRECT", "model_id": "ghost"}',  # id fuera del registro
        '{"mode":"DIRECT","model_id":"local-mini"}',  # sin espacios canónicos
        '{"model_id": "local-mini", "mode": "DIRECT"}',  # orden de claves distinto
        '{"mode": "DIRECT", "model_id": "local-mini"} ',  # basura al final
        'prefix {"mode": "DIRECT", "model_id": "local-mini"}',  # basura al inicio
    ]
    for text in invalid:
        assert regex.match(text) is None, text


def test_regex_ids_are_escaped() -> None:
    # Un id con caracteres especiales de regex no debe romper ni ampliar el match.
    from routerpolicy.schema.core import Locality, ModelSpec, Registry

    reg = Registry(
        models=(
            ModelSpec(
                id="a.b+c",
                tags=("code",),
                context_window=1024,
                cost=1,
                locality=Locality.LOCAL,
                supports_tools=False,
            ),
        )
    )
    regex = build_decision_regex(reg)
    assert regex.match('{"mode": "DIRECT", "model_id": "a.b+c"}')
    assert regex.match('{"mode": "DIRECT", "model_id": "axbxc"}') is None


def test_json_schema_structure(golden_registry: Registry) -> None:
    schema = build_json_schema(golden_registry)
    props = schema["properties"]
    assert isinstance(props, dict)
    assert props["mode"]["enum"] == ["DIRECT", "TOOL_CALL", "PLAN"]
    assert props["model_id"]["enum"] == list(golden_registry.model_ids)
    assert schema["required"] == ["mode", "model_id"]
    assert schema["additionalProperties"] is False
