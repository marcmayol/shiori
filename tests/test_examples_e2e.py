"""DoD Fase 1: 5 ejemplos a mano pasan esquema + gramática de punta a punta.

Cada ejemplo: (registro, tarea, decisión). Se construye el chat, se extrae el
turno del assistant y se comprueba que (1) valida contra el esquema/registro y
(2) lo admite el regex del decoding constreñido del mismo registro.
"""

from __future__ import annotations

from routerpolicy.dataset.format import SYSTEM_PROMPT, build_chat_example
from routerpolicy.inference.constraints import build_decision_regex
from routerpolicy.schema.core import Locality, Mode, ModelSpec, Registry, RoutingDecision
from routerpolicy.schema.validation import validate_decision_json


def _reg(*specs: ModelSpec) -> Registry:
    return Registry(models=specs)


def _m(
    mid: str,
    tags: tuple[str, ...],
    ctx: int,
    cost: float,
    loc: Locality,
    tools: bool,
) -> ModelSpec:
    return ModelSpec(
        id=mid,
        tags=tags,
        context_window=ctx,
        cost=cost,
        locality=loc,
        supports_tools=tools,
    )


# 5 ejemplos heterogéneos construidos a mano (registros, tareas y modos variados).
HAND_EXAMPLES: list[tuple[Registry, str, RoutingDecision]] = [
    (
        _reg(
            _m("mini", ("code", "general"), 8192, 1, Locality.LOCAL, False),
            _m("big-api", ("code", "reasoning", "plan"), 200000, 20, Locality.API, True),
        ),
        "What is the capital of France?",
        RoutingDecision(mode=Mode.DIRECT, model_id="mini"),
    ),
    (
        _reg(
            _m("local-tools", ("tools", "code"), 16384, 2, Locality.LOCAL, True),
            _m("cheap", ("general",), 4096, 1, Locality.LOCAL, False),
        ),
        "Call the weather API for Tokyo and return today's temperature.",
        RoutingDecision(mode=Mode.TOOL_CALL, model_id="local-tools"),
    ),
    (
        _reg(
            _m("q-mini", ("code",), 8192, 1, Locality.LOCAL, False),
            _m("q-med", ("code", "reasoning"), 32768, 4, Locality.LOCAL, True),
            _m("opus", ("code", "reasoning", "plan"), 200000, 30, Locality.API, True),
        ),
        "Design and implement a distributed rate limiter with tests and a rollout plan.",
        RoutingDecision(mode=Mode.PLAN, model_id="opus"),
    ),
    (
        _reg(
            _m("a", ("code",), 8192, 1, Locality.LOCAL, False),
            _m("b", ("code", "reasoning"), 32768, 3, Locality.LOCAL, False),
            _m("c", ("code", "reasoning", "math"), 65536, 6, Locality.LOCAL, True),
        ),
        "Write a Python function that returns the nth Fibonacci number.",
        RoutingDecision(mode=Mode.DIRECT, model_id="a"),
    ),
    (
        _reg(
            _m("id.with+special", ("tools",), 16384, 2, Locality.LOCAL, True),
            _m("plain", ("general",), 4096, 1, Locality.LOCAL, False),
        ),
        "Fetch the user's calendar and schedule a meeting tomorrow at 10am.",
        RoutingDecision(mode=Mode.TOOL_CALL, model_id="id.with+special"),
    ),
]


def test_exactly_five_examples() -> None:
    assert len(HAND_EXAMPLES) == 5


def test_hand_examples_pass_schema_and_grammar_end_to_end() -> None:
    for registry, task, decision in HAND_EXAMPLES:
        # la decisión etiquetada debe referirse a un modelo del propio registro
        assert registry.contains(decision.model_id)

        messages = build_chat_example(registry, task, decision)
        assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert messages[1]["role"] == "user"
        assert task in messages[1]["content"]
        assistant = messages[2]["content"]

        # (1) valida contra el esquema y el registro del propio ejemplo
        parsed = validate_decision_json(assistant, registry)
        assert parsed == decision

        # (2) lo admite el decoding constreñido generado desde ese registro
        regex = build_decision_regex(registry)
        assert regex.match(assistant), assistant


def test_inference_example_omits_assistant() -> None:
    registry, task, _ = HAND_EXAMPLES[0]
    messages = build_chat_example(registry, task)
    assert [m["role"] for m in messages] == ["system", "user"]
