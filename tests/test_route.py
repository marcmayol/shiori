"""Tests de route() y sus helpers (Fase 6), incluido un test de integración real."""

from __future__ import annotations

import random
import urllib.error

import pytest

from routerpolicy.inference.constraints import build_gbnf
from routerpolicy.inference.route import (
    apply_capability_prior,
    apply_mode_prior,
    mode_dist_from_logprobs,
    most_capable_id,
    route,
)
from routerpolicy.registry.synthetic import generate_pool
from routerpolicy.schema.core import Locality, Mode, ModelSpec, Registry


def _pool() -> Registry:
    return Registry(
        models=tuple(m.spec for m in generate_pool(random.Random(0), capabilities=[1, 2, 4]))
    )


def test_build_gbnf_structure() -> None:
    reg = _pool()
    g = build_gbnf(reg)
    assert "root ::=" in g
    assert "mode ::=" in g and '"DIRECT"' in g and '"PLAN"' in g and '"TOOL_CALL"' in g
    for mid in reg.model_ids:
        assert f'"{mid}"' in g


def test_mode_dist_from_logprobs() -> None:
    logprobs = [
        {"token": '{"', "logprob": 0.0, "top_logprobs": []},
        {"token": "mode", "logprob": 0.0, "top_logprobs": []},
        {
            "token": "DIRECT",
            "logprob": -0.1,
            "top_logprobs": [
                {"token": "DIRECT", "logprob": -0.1},
                {"token": "PLAN", "logprob": -2.0},
                {"token": "TOOL", "logprob": -5.0},
            ],
        },
    ]
    dist = mode_dist_from_logprobs(logprobs)
    assert dist[Mode.DIRECT] == pytest.approx(-0.1)
    assert dist[Mode.PLAN] == pytest.approx(-2.0)
    assert dist[Mode.TOOL_CALL] == pytest.approx(-5.0)


def test_apply_mode_prior_flips_to_plan() -> None:
    dist = {Mode.DIRECT: -0.5, Mode.PLAN: -1.0, Mode.TOOL_CALL: -3.0}
    assert apply_mode_prior(dist, 0.0) is Mode.DIRECT
    assert apply_mode_prior(dist, 0.6) is Mode.PLAN  # prior supera la brecha 0.5
    assert apply_mode_prior({}, 0.5) is None


def _m(mid: str, tags: tuple[str, ...], cost: float) -> ModelSpec:
    return ModelSpec(
        id=mid,
        tags=tags,
        context_window=8192,
        cost=cost,
        locality=Locality.LOCAL,
        supports_tools=False,
    )


def test_apply_capability_prior_bumps_up() -> None:
    reg = Registry(
        models=(
            _m("weak", ("code",), 1.0),  # cap 1
            _m("mid", ("code", "reasoning"), 3.0),  # cap 2
            _m("strong", ("code", "reasoning", "planning"), 6.0),  # cap 3
        )
    )
    assert apply_capability_prior(reg, "weak", 0.0) == "weak"  # sin prior, sin cambio
    assert apply_capability_prior(reg, "weak", 1.0) == "mid"  # sube 1 nivel -> el más barato cap>=2
    assert apply_capability_prior(reg, "weak", 2.0) == "strong"  # sube 2 -> cap>=3


def test_most_capable_id() -> None:
    reg = Registry(
        models=(
            _m("a", ("code",), 1.0),
            _m("b", ("code", "reasoning", "planning", "expert"), 9.0),
        )
    )
    assert most_capable_id(reg) == "b"


def _ollama_up() -> bool:
    import urllib.request

    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3).read()
        return True
    except (urllib.error.URLError, OSError):
        return False


def test_route_integration_real_ollama() -> None:
    """Integración REAL contra Ollama local (sin red externa). Skip si no está."""
    if not _ollama_up():
        pytest.skip("Ollama no disponible en localhost:11434")
    reg = _pool()
    try:
        decision = route("Write a function that returns the nth prime.", reg, plan_prior=0.5)
    except (urllib.error.URLError, OSError) as exc:
        pytest.skip(f"modelo shiori-router no disponible: {exc}")
    assert decision.mode in set(Mode)
    assert reg.contains(decision.model_id)  # siempre una decisión válida del pool
