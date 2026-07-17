"""Tests del render del registro: golden-file y presupuesto de tokens."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from routerpolicy.registry.render import RENDER_TOKEN_BUDGET, render_registry_prompt
from routerpolicy.schema.core import Registry
from routerpolicy.tokens import estimate_tokens

GOLDEN = Path(__file__).parent / "golden" / "registry_render.txt"


def test_render_matches_golden(golden_registry: Registry) -> None:
    expected = GOLDEN.read_text(encoding="utf-8").rstrip("\n")
    rendered = render_registry_prompt(golden_registry)
    assert rendered == expected


def test_render_has_no_trailing_newline(golden_registry: Registry) -> None:
    # El render no debe terminar en salto: el golden se compara con rstrip, así
    # que esta aserción evita que una regresión de salto final pase inadvertida.
    assert not render_registry_prompt(golden_registry).endswith("\n")


def test_render_one_line_per_model_plus_header(golden_registry: Registry) -> None:
    lines = render_registry_prompt(golden_registry).splitlines()
    assert lines[0] == "Available models:"
    assert len(lines) == 1 + len(golden_registry.models)


def test_render_budget_8_models(big_registry: Callable[[int], Registry]) -> None:
    rendered = render_registry_prompt(big_registry(8))
    assert estimate_tokens(rendered) <= RENDER_TOKEN_BUDGET
