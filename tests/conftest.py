"""Fixtures compartidas de los tests (evita imports entre módulos de test)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from routerpolicy.schema.core import Locality, ModelSpec, Registry


def _golden_registry() -> Registry:
    """Registro determinista usado por el golden-file del render."""
    return Registry(
        models=(
            ModelSpec(
                id="local-mini",
                tags=("code", "general"),
                context_window=8192,
                cost=1,
                locality=Locality.LOCAL,
                supports_tools=False,
            ),
            ModelSpec(
                id="local-med",
                tags=("code", "reasoning", "tools"),
                context_window=32768,
                cost=3,
                locality=Locality.LOCAL,
                supports_tools=True,
            ),
            ModelSpec(
                id="api-large",
                tags=("code", "reasoning", "plan", "tools", "general"),
                context_window=200000,
                cost=20,
                locality=Locality.API,
                supports_tools=True,
            ),
        )
    )


def _big_registry(n: int) -> Registry:
    """Registro sintético de `n` modelos para el test de presupuesto."""
    all_tags = ("code", "reasoning", "plan", "tools", "general", "math", "vision")
    models = tuple(
        ModelSpec(
            id=f"provider-model-variant-{i:02d}",
            tags=all_tags[: (i % len(all_tags)) + 2],
            context_window=4096 * (i + 1),
            cost=float(i + 1),
            locality=Locality.LOCAL if i % 2 == 0 else Locality.API,
            supports_tools=i % 3 != 0,
        )
        for i in range(n)
    )
    return Registry(models=models)


@pytest.fixture
def golden_registry() -> Registry:
    return _golden_registry()


@pytest.fixture
def big_registry() -> Callable[[int], Registry]:
    return _big_registry
