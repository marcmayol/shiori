"""Tests del generador de pools sintéticos (Fase 3)."""

from __future__ import annotations

import random

from routerpolicy.registry.render import render_registry_prompt
from routerpolicy.registry.synthetic import (
    MAX_CAPABILITY,
    generate_pool,
    pool_signature,
)
from routerpolicy.schema.core import Registry


def test_generate_pool_deterministic() -> None:
    a = generate_pool(random.Random(42))
    b = generate_pool(random.Random(42))
    assert [m.spec.id for m in a] == [m.spec.id for m in b]
    assert [m.capability for m in a] == [m.capability for m in b]


def test_pool_size_in_range() -> None:
    for seed in range(30):
        pool = generate_pool(random.Random(seed), min_models=2, max_models=8)
        assert 2 <= len(pool) <= 8


def test_ids_unique_within_pool() -> None:
    pool = generate_pool(random.Random(1), capabilities=[0, 1, 2, 3, 4, 0, 1, 2])
    ids = [m.spec.id for m in pool]
    assert len(ids) == len(set(ids))


def test_capability_encoded_in_tags() -> None:
    # reasoning ⟺ cap>=2, planning ⟺ cap>=3, expert ⟺ cap==4
    for cap in range(MAX_CAPABILITY + 1):
        pool = generate_pool(random.Random(7), capabilities=[cap])
        tags = set(pool[0].spec.tags)
        assert ("reasoning" in tags) == (cap >= 2)
        assert ("planning" in tags) == (cap >= 3)
        assert ("expert" in tags) == (cap >= 4)


def test_synthetic_specs_render_with_fixed_format() -> None:
    pool = generate_pool(random.Random(3), capabilities=[0, 2, 4])
    reg = Registry(models=tuple(m.spec for m in pool))
    rendered = render_registry_prompt(reg)
    assert rendered.startswith("Available models:")
    assert len(rendered.splitlines()) == 1 + len(pool)


def test_pool_signature_ignores_names() -> None:
    # dos pools con mismas capacidades/tools pero distintos seeds de nombres
    p1 = generate_pool(random.Random(10), capabilities=[1, 3])
    p2 = generate_pool(random.Random(11), capabilities=[1, 3])
    # las firmas comparten tamaño y capacidades (tools puede variar por azar)
    s1, s2 = pool_signature(p1), pool_signature(p2)
    assert s1[0] == s2[0] == 2
    assert s1[1] == s2[1] == (1, 3)
