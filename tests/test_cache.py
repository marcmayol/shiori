"""Tests de la cache de completions y el runner con cache."""

from __future__ import annotations

from pathlib import Path

from routerpolicy.harness.cache import (
    CachingRunner,
    CompletionCache,
    completion_key,
)
from routerpolicy.harness.runner import Completion, FakeRunner


def test_key_is_stable_and_distinguishes() -> None:
    k1 = completion_key("m", "prompt")
    assert k1 == completion_key("m", "prompt")
    assert k1 != completion_key("m2", "prompt")
    assert k1 != completion_key("m", "prompt2")


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = CompletionCache(tmp_path)
    key = completion_key("m", "p")
    assert cache.get(key) is None
    c = Completion("m", "text", 3, 4)
    cache.put(key, c)
    got = cache.get(key)
    assert got == c


def test_caching_runner_serves_from_cache(tmp_path: Path) -> None:
    inner = FakeRunner("m", default="out")
    cache = CompletionCache(tmp_path)
    runner = CachingRunner(inner, cache)

    first = runner.complete("hello")
    assert first.text == "out"
    assert runner.misses == 1 and runner.hits == 0
    assert inner.calls == ["hello"]

    second = runner.complete("hello")
    assert second == first
    assert runner.hits == 1
    assert inner.calls == ["hello"]  # no se volvió a llamar al inner


def test_caching_runner_persists_across_instances(tmp_path: Path) -> None:
    cache = CompletionCache(tmp_path)
    CachingRunner(FakeRunner("m", default="out"), cache).complete("p")

    # simula reinicio: nuevo inner que fallaría si se llamara, pero hay cache
    fresh = CachingRunner(FakeRunner("m", default="DIFFERENT"), cache)
    got = fresh.complete("p")
    assert got.text == "out"  # sirve desde disco, no re-ejecuta
    assert fresh.hits == 1
