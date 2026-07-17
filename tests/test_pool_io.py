"""Tests de carga del pool (construcción, sin red)."""

from __future__ import annotations

from pathlib import Path

import pytest

from routerpolicy.harness.cache import CachingRunner
from routerpolicy.harness.pool_io import load_pool

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_POOL = REPO_ROOT / "configs" / "pool.local.yaml"


def test_load_local_pool(tmp_path: Path) -> None:
    runners = load_pool(LOCAL_POOL, cache_dir=tmp_path)
    assert [r.model_id for r in runners] == [
        "qwen2.5-coder:1.5b",
        "qwen2.5-coder:7b",
        "esdrac:latest",
    ]
    assert all(isinstance(r, CachingRunner) for r in runners)


def test_openai_compat_requires_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "pool.yaml"
    cfg.write_text(
        "tiers:\n"
        "  - id: local\n"
        "    backend: ollama\n"
        "  - backend: openai_compat\n"
        "    model: gpt-x\n"
        "    base_url: https://api.example.com/v1\n"
        "    api_key_env: SHIORI_TEST_KEY\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SHIORI_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SHIORI_TEST_KEY"):
        load_pool(cfg, cache_dir=tmp_path)

    monkeypatch.setenv("SHIORI_TEST_KEY", "secret")
    runners = load_pool(cfg, cache_dir=tmp_path)
    assert [r.model_id for r in runners] == ["local", "gpt-x"]


def test_rejects_missing_tiers(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("foo: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tiers"):
        load_pool(cfg, cache_dir=tmp_path)
