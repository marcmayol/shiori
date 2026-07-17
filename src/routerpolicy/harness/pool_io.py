"""Carga del pool de generación desde YAML (modelos reales que se ejecutan).

Construye la lista ordenada de runners (barato -> capaz) desde una config,
envolviendo cada uno en CachingRunner con una cache compartida. Soporta backend
`ollama` (local) y `openai_compat` (API; la clave se lee de una variable de
entorno indicada en la config, nunca se hardcodea).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from routerpolicy.harness.backends import OllamaRunner, OpenAICompatRunner
from routerpolicy.harness.cache import CachingRunner, CompletionCache
from routerpolicy.harness.runner import ModelRunner

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_CACHE_DIR = "data/completions_cache"


def _build_inner(tier: dict[str, Any], host: str) -> ModelRunner:
    backend = str(tier.get("backend", "ollama"))
    if backend == "ollama":
        return OllamaRunner(str(tier["id"]), host=host)
    if backend == "openai_compat":
        key_env = str(tier["api_key_env"])
        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(f"falta la variable de entorno {key_env} para el tier de API")
        return OpenAICompatRunner(
            str(tier["model"]),
            base_url=str(tier["base_url"]),
            api_key=api_key,
        )
    raise ValueError(f"backend desconocido: {backend!r}")


def load_pool(path: Path, cache_dir: Path | None = None) -> list[ModelRunner]:
    """Carga el pool ordenado desde `path`, con cache de completions."""
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "tiers" not in data:
        raise ValueError("la config de pool debe tener 'tiers'")
    host = str(data.get("host", DEFAULT_HOST))
    resolved_cache = cache_dir or Path(str(data.get("cache_dir", DEFAULT_CACHE_DIR)))
    cache = CompletionCache(resolved_cache)
    runners: list[ModelRunner] = []
    for tier in data["tiers"]:
        runners.append(CachingRunner(_build_inner(tier, host), cache))
    if not runners:
        raise ValueError("la config de pool no define ningún tier")
    return runners
