"""Cache agresiva de completions en disco (hash de modelo+prompt).

Permite re-etiquetar (p. ej. al recalcular etiquetas por pool en Fase 3) sin
re-ejecutar los modelos, y reanudar runs largos desde donde se quedaron.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from routerpolicy.harness.runner import Completion, ModelRunner


def completion_key(model_id: str, prompt: str) -> str:
    """Clave estable sha256 de (model_id, prompt)."""
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


class CompletionCache:
    """Cache de completions basada en archivos JSON, una por clave."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> Completion | None:
        path = self._path(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Completion(
            model_id=data["model_id"],
            text=data["text"],
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
        )

    def put(self, key: str, completion: Completion) -> None:
        payload = {
            "model_id": completion.model_id,
            "text": completion.text,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
        }
        # escritura atómica: tmp + replace, para no dejar JSON a medias si se
        # interrumpe (el equipo puede reiniciarse sin avisar).
        tmp = self._path(key).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        tmp.replace(self._path(key))


class CachingRunner:
    """Envuelve un ModelRunner y sirve desde cache cuando hay acierto."""

    def __init__(self, inner: ModelRunner, cache: CompletionCache) -> None:
        self._inner = inner
        self._cache = cache
        self.hits = 0
        self.misses = 0

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def complete(self, prompt: str) -> Completion:
        key = completion_key(self._inner.model_id, prompt)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        completion = self._inner.complete(prompt)
        self._cache.put(key, completion)
        return completion
