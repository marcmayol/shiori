"""Backends reales que implementan ModelRunner.

- OllamaRunner: modelos locales vía el endpoint HTTP de Ollama (localhost).
- OpenAICompatRunner: cualquier proveedor con API estilo OpenAI (OpenAI,
  OpenRouter, vLLM local, endpoint compat de Anthropic...). El proveedor/modelo
  concreto del tier de API es decisión abierta del pool de generación.

Ambos usan urllib (stdlib): sin dependencias pesadas. El transporte HTTP es
inyectable para poder testear sin red.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from typing import Any

from routerpolicy.harness.runner import Completion
from routerpolicy.tokens import estimate_tokens

# (url, payload, headers, timeout) -> respuesta JSON decodificada
PostJson = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


def urllib_post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    """POST JSON con urllib (transporte por defecto)."""
    data = json.dumps(payload).encode("utf-8")
    all_headers = {"Content-Type": "application/json", **headers}
    req = urllib.request.Request(url, data=data, headers=all_headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    parsed: dict[str, Any] = json.loads(body)
    return parsed


class OllamaRunner:
    """Runner de un modelo local servido por Ollama."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        post: PostJson = urllib_post_json,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._post = post

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, prompt: str) -> Completion:
        resp = self._post(
            f"{self._host}/api/generate",
            {"model": self._model, "prompt": prompt, "stream": False},
            {},
            self._timeout,
        )
        text = str(resp.get("response", ""))
        prompt_tokens = resp.get("prompt_eval_count")
        completion_tokens = resp.get("eval_count")
        return Completion(
            model_id=self._model,
            text=text,
            prompt_tokens=int(prompt_tokens)
            if isinstance(prompt_tokens, int)
            else estimate_tokens(prompt),
            completion_tokens=int(completion_tokens)
            if isinstance(completion_tokens, int)
            else estimate_tokens(text),
        )


class OpenAICompatRunner:
    """Runner para APIs estilo OpenAI (chat/completions)."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        post: PostJson = urllib_post_json,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._post = post

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, prompt: str) -> Completion:
        resp = self._post(
            f"{self._base_url}/chat/completions",
            {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
            },
            {"Authorization": f"Bearer {self._api_key}"},
            self._timeout,
        )
        choices = resp.get("choices") or []
        text = ""
        if choices:
            text = str(choices[0].get("message", {}).get("content", ""))
        usage = resp.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return Completion(
            model_id=self._model,
            text=text,
            prompt_tokens=int(prompt_tokens)
            if isinstance(prompt_tokens, int)
            else estimate_tokens(prompt),
            completion_tokens=int(completion_tokens)
            if isinstance(completion_tokens, int)
            else estimate_tokens(text),
        )
