"""Tests de los backends reales con transporte HTTP fake (sin red)."""

from __future__ import annotations

from typing import Any

from routerpolicy.harness.backends import OllamaRunner, OpenAICompatRunner
from routerpolicy.harness.runner import ModelRunner


class _FakePost:
    """Transporte fake: registra la llamada y devuelve una respuesta fija."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.url: str | None = None
        self.payload: dict[str, Any] | None = None
        self.headers: dict[str, str] | None = None

    def __call__(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, Any]:
        self.url = url
        self.payload = payload
        self.headers = headers
        return self.response


def test_ollama_runner_parses_response_and_counts() -> None:
    post = _FakePost({"response": "hello", "prompt_eval_count": 7, "eval_count": 3})
    runner = OllamaRunner("esdrac", post=post)
    assert isinstance(runner, ModelRunner)
    c = runner.complete("hi")
    assert c.model_id == "esdrac"
    assert c.text == "hello"
    assert c.prompt_tokens == 7
    assert c.completion_tokens == 3
    assert post.url == "http://localhost:11434/api/generate"
    assert post.payload is not None and post.payload["stream"] is False


def test_ollama_estimates_tokens_when_missing() -> None:
    post = _FakePost({"response": "some text output"})
    c = OllamaRunner("m", post=post).complete("a prompt")
    assert c.prompt_tokens > 0
    assert c.completion_tokens > 0


def test_ollama_passes_options() -> None:
    post = _FakePost({"response": "x"})
    OllamaRunner("m", post=post, options={"num_predict": 32, "temperature": 0.0}).complete("p")
    assert post.payload is not None
    assert post.payload["options"] == {"num_predict": 32, "temperature": 0.0}


def test_ollama_omits_options_when_none() -> None:
    post = _FakePost({"response": "x"})
    OllamaRunner("m", post=post).complete("p")
    assert post.payload is not None
    assert "options" not in post.payload


def test_openai_compat_runner_parses_and_auths() -> None:
    post = _FakePost(
        {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )
    runner = OpenAICompatRunner(
        "gpt-x", base_url="https://api.example.com/v1", api_key="secret", post=post
    )
    assert isinstance(runner, ModelRunner)
    c = runner.complete("question")
    assert c.text == "answer"
    assert c.prompt_tokens == 10
    assert c.completion_tokens == 5
    assert post.url == "https://api.example.com/v1/chat/completions"
    assert post.headers is not None
    assert post.headers["Authorization"] == "Bearer secret"


def test_openai_compat_handles_empty_choices() -> None:
    post = _FakePost({"choices": []})
    c = OpenAICompatRunner("m", base_url="http://x/v1", api_key="k", post=post).complete("p")
    assert c.text == ""
    assert c.prompt_tokens > 0  # estimado
