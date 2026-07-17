"""Tests del runner de modelos y utilidades de extracción de código."""

from __future__ import annotations

from routerpolicy.harness.runner import (
    Completion,
    FakeRunner,
    ModelRunner,
    extract_code,
)


def test_completion_estimated_counts_tokens() -> None:
    c = Completion.estimated("m", "a prompt here", "some output text")
    assert c.model_id == "m"
    assert c.prompt_tokens > 0
    assert c.completion_tokens > 0


def test_fake_runner_records_calls_and_maps_response() -> None:
    runner = FakeRunner("fake", responder={"hi": "hello"}, default="?")
    assert isinstance(runner, ModelRunner)
    assert runner.model_id == "fake"
    assert runner.complete("hi").text == "hello"
    assert runner.complete("other").text == "?"
    assert runner.calls == ["hi", "other"]


def test_extract_code_from_fence() -> None:
    text = "Here you go:\n```python\ndef f():\n    return 1\n```\nDone."
    assert extract_code(text) == "def f():\n    return 1"


def test_extract_code_plain() -> None:
    assert extract_code("  def f():\n    return 1  ") == "def f():\n    return 1"


def test_extract_code_first_fence_wins() -> None:
    text = "```\na\n```\nmid\n```\nb\n```"
    assert extract_code(text) == "a"
