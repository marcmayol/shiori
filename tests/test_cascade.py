"""Tests de la cascada offline de suficiencia."""

from __future__ import annotations

from routerpolicy.harness.cascade import build_code_prompt, run_cascade
from routerpolicy.harness.runner import FakeRunner
from routerpolicy.harness.tasks import CodeTask, TaskSource

TASK = CodeTask(
    task_id="add-1",
    source=TaskSource.MBPP_PLUS,
    prompt="Write add(a, b) that returns a + b.",
    entry_point="add",
    test_code="assert add(2, 3) == 5",
)

GOOD = "```python\ndef add(a, b):\n    return a + b\n```"
BAD = "```python\ndef add(a, b):\n    return a - b\n```"


def _runner(model_id: str, text: str) -> FakeRunner:
    prompt = build_code_prompt(TASK)
    return FakeRunner(model_id, responder={prompt: text})


def test_cheapest_sufficient_stops_early() -> None:
    cheap = _runner("cheap", GOOD)
    mid = _runner("mid", GOOD)
    result = run_cascade(TASK, [cheap, mid])
    assert result.sufficient_model_id == "cheap"
    assert result.any_sufficient
    assert len(result.attempts) == 1  # no llega al segundo
    assert mid.calls == []  # el modelo caro no se ejecutó


def test_escalates_to_next_when_cheap_fails() -> None:
    cheap = _runner("cheap", BAD)
    mid = _runner("mid", GOOD)
    result = run_cascade(TASK, [cheap, mid])
    assert result.sufficient_model_id == "mid"
    assert len(result.attempts) == 2
    assert result.attempts[0].passed is False
    assert result.attempts[0].error is not None
    assert result.attempts[1].passed is True


def test_none_sufficient() -> None:
    cheap = _runner("cheap", BAD)
    mid = _runner("mid", BAD)
    result = run_cascade(TASK, [cheap, mid])
    assert result.sufficient_model_id is None
    assert not result.any_sufficient
    assert len(result.attempts) == 2


def test_attempts_carry_token_counts() -> None:
    result = run_cascade(TASK, [_runner("cheap", GOOD)])
    att = result.attempts[0]
    assert att.prompt_tokens > 0
    assert att.completion_tokens > 0


class _RaisingRunner:
    """Runner que siempre falla de infra (simula timeout/red)."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self.calls = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete(self, prompt: str):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise TimeoutError("boom")


def test_infra_error_escalates_instead_of_crashing() -> None:
    raising = _RaisingRunner("cheap")
    good = _runner("mid", GOOD)
    result = run_cascade(TASK, [raising, good], max_attempts=2)
    assert result.attempts[0].errored is True
    assert result.attempts[0].passed is False
    assert raising.calls == 2  # reintentó antes de rendirse
    assert result.sufficient_model_id == "mid"


def test_all_infra_errors_yield_none_sufficient() -> None:
    result = run_cascade(TASK, [_RaisingRunner("a"), _RaisingRunner("b")], max_attempts=1)
    assert result.sufficient_model_id is None
    assert all(a.errored for a in result.attempts)
