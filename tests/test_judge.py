"""Tests del LLM juez de modo (parseo + fake + respaldado por runner)."""

from __future__ import annotations

import pytest

from routerpolicy.harness.runner import FakeRunner
from routerpolicy.labeling.judge import (
    FakeJudge,
    JudgeError,
    LlmModeJudge,
    ModeJudge,
    build_judge_prompt,
    parse_judge_output,
)
from routerpolicy.schema.core import Mode


def test_parse_clean_json() -> None:
    assert parse_judge_output('{"mode": "PLAN"}') is Mode.PLAN


def test_parse_json_embedded_in_text() -> None:
    text = 'Sure, here is my answer: {"mode": "TOOL_CALL"} done.'
    assert parse_judge_output(text) is Mode.TOOL_CALL


def test_parse_rejects_missing_json() -> None:
    with pytest.raises(JudgeError):
        parse_judge_output("no json here")


def test_parse_rejects_invalid_mode() -> None:
    with pytest.raises(JudgeError):
        parse_judge_output('{"mode": "NOPE"}')


def test_fake_judge_conforms_to_protocol() -> None:
    judge: ModeJudge = FakeJudge({"q": Mode.PLAN}, default=Mode.DIRECT)
    assert judge.judge("q") is Mode.PLAN
    assert judge.judge("other") is Mode.DIRECT


def test_llm_judge_uses_runner_and_prompt() -> None:
    prompt = build_judge_prompt("Design a system.")
    runner = FakeRunner("judge-model", responder={prompt: '{"mode": "PLAN"}'})
    judge = LlmModeJudge(runner)
    assert judge.judge("Design a system.") is Mode.PLAN
    assert runner.calls == [prompt]
