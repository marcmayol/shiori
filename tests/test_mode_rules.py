"""Tests de las reglas de modo."""

from __future__ import annotations

from routerpolicy.harness.tasks import ChatTask, TaskSource, ToolTask
from routerpolicy.labeling.mode_rules import (
    is_short_factual,
    rule_mode_for_chat_task,
    rule_mode_for_tool_task,
)
from routerpolicy.schema.core import Mode


def _chat(prompt: str) -> ChatTask:
    return ChatTask(task_id="c", source=TaskSource.WILDCHAT, prompt=prompt)


def test_tool_task_is_tool_call() -> None:
    task = ToolTask(
        task_id="t",
        source=TaskSource.XLAM,
        prompt="Get the weather in Tokyo.",
        tool_names=("get_weather",),
    )
    assert rule_mode_for_tool_task(task) is Mode.TOOL_CALL


def test_short_factual_is_direct() -> None:
    assert rule_mode_for_chat_task(_chat("What is the capital of France?")) is Mode.DIRECT


def test_long_or_planning_defers_to_judge() -> None:
    assert rule_mode_for_chat_task(_chat("Design a scalable chat system.")) is None
    long_q = "Could you please help me understand " + "really " * 20 + "well?"
    assert rule_mode_for_chat_task(_chat(long_q)) is None


def test_is_short_factual_edge_cases() -> None:
    assert not is_short_factual("Not a question.")
    assert not is_short_factual("Design the plan?")  # keyword de planificación
    assert is_short_factual("Who wrote Hamlet?")
