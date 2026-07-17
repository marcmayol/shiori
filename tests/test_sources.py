"""Tests de los mapeadores puros de fuentes (esquemas documentados en fixtures).

Además, se comprueba que el test_code reconstruido para código realmente ejecuta
y verifica (integración con el verificador), que es el punto crítico del eje de
suficiencia.
"""

from __future__ import annotations

from routerpolicy.dataset.sources import (
    chat_task_from_wildchat,
    code_task_from_humaneval,
    code_task_from_mbpp,
    tool_task_from_xlam,
)
from routerpolicy.harness.tasks import TaskSource
from routerpolicy.harness.verify import verify_code
from routerpolicy.schema.core import Mode  # noqa: F401  (fija dependencia de contrato)

HUMANEVAL_REC = {
    "task_id": "HumanEval/0",
    "prompt": "def add(a, b):\n    '''return a+b'''\n",
    "entry_point": "add",
    "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
    "canonical_solution": "    return a + b\n",
}

# Esquema evalplus MBPP+: prompt + entry_point + assertion (asserts directos).
MBPP_REC = {
    "task_id": "Mbpp/601",
    "prompt": "Write a function to add two numbers.",
    "entry_point": "add",
    "assertion": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n",
}

XLAM_REC = {
    "id": 42,
    "query": "What's the weather in Tokyo?",
    "tools": '[{"name": "get_weather", "parameters": {}}]',
}


def test_humaneval_mapper_and_verification() -> None:
    task = code_task_from_humaneval(HUMANEVAL_REC)
    assert task.source is TaskSource.HUMANEVAL_PLUS
    assert task.entry_point == "add"
    assert task.task_id == "HumanEval/0"
    # el test reconstruido pasa con una implementación correcta y falla con una mala
    assert verify_code("def add(a, b):\n    return a + b", task.test_code).passed
    assert not verify_code("def add(a, b):\n    return a - b", task.test_code).passed


def test_mbpp_mapper_and_verifies() -> None:
    task = code_task_from_mbpp(MBPP_REC)
    assert task.source is TaskSource.MBPP_PLUS
    assert task.entry_point == "add"
    assert task.task_id == "Mbpp/601"
    assert verify_code("def add(a, b):\n    return a + b", task.test_code).passed
    assert not verify_code("def add(a, b):\n    return 0", task.test_code).passed


def test_xlam_mapper_extracts_tool_names() -> None:
    task = tool_task_from_xlam(XLAM_REC)
    assert task.source is TaskSource.XLAM
    assert task.tool_names == ("get_weather",)
    assert "Tokyo" in task.prompt


def test_xlam_accepts_list_tools() -> None:
    rec = {**XLAM_REC, "tools": [{"name": "a"}, {"name": "b"}]}
    assert tool_task_from_xlam(rec).tool_names == ("a", "b")


def test_wildchat_filters_and_maps() -> None:
    rec = {
        "conversation_hash": "abc",
        "language": "English",
        "toxic": False,
        "conversation": [
            {"role": "user", "content": "Explain gradient descent."},
            {"role": "assistant", "content": "..."},
        ],
    }
    task = chat_task_from_wildchat(rec)
    assert task is not None
    assert task.prompt == "Explain gradient descent."


def test_wildchat_rejects_toxic_and_non_english() -> None:
    base = {
        "conversation_hash": "x",
        "conversation": [{"role": "user", "content": "hi"}],
    }
    assert chat_task_from_wildchat({**base, "toxic": True, "language": "English"}) is None
    assert chat_task_from_wildchat({**base, "toxic": False, "language": "Spanish"}) is None
