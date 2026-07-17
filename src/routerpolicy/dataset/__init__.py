"""dataset: formato de ejemplo, ingesta de fuentes y (Fase 3) augmentación."""

from routerpolicy.dataset.format import (
    SYSTEM_PROMPT,
    ChatMessage,
    build_chat_example,
    build_user_message,
)
from routerpolicy.dataset.sources import (
    chat_task_from_wildchat,
    code_task_from_humaneval,
    code_task_from_mbpp,
    ingest_humaneval_plus,
    ingest_mbpp_plus,
    ingest_wildchat,
    ingest_xlam,
    tool_task_from_xlam,
)
from routerpolicy.dataset.tasks_io import load_jsonl, write_jsonl

__all__ = [
    "SYSTEM_PROMPT",
    "ChatMessage",
    "build_chat_example",
    "build_user_message",
    "chat_task_from_wildchat",
    "code_task_from_humaneval",
    "code_task_from_mbpp",
    "ingest_humaneval_plus",
    "ingest_mbpp_plus",
    "ingest_wildchat",
    "ingest_xlam",
    "load_jsonl",
    "tool_task_from_xlam",
    "write_jsonl",
]
