"""dataset: formato de ejemplo y (Fase 3) ensamblado, augmentación y splits."""

from routerpolicy.dataset.format import (
    SYSTEM_PROMPT,
    ChatMessage,
    build_chat_example,
    build_user_message,
)

__all__ = [
    "SYSTEM_PROMPT",
    "ChatMessage",
    "build_chat_example",
    "build_user_message",
]
