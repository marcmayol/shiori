"""training: full FT (mini) y (Fase 4b) QLoRA, con máscara solo del assistant."""

from routerpolicy.training.config import TrainConfig, load_train_config
from routerpolicy.training.data import (
    gold_completion,
    load_rows,
    messages_of,
    prompt_messages,
)
from routerpolicy.training.prepare import (
    IGNORE_INDEX,
    merge_system_into_user,
    tokenize_with_completion_mask,
)

__all__ = [
    "IGNORE_INDEX",
    "TrainConfig",
    "gold_completion",
    "load_rows",
    "load_train_config",
    "merge_system_into_user",
    "messages_of",
    "prompt_messages",
    "tokenize_with_completion_mask",
]
