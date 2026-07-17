"""Config de entrenamiento por peldaño (YAML versionado, nunca hardcode)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class TrainConfig(BaseModel):
    """Hiperparámetros de un peldaño. Serializable a/desde YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_name: str  # p. ej. "gemma-3-270m"
    model_id: str  # base de HF (p. ej. google/gemma-3-270m)
    max_seq_len: int = 2048
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 3.0
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 20
    save_steps: int = 200
    save_total_limit: int = 2  # retención de los 2 mejores/últimos checkpoints
    seed: int = 20260717
    bf16: bool = True
    gradient_checkpointing: bool = False


def load_train_config(path: Path) -> TrainConfig:
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrainConfig.model_validate(data)
