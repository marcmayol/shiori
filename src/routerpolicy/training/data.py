"""Carga del dataset de entrenamiento (filas con `messages` de Fase 1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ChatRow = dict[str, Any]


def load_rows(path: Path, limit: int | None = None) -> list[ChatRow]:
    """Carga filas del JSONL del dataset; hasta `limit` si se indica."""
    out: list[ChatRow] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        out.append(json.loads(line))
        if limit is not None and len(out) >= limit:
            break
    return out


def messages_of(row: ChatRow) -> list[dict[str, str]]:
    """Los 3 turnos (system/user/assistant) de la fila."""
    return list(row["messages"])


def prompt_messages(row: ChatRow) -> list[dict[str, str]]:
    """Turnos de entrada para inferencia (sin el assistant)."""
    return [m for m in row["messages"] if m["role"] != "assistant"]


def gold_completion(row: ChatRow) -> str:
    """Contenido del turno assistant (JSON canónico esperado)."""
    for message in reversed(row["messages"]):
        if message["role"] == "assistant":
            return str(message["content"])
    raise ValueError(f"fila sin turno assistant: {row.get('task_id')}")
