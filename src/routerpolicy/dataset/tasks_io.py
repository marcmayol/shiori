"""E/S canónica de tareas en JSONL (una tarea por línea).

Formato intermedio del pipeline: las fuentes (HF datasets) se ingieren a este
JSONL una vez, y el resto del harness (cascada, etiquetado) lo consume. Soporta
`limit` para la convención `--sample N` de ensayos baratos.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write_jsonl(path: Path, items: Iterable[BaseModel]) -> int:
    """Escribe items como JSONL; devuelve cuántos escribió."""
    lines = [item.model_dump_json() for item in items]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(text, encoding="utf-8")
    return len(lines)


def load_jsonl(path: Path, model_cls: type[T], limit: int | None = None) -> list[T]:
    """Carga hasta `limit` tareas del JSONL, validándolas con `model_cls`."""
    out: list[T] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        out.append(model_cls.model_validate_json(line))
        if limit is not None and len(out) >= limit:
            break
    return out
