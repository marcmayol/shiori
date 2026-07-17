"""Ingesta de las fuentes de tareas hacia el formato canónico.

Cada fuente tiene un MAPEADOR PURO (record dict -> Task), testeado con fixtures
que documentan el esquema asumido, y un INGEST perezoso que importa `datasets`/
`evalplus` dentro de la función (extra `label`, no en CI) y aplica el mapeador.

NOTA: los ingest_* no se ejercitan en CI (requieren descarga). Su esquema real
se valida en la primera descarga, durante el run de etiquetado (post-gate). Los
mapeadores puros SÍ están testeados y son el punto de verdad del formato.
"""

from __future__ import annotations

import json
import re
from typing import Any

from routerpolicy.harness.tasks import ChatTask, CodeTask, TaskSource, ToolTask

# ----------------------------- mapeadores puros -----------------------------

_DEF_NAME = re.compile(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def code_task_from_humaneval(rec: dict[str, Any]) -> CodeTask:
    """openai_humaneval: {task_id, prompt, entry_point, test, canonical_solution}.

    El campo `test` define `check(candidate)`; el test ejecutable lo invoca con
    el entry_point.
    """
    entry_point = str(rec["entry_point"])
    test_code = f"{rec['test']}\n\ncheck({entry_point})\n"
    return CodeTask(
        task_id=f"humaneval/{rec['task_id']}",
        source=TaskSource.HUMANEVAL_PLUS,
        prompt=str(rec["prompt"]),
        entry_point=entry_point,
        test_code=test_code,
    )


def code_task_from_mbpp(rec: dict[str, Any]) -> CodeTask:
    """mbpp: {task_id, text, code, test_list, test_setup_code}.

    entry_point se deriva del primer assert; el test ejecutable concatena el
    setup y los asserts.
    """
    test_list = list(rec.get("test_list", []))
    setup = str(rec.get("test_setup_code", "") or "")
    entry_point = "candidate"
    for assertion in test_list:
        m = _DEF_NAME.search(str(assertion))
        if m:
            entry_point = m.group(1)
            break
    body = "\n".join(str(a) for a in test_list)
    test_code = f"{setup}\n{body}\n" if setup else f"{body}\n"
    return CodeTask(
        task_id=f"mbpp/{rec['task_id']}",
        source=TaskSource.MBPP_PLUS,
        prompt=str(rec["text"]),
        entry_point=entry_point,
        test_code=test_code,
    )


def tool_task_from_xlam(rec: dict[str, Any]) -> ToolTask:
    """xlam-function-calling: {id, query, tools(JSON str o list)}."""
    tools_raw = rec["tools"]
    tools = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
    names = tuple(str(t["name"]) for t in tools)
    return ToolTask(
        task_id=f"xlam/{rec['id']}",
        source=TaskSource.XLAM,
        prompt=str(rec["query"]),
        tool_names=names,
    )


def chat_task_from_wildchat(rec: dict[str, Any]) -> ChatTask | None:
    """WildChat: toma el primer turno de usuario; filtra idioma/toxicidad.

    Devuelve None si no cumple el filtro (no inglés, tóxico, o sin turno user).
    """
    if rec.get("toxic") is True:
        return None
    if rec.get("language") not in (None, "English"):
        return None
    conversation = rec.get("conversation") or []
    for turn in conversation:
        if turn.get("role") == "user":
            content = str(turn.get("content", "")).strip()
            if content:
                return ChatTask(
                    task_id=f"wildchat/{rec.get('conversation_hash', rec.get('id'))}",
                    source=TaskSource.WILDCHAT,
                    prompt=content,
                )
            return None
    return None


# ------------------------------ ingest perezoso ------------------------------


def ingest_humaneval_plus(limit: int | None = None) -> list[CodeTask]:
    """Carga HumanEval+ vía evalplus (extra `label`). Import perezoso."""
    from evalplus.data import get_human_eval_plus

    problems = get_human_eval_plus()
    tasks = [code_task_from_humaneval(dict(p)) for p in problems.values()]
    return tasks[:limit] if limit is not None else tasks


def ingest_mbpp_plus(limit: int | None = None) -> list[CodeTask]:
    """Carga MBPP+ vía evalplus (extra `label`). Import perezoso."""
    from evalplus.data import get_mbpp_plus

    problems = get_mbpp_plus()
    tasks = [code_task_from_mbpp(dict(p)) for p in problems.values()]
    return tasks[:limit] if limit is not None else tasks


def ingest_xlam(limit: int | None = None) -> list[ToolTask]:
    """Carga xLAM function-calling vía datasets streaming (extra `label`, gated)."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train", streaming=True)
    tasks: list[ToolTask] = []
    for rec in ds:
        tasks.append(tool_task_from_xlam(dict(rec)))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def ingest_wildchat(limit: int | None = None) -> list[ChatTask]:
    """Carga WildChat filtrado vía datasets streaming (extra `label`, gated)."""
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    tasks: list[ChatTask] = []
    for rec in ds:
        task = chat_task_from_wildchat(dict(rec))
        if task is None:
            continue
        tasks.append(task)
        if limit is not None and len(tasks) >= limit:
            break
    return tasks
