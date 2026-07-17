"""Ingesta de las fuentes de tareas hacia el formato canónico.

Cada fuente tiene un MAPEADOR PURO (record dict -> Task), testeado con fixtures
que documentan el esquema asumido, y un INGEST perezoso que importa `datasets`/
`evalplus` dentro de la función (extra `label`, no en CI) y aplica el mapeador.

NOTA: los ingest_* no se ejercitan en CI (requieren descarga). Su esquema real
se valida en la primera descarga, durante el run de etiquetado (post-gate). Los
mapeadores puros SÍ están testeados y son el punto de verdad del formato.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from routerpolicy.harness.tasks import ChatTask, CodeTask, TaskSource, ToolTask

# ----------------------------- mapeadores puros -----------------------------
#
# Esquema real de evalplus (validado en la descarga):
#   HumanEval+: {task_id, prompt, entry_point, test (def check(candidate)), ...}
#   MBPP+:      {task_id, prompt, entry_point, assertion (asserts directos), ...}
# El task_id ya viene namespaced ("HumanEval/0", "Mbpp/2").


def code_task_from_humaneval(rec: dict[str, Any]) -> CodeTask:
    """HumanEval+: el campo `test` define `check(candidate)`; se invoca con el
    entry_point para obtener un script ejecutable."""
    entry_point = str(rec["entry_point"])
    test_code = f"{rec['test']}\n\ncheck({entry_point})\n"
    return CodeTask(
        task_id=str(rec["task_id"]),
        source=TaskSource.HUMANEVAL_PLUS,
        prompt=str(rec["prompt"]),
        entry_point=entry_point,
        test_code=test_code,
    )


def code_task_from_mbpp(rec: dict[str, Any]) -> CodeTask:
    """MBPP+: el campo `assertion` son asserts directos que llaman al
    entry_point; sirven de verificación mecánica tal cual."""
    return CodeTask(
        task_id=str(rec["task_id"]),
        source=TaskSource.MBPP_PLUS,
        prompt=str(rec["prompt"]),
        entry_point=str(rec["entry_point"]),
        test_code=str(rec["assertion"]),
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


def tool_task_from_hermes(rec: dict[str, Any]) -> ToolTask | None:
    """Hermes function-calling (no-gated): {id, tools(JSON), conversations}.

    tool_names salen de `tools`; el prompt es el primer turno 'human'. Devuelve
    None si no hay turno humano o herramientas.
    """
    tools_raw = rec.get("tools")
    tools = json.loads(tools_raw) if isinstance(tools_raw, str) else (tools_raw or [])
    names = tuple(str(t["function"]["name"]) for t in tools if "function" in t)
    if not names:
        return None
    human = next(
        (t.get("value") for t in rec.get("conversations", []) if t.get("from") == "human"),
        None,
    )
    if not human:
        return None
    return ToolTask(
        task_id=f"hermes/{rec['id']}",
        source=TaskSource.HERMES,
        prompt=str(human),
        tool_names=names,
    )


def chat_task_from_dolly(rec: dict[str, Any]) -> ChatTask | None:
    """Dolly-15k (no-gated): {instruction, context, response, category}.

    Usa `instruction` como prompt. Devuelve None si está vacía. El id se deriva
    del hash de la instrucción (dolly no trae id).
    """
    instruction = str(rec.get("instruction", "")).strip()
    if not instruction:
        return None
    task_id = "dolly/" + hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:12]
    return ChatTask(task_id=task_id, source=TaskSource.DOLLY, prompt=instruction)


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


def ingest_hermes(limit: int | None = None) -> list[ToolTask]:
    """Carga Hermes function-calling (no-gated) vía datasets streaming."""
    from datasets import load_dataset

    ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train", streaming=True)
    tasks: list[ToolTask] = []
    for rec in ds:
        task = tool_task_from_hermes(dict(rec))
        if task is None:
            continue
        tasks.append(task)
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def ingest_dolly(limit: int | None = None) -> list[ChatTask]:
    """Carga Dolly-15k (no-gated) vía datasets streaming."""
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    tasks: list[ChatTask] = []
    for rec in ds:
        task = chat_task_from_dolly(dict(rec))
        if task is None:
            continue
        tasks.append(task)
        if limit is not None and len(tasks) >= limit:
            break
    return tasks
