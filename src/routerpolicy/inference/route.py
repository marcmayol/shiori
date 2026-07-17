"""route(task, registry) — la función pública del artefacto (Fase 6).

Habla con Ollama por defecto (endpoint local), genera el JSON schema desde el
registro recibido y decodifica contra él (structured outputs). Integra la
CALIBRACIÓN de la Fase 5b:

- `plan_prior` (público, por defecto 0.50): sesga la decisión de MODO hacia PLAN
  sumando el prior al logprob de PLAN (recupera el recall de PLAN sin reentrenar).
- `capability_prior` (por defecto 0.0): sesga el MODEL_ID hacia modelos más
  capaces para contrarrestar la infra-provisión del regret (elige más barato que
  el mínimo suficiente).

Fallback documentado: si la salida es inválida, no hay logprobs de modo, o hay
empate/gramática vacía, se cae al **modelo más capaz del pool** (regla de Fase 6).

Sin ninguna llamada de red externa: solo el endpoint local de Ollama.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from routerpolicy.dataset.format import SYSTEM_PROMPT, build_user_message
from routerpolicy.inference.constraints import build_json_schema
from routerpolicy.registry.synthetic import MAX_CAPABILITY, capability_from_tags
from routerpolicy.schema.core import Mode, Registry, RoutingDecision
from routerpolicy.schema.validation import DecisionValidationError, validate_decision_json

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "shiori-router"
DEFAULT_PLAN_PRIOR = 0.50

# El primer token del valor de modo distingue las clases (DIRECT/PLAN/TOOL_CALL).
_MODE_PREFIXES: tuple[tuple[str, Mode], ...] = (
    ("DIRECT", Mode.DIRECT),
    ("PLAN", Mode.PLAN),
    ("TOOL", Mode.TOOL_CALL),
)


def mode_dist_from_logprobs(logprobs: list[dict[str, Any]]) -> dict[Mode, float]:
    """Distribución de modo (logprob por clase) leída del token del valor de modo."""
    for entry in logprobs:
        tok = str(entry.get("token", "")).strip()
        if not any(tok.startswith(p) for p, _ in _MODE_PREFIXES):
            continue
        alts = entry.get("top_logprobs") or [entry]
        dist: dict[Mode, float] = {}
        for cand in alts:
            ct = str(cand.get("token", "")).strip()
            lp = float(cand.get("logprob", float("-inf")))
            for pref, mode in _MODE_PREFIXES:
                if ct.startswith(pref):
                    dist[mode] = max(dist.get(mode, float("-inf")), lp)
        return dist
    return {}


def apply_mode_prior(dist: dict[Mode, float], plan_prior: float) -> Mode | None:
    """Modo elegido tras sumar `plan_prior` al logprob de PLAN. None si vacío."""
    if not dist:
        return None
    adjusted = {m: lp + (plan_prior if m is Mode.PLAN else 0.0) for m, lp in dist.items()}
    return max(adjusted, key=lambda m: adjusted[m])


def most_capable_id(registry: Registry) -> str:
    """Fallback: el modelo más capaz del pool (desempate por más barato)."""
    return max(registry.models, key=lambda m: (capability_from_tags(m.tags), -m.cost)).id


def apply_capability_prior(registry: Registry, model_id: str, capability_prior: float) -> str:
    """Sube el model_id hacia más capaz (contra infra-provisión). Devuelve el nuevo id."""
    if capability_prior <= 0:
        return model_id
    spec = registry.get(model_id)
    if spec is None:
        return model_id
    target = min(capability_from_tags(spec.tags) + round(capability_prior), MAX_CAPABILITY)
    sufficient = [m for m in registry.models if capability_from_tags(m.tags) >= target]
    if not sufficient:
        return model_id
    return min(sufficient, key=lambda m: (m.cost, m.id)).id


def _ollama_generate(
    prompt: str, schema: dict[str, object], host: str, model: str, timeout: float, logprobs: bool
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "options": {"temperature": 0},
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 10
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
    return data


def route(
    task: str,
    registry: Registry,
    plan_prior: float = DEFAULT_PLAN_PRIOR,
    capability_prior: float = 0.0,
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
    timeout: float = 60.0,
) -> RoutingDecision:
    """Elige {mode, model_id} para la tarea y el registro, vía Ollama local.

    Aplica el prior de modo (default 0.50) y el prior de capacidad. Fallback al
    modelo más capaz del pool si algo falla.
    """
    prompt = f"{SYSTEM_PROMPT}\n\n{build_user_message(registry, task)}"
    schema = build_json_schema(registry)
    data = _ollama_generate(prompt, schema, host, model, timeout, logprobs=True)

    try:
        raw = validate_decision_json(str(data.get("response", "")), registry)
    except DecisionValidationError:
        raw = None

    dist = mode_dist_from_logprobs(data.get("logprobs") or [])
    chosen_mode = apply_mode_prior(dist, plan_prior)
    if chosen_mode is None:
        chosen_mode = raw.mode if raw is not None else Mode.DIRECT

    if raw is not None and raw.mode is chosen_mode:
        model_id = raw.model_id
    else:
        # el prior cambió el modo (o la salida fue inválida): decide el model_id
        # con el modo fijado.
        fixed = build_json_schema(registry)
        props = fixed["properties"]
        assert isinstance(props, dict)
        props["mode"]["enum"] = [chosen_mode.value]
        data2 = _ollama_generate(prompt, fixed, host, model, timeout, logprobs=False)
        try:
            model_id = validate_decision_json(str(data2.get("response", "")), registry).model_id
        except DecisionValidationError:
            model_id = most_capable_id(registry)  # fallback

    model_id = apply_capability_prior(registry, model_id, capability_prior)
    return RoutingDecision(mode=chosen_mode, model_id=model_id)
