"""Inferencia con transformers para evaluación (Fase 5).

- generate_decision: generación libre (SIN constraint); puede dar salida inválida.
- constrained_decision: decoding constreñido efectivo por SCORING — puntúa cada
  decisión válida del registro (enum de modos x ids del pool) por log-prob y
  elige la mejor. Garantiza salida válida (equivale al constraint del artefacto).
"""

from __future__ import annotations

from typing import Any

import torch

from routerpolicy.schema.core import Mode, Registry, RoutingDecision
from routerpolicy.training.prepare import merge_system_into_user

Message = dict[str, str]


def _eos_ids(tokenizer: Any) -> list[int]:
    ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot, int) and eot >= 0 and eot not in ids:
        ids.append(eot)
    return ids


def _prompt_ids(tokenizer: Any, messages: list[Message]) -> list[int]:
    prepared = merge_system_into_user([m for m in messages if m["role"] != "assistant"])
    out: list[int] = tokenizer.apply_chat_template(
        prepared, tokenize=True, add_generation_prompt=True, return_dict=True
    )["input_ids"]
    return out


def generate_decision(model: Any, tokenizer: Any, messages: list[Message]) -> str:
    """Generación libre; devuelve el texto (para parsear/validar aparte)."""
    ids = torch.tensor([_prompt_ids(tokenizer, messages)], device=model.device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=48,
            do_sample=False,
            eos_token_id=_eos_ids(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
    return str(tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip())


def _candidates(registry: Registry) -> list[RoutingDecision]:
    return [RoutingDecision(mode=mode, model_id=mid) for mode in Mode for mid in registry.model_ids]


def constrained_scores(
    model: Any, tokenizer: Any, messages: list[Message], registry: Registry
) -> list[tuple[RoutingDecision, float]]:
    """Log-prob (normalizado por longitud) de cada decisión válida del registro.

    Un solo forward; el lm_head se aplica solo en las posiciones de la completion.
    Base para el argmax y para la calibración con prior a PLAN.
    """
    prompt = _prompt_ids(tokenizer, messages)
    candidates = _candidates(registry)

    seqs: list[list[int]] = []
    comp_lens: list[int] = []
    for cand in candidates:
        comp = tokenizer(cand.to_canonical_json(), add_special_tokens=False)["input_ids"]
        seqs.append(prompt + comp)
        comp_lens.append(len(comp))

    maxlen = max(len(s) for s in seqs)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.tensor([s + [pad_id] * (maxlen - len(s)) for s in seqs], device=model.device)
    attn = torch.tensor([[1] * len(s) + [0] * (maxlen - len(s)) for s in seqs], device=model.device)
    plen = len(prompt)
    with torch.no_grad():
        # cuerpo del modelo -> hidden; lm_head SOLO en las posiciones que predicen
        # la completion (~25), no en las ~320 del prompt (evita OOM con vocab 262k).
        hidden = model.model(input_ids=input_ids, attention_mask=attn).last_hidden_state
        tail = hidden[:, plen - 1 :, :]  # tail[:, t] predice el token en plen+t
        log_probs = torch.log_softmax(model.lm_head(tail).float(), dim=-1)

    scored: list[tuple[RoutingDecision, float]] = []
    for i, comp_len in enumerate(comp_lens):
        score = 0.0
        for t in range(comp_len):
            tok = int(input_ids[i, plen + t])
            score += float(log_probs[i, t, tok])
        scored.append((candidates[i], score / max(comp_len, 1)))
    return scored


def constrained_decision(
    model: Any, tokenizer: Any, messages: list[Message], registry: Registry
) -> RoutingDecision:
    """Elige la decisión válida de mayor log-prob (constraint por scoring)."""
    scored = constrained_scores(model, tokenizer, messages, registry)
    return max(scored, key=lambda ds: ds[1])[0]
