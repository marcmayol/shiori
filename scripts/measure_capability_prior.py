"""Mide el prior de capacidad contra la infra-provisión del regret (Fase 6/cond 2).

Sobre el test congelado: decisión base (argmax constreñido) + prior de capacidad
creciente; reporta regret medio, nº de infra/sobre-provisiones y coste medio.
Debe mostrar que subir el prior reduce la infra-provisión (regret -> 0+) a costa
de más coste.

Uso: uv run --extra train python scripts/measure_capability_prior.py --sample 400
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from routerpolicy.evaluation.infer import constrained_decision
from routerpolicy.inference.route import apply_capability_prior
from routerpolicy.registry.render import parse_registry_prompt
from routerpolicy.registry.synthetic import capability_from_tags
from routerpolicy.training.data import load_rows
from routerpolicy.training.prepare import ensure_chat_template

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260717


def _force_utf8() -> None:
    rc = getattr(sys.stdout, "reconfigure", None)
    if callable(rc):
        rc(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "gemma-3-270m" / "final"
    )
    ap.add_argument("--base", type=str, default="google/gemma-3-270m")
    ap.add_argument("--sample", type=int, default=400)
    args = ap.parse_args(argv)

    tok: Any = AutoTokenizer.from_pretrained(args.base)
    ensure_chat_template(tok)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()

    rows = load_rows(REPO_ROOT / "data" / "dataset" / "test.jsonl")
    random.Random(SEED).shuffle(rows)
    rows = rows[: args.sample]

    print("[calculando decisiones base...]", flush=True)
    cases = []  # (registry, base_model_id, gold_model_id)
    for row in rows:
        reg = parse_registry_prompt(
            next(m["content"] for m in row["messages"] if m["role"] == "user")
        )
        dec = constrained_decision(model, tok, row["messages"], reg)
        cases.append((reg, dec.model_id, row["model_id"]))

    print("\n== PRIOR DE CAPACIDAD vs INFRA-PROVISIÓN DEL REGRET ==")
    print(
        f"  {'prior':>6}{'regret_medio':>14}{'infra(<0)':>11}{'sobre(>0)':>11}{'coste_medio':>13}"
    )
    for prior in [0.0, 1.0, 2.0]:
        reg_sum = cost_sum = 0.0
        under = over = n = 0
        for reg, base_id, gold_id in cases:
            mid = apply_capability_prior(reg, base_id, prior)
            chosen, oracle = reg.get(mid), reg.get(gold_id)
            if chosen is None or oracle is None:
                continue
            n += 1
            diff = chosen.cost - oracle.cost
            reg_sum += diff
            cost_sum += chosen.cost
            if capability_from_tags(chosen.tags) < capability_from_tags(oracle.tags):
                under += 1
            elif capability_from_tags(chosen.tags) > capability_from_tags(oracle.tags):
                over += 1
        print(f"  {prior:>6.1f}{reg_sum / n:>14.3f}{under:>11}{over:>11}{cost_sum / n:>13.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
