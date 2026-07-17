"""Evaluación de un peldaño sobre el test congelado (Fase 5).

Métricas CON y SIN decoding constreñido: exact match, exactitud de modo (por
clase), model_id, tasa de inválidos (sin constraint), regret de coste vs oráculo,
y latencia por decisión (transformers). La latencia con llama.cpp se mide aparte
tras exportar a GGUF (scripts/bench_llamacpp.py).

Uso:
    uv run --extra train python scripts/evaluate.py \
        --checkpoint checkpoints/gemma-3-270m/final --sample 500
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from routerpolicy.evaluation.infer import constrained_decision, generate_decision
from routerpolicy.evaluation.metrics import compute_metrics
from routerpolicy.registry.render import parse_registry_prompt
from routerpolicy.schema.core import Mode, RoutingDecision
from routerpolicy.schema.validation import DecisionValidationError, validate_decision_json
from routerpolicy.training.data import load_rows
from routerpolicy.training.prepare import ensure_chat_template

REPO_ROOT = Path(__file__).resolve().parents[1]


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def _user_message(row: dict[str, Any]) -> str:
    for msg in row["messages"]:
        if msg["role"] == "user":
            return str(msg["content"])
    raise ValueError("fila sin user")


def _gold(row: dict[str, Any]) -> RoutingDecision:
    return RoutingDecision(mode=Mode(row["mode"]), model_id=row["model_id"])


def _print_metrics(title: str, metrics: Any) -> None:
    print(f"\n== {title} ==")
    print(f"  n={metrics.n}")
    print(f"  exact match:      {metrics.exact_match:.3f}")
    print(f"  mode accuracy:    {metrics.mode_accuracy:.3f}")
    print(f"  model_id acc:     {metrics.model_id_accuracy:.3f}")
    print(f"  invalid rate:     {metrics.invalid_rate:.3f}")
    for mode, acc in metrics.mode_accuracy_by_class.items():
        print(f"    modo {mode.value:<10} {acc:.3f}")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="Evaluación Fase 5 de un peldaño")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=REPO_ROOT / "data" / "dataset" / "test.jsonl")
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--base-tokenizer", type=str, default="google/gemma-3-270m")
    args = parser.parse_args(argv)

    tokenizer: Any = AutoTokenizer.from_pretrained(args.base_tokenizer)
    ensure_chat_template(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()

    all_rows = load_rows(args.test)
    random.Random(20260717).shuffle(all_rows)  # muestra representativa de todos los modos
    rows = all_rows[: args.sample]
    print(f"evaluando {len(rows)} ejemplos del test (pools no vistos, barajado)", flush=True)

    golds: list[RoutingDecision] = []
    pred_uncon: list[RoutingDecision | None] = []
    pred_con: list[RoutingDecision | None] = []
    regret_sum = 0.0
    regret_n = 0
    con_latencies: list[float] = []

    for row in rows:
        registry = parse_registry_prompt(_user_message(row))
        gold = _gold(row)
        golds.append(gold)

        # SIN constraint: generación libre + validación
        gen = generate_decision(model, tokenizer, row["messages"])
        try:
            pred_uncon.append(validate_decision_json(gen, registry))
        except DecisionValidationError:
            pred_uncon.append(None)

        # CON constraint: scoring (mide latencia)
        t0 = time.perf_counter()
        con = constrained_decision(model, tokenizer, row["messages"], registry)
        con_latencies.append((time.perf_counter() - t0) * 1000)
        pred_con.append(con)

        # regret de coste (sobre la elección constreñida)
        chosen = registry.get(con.model_id)
        oracle = registry.get(gold.model_id)
        if chosen is not None and oracle is not None:
            regret_sum += chosen.cost - oracle.cost
            regret_n += 1

    _print_metrics("SIN constraint (generación libre)", compute_metrics(golds, pred_uncon))
    _print_metrics("CON constraint (scoring)", compute_metrics(golds, pred_con))

    con_latencies.sort()
    print("\n== COSTE Y LATENCIA ==")
    print(f"  regret de coste medio: {regret_sum / regret_n:.3f}" if regret_n else "  regret: n/a")
    p50 = con_latencies[len(con_latencies) // 2]
    p95 = con_latencies[int(len(con_latencies) * 0.95)]
    print(f"  latencia constreñida (transformers, GPU): p50={p50:.0f}ms p95={p95:.0f}ms")
    print("  (latencia con llama.cpp: ver scripts/bench_llamacpp.py tras exportar GGUF)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
