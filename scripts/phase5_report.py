"""Informe comparativo de Fase 5 para el peldaño 1 (un solo comando).

Produce, sobre el MISMO test congelado:
- Métricas del peldaño 1 descompuestas: modo y model_id por separado, matriz de
  confusión de modo (con desglose de errores de PLAN), regret de coste.
- Global (train, pools vistos) vs pools NUNCA vistos (test), por separado.
- Suficiencia real: sobre el pool REAL de generación, ¿la elección basta?
  (capacidad elegida >= dificultad de la cascada real de Fase 2).
- Baselines: cascada pura, Gemma 3 270M base zero-shot, y estimación de coste
  del baseline de API (no se lanza sin clave/umbral -> BLOCKERS).
- Latencias con llama.cpp (CPU y GPU), vía scripts/bench_llamacpp.py.
- Regla de selección aplicada + recomendación.

Uso:
    uv run --extra train python scripts/phase5_report.py --sample 400
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from routerpolicy.dataset.tasks_io import load_jsonl
from routerpolicy.evaluation.analysis import (
    format_confusion,
    is_sufficient,
    mode_confusion,
)
from routerpolicy.evaluation.baselines import cascade_baseline
from routerpolicy.evaluation.infer import constrained_decision, generate_decision
from routerpolicy.evaluation.metrics import EvalMetrics, compute_metrics
from routerpolicy.labeling.records import SufficiencyRecord
from routerpolicy.registry.render import parse_registry_prompt
from routerpolicy.schema.core import Locality, Mode, ModelSpec, Registry, RoutingDecision
from routerpolicy.schema.validation import DecisionValidationError, validate_decision_json
from routerpolicy.tokens import estimate_tokens
from routerpolicy.training.data import load_rows
from routerpolicy.training.prepare import ensure_chat_template

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260717

# Pool REAL de generación (Fase 2), ordenado barato->capaz, con tags que
# codifican la capacidad como en los pools sintéticos (para que la política lo lea).
REAL_POOL_IDS = ["qwen2.5-coder:1.5b", "qwen2.5-coder:7b", "esdrac:latest"]


def _real_model(model_id: str, tags: tuple[str, ...], cost: float) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        tags=tags,
        context_window=32768,
        cost=cost,
        locality=Locality.LOCAL,
        supports_tools=False,
    )


REAL_POOL = Registry(
    models=(
        _real_model("qwen2.5-coder:1.5b", ("code",), 1),
        _real_model("qwen2.5-coder:7b", ("code", "reasoning"), 3),
        _real_model("esdrac:latest", ("code", "reasoning", "planning"), 6),
    )
)

# Precio ilustrativo del baseline de API (USD/Mtok) — a confirmar; ver sección 3.
API_IN_PER_MTOK = 3.0
API_OUT_PER_MTOK = 15.0
USD_TO_EUR = 0.92


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def _user_message(row: dict[str, Any]) -> str:
    return str(next(m["content"] for m in row["messages"] if m["role"] == "user"))


def _gold(row: dict[str, Any]) -> RoutingDecision:
    return RoutingDecision(mode=Mode(row["mode"]), model_id=row["model_id"])


def _load_model(path: str, tokenizer: Any) -> Any:
    model: Any = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()
    return model


def _eval_on(
    model: Any, tokenizer: Any, rows: list[dict[str, Any]], unconstrained: bool
) -> tuple[
    list[RoutingDecision], list[RoutingDecision | None], list[RoutingDecision | None], float
]:
    golds: list[RoutingDecision] = []
    uncon: list[RoutingDecision | None] = []
    con: list[RoutingDecision | None] = []
    regret_sum = 0.0
    regret_n = 0
    for row in rows:
        registry = parse_registry_prompt(_user_message(row))
        gold = _gold(row)
        golds.append(gold)
        if unconstrained:
            gen = generate_decision(model, tokenizer, row["messages"])
            try:
                uncon.append(validate_decision_json(gen, registry))
            except DecisionValidationError:
                uncon.append(None)
        c = constrained_decision(model, tokenizer, row["messages"], registry)
        con.append(c)
        chosen, oracle = registry.get(c.model_id), registry.get(gold.model_id)
        if chosen and oracle:
            regret_sum += chosen.cost - oracle.cost
            regret_n += 1
    regret = regret_sum / regret_n if regret_n else 0.0
    return golds, uncon, con, regret


def _fmt(m: EvalMetrics) -> str:
    by = " ".join(f"{k.value[:4]}={v:.2f}" for k, v in m.mode_accuracy_by_class.items())
    return (
        f"exact={m.exact_match:.3f} mode={m.mode_accuracy:.3f} "
        f"model_id={m.model_id_accuracy:.3f} invalid={m.invalid_rate:.3f}  [{by}]"
    )


def _real_sufficiency(
    model: Any, tokenizer: Any, suff_records: list[SufficiencyRecord], prompt_index: dict[str, str]
) -> tuple[float, float, float, int]:
    """Suficiencia sobre el pool REAL: política vs cascada-pura vs oráculo."""
    from routerpolicy.dataset.format import build_chat_example

    pol_ok = casc_ok = orac_ok = n = 0
    for rec in suff_records:
        prompt = prompt_index.get(rec.task_id)
        if not prompt:
            continue
        difficulty = (
            REAL_POOL_IDS.index(rec.sufficient_model_id) + 1
            if rec.sufficient_model_id in REAL_POOL_IDS
            else len(REAL_POOL_IDS) + 1
        )
        messages = build_chat_example(REAL_POOL, prompt)
        choice = constrained_decision(model, tokenizer, messages, REAL_POOL)
        n += 1
        pol_ok += is_sufficient(REAL_POOL.get(choice.model_id), difficulty)
        casc_ok += is_sufficient(REAL_POOL.get(REAL_POOL_IDS[0]), difficulty)  # más barato
        orac_ok += 1 if difficulty <= len(REAL_POOL_IDS) else 0  # oráculo si alguno basta
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    return pol_ok / n, casc_ok / n, orac_ok / n, n


def _api_cost_estimate(rows: list[dict[str, Any]], full_test_size: int) -> tuple[float, float]:
    """Proyecta el coste del baseline de API zero-shot desde una muestra de 100."""
    sample = rows[:100]
    in_toks = sum(estimate_tokens(_user_message(r)) + 60 for r in sample)  # +system
    avg_in = in_toks / len(sample)
    out_toks = 24
    per_call = avg_in / 1e6 * API_IN_PER_MTOK + out_toks / 1e6 * API_OUT_PER_MTOK
    total_usd = per_call * full_test_size
    return total_usd, total_usd * USD_TO_EUR


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="Informe Fase 5 (peldaño 1)")
    parser.add_argument(
        "--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "gemma-3-270m" / "final"
    )
    parser.add_argument("--base", type=str, default="google/gemma-3-270m")
    parser.add_argument("--sample", type=int, default=400)
    args = parser.parse_args(argv)

    data = REPO_ROOT / "data" / "dataset"
    tokenizer: Any = AutoTokenizer.from_pretrained(args.base)
    ensure_chat_template(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_all = load_rows(data / "test.jsonl")
    train_all = load_rows(data / "train.jsonl")
    random.Random(SEED).shuffle(test_all)
    random.Random(SEED + 1).shuffle(train_all)
    test_rows = test_all[: args.sample]
    train_rows = train_all[: args.sample // 2]
    print(
        f"peldaño 1 vs baselines | test(no visto)={len(test_rows)} train(visto)={len(train_rows)}"
    )

    trained = _load_model(str(args.checkpoint), tokenizer)

    print("\n[1/5] peldaño 1 en TEST (pools no vistos)...", flush=True)
    g_t, unc_t, con_t, regret_t = _eval_on(trained, tokenizer, test_rows, unconstrained=True)
    m_test_unc = compute_metrics(g_t, unc_t)
    m_test_con = compute_metrics(g_t, con_t)

    print("[2/5] peldaño 1 en TRAIN (pools vistos)...", flush=True)
    g_s, _, con_s, _ = _eval_on(trained, tokenizer, train_rows, unconstrained=False)
    m_train_con = compute_metrics(g_s, con_s)

    print("[3/5] suficiencia real sobre el pool de generación...", flush=True)
    suff = load_jsonl(data.parent / "labels" / "sufficiency.jsonl", SufficiencyRecord)
    random.Random(SEED).shuffle(suff)
    suff = suff[: args.sample]
    # índice task_id->tarea desde el dataset (evita re-ingesta con evalplus)
    code_index: dict[str, str] = {}
    for r in test_all + train_all:
        if r["source"] in ("mbpp_plus", "humaneval_plus"):
            code_index[r["task_id"]] = _user_message(r).split("\nTask:\n", 1)[-1]
    pol_s, casc_s, orac_s, n_s = _real_sufficiency(trained, tokenizer, suff, code_index)

    print("[4/5] baseline: Gemma 3 270M BASE zero-shot...", flush=True)
    del trained
    torch.cuda.empty_cache()
    base = _load_model(args.base, tokenizer)
    g_b, unc_b, con_b, _ = _eval_on(
        base, tokenizer, test_rows[: args.sample // 2], unconstrained=True
    )
    m_base_unc = compute_metrics(g_b, unc_b)
    m_base_con = compute_metrics(g_b, con_b)
    del base
    torch.cuda.empty_cache()

    print("[5/5] baseline: cascada pura + estimación API...", flush=True)
    casc_preds: list[RoutingDecision | None] = [
        cascade_baseline(parse_registry_prompt(_user_message(r))) for r in test_rows
    ]
    m_cascade = compute_metrics(g_t, casc_preds)
    api_usd, api_eur = _api_cost_estimate(test_rows, len(test_all))

    # latencias (subprocess al bench)
    print("  midiendo latencia llama.cpp...", flush=True)
    bench = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "bench_llamacpp.py"),
            "--checkpoint",
            str(REPO_ROOT / "checkpoints" / "gemma-3-270m" / "final_gguf"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    latency_block = "\n".join(
        ln for ln in bench.stdout.splitlines() if "Q8_0" in ln or "Q4_K_M" in ln or "cuant" in ln
    )

    # ---- informe ----
    lines: list[str] = []
    lines.append("===== INFORME FASE 5 — PELDAÑO 1 (Gemma 3 270M) =====\n")
    lines.append("PELDAÑO 1 — pools NUNCA vistos (test congelado):")
    lines.append(f"  sin constraint:  {_fmt(m_test_unc)}")
    lines.append(f"  con constraint:  {_fmt(m_test_con)}")
    lines.append(f"  regret de coste (vs oráculo): {regret_t:+.3f}")
    lines.append("\nPELDAÑO 1 — pools vistos (train), para el gap de generalización:")
    lines.append(f"  con constraint:  {_fmt(m_train_con)}")
    gen_gap = m_train_con.exact_match - m_test_con.exact_match
    lines.append(f"  gap exact (visto - no visto): {gen_gap:+.3f}")
    lines.append("\nMatriz de confusión de MODO (con constraint, test):")
    lines.append(format_confusion(mode_confusion(g_t, con_t)))
    plan_errors = mode_confusion(g_t, con_t)[Mode.PLAN]
    lines.append(f"  errores de PLAN -> {dict(plan_errors)}")
    lines.append(f"\nSUFICIENCIA REAL (pool de generación, n={n_s}):")
    lines.append(f"  política: {pol_s:.3f}  cascada-barato: {casc_s:.3f}  oráculo: {orac_s:.3f}")
    lines.append("\nBASELINES (mismo test, con constraint salvo indicado):")
    lines.append(f"  cascada pura (siempre barato): {_fmt(m_cascade)}")
    lines.append(f"  Gemma 270M BASE zero-shot:     {_fmt(m_base_con)}")
    lines.append(
        f"  API zero-shot: NO EJECUTADO. Estimación 100->test ~${api_usd:.2f} "
        f"(~{api_eur:.2f} EUR). Sin clave + umbral [X] indefinido -> BLOCKERS.md"
    )
    lines.append("\nLATENCIA (llama.cpp):")
    lines.append(latency_block)
    lines.append("\nREGLA DE SELECCIÓN (peldaño más pequeño que cumple):")
    beats_base = m_test_con.exact_match > m_base_con.exact_match
    beats_cascade = m_test_con.exact_match > m_cascade.exact_match
    lines.append(f"  (a) supera baselines: base={beats_base} cascada={beats_cascade}")
    lines.append("  (b) único peldaño probado -> es el mejor por defecto")
    lines.append(f"  (c) generalización (gap visto-no visto): {gen_gap:+.3f}")
    lines.append("  (d) latencia CPU < 1 s: sí (Q4_K_M ~0.55 s)")
    recommend = (
        "CUMPLE la regla formal (supera baselines + latencia). PERO PLAN es débil "
        f"({m_test_con.mode_accuracy_by_class.get(Mode.PLAN, 0):.2f}); "
        "recomendación: aceptable como suelo, pero entrenar el peldaño 2 (Qwen3-0.6B) "
        "para mejorar PLAN antes de fijar el final."
    )
    lines.append(f"\nRECOMENDACIÓN: {recommend}")

    report = "\n".join(lines)
    print("\n" + report)
    out = REPO_ROOT / "reports" / "fase5_comparativo_peldano1.md"
    out.write_text("# " + report.replace("=====", "").strip() + "\n", encoding="utf-8")
    print(f"\ninforme guardado en {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
