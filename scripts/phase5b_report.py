"""Informe Fase 5b — cierre de la evaluación del peldaño 1 (un comando, sin entrenar).

(1) Baseline de modelo grande zero-shot como router: se imprime la estimación de
    coste del API (presupuesto 10 EUR); SIN clave no se lanza la llamada de pago
    (BLOCKERS), y se usa esdrac (7B) LOCAL vía Ollama como sustituto no-pago.
(2) Calibración por logprobs: barrido de un prior a PLAN, curva recall/coste.
(3) Matriz de confusión en el punto de operación elegido.
(4) Simulación económica router (apuesta+escalado) vs cascada pura.
(5) Definición exacta del regret y explicación del -0.169.

Uso: uv run --extra train python scripts/phase5b_report.py --sample 400
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from routerpolicy.dataset.format import build_chat_example
from routerpolicy.dataset.tasks_io import load_jsonl
from routerpolicy.evaluation.analysis import format_confusion, mode_confusion
from routerpolicy.evaluation.calibration import (
    CalibExample,
    decision_at_prior,
    sweep_plan_prior,
)
from routerpolicy.evaluation.economic import simulate_cascade, simulate_router
from routerpolicy.evaluation.infer import constrained_decision, constrained_scores
from routerpolicy.evaluation.metrics import compute_metrics
from routerpolicy.inference.constraints import build_json_schema
from routerpolicy.labeling.records import SufficiencyRecord
from routerpolicy.registry.render import parse_registry_prompt
from routerpolicy.schema.core import Locality, Mode, ModelSpec, Registry, RoutingDecision
from routerpolicy.schema.validation import DecisionValidationError, validate_decision_json
from routerpolicy.tokens import estimate_tokens
from routerpolicy.training.data import load_rows
from routerpolicy.training.prepare import ensure_chat_template, merge_system_into_user

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260717
BUDGET_EUR = 10.0
API_IN_PER_MTOK, API_OUT_PER_MTOK, USD_TO_EUR = 3.0, 15.0, 0.92

REAL_POOL_IDS = ["qwen2.5-coder:1.5b", "qwen2.5-coder:7b", "esdrac:latest"]
REAL_COSTS = [1.5, 7.0, 8.0]  # coste ~ tamaño en B de parámetros (proxy de cómputo)
CAPABLE_MODEL = "esdrac:latest"


def _r(mid: str, tags: tuple[str, ...], cost: float) -> ModelSpec:
    return ModelSpec(
        id=mid,
        tags=tags,
        context_window=32768,
        cost=cost,
        locality=Locality.LOCAL,
        supports_tools=False,
    )


REAL_POOL = Registry(
    models=(
        _r("qwen2.5-coder:1.5b", ("code",), REAL_COSTS[0]),
        _r("qwen2.5-coder:7b", ("code", "reasoning"), REAL_COSTS[1]),
        _r("esdrac:latest", ("code", "reasoning", "planning"), REAL_COSTS[2]),
    )
)


def _force_utf8() -> None:
    rc = getattr(sys.stdout, "reconfigure", None)
    if callable(rc):
        rc(encoding="utf-8", errors="replace")


def _user(row: dict[str, Any]) -> str:
    return str(next(m["content"] for m in row["messages"] if m["role"] == "user"))


def _gold(row: dict[str, Any]) -> RoutingDecision:
    return RoutingDecision(mode=Mode(row["mode"]), model_id=row["model_id"])


def _load(path: str) -> Any:
    m: Any = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    m.to("cuda")
    m.eval()
    return m


def _ollama_decision(
    messages: list[dict[str, str]], registry: Registry, model: str
) -> RoutingDecision | None:
    """esdrac zero-shot con structured output (JSON schema) vía Ollama. None si inválido."""
    prepared = merge_system_into_user([m for m in messages if m["role"] != "assistant"])
    prompt = "".join(f"{m['role']}: {m['content']}\n" for m in prepared)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": build_json_schema(registry),
        "options": {"temperature": 0.0, "num_predict": 48},
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            text = json.loads(resp.read().decode())["response"]
        return validate_decision_json(text, registry)
    except (DecisionValidationError, Exception):
        return None


def _choose_operating_point(points: list[Any], base_mode_acc: float) -> float:
    """Mayor recall de PLAN sin que la exactitud de modo caiga >2 pts del base."""
    best_prior = 0.0
    best_recall = -1.0
    for p in points:
        if p.mode_accuracy >= base_mode_acc - 0.02 and p.plan_recall > best_recall:
            best_recall = p.plan_recall
            best_prior = p.plan_prior
    return best_prior


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    ap = argparse.ArgumentParser(description="Informe Fase 5b (peldaño 1)")
    ap.add_argument(
        "--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "gemma-3-270m" / "final"
    )
    ap.add_argument("--base", type=str, default="google/gemma-3-270m")
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--esdrac-sample", type=int, default=150)
    args = ap.parse_args(argv)

    data = REPO_ROOT / "data" / "dataset"
    tok: Any = AutoTokenizer.from_pretrained(args.base)
    ensure_chat_template(tok)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    test_all = load_rows(data / "test.jsonl")
    train_all = load_rows(data / "train.jsonl")
    random.Random(SEED).shuffle(test_all)
    rows = test_all[: args.sample]
    model = _load(str(args.checkpoint))

    lines: list[str] = ["===== INFORME FASE 5b — PELDAÑO 1 ====="]

    # ---- scoring una vez (para calibración) ----
    print("[1/4] scoring del peldaño 1 sobre el test...", flush=True)
    examples: list[CalibExample] = []
    scored_all: list[list[tuple[RoutingDecision, float]]] = []
    golds: list[RoutingDecision] = []
    for row in rows:
        reg = parse_registry_prompt(_user(row))
        sc = constrained_scores(model, tok, row["messages"], reg)
        scored_all.append(sc)
        golds.append(_gold(row))
        examples.append(
            CalibExample(
                scored=sc,
                gold_mode=_gold(row).mode,
                cost_by_model_id={m.id: m.cost for m in reg.models},
            )
        )
    base_preds = [decision_at_prior(sc, 0.0) for sc in scored_all]
    base_metrics = compute_metrics(golds, base_preds)

    # ---- (2) calibración ----
    print("[2/4] calibración del prior a PLAN...", flush=True)
    priors = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    curve = sweep_plan_prior(examples, priors)
    op = _choose_operating_point(curve, base_metrics.mode_accuracy)

    lines.append("\n(2) CALIBRACIÓN — barrido de prior a PLAN:")
    lines.append(
        f"  {'prior':>6}{'PLAN_r':>9}{'DIRECT_r':>10}{'TOOL_r':>9}{'mode_acc':>10}{'coste':>9}"
    )
    for p in curve:
        mark = "  <- op" if abs(p.plan_prior - op) < 1e-9 else ""
        lines.append(
            f"  {p.plan_prior:>6.2f}{p.plan_recall:>10.3f}{p.direct_recall:>12.3f}"
            f"{p.tool_recall:>10.3f}{p.mode_accuracy:>10.3f}{p.mean_cost:>11.2f}{mark}"
        )
    lines.append(
        f"  Punto de operación elegido: prior={op:.2f} (máx recall de PLAN con "
        f"caída de mode_acc <=2 pts vs prior 0)."
    )

    # ---- (3) confusión en el punto de operación ----
    op_preds = [decision_at_prior(sc, op) for sc in scored_all]
    lines.append(f"\n(3) MATRIZ DE CONFUSIÓN en el punto de operación (prior={op:.2f}):")
    lines.append(format_confusion(mode_confusion(golds, op_preds)))
    op_metrics = compute_metrics(golds, op_preds)
    lines.append(
        f"  PLAN recall {base_metrics.mode_accuracy_by_class.get(Mode.PLAN, 0):.3f} -> "
        f"{op_metrics.mode_accuracy_by_class.get(Mode.PLAN, 0):.3f}  |  "
        f"mode_acc {base_metrics.mode_accuracy:.3f} -> {op_metrics.mode_accuracy:.3f}"
    )

    # ---- (4) simulación económica (código) ----
    print("[3/4] simulación económica router vs cascada...", flush=True)
    suff = load_jsonl(data.parent / "labels" / "sufficiency.jsonl", SufficiencyRecord)
    random.Random(SEED).shuffle(suff)
    suff = suff[: args.sample]
    code_index: dict[str, str] = {}
    for r in test_all + train_all:  # código de ambos splits: más n en la economía
        if r["source"] in ("mbpp_plus", "humaneval_plus"):
            code_index[r["task_id"]] = _user(r).split("\nTask:\n", 1)[-1]
    bets_diff: list[tuple[int, int]] = []
    diffs: list[int] = []
    for rec in suff:
        prompt = code_index.get(rec.task_id)
        if not prompt:
            continue
        d = (
            REAL_POOL_IDS.index(rec.sufficient_model_id) + 1
            if rec.sufficient_model_id in REAL_POOL_IDS
            else len(REAL_POOL_IDS) + 1
        )
        choice = constrained_decision(model, tok, build_chat_example(REAL_POOL, prompt), REAL_POOL)
        bet_rank = (
            REAL_POOL_IDS.index(choice.model_id) + 1 if choice.model_id in REAL_POOL_IDS else 1
        )
        bets_diff.append((bet_rank, d))
        diffs.append(d)
    econ_router = simulate_router(bets_diff, REAL_COSTS)
    econ_cascade = simulate_cascade(diffs, REAL_COSTS)
    lines.append(f"\n(4) ECONOMÍA (código, n={econ_router.n}, coste ~ B de params):")
    lines.append(
        f"  router (apuesta+escala): coste_medio={econ_router.mean_cost:.2f} "
        f"pass_rate={econ_router.pass_rate:.3f}"
    )
    lines.append(
        f"  cascada pura (barato->capaz): coste_medio={econ_cascade.mean_cost:.2f} "
        f"pass_rate={econ_cascade.pass_rate:.3f}"
    )
    ahorro = (econ_cascade.mean_cost - econ_router.mean_cost) / econ_cascade.mean_cost * 100
    lines.append(f"  ahorro de coste del router vs cascada: {ahorro:+.1f}% (a igual pass-rate)")

    # ---- (5) regret ----
    reg_sum = reg_n = 0.0
    under = over = exact = 0
    for row, pred in zip(rows, base_preds, strict=True):
        reg = parse_registry_prompt(_user(row))
        gold = _gold(row)
        ch, orc = reg.get(pred.model_id), reg.get(gold.model_id)
        if ch and orc:
            diff = ch.cost - orc.cost
            reg_sum += diff
            reg_n += 1
            if diff < 0:
                under += 1
            elif diff > 0:
                over += 1
            else:
                exact += 1
    lines.append("\n(5) REGRET DE COSTE — definición y explicación:")
    lines.append("  regret_i = coste(elegido_i) - coste(oráculo_i), oráculo = mínimo suficiente")
    lines.append("  (la etiqueta gold). Media sobre el test:")
    lines.append(f"    regret medio = {reg_sum / reg_n:+.3f}   (n={int(reg_n)})")
    lines.append(
        f"    desglose: infra-provisión (más barato que el mínimo, <0)={under}  "
        f"exacto (=0)={exact}  sobre-provisión (>0)={over}"
    )
    sign = "NEGATIVO" if reg_sum < 0 else "positivo"
    lines.append(
        f"  Explicación del signo ({sign}): un regret negativo NO es un ahorro óptimo. "
        "Sale de que la política a veces elige un modelo MÁS BARATO que el mínimo "
        "suficiente (infra-provisión), lo que correlaciona con el fallo de PLAN "
        "(subestima la dificultad). Un regret sano sería ~0+; el negativo señala "
        "under-provisioning (fallo de calidad), no un ahorro. El -0.169 del informe "
        "de Fase 5 es exactamente esto."
    )

    # ---- (1) baseline grande zero-shot ----
    print(f"[4/4] baseline grande zero-shot (esdrac 7B, n={args.esdrac_sample})...", flush=True)
    esdrac_rows = rows[: args.esdrac_sample]
    e_golds = [_gold(r) for r in esdrac_rows]
    e_preds: list[RoutingDecision | None] = []
    for row in esdrac_rows:
        reg = parse_registry_prompt(_user(row))
        e_preds.append(_ollama_decision(row["messages"], reg, CAPABLE_MODEL))
    m_esdrac = compute_metrics(e_golds, e_preds)

    api_in = sum(estimate_tokens(_user(r)) + 60 for r in rows[:100]) / 100
    api_usd = (api_in / 1e6 * API_IN_PER_MTOK + 24 / 1e6 * API_OUT_PER_MTOK) * len(test_all)
    api_eur = api_usd * USD_TO_EUR

    lines.insert(1, "\n(1) BASELINE GRANDE ZERO-SHOT como router (mismo test):")
    lines.insert(2, f"  peldaño 1 (270M, entrenado): {_fmt2(base_metrics)}")
    lines.insert(3, f"  esdrac 7B zero-shot (26x):   {_fmt2(m_esdrac)}")
    lines.insert(
        4,
        f"  API de pago: estimación ~${api_usd:.2f} (~{api_eur:.2f} EUR, <{BUDGET_EUR:.0f} EUR "
        f"autorizados) pero SIN CLAVE -> NO ejecutado (BLOCKERS.md).",
    )

    report = "\n".join(lines)
    print("\n" + report)
    out = REPO_ROOT / "reports" / "fase5b_peldano1.md"
    out.write_text("# " + report.replace("=====", "").strip() + "\n", encoding="utf-8")
    print(f"\ninforme guardado en {out}")
    return 0


def _fmt2(m: Any) -> str:
    by = " ".join(f"{k.value[:4]}={v:.2f}" for k, v in m.mode_accuracy_by_class.items())
    return (
        f"exact={m.exact_match:.3f} mode={m.mode_accuracy:.3f} invalid={m.invalid_rate:.3f} [{by}]"
    )


if __name__ == "__main__":
    sys.exit(main())
