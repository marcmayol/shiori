"""Baseline API zero-shot como router (OpenAI) sobre el test congelado.

Imprime la estimación de coste ANTES de lanzar; aborta si supera el presupuesto.
Structured output (JSON schema desde el registro) -> salida siempre válida.
Guarda predicciones a un JSONL para el informe comparativo.

Clave: .secrets/openai_key.txt (gitignored) o env OPENAI_API_KEY.

Uso:
    uv run --extra train python scripts/api_baseline.py --sample 400 --model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any

from routerpolicy.dataset.format import SYSTEM_PROMPT, build_user_message
from routerpolicy.inference.constraints import build_json_schema
from routerpolicy.registry.render import parse_registry_prompt
from routerpolicy.schema.core import Mode, Registry, RoutingDecision
from routerpolicy.schema.validation import DecisionValidationError, validate_decision_json
from routerpolicy.tokens import estimate_tokens
from routerpolicy.training.data import load_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260717
BUDGET_EUR = 10.0
USD_TO_EUR = 0.92

# Precios OpenAI (USD/Mtok) — a la fecha; confirmar en la doc del proveedor.
PRICES = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
}


def _force_utf8() -> None:
    rc = getattr(sys.stdout, "reconfigure", None)
    if callable(rc):
        rc(encoding="utf-8", errors="replace")


def _api_key() -> str:
    import os

    env = os.environ.get("OPENAI_API_KEY")
    if env:
        return env
    f = REPO_ROOT / ".secrets" / "openai_key.txt"
    if f.exists():
        return f.read_text(encoding="utf-8").strip()
    raise RuntimeError("sin clave: pon OPENAI_API_KEY o .secrets/openai_key.txt")


def _user(row: dict[str, Any]) -> str:
    return str(next(m["content"] for m in row["messages"] if m["role"] == "user"))


def _task_only(user_msg: str) -> str:
    return user_msg.split("\nTask:\n", 1)[-1]


def _openai_decision(
    key: str, model: str, registry: Registry, user_msg: str
) -> tuple[RoutingDecision | None, int, int]:
    """Llama a OpenAI con structured output. Devuelve (decisión|None, in_toks, out_toks)."""
    schema = build_json_schema(registry)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0,
        "max_tokens": 40,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "routing_decision", "schema": schema, "strict": True},
        },
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    in_t, out_t = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    try:
        return validate_decision_json(text, registry), in_t, out_t
    except DecisionValidationError:
        return None, in_t, out_t


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    ap = argparse.ArgumentParser(description="Baseline API zero-shot")
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--model", type=str, default="gpt-4o")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "api_baseline_preds.jsonl")
    args = ap.parse_args(argv)

    key = _api_key()
    rows = load_rows(REPO_ROOT / "data" / "dataset" / "test.jsonl")
    random.Random(SEED).shuffle(rows)
    rows = rows[: args.sample]

    # ---- estimación de coste ANTES de lanzar ----
    in_price, out_price = PRICES.get(args.model, (2.5, 10.0))
    avg_in = sum(
        estimate_tokens(_user(r)) + estimate_tokens(SYSTEM_PROMPT) for r in rows[:100]
    ) / min(100, len(rows))
    est_in = avg_in * len(rows)
    est_out = 24 * len(rows)
    est_usd = est_in / 1e6 * in_price + est_out / 1e6 * out_price
    est_eur = est_usd * USD_TO_EUR
    print(f"Modelo API: {args.model}  |  muestra: {len(rows)}")
    print(f"ESTIMACIÓN: ~${est_usd:.2f} (~{est_eur:.2f} EUR)  presupuesto={BUDGET_EUR:.0f} EUR")
    if est_eur > BUDGET_EUR:
        print(f"ABORTA: estimación {est_eur:.2f} EUR > presupuesto {BUDGET_EUR:.0f} EUR.")
        return 2
    print("dentro de presupuesto -> ejecutando...", flush=True)

    preds: list[dict[str, Any]] = []
    real_in = real_out = 0
    for i, row in enumerate(rows):
        reg = parse_registry_prompt(_user(row))
        # prompt = registro renderizado + tarea (mismo formato que el peldaño 1)
        user_msg = build_user_message(reg, _task_only(_user(row)))
        dec, it, ot = _openai_decision(key, args.model, reg, user_msg)
        real_in += it
        real_out += ot
        preds.append(
            {
                "task_id": row["task_id"],
                "gold_mode": row["mode"],
                "gold_model_id": row["model_id"],
                "pred": dec.to_canonical_json() if dec else None,
                "pred_mode": dec.mode.value if dec else None,
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}...", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(p, ensure_ascii=True) + "\n")

    real_usd = real_in / 1e6 * in_price + real_out / 1e6 * out_price
    valid = sum(1 for p in preds if p["pred"] is not None)
    exact = sum(1 for p in preds if p["pred"] == _canon(p))
    mode_ok = sum(1 for p in preds if p["pred_mode"] == p["gold_mode"])
    plan_total = sum(1 for p in preds if p["gold_mode"] == Mode.PLAN.value)
    plan_ok = sum(
        1 for p in preds if p["gold_mode"] == Mode.PLAN.value and p["pred_mode"] == Mode.PLAN.value
    )
    n = len(preds)
    print(
        f"\ncoste REAL: ~${real_usd:.2f} (~{real_usd * USD_TO_EUR:.2f} EUR) "
        f"[in={real_in} out={real_out} tokens]"
    )
    print(
        f"n={n}  exact={exact / n:.3f}  mode_acc={mode_ok / n:.3f}  "
        f"invalid={(n - valid) / n:.3f}  PLAN_acc={(plan_ok / plan_total) if plan_total else 0:.3f}"
    )
    print(f"predicciones guardadas en {args.out}")
    return 0


def _canon(p: dict[str, Any]) -> str:
    return RoutingDecision(
        mode=Mode(p["gold_mode"]), model_id=p["gold_model_id"]
    ).to_canonical_json()


if __name__ == "__main__":
    sys.exit(main())
