"""Run de etiquetado offline (Fase 2): ingesta -> cascada -> modo -> reporte.

100% reanudable: los registros se anexan por tarea; relanzar salta lo hecho.
Usa `--sample N` para un ensayo barato antes del run completo.

Uso:
    uv run python scripts/run_labeling.py --sample 100
    uv run python scripts/run_labeling.py            # run completo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from routerpolicy.dataset.sources import (
    ingest_humaneval_plus,
    ingest_mbpp_plus,
    ingest_wildchat,
    ingest_xlam,
)
from routerpolicy.dataset.tasks_io import load_jsonl
from routerpolicy.harness.pool_io import load_pool
from routerpolicy.harness.tasks import ChatTask, CodeTask, ToolTask
from routerpolicy.labeling.judge import LlmModeJudge
from routerpolicy.labeling.pipeline import (
    build_report,
    run_mode_labeling,
    run_sufficiency_labeling,
)
from routerpolicy.labeling.records import ModeRecord, SufficiencyRecord

REPO_ROOT = Path(__file__).resolve().parents[1]


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def _ingest_code(sample: int | None) -> list[CodeTask]:
    print("Ingesta de código (HumanEval+, MBPP+)...", flush=True)
    tasks = ingest_humaneval_plus() + ingest_mbpp_plus()
    return tasks[:sample] if sample is not None else tasks


def _ingest_mode(sample: int | None) -> list[ToolTask | ChatTask]:
    print("Ingesta de modo (xLAM, WildChat)...", flush=True)
    half = None if sample is None else max(1, sample // 2)
    tool: list[ToolTask] = ingest_xlam(limit=half)
    chat: list[ChatTask] = ingest_wildchat(limit=half)
    out: list[ToolTask | ChatTask] = [*tool, *chat]
    return out


def _print_report(code_out: Path, mode_out: Path) -> None:
    suff = load_jsonl(code_out, SufficiencyRecord) if code_out.exists() else []
    modes = load_jsonl(mode_out, ModeRecord) if mode_out.exists() else []
    report = build_report(suff, modes)
    print("\n===== DISTRIBUCIÓN DE ETIQUETAS =====")
    print(f"código: {report.n_code}  |  modo: {report.n_mode}")
    print("por modelo-suficiente:")
    for model, n in sorted(report.by_sufficient_model.items(), key=lambda kv: -kv[1]):
        print(f"  {model:<22} {n}")
    print("por modo:")
    for mode, n in report.by_mode.items():
        print(f"  {mode.value:<12} {n}")
    print("por procedencia de modo:")
    for src, n in report.by_mode_source.items():
        print(f"  {src.value:<12} {n}")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="Run de etiquetado (Fase 2)")
    parser.add_argument("--sample", type=int, default=None, help="límite de tareas por eje")
    parser.add_argument("--pool", type=Path, default=REPO_ROOT / "configs" / "pool.local.yaml")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "labels")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--skip-code", action="store_true")
    parser.add_argument("--skip-mode", action="store_true")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    code_out = args.out_dir / "sufficiency.jsonl"
    mode_out = args.out_dir / "mode.jsonl"

    pool = load_pool(args.pool)
    print(f"Pool: {[r.model_id for r in pool]}", flush=True)

    if not args.skip_code:
        code_tasks = _ingest_code(args.sample)
        print(f"Etiquetando suficiencia de {len(code_tasks)} tareas...", flush=True)
        t0 = time.time()
        n = run_sufficiency_labeling(code_tasks, pool, code_out, timeout_s=args.timeout)
        print(f"  nuevas: {n}  ({time.time() - t0:.1f}s)", flush=True)

    if not args.skip_mode:
        mode_tasks = _ingest_mode(args.sample)
        judge = LlmModeJudge(pool[-1])  # el tier más capaz actúa de juez
        print(f"Etiquetando modo de {len(mode_tasks)} tareas...", flush=True)
        t0 = time.time()
        n = run_mode_labeling(mode_tasks, judge, mode_out)
        print(f"  nuevas: {n}  ({time.time() - t0:.1f}s)", flush=True)

    _print_report(code_out, mode_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
