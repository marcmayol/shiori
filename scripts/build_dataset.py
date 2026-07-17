"""Ensambla el dataset final de entrenamiento (Fase 3).

Pipeline: labels (Fase 2) + prompts -> BaseTask -> dedup near-dup -> split
estratificado -> augmentación por pool (train con firmas no reservadas, test con
las reservadas) -> train.jsonl / test.jsonl + informe de balance + leakage.

Determinista (semilla fija): el dataset se regenera desde los labels versionados.

Uso:
    uv run python scripts/build_dataset.py --factor 8
    uv run python scripts/build_dataset.py --sample 300   # ensayo barato
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

from routerpolicy.dataset.augment import AugmentedExample, augment_task
from routerpolicy.dataset.dedup import dedup_tasks
from routerpolicy.dataset.format import build_chat_example
from routerpolicy.dataset.materialize import build_base_tasks, build_prompt_index
from routerpolicy.dataset.splits import check_leakage, is_test_signature, stratified_split
from routerpolicy.dataset.tasks_io import load_jsonl
from routerpolicy.labeling.records import ModeRecord, SufficiencyRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260717


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def _ordered_pool_ids(pool_cfg: Path) -> list[str]:
    data = yaml.safe_load(pool_cfg.read_text(encoding="utf-8"))
    return [str(t["id"]) for t in data["tiers"]]


def _write_examples(path: Path, examples: list[AugmentedExample], split: str) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            row = {
                "task_id": ex.task_id,
                "source": ex.source,
                "split": split,
                "mode": ex.decision.mode.value,
                "model_id": ex.decision.model_id,
                "difficulty": ex.difficulty,
                "n_models": ex.n_models,
                "messages": build_chat_example(ex.registry, ex.prompt, ex.decision),
            }
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def _balance(examples: list[AugmentedExample]) -> dict[str, Counter[str]]:
    return {
        "mode": Counter(ex.decision.mode.value for ex in examples),
        "difficulty": Counter(str(ex.difficulty) for ex in examples),
        "n_models": Counter(str(ex.n_models) for ex in examples),
    }


def _print_balance(name: str, examples: list[AugmentedExample]) -> None:
    print(f"\n== {name}: {len(examples)} filas ==")
    for dim, counter in _balance(examples).items():
        pretty = ", ".join(f"{k}:{v}" for k, v in sorted(counter.items()))
        print(f"  {dim}: {pretty}")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="Ensambla el dataset final (Fase 3)")
    parser.add_argument("--factor", type=int, default=8, help="augmentación por tarea (5-8)")
    parser.add_argument("--dedup-threshold", type=float, default=0.8, help="similitud near-dup")
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--sample", type=int, default=None, help="límite de fuentes de modo")
    parser.add_argument("--labels-dir", type=Path, default=REPO_ROOT / "data" / "labels")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "dataset")
    parser.add_argument("--pool", type=Path, default=REPO_ROOT / "configs" / "pool.local.yaml")
    args = parser.parse_args(argv)

    ordered_ids = _ordered_pool_ids(args.pool)
    suff = load_jsonl(args.labels_dir / "sufficiency.jsonl", SufficiencyRecord)
    modes = load_jsonl(args.labels_dir / "mode.jsonl", ModeRecord)
    print(f"labels: {len(suff)} suficiencia + {len(modes)} modo", flush=True)

    print("construyendo índice de prompts (re-ingesta de fuentes)...", flush=True)
    prompt_index = build_prompt_index(sample=args.sample)
    base = build_base_tasks(suff, modes, prompt_index, ordered_ids)
    print(f"tareas base con prompt: {len(base)}", flush=True)

    deduped, dedup_report = dedup_tasks(base, threshold=args.dedup_threshold)
    print(f"dedup: {dedup_report.kept} mantenidas, {dedup_report.removed} eliminadas", flush=True)

    train_tasks, test_tasks = stratified_split(deduped, args.test_frac, random.Random(SEED))
    print(f"split: {len(train_tasks)} train / {len(test_tasks)} test", flush=True)

    rng = random.Random(SEED)
    train_ex: list[AugmentedExample] = []
    for task in train_tasks:
        train_ex += augment_task(
            task, rng, args.factor, allow_signature=lambda s: not is_test_signature(s)
        )
    test_ex: list[AugmentedExample] = []
    for task in test_tasks:
        test_ex += augment_task(task, rng, args.factor, allow_signature=is_test_signature)

    _write_examples(args.out_dir / "train.jsonl", train_ex, "train")
    _write_examples(args.out_dir / "test.jsonl", test_ex, "test")

    _print_balance("TRAIN", train_ex)
    _print_balance("TEST (pools no vistos)", test_ex)

    leakage = check_leakage(train_ex, test_ex)
    print("\n== LEAKAGE ==")
    print(f"  solape de tareas: {leakage.task_overlap}")
    print(f"  solape de firmas de pool: {leakage.signature_overlap}")
    print(f"  limpio: {leakage.clean}")
    print(f"\nTOTAL dataset: {len(train_ex) + len(test_ex)} filas", flush=True)
    return 0 if leakage.clean else 1


if __name__ == "__main__":
    sys.exit(main())
