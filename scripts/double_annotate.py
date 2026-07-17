"""Control de ruido: doble anotación del juez sobre una muestra (Fase 2).

Juzga las mismas tareas de chat con DOS configuraciones del juez (determinista
vs con temperatura) y reporta el acuerdo (bruto, kappa de Cohen, por clase). Si
el acuerdo es bajo en alguna clase, hay que revisar la rúbrica antes de escalar.

Solo se anotan las tareas que NO resuelve la regla (las que van al juez).

Uso:
    uv run python scripts/double_annotate.py --sample 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from routerpolicy.dataset.sources import ingest_dolly
from routerpolicy.harness.backends import OllamaRunner
from routerpolicy.labeling.agreement import agreement_report
from routerpolicy.labeling.judge import JudgeError, LlmModeJudge
from routerpolicy.labeling.mode_rules import rule_mode_for_chat_task
from routerpolicy.schema.core import Mode

REPO_ROOT = Path(__file__).resolve().parents[1]


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="Doble anotación del juez")
    parser.add_argument("--sample", type=int, default=100, help="tareas de chat a muestrear")
    parser.add_argument("--model", type=str, default="esdrac:latest")
    args = parser.parse_args(argv)

    # Dos anotadores: mismo modelo, distinta temperatura (mide estabilidad del juez).
    judge_a = LlmModeJudge(
        OllamaRunner(args.model, options={"num_predict": 32, "temperature": 0.0})
    )
    judge_b = LlmModeJudge(
        OllamaRunner(args.model, options={"num_predict": 32, "temperature": 0.8})
    )

    tasks = ingest_dolly(limit=args.sample)
    judge_tasks = [t for t in tasks if rule_mode_for_chat_task(t) is None]
    print(f"muestra: {len(tasks)} chat; van al juez: {len(judge_tasks)}", flush=True)

    annot_a: list[Mode] = []
    annot_b: list[Mode] = []
    for task in judge_tasks:
        try:
            a = judge_a.judge(task.prompt)
            b = judge_b.judge(task.prompt)
        except JudgeError:
            continue
        annot_a.append(a)
        annot_b.append(b)

    if not annot_a:
        print("no hubo anotaciones válidas", flush=True)
        return 1

    report = agreement_report(annot_a, annot_b)
    print("\n===== ACUERDO (doble anotación) =====")
    print(f"n anotadas: {report.n}")
    print(f"acuerdo bruto: {report.raw_agreement:.3f}")
    print(f"kappa de Cohen: {report.cohen_kappa:.3f}")
    print("por clase (según anotador A):")
    for mode, agr in report.per_class_agreement.items():
        print(f"  {mode.value:<12} {agr:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
