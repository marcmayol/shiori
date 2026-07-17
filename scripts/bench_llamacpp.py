"""Latencia con el runtime local real (llama.cpp) — Fase 5/6.

Convierte el checkpoint a GGUF, cuantiza a Q8_0 y Q4_K_M, y mide la latencia por
decisión con llama-cli en CPU (8 hilos). Es la latencia que exige la sección 2
(< 1 s en CPU), medida con llama.cpp, no con transformers.

Requiere el repo llama.cpp (convert_hf_to_gguf.py) y los binarios (llama-quantize,
llama-cli). Rutas por defecto: ~/llama.cpp y ~/llamacpp-bin.

Uso:
    uv run --extra train python scripts/bench_llamacpp.py \
        --checkpoint checkpoints/gemma-3-270m/final --threads 8
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home()


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _build_prompt(checkpoint: Path, test_path: Path) -> str:
    """Construye el texto de prompt de un ejemplo real con la plantilla del modelo."""
    from transformers import AutoTokenizer

    from routerpolicy.training.data import load_rows, prompt_messages
    from routerpolicy.training.prepare import ensure_chat_template, merge_system_into_user

    tok: Any = AutoTokenizer.from_pretrained(str(checkpoint))
    ensure_chat_template(tok)
    row = load_rows(test_path, limit=1)[0]
    prepared = merge_system_into_user(prompt_messages(row))
    return str(tok.apply_chat_template(prepared, tokenize=False, add_generation_prompt=True))


_TOTAL_RE = re.compile(r"total time\s*=\s*([\d.]+)\s*ms")


def _bench_one(llama_cli: Path, gguf: Path, prompt_file: Path, threads: int, repeats: int) -> float:
    """Latencia mediana (ms) de generar una decisión (~24 tokens) en CPU."""
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        proc = _run(
            [
                str(llama_cli),
                "-m",
                str(gguf),
                "-f",
                str(prompt_file),
                "-n",
                "24",
                "-t",
                str(threads),
                "--no-display-prompt",
                "-no-cnv",
                "--temp",
                "0",
            ]
        )
        wall = (time.perf_counter() - t0) * 1000
        match = _TOTAL_RE.search(proc.stderr)
        times.append(float(match.group(1)) if match else wall)
    times.sort()
    return times[len(times) // 2]


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="Latencia con llama.cpp")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--llama-repo", type=Path, default=HOME / "llama.cpp")
    parser.add_argument("--llama-bin", type=Path, default=HOME / "llamacpp-bin")
    parser.add_argument("--gguf-dir", type=Path, default=REPO_ROOT / "artifacts" / "gguf")
    parser.add_argument("--test", type=Path, default=REPO_ROOT / "data" / "dataset" / "test.jsonl")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)

    args.gguf_dir.mkdir(parents=True, exist_ok=True)
    f16 = args.gguf_dir / "shiori-270m-f16.gguf"
    quants = {
        "Q8_0": args.gguf_dir / "shiori-270m-Q8_0.gguf",
        "Q4_K_M": args.gguf_dir / "shiori-270m-Q4_K_M.gguf",
    }
    quantize = args.llama_bin / "llama-quantize.exe"
    llama_cli = args.llama_bin / "llama-cli.exe"

    if not f16.exists():
        print("convirtiendo a GGUF f16...", flush=True)
        conv = _run(
            [
                sys.executable,
                str(args.llama_repo / "convert_hf_to_gguf.py"),
                str(args.checkpoint),
                "--outfile",
                str(f16),
                "--outtype",
                "f16",
            ]
        )
        if not f16.exists():
            print(conv.stderr[-2000:], flush=True)
            return 1
    for name, path in quants.items():
        if not path.exists():
            print(f"cuantizando {name}...", flush=True)
            _run([str(quantize), str(f16), str(path), name])

    prompt_file = args.gguf_dir / "sample_prompt.txt"
    prompt_file.write_text(_build_prompt(args.checkpoint, args.test), encoding="utf-8")

    print(f"\n===== LATENCIA llama.cpp (CPU, {args.threads} hilos) =====", flush=True)
    for name, path in quants.items():
        if not path.exists():
            print(f"  {name}: no generado")
            continue
        median = _bench_one(llama_cli, path, prompt_file, args.threads, args.repeats)
        size_mb = path.stat().st_size / (1024**2)
        print(f"  {name:<8} {size_mb:6.0f} MB   latencia/decisión ≈ {median:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
