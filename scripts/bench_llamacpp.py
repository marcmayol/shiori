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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home()


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    )


PROMPT_TOKENS = 320  # tamaño representativo (registro ~250 + tarea)
GEN_TOKENS = 24  # una decisión JSON

# fila de llama-bench: "| ... | testNNN | t/s ± err |" (sin depender del ±)
_ROW_RE = re.compile(r"\|\s*(pp|tg)(\d+)\s*\|\s*([\d.]+)")


def _bench_one(llama_bench: Path, gguf: Path, threads: int) -> float:
    """Latencia (ms) de una decisión en CPU: prompt + generación, vía llama-bench."""
    proc = _run(
        [
            str(llama_bench),
            "-m",
            str(gguf),
            "-t",
            str(threads),
            "-p",
            str(PROMPT_TOKENS),
            "-n",
            str(GEN_TOKENS),
        ]
    )
    pp_tps = tg_tps = 0.0
    for kind, _n, tps in _ROW_RE.findall(proc.stdout):
        if kind == "pp":
            pp_tps = float(tps)
        elif kind == "tg":
            tg_tps = float(tps)
    if pp_tps <= 0 or tg_tps <= 0:
        return -1.0
    return (PROMPT_TOKENS / pp_tps + GEN_TOKENS / tg_tps) * 1000


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
    llama_bench = args.llama_bin / "llama-bench.exe"

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

    print(f"\n===== LATENCIA llama.cpp (CPU, {args.threads} hilos) =====", flush=True)
    print(f"  (decisión = {PROMPT_TOKENS} tokens de prompt + {GEN_TOKENS} de salida)")
    for name, path in quants.items():
        if not path.exists():
            print(f"  {name}: no generado")
            continue
        latency = _bench_one(llama_bench, path, args.threads)
        size_mb = path.stat().st_size / (1024**2)
        budget = "OK <1s" if 0 < latency < 1000 else "EXCEDE"
        print(f"  {name:<8} {size_mb:6.0f} MB   latencia/decisión ≈ {latency:.0f} ms  [{budget}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
