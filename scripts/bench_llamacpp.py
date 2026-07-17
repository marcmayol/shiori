"""Latencia con el runtime local real (llama.cpp) — Fase 5/6.

Convierte el checkpoint a GGUF, cuantiza a Q8_0 y Q4_K_M, y mide la latencia por
decisión con llama-bench en CPU (8 hilos) y GPU (CUDA, -ngl 99). Es la latencia
que exige la sección 2 (< 1 s CPU, < 50 ms GPU), medida con llama.cpp.

Requiere el repo llama.cpp (convert_hf_to_gguf.py), los binarios CPU
(~/llamacpp-bin: llama-quantize, llama-bench) y CUDA (~/llamacpp-cuda).

Uso:
    uv run --extra train python scripts/bench_llamacpp.py \
        --checkpoint checkpoints/gemma-3-270m/final_gguf --threads 8
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


def _bench_one(llama_bench: Path, gguf: Path, threads: int, ngl: int = 0) -> float:
    """Latencia (ms) de una decisión: prompt + generación, vía llama-bench.

    ngl>0 offloada capas a la GPU (CUDA); ngl=0 es CPU puro.
    """
    cmd = [
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
    if ngl > 0:
        cmd += ["-ngl", str(ngl)]
    proc = _run(cmd)
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
    parser.add_argument("--llama-bin-cuda", type=Path, default=HOME / "llamacpp-cuda")
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

    llama_bench_cuda = args.llama_bin_cuda / "llama-bench.exe"
    print(
        f"\n===== LATENCIA llama.cpp (decisión = {PROMPT_TOKENS} prompt + {GEN_TOKENS} salida) ====="
    )
    print(f"  {'cuant':<8}{'tamaño':>9}{'CPU 8h':>12}{'GPU':>10}   presupuesto")
    for name, path in quants.items():
        if not path.exists():
            print(f"  {name}: no generado")
            continue
        cpu = _bench_one(llama_bench, path, args.threads)
        gpu = (
            _bench_one(llama_bench_cuda, path, args.threads, ngl=99)
            if llama_bench_cuda.exists()
            else -1.0
        )
        size_mb = path.stat().st_size / (1024**2)
        cpu_ok = "CPU<1s ✓" if 0 < cpu < 1000 else "CPU ✗"
        gpu_ok = "GPU<50ms ✓" if 0 < gpu < 50 else ("GPU " + ("n/a" if gpu < 0 else "✗"))
        gpu_s = f"{gpu:.0f} ms" if gpu > 0 else "n/a"
        print(f"  {name:<8}{size_mb:6.0f} MB{cpu:>8.0f} ms{gpu_s:>10}   {cpu_ok} {gpu_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
