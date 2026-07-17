"""Publica el artefacto en HF Hub (Fase 6): pesos + GGUF Q8_0/Q4_K_M + model card.

Ensambla artifacts/hf/ (pesos del checkpoint final_gguf, los GGUF cuantizados, el
Modelfile de Ollama y la model card) y lo sube a un repo público Apache-2.0.
Reproducible; requiere estar autenticado en HF con permiso de escritura.

Uso:
    uv run --extra train python scripts/publish.py --repo natzx94/shiori-router-270m
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _force_utf8() -> None:
    rc = getattr(sys.stdout, "reconfigure", None)
    if callable(rc):
        rc(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    ap = argparse.ArgumentParser(description="Publica Shiori en HF Hub")
    ap.add_argument("--repo", type=str, required=True)
    ap.add_argument(
        "--weights", type=Path, default=REPO_ROOT / "checkpoints" / "gemma-3-270m" / "final_gguf"
    )
    ap.add_argument("--gguf-dir", type=Path, default=REPO_ROOT / "artifacts" / "gguf")
    ap.add_argument("--staging", type=Path, default=REPO_ROOT / "artifacts" / "hf")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi, create_repo

    args.staging.mkdir(parents=True, exist_ok=True)
    # pesos transformers
    for name in [
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]:
        src = args.weights / name
        if src.exists():
            shutil.copy2(src, args.staging / name)
    # GGUF cuantizados
    for q in ["Q8_0", "Q4_K_M"]:
        g = args.gguf_dir / f"shiori-270m-{q}.gguf"
        if g.exists():
            shutil.copy2(g, args.staging / g.name)
    # Modelfile + model card
    shutil.copy2(REPO_ROOT / "deploy" / "Modelfile", args.staging / "Modelfile")
    shutil.copy2(REPO_ROOT / "reports" / "model_card.md", args.staging / "README.md")

    print(f"subiendo {args.staging} -> {args.repo} (private={args.private})", flush=True)
    create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
    HfApi().upload_folder(
        folder_path=str(args.staging),
        repo_id=args.repo,
        repo_type="model",
        commit_message="Shiori router 270m (peldaño 1): pesos + GGUF + model card",
    )
    print(f"publicado: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
