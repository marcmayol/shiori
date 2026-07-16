"""Detección de entorno GPU para el pipeline de Shiori.

El requisito duro del plan (sección 2) es que todo el entrenamiento quepa en una
GPU de 16 GB (RTX 5070 Ti). Este módulo reporta la GPU disponible y su VRAM, y
evalúa si cumple el presupuesto, con lógica pura y testeable sin GPU física.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

# VRAM mínima requerida por el plan. La 5070 Ti expone ~16303 MiB; usamos un
# umbral con margen para no fallar por la memoria reservada por el driver.
MIN_VRAM_MB = 15000


@dataclass(frozen=True)
class GpuInfo:
    """Información de la GPU detectada."""

    available: bool
    name: str | None
    vram_mb: int | None
    source: str  # "nvidia-smi" | "torch" | "none"

    @property
    def meets_requirement(self) -> bool:
        """True si hay GPU con al menos MIN_VRAM_MB de VRAM."""
        return self.available and self.vram_mb is not None and self.vram_mb >= MIN_VRAM_MB


def parse_nvidia_smi(output: str) -> GpuInfo:
    """Parsea la salida CSV de `nvidia-smi --query-gpu=name,memory.total`.

    Espera líneas del tipo ``NVIDIA GeForce RTX 5070 Ti, 16303 MiB``.
    Se queda con la GPU de mayor VRAM si hay varias.
    """
    best: GpuInfo | None = None
    for raw_line in output.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        mem_token = parts[1].split()[0]  # "16303" de "16303 MiB"
        try:
            vram = int(mem_token)
        except ValueError:
            continue
        candidate = GpuInfo(available=True, name=name, vram_mb=vram, source="nvidia-smi")
        if best is None or (candidate.vram_mb or 0) > (best.vram_mb or 0):
            best = candidate
    if best is not None:
        return best
    return GpuInfo(available=False, name=None, vram_mb=None, source="none")


def _detect_via_nvidia_smi() -> GpuInfo | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    info = parse_nvidia_smi(result.stdout)
    return info if info.available else None


def _detect_via_torch() -> GpuInfo | None:
    try:
        import torch  # import perezoso: torch es opcional (extra `train`)
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return GpuInfo(
        available=True,
        name=props.name,
        vram_mb=int(props.total_memory // (1024 * 1024)),
        source="torch",
    )


def detect_gpu() -> GpuInfo:
    """Detecta la GPU: primero nvidia-smi (no requiere torch), luego torch."""
    info = _detect_via_nvidia_smi()
    if info is not None:
        return info
    info = _detect_via_torch()
    if info is not None:
        return info
    return GpuInfo(available=False, name=None, vram_mb=None, source="none")


def format_report(info: GpuInfo) -> str:
    """Mensaje humano del estado del entorno."""
    if not info.available:
        return (
            "❌ No se detectó GPU NVIDIA. El entrenamiento requiere una GPU de "
            f"≥ {MIN_VRAM_MB} MiB de VRAM (plan: RTX 5070 Ti, 16 GB)."
        )
    lines = [
        f"GPU: {info.name}",
        f"VRAM: {info.vram_mb} MiB (fuente: {info.source})",
    ]
    if info.meets_requirement:
        lines.append(f"✅ Cumple el requisito (≥ {MIN_VRAM_MB} MiB).")
    else:
        lines.append(f"⚠️  VRAM insuficiente: {info.vram_mb} MiB < {MIN_VRAM_MB} MiB requeridos.")
    return "\n".join(lines)
