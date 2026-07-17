"""Verificación mecánica de código: ejecuta candidato + tests en subproceso.

La etiqueta de suficiencia se obtiene EJECUTANDO y verificando (sección 2), no
opinando. El aislamiento es un subproceso con timeout en un directorio temporal
propio. NO es un sandbox de seguridad completo (sin límites de red/FS): es el MVP
pragmático de evalplus para tareas de benchmark; endurecerlo es trabajo futuro.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT_S = 15.0


@dataclass(frozen=True)
class VerifyResult:
    """Resultado de verificar un candidato contra sus tests."""

    passed: bool
    error: str | None
    duration_s: float
    timed_out: bool = False


def build_script(candidate_code: str, test_code: str) -> str:
    """Ensambla el script a ejecutar: definición candidata + verificación."""
    return f"{candidate_code}\n\n# --- verification ---\n{test_code}\n"


def verify_code(
    candidate_code: str,
    test_code: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> VerifyResult:
    """Ejecuta el candidato con sus tests; pasa si termina con código 0.

    Se ejecuta con `sys.executable` en un cwd temporal aislado. Cualquier
    excepción/assert fallido produce returncode != 0 -> passed=False.
    """
    script = build_script(candidate_code, test_code)
    with tempfile.TemporaryDirectory(prefix="shiori_verify_") as tmp:
        script_path = Path(tmp) / "candidate.py"
        script_path.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tmp,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return VerifyResult(
                passed=False,
                error=f"timeout tras {timeout_s}s",
                duration_s=timeout_s,
                timed_out=True,
            )
        if proc.returncode == 0:
            return VerifyResult(passed=True, error=None, duration_s=0.0)
        err = (proc.stderr or proc.stdout or "").strip()
        # recorta trazas largas para no inflar el dataset de procedencia
        if len(err) > 2000:
            err = err[:2000] + "\n...[truncado]"
        return VerifyResult(passed=False, error=err, duration_s=0.0)
