"""Entrypoints de consola declarados en pyproject `[project.scripts]`."""

from __future__ import annotations

import sys

from routerpolicy.env import detect_gpu, format_report


def _force_utf8_stdout() -> None:
    """Evita UnicodeEncodeError en la consola cp1252 de Windows.

    El reporte usa caracteres no-ASCII (✅/⚠️); la consola por defecto de
    Windows no los codifica. reconfigure existe en 3.7+; si no, se ignora.
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def check_env_main() -> int:
    """`check-env`: reporta la GPU/VRAM y devuelve código de salida.

    Código 0 si cumple el requisito de VRAM, 1 en caso contrario. Así sirve
    tanto para inspección manual como para gates en scripts/CI locales.
    """
    _force_utf8_stdout()
    info = detect_gpu()
    print(format_report(info))
    return 0 if info.meets_requirement else 1


if __name__ == "__main__":
    sys.exit(check_env_main())
