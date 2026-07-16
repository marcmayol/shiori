"""CLI de fase: `uv run python scripts/check_env.py`.

Reexporta el entrypoint de consola para que exista un script por fase bajo
`scripts/` (convención de la sección 4 del plan).
"""

from __future__ import annotations

import sys

from routerpolicy.scripts_entry import check_env_main

if __name__ == "__main__":
    sys.exit(check_env_main())
