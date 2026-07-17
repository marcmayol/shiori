"""Carga de la tabla de precios desde YAML (configs versionadas, no hardcode)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from routerpolicy.harness.cost import PriceSpec, PricingTable


def load_tiers(path: Path) -> list[PriceSpec]:
    """Carga los tiers ordenados (de más barato a más capaz) desde un YAML."""
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "tiers" not in data:
        raise ValueError("el YAML de precios debe tener una clave 'tiers'")
    tiers: list[PriceSpec] = []
    for entry in data["tiers"]:
        tiers.append(
            PriceSpec(
                model_id=str(entry["model_id"]),
                input_per_mtok=float(entry["input_per_mtok"]),
                output_per_mtok=float(entry["output_per_mtok"]),
                is_local=bool(entry.get("is_local", False)),
            )
        )
    if not tiers:
        raise ValueError("el YAML de precios no define ningún tier")
    return tiers


def load_pricing(path: Path) -> tuple[list[PriceSpec], PricingTable]:
    """Devuelve (tiers ordenados, tabla indexada)."""
    tiers = load_tiers(path)
    return tiers, PricingTable(tiers)
