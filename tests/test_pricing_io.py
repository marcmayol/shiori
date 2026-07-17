"""Tests de carga de precios y del script de estimación de coste."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from routerpolicy.harness.pricing_io import load_pricing, load_tiers

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "configs" / "pricing.example.yaml"


def test_load_example_pricing() -> None:
    tiers, table = load_pricing(EXAMPLE)
    assert [t.model_id for t in tiers] == [
        "local-small-code",
        "local-medium",
        "api-capable",
    ]
    assert tiers[0].is_local
    assert not tiers[-1].is_local
    assert table.cost_of("api-capable", 1_000_000, 0) == pytest.approx(3.0)


def test_load_rejects_missing_tiers(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("foo: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tiers"):
        load_tiers(bad)


def test_estimate_cost_script_runs() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "estimate_cost.py"),
            "--n-tasks",
            "100",
            "--pricing",
            str(EXAMPLE),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "TOTAL" in proc.stdout
    assert "esperado" in proc.stdout
