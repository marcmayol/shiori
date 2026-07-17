"""Tests del control de ruido (acuerdo entre anotadores)."""

from __future__ import annotations

import pytest

from routerpolicy.labeling.agreement import agreement_report
from routerpolicy.schema.core import Mode

D, T, P = Mode.DIRECT, Mode.TOOL_CALL, Mode.PLAN


def test_perfect_agreement() -> None:
    a = [D, T, P, D]
    report = agreement_report(a, list(a))
    assert report.raw_agreement == 1.0
    assert report.cohen_kappa == pytest.approx(1.0)
    assert report.per_class_agreement[D] == 1.0


def test_partial_agreement() -> None:
    a = [D, T, P, D]
    b = [D, T, D, P]  # coincide en 2 de 4
    report = agreement_report(a, b)
    assert report.raw_agreement == pytest.approx(0.5)
    assert report.n == 4
    # clase D: 2 ítems según a, coinciden 1 -> 0.5
    assert report.per_class_agreement[D] == pytest.approx(0.5)


def test_chance_level_gives_low_kappa() -> None:
    a = [D, D, T, T]
    b = [D, T, D, T]  # acuerdo bruto 0.5 con marginales balanceadas
    report = agreement_report(a, b)
    assert report.raw_agreement == pytest.approx(0.5)
    assert report.cohen_kappa == pytest.approx(0.0, abs=1e-9)


def test_length_and_empty_validation() -> None:
    with pytest.raises(ValueError, match="misma longitud"):
        agreement_report([D], [D, T])
    with pytest.raises(ValueError, match="no hay anotaciones"):
        agreement_report([], [])
