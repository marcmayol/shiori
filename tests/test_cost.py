"""Tests de proyección y contabilidad de coste."""

from __future__ import annotations

import pytest

from routerpolicy.harness.cascade import CascadeAttempt, CascadeResult
from routerpolicy.harness.cost import (
    PriceSpec,
    PricingTable,
    cost_of_results,
    project_cost,
)

LOCAL = PriceSpec("local", 0.0, 0.0, is_local=True)
API = PriceSpec("api", input_per_mtok=3.0, output_per_mtok=15.0)


def test_pricespec_local_is_free() -> None:
    assert LOCAL.cost(1000, 1000) == 0.0


def test_pricespec_api_cost() -> None:
    # 1M input a 3.0 + 1M output a 15.0 = 18.0
    assert API.cost(1_000_000, 1_000_000) == pytest.approx(18.0)


def test_pricing_table_unknown_model_raises() -> None:
    table = PricingTable([LOCAL])
    with pytest.raises(KeyError):
        table.cost_of("nope", 10, 10)


def test_cost_of_results_measured() -> None:
    table = PricingTable([LOCAL, API])
    results = [
        CascadeResult(
            task_id="t1",
            sufficient_model_id="local",
            attempts=(CascadeAttempt("local", True, None, 100, 50),),
        ),
        CascadeResult(
            task_id="t2",
            sufficient_model_id="api",
            attempts=(
                CascadeAttempt("local", False, "err", 100, 50),
                CascadeAttempt("api", True, None, 1_000_000, 1_000_000),
            ),
        ),
    ]
    report = cost_of_results(results, table)
    assert report.n_tasks == 2
    assert report.per_model_calls == {"local": 2, "api": 1}
    assert report.total_usd == pytest.approx(18.0)  # solo la llamada de API cuesta


def test_project_cost_escalation_math() -> None:
    # 100 tareas, el local falla el 30% -> 30 llegan a la API.
    report = project_cost(
        n_tasks=100,
        avg_input_tokens=1_000_000,
        avg_output_tokens=1_000_000,
        tiers=[LOCAL, API],
        fail_rates=[0.3, 1.0],
    )
    assert report.per_model_calls == {"local": 100, "api": 30}
    # coste = 30 llamadas * 18.0 USD = 540
    assert report.total_usd == pytest.approx(540.0)
    assert report.per_task_usd() == pytest.approx(5.4)


def test_project_cost_validates_lengths_and_ranges() -> None:
    with pytest.raises(ValueError, match="misma longitud"):
        project_cost(10, 100, 100, [LOCAL, API], [0.5])
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        project_cost(10, 100, 100, [LOCAL], [1.5])
