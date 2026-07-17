"""Tests del verificador de código en subproceso."""

from __future__ import annotations

from routerpolicy.harness.verify import verify_code


def test_passing_candidate() -> None:
    candidate = "def add(a, b):\n    return a + b"
    tests = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
    result = verify_code(candidate, tests)
    assert result.passed
    assert result.error is None
    assert not result.timed_out


def test_failing_candidate_reports_error() -> None:
    candidate = "def add(a, b):\n    return a - b"  # mal
    tests = "assert add(2, 3) == 5"
    result = verify_code(candidate, tests)
    assert not result.passed
    assert result.error is not None
    assert "AssertionError" in result.error


def test_syntax_error_is_failure() -> None:
    result = verify_code("def broken(:\n  pass", "assert True")
    assert not result.passed
    assert result.error is not None


def test_timeout() -> None:
    candidate = "def f():\n    while True:\n        pass"
    tests = "f()"
    result = verify_code(candidate, tests, timeout_s=1.0)
    assert not result.passed
    assert result.timed_out
