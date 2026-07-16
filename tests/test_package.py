"""Test de humo: el paquete y todos sus subpaquetes importan limpiamente."""

from __future__ import annotations

import importlib

import routerpolicy

SUBPACKAGES = [
    "schema",
    "registry",
    "harness",
    "labeling",
    "dataset",
    "training",
    "evaluation",
    "inference",
]


def test_version_exposed() -> None:
    assert isinstance(routerpolicy.__version__, str)


def test_all_subpackages_import() -> None:
    for name in SUBPACKAGES:
        mod = importlib.import_module(f"routerpolicy.{name}")
        assert mod is not None
