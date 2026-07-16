"""Tests de detección de entorno GPU. No requieren GPU: prueban el parser puro."""

from __future__ import annotations

from routerpolicy.env import MIN_VRAM_MB, GpuInfo, format_report, parse_nvidia_smi


def test_parse_single_gpu() -> None:
    out = "NVIDIA GeForce RTX 5070 Ti, 16303 MiB\n"
    info = parse_nvidia_smi(out)
    assert info.available is True
    assert info.name == "NVIDIA GeForce RTX 5070 Ti"
    assert info.vram_mb == 16303
    assert info.source == "nvidia-smi"
    assert info.meets_requirement is True


def test_parse_picks_largest_vram() -> None:
    out = "GPU-A, 8192 MiB\nGPU-B, 24576 MiB\n"
    info = parse_nvidia_smi(out)
    assert info.name == "GPU-B"
    assert info.vram_mb == 24576


def test_parse_empty_output_means_no_gpu() -> None:
    info = parse_nvidia_smi("")
    assert info.available is False
    assert info.vram_mb is None
    assert info.meets_requirement is False


def test_parse_ignores_malformed_lines() -> None:
    out = "garbage line\nNVIDIA X, 16000 MiB\n\n"
    info = parse_nvidia_smi(out)
    assert info.available is True
    assert info.vram_mb == 16000


def test_meets_requirement_threshold() -> None:
    just_below = GpuInfo(True, "X", MIN_VRAM_MB - 1, "nvidia-smi")
    exactly = GpuInfo(True, "X", MIN_VRAM_MB, "nvidia-smi")
    assert just_below.meets_requirement is False
    assert exactly.meets_requirement is True


def test_format_report_no_gpu() -> None:
    report = format_report(GpuInfo(False, None, None, "none"))
    assert "No se detectó GPU" in report


def test_format_report_ok() -> None:
    report = format_report(GpuInfo(True, "RTX 5070 Ti", 16303, "nvidia-smi"))
    assert "RTX 5070 Ti" in report
    assert "16303" in report
    assert "✅" in report
