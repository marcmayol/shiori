"""Dedupe near-duplicate de tareas (MinHash + banding LSH), sin dependencias.

Se aplica ANTES del split: tareas casi idénticas no deben quedar repartidas entre
train y test (sería leakage). Determinista (semillas fijas).
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass

from routerpolicy.dataset.augment import BaseTask

_PRIME = (1 << 61) - 1
_NUM_PERM = 64
_BANDS = 16
_ROWS = _NUM_PERM // _BANDS  # 4
_SHINGLE_K = 3


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _shingles(text: str, k: int = _SHINGLE_K) -> set[str]:
    toks = _normalize(text).split()
    if not toks:
        return set()
    if len(toks) < k:
        return {" ".join(toks)}
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def _hash(shingle: str) -> int:
    return int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")


def _seeds(num_perm: int = _NUM_PERM) -> list[tuple[int, int]]:
    rng = random.Random(20260717)  # semilla fija -> reproducible
    return [(rng.randint(1, _PRIME - 1), rng.randint(0, _PRIME - 1)) for _ in range(num_perm)]


_SEEDS = _seeds()


def _signature(text: str) -> tuple[int, ...]:
    shingles = _shingles(text)
    if not shingles:
        return tuple(0 for _ in _SEEDS)
    hashes = [_hash(s) for s in shingles]
    return tuple(min((a * h + b) % _PRIME for h in hashes) for a, b in _SEEDS)


def _similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    return sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # une hacia el índice menor


@dataclass(frozen=True)
class DedupReport:
    kept: int
    removed: int


def dedup_tasks(
    tasks: Sequence[BaseTask], threshold: float = 0.7
) -> tuple[list[BaseTask], DedupReport]:
    """Elimina tareas near-duplicate por prompt. Mantiene la primera de cada grupo."""
    n = len(tasks)
    signatures = [_signature(t.prompt) for t in tasks]
    uf = _UnionFind(n)

    # candidatos por banda: mismo bucket en alguna banda -> comparar
    for band in range(_BANDS):
        buckets: dict[tuple[int, ...], list[int]] = {}
        lo = band * _ROWS
        key_slice = slice(lo, lo + _ROWS)
        for i in range(n):
            buckets.setdefault(signatures[i][key_slice], []).append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            first = members[0]
            for other in members[1:]:
                if _similarity(signatures[first], signatures[other]) >= threshold:
                    uf.union(first, other)

    seen_roots: set[int] = set()
    kept: list[BaseTask] = []
    for i in range(n):
        root = uf.find(i)
        if root not in seen_roots:
            seen_roots.add(root)
            kept.append(tasks[i])
    return kept, DedupReport(kept=len(kept), removed=n - len(kept))
