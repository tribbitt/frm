"""Contamination filter: canonical digests of held-out pairs and informative grids, up to dihedral + recolouring."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from multiprocessing import Pool

import numpy as np

from frm.data.arc import Task
from frm.data.augment import dihedral

INFORMATIVE_CELLS = 36  # a grid needs at least this many cells ...
INFORMATIVE_FG = 8  # ... and this many cells off its most frequent colour

Pair = tuple[np.ndarray, np.ndarray]


def _relabel(gs: list[np.ndarray]) -> list[np.ndarray]:
    """Colours renumbered 0, 1, ... by first appearance over the grids, in order (row-major)."""
    flat = np.concatenate([g.reshape(-1) for g in gs])
    vals, first = np.unique(flat, return_index=True)
    lut = np.zeros(256, np.uint8)
    lut[vals[np.argsort(first)]] = np.arange(len(vals), dtype=np.uint8)
    return [lut[g] for g in gs]


def canon(*grids: np.ndarray) -> bytes:
    """16-byte digest of the grids' joint form, minimised over dihedrals after first-appearance relabelling."""
    gs = [np.asarray(g, dtype=np.uint8) for g in grids]
    best = None
    for k in range(8):
        ts = _relabel([np.ascontiguousarray(dihedral(g, k)) for g in gs])
        key = b"".join(bytes(t.shape) + t.tobytes() for t in ts)
        if best is None or key < best:
            best = key
    return hashlib.blake2b(best, digest_size=16).digest()


def informative(g: np.ndarray) -> bool:
    g = np.asarray(g)
    bg = np.bincount(g.reshape(-1).astype(np.int64), minlength=10).argmax()
    return g.size >= INFORMATIVE_CELLS and int((g != bg).sum()) >= INFORMATIVE_FG


class HeldOut:
    """Digests of held-out pairs (demos too if ``demos``) and informative outputs (inputs too if ``inputs``)."""

    def __init__(self, tasks: dict[str, Task], demos: bool = True, inputs: bool = True):
        self.ids: set[str] = set(tasks)
        self.pairs: set[bytes] = set()
        self.grids: set[bytes] = set()
        for t in tasks.values():
            for ex in (list(t["train"]) if demos else []) + list(t["test"]):
                a = np.asarray(ex["input"], dtype=np.uint8)
                b = None if ex.get("output") is None else np.asarray(ex["output"], dtype=np.uint8)
                if b is not None:
                    self.pairs.add(canon(a, b))
                for g in ((a,) if inputs else ()) + (() if b is None else (b,)):
                    if informative(g):
                        self.grids.add(canon(g))

    def grid_hit(self, g: np.ndarray) -> bool:
        return informative(g) and canon(g) in self.grids

    def hit(self, a: np.ndarray, b: np.ndarray) -> str | None:
        """``'pair'`` or ``'grid'`` if (a, b) is contaminated, else None."""
        if canon(a, b) in self.pairs:
            return "pair"
        if self.grid_hit(a) or self.grid_hit(b):
            return "grid"
        return None


_HELD: HeldOut | None = None


def _init(held: HeldOut) -> None:
    global _HELD
    _HELD = held


def _group_hits(group: Sequence[Pair]) -> list[str | None]:
    assert _HELD is not None
    return [_HELD.hit(a, b) for a, b in group]


def hits(groups: Sequence[Sequence[Pair]], held: HeldOut, workers: int = 8) -> list[list[str | None]]:
    """``held.hit`` over every pair of every group, in ``workers`` processes (inline when ``workers <= 1``)."""
    if workers <= 1:
        return [[held.hit(a, b) for a, b in g] for g in groups]
    chunk = max(1, len(groups) // (workers * 16))
    with Pool(workers, initializer=_init, initargs=(held,)) as pool:
        return pool.map(_group_hits, groups, chunksize=chunk)
