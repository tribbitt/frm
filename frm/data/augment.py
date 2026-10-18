"""Augmented views of a task: dihedral transform + colour permutation (colour 0 fixed), TRM-identical."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from frm.data.arc import Task

DIHEDRAL_INVERSE = (0, 3, 2, 1, 4, 5, 6, 7)
SEP = "|||"
RETRIES_FACTOR = 5


def dihedral(grid: np.ndarray, tid: int) -> np.ndarray:
    """The 8 dihedral symmetries, numbered as in TRM `dataset/common.py`."""
    if tid == 0:
        return grid
    if tid == 1:
        return np.rot90(grid, k=1)
    if tid == 2:
        return np.rot90(grid, k=2)
    if tid == 3:
        return np.rot90(grid, k=3)
    if tid == 4:
        return np.fliplr(grid)
    if tid == 5:
        return np.flipud(grid)
    if tid == 6:
        return grid.T
    if tid == 7:
        return np.fliplr(np.rot90(grid, k=1))
    raise ValueError(f"dihedral id {tid} not in 0..7")


def inverse_dihedral(grid: np.ndarray, tid: int) -> np.ndarray:
    return dihedral(grid, DIHEDRAL_INVERSE[tid])


@dataclass(frozen=True)
class Version:
    """One augmented view: ``apply(g) = dihedral(perm[g], tid)``; ``perm[c]`` is the new colour of colour c."""

    tid: int
    perm: tuple[int, ...]

    @staticmethod
    def identity() -> Version:
        return Version(0, tuple(range(10)))

    @staticmethod
    def parse(name: str) -> tuple[str, Version]:
        """Inverse of :meth:`name`: ``"task|||t3|||0123456789"`` -> (task_id, Version)."""
        task_id, t, perm = name.rsplit(SEP, 2)
        return task_id, Version(int(t[1:]), tuple(int(c) for c in perm))

    def name(self, task_id: str) -> str:
        return f"{task_id}{SEP}t{self.tid}{SEP}{''.join(str(c) for c in self.perm)}"

    def apply(self, grid: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(dihedral(np.asarray(self.perm, dtype=np.uint8)[grid], self.tid))

    def invert(self, grid: np.ndarray) -> np.ndarray:
        inv = np.argsort(np.asarray(self.perm)).astype(np.uint8)
        return np.ascontiguousarray(inv[inverse_dihedral(np.asarray(grid, dtype=np.uint8), self.tid)])


def _grid_hash(grid: np.ndarray) -> str:
    return hashlib.sha256(bytes(grid.shape) + grid.tobytes()).hexdigest()


def _puzzle_hash(pairs: list[tuple[np.ndarray, np.ndarray | None]]) -> str:
    hashes = sorted(f"{_grid_hash(i)}|{'' if o is None else _grid_hash(o)}" for i, o in pairs)
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def make_versions(task: Task, n: int, rng: np.random.Generator) -> list[Version]:
    """Identity + up to ``n`` random distinct versions (TRM `dataset/build_arc_dataset.py`); dedup ignores test
    outputs."""
    pairs = [(np.asarray(ex["input"], dtype=np.uint8), np.asarray(ex["output"], dtype=np.uint8))
             for ex in task["train"]]
    pairs += [(np.asarray(ex["input"], dtype=np.uint8), None) for ex in task["test"]]
    versions = [Version.identity()]
    if n <= 0:
        return versions
    seen = {_puzzle_hash(pairs)}
    for _ in range(RETRIES_FACTOR * n):
        tid = int(rng.integers(0, 8))
        perm = (0, *(int(c) for c in rng.permutation(np.arange(1, 10))))
        v = Version(tid, perm)
        h = _puzzle_hash([(v.apply(i), None if o is None else v.apply(o)) for i, o in pairs])
        if h not in seen:
            seen.add(h)
            versions.append(v)
            if len(versions) == n + 1:
                break
    return versions
