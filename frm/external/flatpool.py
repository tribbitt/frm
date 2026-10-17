"""FlatPool: pairs grouped by task, stored flat in one npz (the format ``scripts/ingest_external.py`` reads).

    data      int8 [cells], every grid row-major, concatenated
    off       int64 [G+1], grid g is data[off[g]:off[g+1]]
    hw        int16 [G, 2]
    pair      int32 [P, 2], (input grid, output grid)
    pool_off  int32 [T+1], task t owns pair[pool_off[t]:pool_off[t+1]]
    ids       str [T]
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Pair = tuple[np.ndarray, np.ndarray]
MAX_SIDE = 30


@dataclass
class FlatPool:
    data: np.ndarray
    off: np.ndarray
    hw: np.ndarray
    pair: np.ndarray
    pool_off: np.ndarray
    ids: list[str]

    def grid(self, g: int) -> np.ndarray:
        h, w = self.hw[g]
        return self.data[self.off[g] : self.off[g + 1]].reshape(int(h), int(w))

    def n_tasks(self) -> int:
        return len(self.ids)

    def size(self, t: int) -> int:
        return int(self.pool_off[t + 1] - self.pool_off[t])

    def pair_at(self, t: int, i: int) -> Pair:
        gi, go = self.pair[self.pool_off[t] + i]
        return self.grid(int(gi)), self.grid(int(go))

    def pairs(self, t: int) -> list[Pair]:
        return [self.pair_at(t, i) for i in range(self.size(t))]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".partial.npz")  # readers never see half a file
        np.savez_compressed(tmp, data=self.data, off=self.off, hw=self.hw, pair=self.pair, pool_off=self.pool_off,
                            ids=np.array(self.ids, dtype=str))
        os.replace(tmp, path)

    @staticmethod
    def load(path: Path) -> FlatPool:
        z = np.load(path, allow_pickle=False)
        return FlatPool(z["data"], z["off"], z["hw"], z["pair"], z["pool_off"], [str(x) for x in z["ids"]])


def build_flat(tasks: list[tuple[str, list[Pair]]]) -> FlatPool:
    """(task id, pairs) in order -> FlatPool; each pair's grids stored input then output."""
    data, off, hw, pair, pool_off, ids = [], [0], [], [], [0], []
    for tid, prs in tasks:
        ids.append(tid)
        for a, b in prs:
            for g in (a, b):
                g = np.ascontiguousarray(g, dtype=np.int8)
                data.append(g.reshape(-1))
                off.append(off[-1] + g.size)
                hw.append(g.shape)
            pair.append((len(hw) - 2, len(hw) - 1))
        pool_off.append(len(pair))
    cells = np.concatenate(data).astype(np.int8) if data else np.zeros(0, np.int8)
    return FlatPool(cells, np.asarray(off, np.int64), np.asarray(hw, np.int16).reshape(-1, 2),
                    np.asarray(pair, np.int32).reshape(-1, 2), np.asarray(pool_off, np.int32), ids)


def fits(a: np.ndarray, b: np.ndarray) -> bool:
    """Both grids 2-D with sides in 1..MAX_SIDE."""
    return a.ndim == 2 and b.ndim == 2 and max(a.shape + b.shape) <= MAX_SIDE and min(a.shape + b.shape) >= 1


def dedup(prs: list[Pair]) -> list[Pair]:
    """Exact duplicate pairs once, first occurrence kept: a copy of the query in context would give its answer."""
    seen, out = set(), []
    for a, b in prs:
        k = (a.shape, a.tobytes(), b.shape, b.tobytes())
        if k not in seen:
            seen.add(k)
            out.append((a, b))
    return out
