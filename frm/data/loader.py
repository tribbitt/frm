"""Dataset access: group-balanced training stream with on-the-fly augmentation/translation, and eval batches."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from frm.data.build import M, SOURCE_CODES
from frm.data.encode import IGNORE, augmented_dims, labels_from, render


@dataclass
class SplitData:
    """Arrays of one bucket/split (see ``frm/data/build.py``); canvases stacked [plain, transposed]."""

    S: int
    tasks: list[str]
    vstart: np.ndarray
    vcount: np.ndarray
    source: np.ndarray
    ex_start: np.ndarray
    ex_index: np.ndarray
    inp: np.ndarray
    inp_hw: np.ndarray
    out: np.ndarray
    out_hw: np.ndarray
    sub: np.ndarray | None  # [2, E*M, S, S]
    sub_hw: np.ndarray | None  # [E, M, 2]


def _stack(base: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.stack([base, base.transpose(0, 2, 1)]))


class Dataset:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        with open(self.root / "meta.json") as f:
            self.meta: dict = json.load(f)
        with open(self.root / "identifiers.json") as f:
            self.identifiers: list[str] = json.load(f)
        self.buckets: list[int] = list(self.meta["buckets"])
        self.num_ids: int = self.meta["num_ids"]
        self.version_tid = np.load(self.root / "versions_tid.npy")
        self.version_perm = np.load(self.root / "versions_perm.npy")
        self.split_data = cache(self._load_split)
        self._test_tasks: dict | None = None

    def splits(self) -> list[str]:
        return list(self.meta["splits"])

    def tasks(self, split: str) -> dict[str, dict]:
        """Original un-augmented tasks of an eval split ("dev", "dev_composite", "test"); ``test`` index i is the
        ``test_index`` of eval_batches keys. Grids are lists; unknown outputs are ``None``."""
        if self._test_tasks is None:
            with open(self.root / "test_tasks.json") as f:
                self._test_tasks = json.load(f)
        return self._test_tasks.get(split, {})

    def has_split(self, bucket: int, split: str) -> bool:
        return str(bucket) in self.meta["counts"].get(split, {})

    def num_train_examples(self, bucket: int) -> int:
        """Base (un-augmented) training pairs in the bucket."""
        by_source = self.meta["counts"].get("train", {}).get(str(bucket), {})
        return sum(c["examples"] for c in by_source.values())

    def _load_split(self, bucket: int, split: str) -> SplitData:
        d = self.root / f"b{bucket}" / split

        def ld(name: str) -> np.ndarray:
            return np.load(d / f"{name}.npy")

        with open(d / "tasks.json") as f:
            tasks = json.load(f)
        has_sub = (d / "sub.npy").exists()
        sub = ld("sub") if has_sub else None
        return SplitData(
            S=bucket,
            tasks=tasks,
            vstart=ld("task_vstart"),
            vcount=ld("task_vcount"),
            source=ld("task_source"),
            ex_start=ld("task_ex_start"),
            ex_index=ld("ex_index"),
            inp=_stack(ld("inp")),
            inp_hw=ld("inp_hw"),
            out=_stack(ld("out")),
            out_hw=ld("out_hw"),
            sub=None if sub is None else _stack(sub.reshape(-1, bucket, bucket)),
            sub_hw=ld("sub_hw") if has_sub else None,
        )


def _rows(ds: Dataset, d: SplitData, ex: np.ndarray, vid: np.ndarray, offset: np.ndarray) -> dict[str, np.ndarray]:
    """Render rows (example index, identifier, canvas offset) into model arrays; ``source`` is the row's
    ``SOURCE_CODES`` value (per-source training metrics and fork prices)."""
    tid, perm = ds.version_tid[vid], ds.version_perm[vid]
    batch = {
        "inputs": render(d.inp, ex, d.inp_hw[ex], tid, perm, offset),
        "labels": labels_from(render(d.out, ex, d.out_hw[ex], tid, perm, offset)),
        "puzzle_ids": vid.astype(np.int32),
        "source": d.source[np.searchsorted(d.ex_start, ex, side="right") - 1],
    }
    n, S2 = len(ex), d.S * d.S
    subgoals = np.full((n, M, S2), IGNORE, dtype=np.int16)
    mask = np.zeros((n, M), dtype=bool)
    if d.sub is not None:
        mask = d.sub_hw[ex, :, 0] > 0
        r, m = np.nonzero(mask)
        if len(r):
            g = ex[r] * M + m
            subgoals[r, m] = labels_from(render(d.sub, g, d.sub_hw[ex[r], m], tid[r], perm[r], offset[r]))
    batch["subgoals"] = subgoals
    batch["subgoal_mask"] = mask
    return batch


class TrainStream:
    """Group-balanced stream (TRM ``puzzle_dataset.py``) over the training tasks of one bucket.

    Each source (``SOURCE_CODES``) keeps its own epoch (a permutation of its base tasks). A draw picks a source with
    probability set by ``source_weights`` (row fractions; ``None`` = proportional to the source's rows per epoch), the
    next task of that source's epoch and a random version of it, and yields a random subset of
    min(task size, ``max_pairs_per_draw``) of the task's training pairs in random order (RE-ARC tasks have ~1000 pairs:
    uncapped, one draw would fill a batch with one rule). Row fractions and rows per draw use the capped sizes. Rows of
    a draw not consumed by ``take`` are carried to the next call.
    """

    def __init__(
        self,
        ds: Dataset,
        bucket: int,
        rng: np.random.Generator,
        source_weights: dict[str, float] | None = None,
        translate_p: float = 0.75,
        max_pairs_per_draw: int = 16,
    ):
        if max_pairs_per_draw < 1:
            raise ValueError(f"max_pairs_per_draw must be >= 1, got {max_pairs_per_draw}")
        self.ds, self.rng, self.translate_p = ds, rng, translate_p
        self.d = d = ds.split_data(bucket, "train")
        self.task_size = np.diff(d.ex_start)
        self.draw_size = np.minimum(self.task_size, max_pairs_per_draw)
        self.sources = [s for s, c in SOURCE_CODES.items() if np.any(d.source == c)]
        self.source_tasks = {s: np.flatnonzero(d.source == SOURCE_CODES[s]) for s in self.sources}
        mean_rows = np.array([self.draw_size[self.source_tasks[s]].mean() for s in self.sources])
        if source_weights is None:
            w = np.array([self.draw_size[self.source_tasks[s]].sum() for s in self.sources], dtype=np.float64)
        else:
            w = np.array([float(source_weights.get(s, 0.0)) for s in self.sources])
        if w.sum() <= 0:
            raise ValueError(f"source_weights {source_weights} give no weight to sources {self.sources} of bucket {bucket}")
        p = (w / w.sum()) / mean_rows  # per-draw probability so that row fractions match w
        self.p = p / p.sum()
        self.rows_per_draw = float(self.p @ mean_rows)
        self.perm = {s: np.zeros(0, dtype=np.int64) for s in self.sources}
        self.pos = {s: 0 for s in self.sources}
        self.pending_ex = np.zeros(0, dtype=np.int64)
        self.pending_vid = np.zeros(0, dtype=np.int64)

    def _next_tasks(self, s: str, k: int) -> np.ndarray:
        out = []
        while k > 0:
            if self.pos[s] >= len(self.perm[s]):
                self.perm[s] = self.source_tasks[s][self.rng.permutation(len(self.source_tasks[s]))]
                self.pos[s] = 0
            take = self.perm[s][self.pos[s] : self.pos[s] + k]
            self.pos[s] += len(take)
            k -= len(take)
            out.append(take)
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)

    def _draw(self, k: int) -> None:
        src = self.rng.choice(len(self.sources), size=k, p=self.p)
        task = np.empty(k, dtype=np.int64)
        for i, s in enumerate(self.sources):
            sel = src == i
            task[sel] = self._next_tasks(s, int(sel.sum()))
        d = self.d
        vid = d.vstart[task] + (self.rng.random(k) * d.vcount[task]).astype(np.int64)
        size = self.task_size[task]
        draw = np.repeat(np.arange(k), size)
        local = np.arange(len(draw)) - (np.cumsum(size) - size)[draw]  # pair index inside its task
        order = np.lexsort((self.rng.random(len(draw)), draw))  # random order inside each draw; draws stay in order
        # order[i] belongs to draw[i] (sorting by draw keeps each draw's block), so local[i] is its rank there
        sel = order[local < self.draw_size[task][draw]]
        self.pending_ex = np.concatenate([self.pending_ex, d.ex_start[task][draw[sel]] + local[sel]])
        self.pending_vid = np.concatenate([self.pending_vid, vid[draw[sel]]])

    def take(self, n: int) -> dict[str, np.ndarray]:
        while len(self.pending_ex) < n:
            self._draw(max(1, math.ceil((n - len(self.pending_ex)) / self.rows_per_draw)))
        ex, vid = self.pending_ex[:n], self.pending_vid[:n]
        self.pending_ex, self.pending_vid = self.pending_ex[n:], self.pending_vid[n:]
        return _rows(self.ds, self.d, ex, vid, self._offsets(ex, vid))

    def _offsets(self, ex: np.ndarray, vid: np.ndarray) -> np.ndarray:
        """Fresh random offset per row (prob ``translate_p``) keeping input, output and sub-goals with EOS inside."""
        d, tid = self.d, self.ds.version_tid[vid]
        ext = np.maximum(augmented_dims(d.inp_hw[ex], tid), augmented_dims(d.out_hw[ex], tid)).astype(np.int64)
        if d.sub is not None:
            for m in range(M):
                ext = np.maximum(ext, augmented_dims(d.sub_hw[ex, m], tid))
        room = np.maximum(d.S - 1 - ext, 0)  # S=30 grids of side 30 cannot move
        shift = (self.rng.random((len(ex), 2)) * (room + 1)).astype(np.int64)
        return np.where(self.rng.random((len(ex), 1)) < self.translate_p, shift, 0)

    def state_dict(self) -> dict:
        return {
            "rng": self.rng.bit_generator.state,
            "perm": {s: self.perm[s].tolist() for s in self.sources},
            "pos": dict(self.pos),
            "pending_ex": self.pending_ex.tolist(),
            "pending_vid": self.pending_vid.tolist(),
        }

    def load_state_dict(self, d: dict) -> None:
        self.rng.bit_generator.state = d["rng"]
        self.perm = {s: np.asarray(d["perm"][s], dtype=np.int64) for s in self.sources}
        self.pos = {s: int(d["pos"][s]) for s in self.sources}
        self.pending_ex = np.asarray(d["pending_ex"], dtype=np.int64)
        self.pending_vid = np.asarray(d["pending_vid"], dtype=np.int64)


def eval_versions(d, t: int, max_versions: int | None, trained: np.ndarray | None) -> np.ndarray:
    """Identifiers of task ``t``'s versions that vote, in version order (identity first), at most ``max_versions``.

    With ``trained`` (bool per identifier), only versions whose embedding row was trained vote — an untrained row tells
    the model nothing about its task, and a crowd of such blind rows outvotes the informed ones. If no version of the
    task is trained, the identity version alone makes a blind attempt."""
    v0 = int(d.vstart[t])
    vids = np.arange(v0, v0 + int(d.vcount[t]))
    if trained is not None:
        hit = vids[trained[vids]]
        vids = hit if len(hit) else vids[:1]
    return vids if max_versions is None else vids[:max_versions]


def eval_batches(
    ds: Dataset,
    split: str,
    bucket: int,
    batch_rows: int,
    max_versions: int | None = None,
    tasks: set[str] | None = None,
    trained: np.ndarray | None = None,
) -> Iterator[dict]:
    """Untranslated rows of ``split`` in ``bucket``: every eval pair x the task's voting versions (``eval_versions``),
    ordered by task, pair index, version; only tasks in ``tasks`` when given (batches stay full).
    Yields nothing if the bucket has no such split."""
    if not ds.has_split(bucket, split):
        return
    d = ds.split_data(bucket, split)
    ex_l, vid_l, keys = [], [], []
    for t, task_id in enumerate(d.tasks):
        if tasks is not None and task_id not in tasks:
            continue
        vids = eval_versions(d, t, max_versions, trained).tolist()
        for e in range(int(d.ex_start[t]), int(d.ex_start[t + 1])):
            for v in vids:
                ex_l.append(e)
                vid_l.append(v)
                keys.append((task_id, int(d.ex_index[e]), v))
    ex_a, vid_a = np.asarray(ex_l, dtype=np.int64), np.asarray(vid_l, dtype=np.int64)
    for a in range(0, len(ex_a), batch_rows):
        ex, vid = ex_a[a : a + batch_rows], vid_a[a : a + batch_rows]
        batch = _rows(ds, d, ex, vid, np.zeros((len(ex), 2), dtype=np.int64))
        if d.sub is None:
            del batch["subgoals"], batch["subgoal_mask"]
        batch["keys"] = keys[a : a + batch_rows]
        yield batch
