"""Build a dataset directory from ARC sources and composite JSON.

    python -m frm.data.build --raw data/raw --sources arc1 [--composite c.json ...] --out data/built/NAME

Layout of ``out/`` (augmentation and translation happen in the loader, so each base grid is stored once):

- ``meta.json``, ``identifiers.json`` (id -> ``"{task}|||t{tid}|||{perm}"``, id 0 = ``"<blank>"``),
  ``versions_tid.npy`` uint8 [num_ids], ``versions_perm.npy`` uint8 [num_ids, 10],
  ``test_tasks.json`` = {split: {task_id: {"train": demos, "test": [{"input", "output" | None}]}}} for eval splits.
- ``b{S}/{split}/``: ``tasks.json`` (task ids), ``task_vstart``/``task_vcount`` int32 [T] (identifier range of the
  task's versions, identity first), ``task_source`` uint8 [T] (0 arc, 1 composite), ``task_ex_start`` int32 [T+1],
  ``ex_index`` int16 [E] (pair index inside the task), ``inp``/``out`` uint8 [E, S, S] raw colours at top-left,
  ``inp_hw``/``out_hw`` uint8 [E, 2] (``out_hw`` = 0 when the output is unknown), and when any pair has sub-goals
  ``sub`` uint8 [E, M, S, S], ``sub_hw`` uint8 [E, M, 2] (0 = absent).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from frm.data.arc import Task, load_subset, load_tasks
from frm.data.augment import make_versions
from frm.data.sizerule import MAX_SIDE, choose_bucket, fits, smallest_bucket

P = 16
VOCAB = 12
M = 3
SOURCE_CODES = {"arc": 0, "composite": 1}
SPLIT_ORDER = ("train", "dev", "dev_composite", "test")

# source -> (subsets whose test pairs are training data except the dev hold-out, subsets whose test pairs are `test`)
ARC_SOURCES = {
    "arc1": (("training", "concept"), ("evaluation",)),
    "arc2": (("training2", "concept"), ("evaluation2",)),
    "arc2-kaggle": (("training2", "concept"), ()),
}


@dataclass
class Record:
    """One base task as it enters the dataset."""

    task_id: str
    source: str  # "arc" | "composite"
    original: Task  # as loaded (test outputs possibly unknown)
    train: list[dict]  # labelled training pairs (demos, plus test pairs when they are training data)
    eval_split: str | None = None
    eval_pairs: list[dict] = field(default_factory=list)
    bucket: int = 0
    vstart: int = 0
    vcount: int = 0


def is_dev(task_id: str, dev_frac: float) -> bool:
    h = int.from_bytes(hashlib.sha256(task_id.encode()).digest()[:8], "big")
    return h < dev_frac * 2**64


def _check_grid(grid: list[list[int]]) -> np.ndarray:
    a = np.asarray(grid, dtype=np.int64)
    if a.ndim != 2 or not (1 <= a.shape[0] <= MAX_SIDE and 1 <= a.shape[1] <= MAX_SIDE) or a.min() < 0 or a.max() > 9:
        raise ValueError(f"invalid ARC grid of shape {a.shape}")
    return a.astype(np.uint8)


def arc_records(tasks: dict[str, Task], role: str, dev_frac: float) -> list[Record]:
    """role "train": test pairs are training data unless the task is in the dev hold-out; role "test": eval pairs."""
    recs = []
    for tid, t in tasks.items():
        if role == "test":
            recs.append(Record(tid, "arc", t, t["train"], "test", t["test"]))
        elif is_dev(tid, dev_frac):
            recs.append(Record(tid, "arc", t, t["train"], "dev", t["test"]))
        else:
            recs.append(Record(tid, "arc", t, t["train"] + t["test"]))
    return recs


def composite_records(path: str) -> list[Record]:
    with open(path) as f:
        tasks = json.load(f)
    recs = []
    for t in tasks:
        task: Task = {"train": t["train"], "test": t["test"]}
        for ex in t["train"] + t["test"]:
            if len(ex.get("subgoals", [])) > M:
                raise ValueError(f"composite {t['id']}: more than {M} sub-goals")
        recs.append(Record(f"comp:{t['id']}", "composite", task, t["train"], "dev_composite", t["test"]))
    return recs


def _bucket(rec: Record) -> int:
    if rec.source == "composite":
        grids = [g for ex in rec.train + rec.eval_pairs for g in (ex["input"], ex["output"], *ex.get("subgoals", []))]
        return smallest_bucket(max(max(len(g), len(g[0])) for g in grids))
    return choose_bucket({"train": rec.train, "test": [{"input": ex["input"]} for ex in rec.eval_pairs]})


def collect(args: argparse.Namespace) -> list[Record]:
    recs: list[Record] = []
    for src in args.sources:
        if src == "kaggle":
            if not args.challenges:
                raise SystemExit("--sources kaggle needs --challenges")
            recs += arc_records(load_tasks(args.challenges, args.solutions), "test", args.dev_frac)
            continue
        train_subsets, test_subsets = ARC_SOURCES[src]
        for sub in train_subsets:
            recs += arc_records(load_subset(args.raw, sub), "train", args.dev_frac)
        for sub in test_subsets:
            recs += arc_records(load_subset(args.raw, sub), "test", args.dev_frac)
    for path in args.composite:
        recs += composite_records(path)
    seen: set[str] = set()
    for r in recs:
        if r.task_id in seen:
            raise SystemExit(f"task id {r.task_id} appears twice (overlapping sources?)")
        seen.add(r.task_id)
    return recs


def _task_seed(seed: int, task_id: str) -> np.random.Generator:
    return np.random.default_rng([seed, int.from_bytes(hashlib.sha256(task_id.encode()).digest()[:8], "big")])


def _write_split(path: str, S: int, recs: list[Record], pairs_of, overflow: list, split: str) -> None:
    os.makedirs(path, exist_ok=True)
    pairs = [(r, i, ex) for r in recs for i, ex in pairs_of(r)]
    E = len(pairs)
    inp = np.zeros((E, S, S), np.uint8)
    out = np.zeros((E, S, S), np.uint8)
    inp_hw = np.zeros((E, 2), np.uint8)
    out_hw = np.zeros((E, 2), np.uint8)
    has_sub = any(ex.get("subgoals") for _, _, ex in pairs)
    sub = np.zeros((E, M, S, S), np.uint8) if has_sub else None
    sub_hw = np.zeros((E, M, 2), np.uint8) if has_sub else None
    ex_index = np.zeros(E, np.int16)
    for e, (r, i, ex) in enumerate(pairs):
        ex_index[e] = i
        g = _check_grid(ex["input"])
        inp[e, : g.shape[0], : g.shape[1]] = g
        inp_hw[e] = g.shape
        if ex.get("output") is not None:
            g = _check_grid(ex["output"])
            if not fits(max(g.shape), S):
                if split == "train":
                    raise AssertionError(f"train grid of {r.task_id} overflows its bucket {S}")
                overflow.append([split, r.task_id, int(i), *g.shape, S])
                g = g[:S, :S]  # canvas cannot hold it: stored as a full-canvas crop without EOS (always wrong)
            out[e, : g.shape[0], : g.shape[1]] = g
            out_hw[e] = g.shape
        for m, sg in enumerate(ex.get("subgoals", [])):
            g = _check_grid(sg)
            sub[e, m, : g.shape[0], : g.shape[1]] = g
            sub_hw[e, m] = g.shape
    counts = np.array([len(pairs_of(r)) for r in recs], dtype=np.int32)
    arrays = {
        "task_vstart": np.array([r.vstart for r in recs], np.int32),
        "task_vcount": np.array([r.vcount for r in recs], np.int32),
        "task_source": np.array([SOURCE_CODES[r.source] for r in recs], np.uint8),
        "task_ex_start": np.concatenate([[0], np.cumsum(counts)]).astype(np.int32),
        "ex_index": ex_index,
        "inp": inp,
        "inp_hw": inp_hw,
        "out": out,
        "out_hw": out_hw,
    }
    if has_sub:
        arrays |= {"sub": sub, "sub_hw": sub_hw}
    for k, v in arrays.items():
        np.save(os.path.join(path, f"{k}.npy"), v)
    with open(os.path.join(path, "tasks.json"), "w") as f:
        json.dump([r.task_id for r in recs], f)


def build(args: argparse.Namespace) -> dict:
    recs = collect(args)
    identifiers = ["<blank>"]
    tids, perms = [0], [list(range(10))]
    for r in recs:
        r.bucket = _bucket(r)
        n = args.n_aug_composite if r.source == "composite" else args.n_aug
        versions = make_versions(r.original, n, _task_seed(args.seed, r.task_id))
        r.vstart, r.vcount = len(identifiers), len(versions)
        for v in versions:
            identifiers.append(v.name(r.task_id))
            tids.append(v.tid)
            perms.append(list(v.perm))

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "versions_tid.npy"), np.array(tids, np.uint8))
    np.save(os.path.join(args.out, "versions_perm.npy"), np.array(perms, np.uint8))
    with open(os.path.join(args.out, "identifiers.json"), "w") as f:
        json.dump(identifiers, f)

    def train_pairs(r: Record) -> list[tuple[int, dict]]:
        return list(enumerate(r.train))

    def eval_pairs(r: Record) -> list[tuple[int, dict]]:
        return list(enumerate(r.eval_pairs))

    buckets = sorted({r.bucket for r in recs})
    counts: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    overflow: list = []
    for S in buckets:
        in_bucket = [r for r in recs if r.bucket == S]
        for split in SPLIT_ORDER:
            if split == "train":
                srecs, pairs_of = [r for r in in_bucket if r.train], train_pairs
            else:
                srecs, pairs_of = [r for r in in_bucket if r.eval_split == split and r.eval_pairs], eval_pairs
            if not srecs:
                continue
            _write_split(os.path.join(args.out, f"b{S}", split), S, srecs, pairs_of, overflow, split)
            for r in srecs:
                c = counts.setdefault(split, {}).setdefault(str(S), {}).setdefault(
                    r.source, {"tasks": 0, "examples": 0, "rows": 0}
                )
                c["tasks"] += 1
                c["examples"] += len(pairs_of(r))
                c["rows"] += len(pairs_of(r)) * r.vcount

    test_tasks: dict[str, dict] = {}
    for r in recs:
        if r.eval_split and r.eval_pairs:
            test_tasks.setdefault(r.eval_split, {})[r.task_id] = {
                "train": [{"input": ex["input"], "output": ex["output"]} for ex in r.original["train"]],
                "test": [{"input": ex["input"], "output": ex.get("output")} for ex in r.eval_pairs],
            }
    with open(os.path.join(args.out, "test_tasks.json"), "w") as f:
        json.dump(test_tasks, f)

    meta = {
        "P": P,
        "vocab": VOCAB,
        "M": M,
        "pad": 0,
        "eos": 1,
        "ignore": -100,
        "buckets": buckets,
        "splits": [s for s in SPLIT_ORDER if s in counts],
        "num_ids": len(identifiers),
        "n_aug": args.n_aug,
        "n_aug_composite": args.n_aug_composite,
        "dev_frac": args.dev_frac,
        "seed": args.seed,
        "sources": list(args.sources),
        "composite": list(args.composite),
        "counts": counts,
        "overflow": overflow,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def format_counts(meta: dict) -> str:
    lines = [f"num_ids {meta['num_ids']}  buckets {meta['buckets']}  splits {meta['splits']}"]
    for split, by_bucket in meta["counts"].items():
        for S, by_source in sorted(by_bucket.items(), key=lambda kv: int(kv[0])):
            for src, c in by_source.items():
                lines.append(f"{split:14s} S={S:>2s} {src:9s} tasks {c['tasks']:5d}  examples {c['examples']:6d}  "
                             f"rows {c['rows']:8d}")
    for o in meta["overflow"]:
        lines.append(f"overflow: split {o[0]} task {o[1]} test {o[2]} output {o[3]}x{o[4]} > bucket {o[5]}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--sources", nargs="*", default=[], choices=[*ARC_SOURCES, "kaggle"])
    ap.add_argument("--challenges", help="kaggle source: challenges JSON")
    ap.add_argument("--solutions", help="kaggle source: solutions JSON (optional)")
    ap.add_argument("--composite", nargs="*", default=[], help="composite task JSON files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_aug", type=int, default=300)
    ap.add_argument("--n_aug_composite", type=int, default=20)
    ap.add_argument("--dev_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not args.sources and not args.composite:
        ap.error("nothing to build: give --sources and/or --composite")
    return args


def main(argv: list[str] | None = None) -> None:
    meta = build(parse_args(argv))
    print(format_counts(meta))


if __name__ == "__main__":
    main()
