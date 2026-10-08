"""Ingest external training pools (tinyabstractor FlatPool npz) with frm's own contamination filter.

    python scripts/ingest_external.py --src ../tinyabstractor/data/cache --raw data/raw --out data/raw/external \
        --pools barc rearc a2_new [synth] [--heldout_pools data/raw/external/a2_new.npz] --workers 8

Held-out = every ARC-AGI-1 evaluation and ARC-AGI-2 evaluation task in ``--raw``, demos and test pairs, inputs and
outputs (``frm.data.heldout``), plus every pair of every task of each ``--heldout_pools`` FlatPool (the a2_new
pool: the adaptation-dev candidates, ``scripts/make_adapt_dev.py``). Per pool:

- any task id that is a held-out id: abort;
- ``barc``, ``rearc``, ``synth`` (many instances of one rule): contaminated pairs are dropped; ``a2_new`` (fixed
  ARC tasks): a task with any contaminated pair is dropped whole;
- exact duplicate pairs inside a task are dropped; tasks left with fewer than 3 pairs are dropped.

Writes ``{out}/{pool}.npz`` (FlatPool layout: ``data`` int8, ``off`` int64 [G+1], ``hw`` int16 [G, 2], ``pair`` int32
[P, 2], ``pool_off`` int32 [T+1], ``ids`` [T]; plus ``heldout_ids`` = the sorted held-out ids it was filtered against
and ``source`` = the ``frm.data.build`` source code name) and ``{out}/{pool}_report.json``.
``synth_gap`` (designed by looking at ARC-AGI-1 evaluation) and ARC-GEN (contaminated) are never ingested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frm.data.arc import load_subset  # noqa: E402
from frm.data.heldout import HeldOut, hits  # noqa: E402

POOL_SOURCE = {"barc": "barc", "rearc": "rearc", "a2_new": "arc", "synth": "synth"}
WHOLE_TASK = {"a2_new"}
HELDOUT_SUBSETS = ("evaluation", "evaluation2")
MIN_PAIRS = 3
MAX_SIDE = 30


def load_heldout(raw: str, pools: list[str] = ()) -> HeldOut:
    tasks: dict = {}
    for sub in HELDOUT_SUBSETS:
        for tid, t in load_subset(raw, sub).items():
            if any(ex["output"] is None for ex in t["test"]):
                raise SystemExit(f"held-out task {tid} ({sub}) has no test solution in {raw}: cannot filter against it")
            tasks[tid] = t
    for path in pools:
        z = np.load(path, allow_pickle=False)
        cells, off, hw, pair, pool_off = z["data"].view(np.uint8), z["off"], z["hw"], z["pair"], z["pool_off"]

        def grid(g: int) -> np.ndarray:
            return cells[off[g] : off[g + 1]].reshape(int(hw[g, 0]), int(hw[g, 1]))

        for t, tid in enumerate(str(x) for x in z["ids"]):
            pairs = [{"input": grid(pair[p, 0]), "output": grid(pair[p, 1])}
                     for p in range(pool_off[t], pool_off[t + 1])]
            tasks[tid] = {"train": pairs, "test": []}
    return HeldOut(tasks, demos=True, inputs=True)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _save(path: str, arrays: dict[str, np.ndarray]) -> None:
    tmp = path[: -len(".npz")] + ".partial.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def ingest(pool: str, src: str, out_dir: str, held: HeldOut, workers: int, heldout_pools: list[str] = ()) -> dict:
    t0 = time.time()
    path = os.path.join(src, f"{pool}.npz")
    z = np.load(path, allow_pickle=False)
    data, off, hw, pair, pool_off = (z[k] for k in ("data", "off", "hw", "pair", "pool_off"))
    ids = [str(x) for x in z["ids"]]
    if len(pool_off) != len(ids) + 1 or len(off) != len(hw) + 1 or int(off[-1]) != len(data):
        raise SystemExit(f"{path}: inconsistent FlatPool layout")
    if hw.min() < 1 or hw.max() > MAX_SIDE or data.min() < 0 or data.max() > 9:
        raise SystemExit(f"{path}: grid sides outside 1..{MAX_SIDE} or colours outside 0..9")
    if not np.array_equal(np.diff(off), hw[:, 0].astype(np.int64) * hw[:, 1]):
        raise SystemExit(f"{path}: off does not match hw")
    overlap = sorted(set(ids) & held.ids)
    if overlap:
        raise SystemExit(f"{pool}: {len(overlap)} task ids are held-out ids, e.g. {overlap[:5]}")
    cells = data.view(np.uint8)

    def grid(g: int) -> np.ndarray:
        return cells[off[g] : off[g + 1]].reshape(int(hw[g, 0]), int(hw[g, 1]))

    groups = [[(grid(pair[p, 0]), grid(pair[p, 1])) for p in range(pool_off[t], pool_off[t + 1])]
              for t in range(len(ids))]
    res = hits(groups, held, workers)

    keep: list[int] = []
    new_pool_off = [0]
    ids_out: list[str] = []
    n_hit = {"pair": 0, "grid": 0}
    n_dup = 0
    dropped = {"contaminated": [], "too_few_pairs": []}
    pairs_with_dropped_tasks = 0
    by_task: dict[str, dict] = {}
    for t, tid in enumerate(ids):
        rec = {"pair": 0, "grid": 0, "duplicate": 0}
        for r in res[t]:
            if r is not None:
                rec[r] += 1
                n_hit[r] += 1
        task_pairs = range(int(pool_off[t]), int(pool_off[t + 1]))
        kept: list[int] = []
        if pool in WHOLE_TASK and (rec["pair"] or rec["grid"]):
            dropped["contaminated"].append(tid)
            rec["dropped"] = "contaminated"
        else:
            seen: set[bytes] = set()
            for p, r in zip(task_pairs, res[t]):
                if r is not None:
                    continue
                a, b = groups[t][p - int(pool_off[t])]
                key = bytes(a.shape) + a.tobytes() + bytes(b.shape) + b.tobytes()
                if key in seen:
                    rec["duplicate"] += 1
                    n_dup += 1
                    continue
                seen.add(key)
                kept.append(p)
            if len(kept) < MIN_PAIRS:
                dropped["too_few_pairs"].append(tid)
                rec["dropped"] = "too_few_pairs"
                kept = []
        if rec.get("dropped"):
            pairs_with_dropped_tasks += len(task_pairs) - rec["pair"] - rec["grid"] - rec["duplicate"]
        if any(rec.values()):
            by_task[tid] = rec
        if kept:
            keep += kept
            new_pool_off.append(len(keep))
            ids_out.append(tid)

    kp = np.asarray(keep, dtype=np.int64)
    gidx = pair[kp].reshape(-1).astype(np.int64)  # in, out, in, out, ...
    lens = np.diff(off)[gidx]
    new_off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    src_idx = np.repeat(off[gidx] - new_off[:-1], lens) + np.arange(int(new_off[-1]))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pool}.npz")
    _save(out_path, {
        "data": data[src_idx].astype(np.int8),
        "off": new_off,
        "hw": hw[gidx].astype(np.int16),
        "pair": np.arange(2 * len(kp), dtype=np.int32).reshape(-1, 2),
        "pool_off": np.asarray(new_pool_off, dtype=np.int32),
        "ids": np.array(ids_out, dtype=str),
        "heldout_ids": np.array(sorted(held.ids), dtype=str),
        "source": np.array(POOL_SOURCE[pool]),
    })
    report = {
        "pool": pool,
        "source": POOL_SOURCE[pool],
        "policy": "drop task" if pool in WHOLE_TASK else "drop pair",
        "input": os.path.abspath(path),
        "input_sha256": _sha256(path),
        "output": os.path.abspath(out_path),
        "heldout": {"tasks": len(held.ids), "pairs": len(held.pairs), "grids": len(held.grids),
                    "subsets": list(HELDOUT_SUBSETS),
                    "pools": [{"path": os.path.abspath(p), "sha256": _sha256(p)} for p in heldout_pools]},
        "tasks_in": len(ids),
        "pairs_in": int(pool_off[-1] - pool_off[0]),
        "tasks_out": len(ids_out),
        "pairs_out": len(keep),
        "contaminated_pairs": n_hit,
        "duplicate_pairs": n_dup,
        "tasks_dropped": {k: len(v) for k, v in dropped.items()},
        "pairs_dropped_with_tasks": pairs_with_dropped_tasks,
        "dropped_task_ids": dropped,
        "by_task": by_task,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, f"{pool}_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="../tinyabstractor/data/cache", help="directory of the input {pool}.npz files")
    ap.add_argument("--raw", default="data/raw", help="ARC raw directory (held-out evaluation sets)")
    ap.add_argument("--out", default="data/raw/external")
    ap.add_argument("--pools", nargs="+", default=["barc", "rearc", "a2_new"], choices=list(POOL_SOURCE))
    ap.add_argument("--heldout_pools", nargs="*", default=[],
                    help="FlatPool npz files whose every task is held out too (data/raw/external/a2_new.npz)")
    ap.add_argument("--workers", type=int, default=8)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> list[dict]:
    args = parse_args(argv)
    held = load_heldout(args.raw, args.heldout_pools)
    print(f"held-out: {len(held.ids)} tasks, {len(held.pairs)} pair digests, {len(held.grids)} grid digests")
    reports = []
    for pool in args.pools:
        r = ingest(pool, args.src, args.out, held, args.workers, args.heldout_pools)
        print(f"{pool}: tasks {r['tasks_in']} -> {r['tasks_out']}, pairs {r['pairs_in']} -> {r['pairs_out']}; "
              f"contaminated {r['contaminated_pairs']}, duplicates {r['duplicate_pairs']}, "
              f"tasks dropped {r['tasks_dropped']}; {r['elapsed_s']} s")
        reports.append(r)
    return reports


if __name__ == "__main__":
    main()
