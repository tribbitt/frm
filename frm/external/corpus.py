"""Build the RE-ARC and ARC-AGI-2 extra-training FlatPools: ``{out}/rearc.npz``, ``{out}/a2_new.npz`` and
``{out}/corpus_report.json``.

    python -m frm.external.corpus [--raw data/raw] [--rearc data/external/re_arc/tasks] [--out data/pools] \
        [--workers 8]

Regenerating the pools ``scripts/ingest_external.py`` consumes, in order (from the repository root):

    python scripts/fetch_data.py                                  # data/raw
    git clone --depth 1 https://github.com/michaelhodel/re-arc data/external/re-arc
    unzip -q data/external/re-arc/re_arc.zip -d data/external     # data/external/re_arc/tasks
    python -m frm.external.corpus                              # data/pools/{rearc,a2_new}.npz
    (BARC: the commands in frm/external/barc/__init__.py)      # data/pools/barc.npz
    python scripts/ingest_external.py --src data/pools --pools a2_new
    python scripts/ingest_external.py --src data/pools --pools barc rearc \
        --heldout_pools data/raw/external/a2_new.npz

RE-ARC (MIT): ``re_arc.zip`` of github.com/michaelhodel/re-arc, sha256
4a7c309499f450eb47c2117fa06f3c97fe741cf6e7912579c0b5b8cc759e9ac4 (50,913,541 bytes); ``tasks/`` holds one JSON list
of {"input", "output"} per ARC-AGI-1 training task (1,000 verified instances each).

Held out (``heldout``): every demonstration and test pair of ARC-AGI-1 evaluation and ARC-AGI-2 evaluation, both
versions of the ids the two share. Contamination is ``frm.data.heldout.HeldOut.hit``.
    rearc    per generator, in sorted file order: pairs with sides in 1..30, exact duplicates once, contaminated
             instances dropped.
    a2_new   ARC-AGI-2 training minus ARC-AGI-1 training ids minus held-out ids, sorted by id: demonstrations then
             test pairs, sides in 1..30, exact duplicates once; a task with any contaminated pair is dropped whole,
             and a task needs >= 2 pairs.
The output is a pure function of the input files.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from frm.data.arc import load_subset
from frm.data.heldout import HeldOut, hits
from frm.external.flatpool import Pair, build_flat, dedup, fits

HELDOUT_SUBSETS = ("evaluation", "evaluation2")
FIXED_MIN_PAIRS = 2


def _g(x) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.int8))


def arc_tasks(raw: str | Path, subset: str) -> list[tuple[str, list[Pair], list[Pair]]]:
    """(id, demonstration pairs, test pairs) of an ARC subset in ``raw``, sorted by id."""
    out = []
    for tid, t in sorted(load_subset(str(raw), subset).items()):
        if any(ex["output"] is None for ex in t["test"]):
            raise SystemExit(f"{subset} task {tid} has no test solution in {raw}")
        out.append((tid, [(_g(e["input"]), _g(e["output"])) for e in t["train"]],
                    [(_g(e["input"]), _g(e["output"])) for e in t["test"]]))
    return out


def heldout(raw: str | Path) -> HeldOut:
    """Every pair of the held-out subsets; an id in two subsets contributes both versions."""
    parts = []
    for sub in HELDOUT_SUBSETS:
        tasks = load_subset(str(raw), sub)
        if any(ex["output"] is None for t in tasks.values() for ex in t["test"]):
            raise SystemExit(f"{sub}: held-out tasks without test solutions in {raw}: cannot filter against them")
        parts.append(HeldOut(tasks, demos=True, inputs=True))
    held, *rest = parts
    for h in rest:
        held.ids |= h.ids
        held.pairs |= h.pairs
        held.grids |= h.grids
    return held


def fixed_pool(tasks: list[tuple[str, list[Pair], list[Pair]]], held: HeldOut,
               workers: int) -> tuple[list[tuple[str, list[Pair]]], list[dict]]:
    """Kept (id, pairs) and dropped [{tid, hits}]: any contaminated pair drops the task, >= FIXED_MIN_PAIRS kept."""
    groups = [dedup([p for p in train + test if fits(*p)]) for _, train, test in tasks]
    res = hits(groups, held, workers)
    keep, dropped = [], []
    for (tid, _, _), prs, h in zip(tasks, groups, res, strict=True):
        if any(h):
            dropped.append({"tid": tid, "hits": [x for x in h if x]})
        elif len(prs) >= FIXED_MIN_PAIRS:
            keep.append((tid, prs))
    return keep, dropped


def a2_new_tasks(raw: str | Path, held_ids: set[str]) -> list[tuple[str, list[Pair], list[Pair]]]:
    a1 = set(load_subset(str(raw), "training"))
    return [t for t in arc_tasks(raw, "training2") if t[0] not in a1 and t[0] not in held_ids]


def _rearc_file(f: str) -> tuple[str, list[Pair]]:
    prs = [(_g(p["input"]), _g(p["output"])) for p in json.loads(Path(f).read_text())]
    return Path(f).stem, dedup([p for p in prs if fits(*p)])


def rearc_files(tasks_dir: str | Path) -> list[Path]:
    files = sorted(Path(tasks_dir).glob("*.json"))
    if not files:
        raise SystemExit(f"no RE-ARC tasks under {tasks_dir}: unzip re_arc.zip (this module's docstring)")
    return files


def rearc_pool(files: list[Path], held: HeldOut, workers: int) -> tuple[list[tuple[str, list[Pair]]], dict]:
    """Kept (generator id, pairs) per file, and the drop report."""
    if workers > 1:
        with Pool(workers) as pool:
            tasks = pool.map(_rearc_file, [str(f) for f in files], chunksize=4)
    else:
        tasks = [_rearc_file(str(f)) for f in files]
    res = hits([prs for _, prs in tasks], held, workers)
    keep, n_drop, where = [], {"pair": 0, "grid": 0}, {}
    for (tid, prs), h in zip(tasks, res, strict=True):
        for x in h:
            if x:
                n_drop[x] += 1
                where[tid] = where.get(tid, 0) + 1
        keep.append((tid, [p for p, x in zip(prs, h, strict=True) if x is None]))
    return keep, {"generators": len(keep), "pairs": sum(len(p) for _, p in keep), "instances_dropped": n_drop,
                  "by_generator": where}


def build(raw: str | Path, rearc: str | Path, out: str | Path, workers: int = 8) -> dict:
    out = Path(out)
    files = rearc_files(rearc)
    held = heldout(raw)
    report: dict = {"held_out": {"subsets": list(HELDOUT_SUBSETS), "task_ids": len(held.ids), "pairs": len(held.pairs),
                                 "informative_grids": len(held.grids)}}
    tasks = a2_new_tasks(raw, held.ids)
    keep, dropped = fixed_pool(tasks, held, workers)
    build_flat(keep).save(out / "a2_new.npz")
    report["a2_new"] = {"tasks_in": len(tasks), "tasks_kept": len(keep), "dropped": dropped,
                        "pairs": sum(len(p) for _, p in keep)}
    print(f"a2_new: {len(keep)} of {len(tasks)} tasks kept, {report['a2_new']['pairs']} pairs", flush=True)
    keep, report["rearc"] = rearc_pool(files, held, workers)
    build_flat(keep).save(out / "rearc.npz")
    print(f"rearc: {report['rearc']['pairs']} pairs, dropped {report['rearc']['instances_dropped']}", flush=True)
    (out / "corpus_report.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw", help="arc-agi_{subset}_{challenges,solutions}.json (fetch_data.py)")
    ap.add_argument("--rearc", default="data/external/re_arc/tasks", help="RE-ARC tasks/ directory")
    ap.add_argument("--out", default="data/pools")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    build(a.raw, a.rearc, a.out, a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
