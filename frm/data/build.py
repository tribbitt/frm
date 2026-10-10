"""Build a dataset directory from ARC sources, composite JSON and ingested external pools.

    python -m frm.data.build --raw data/raw --sources arc1 [--composite c.json ...] \
        [--external SOURCE=data/raw/external/POOL.npz ...] --out data/built/NAME

Splits (DESIGN.md §3.4). The dev rule: the ``--dev_n`` tasks of the dev pool (``--dev_subset``, default ARC-AGI-1
evaluation) with >= 3 demos, ordered by the md5 hex digest of the task id (tinyv2.2 / tinyabstractor's development
tasks). ``arc1``: their test pairs -> dev, the other evaluation tasks' test pairs -> test, every training/concept test
pair -> train. ``arc2``/``arc2-kaggle``: training2 tasks with a dev id -> their test pairs -> dev; evaluation2 test pairs
-> test (``arc2``); with ``--dev_subset evaluation2`` the dev tasks are evaluation2 tasks instead. Demos of every
task are training data. External pools (``scripts/ingest_external.py``) are training pairs only; generator pools
(RE-ARC, synth) are split into per-bucket sub-tasks ``{source}:{id}@{S}``, other tasks stay whole.

Content filter: ARC/ConceptARC training tasks and composite pairs are checked against the build's held-out set (dev
and test tasks' test pairs and informative test outputs, up to dihedral transforms and colour relabelling,
``frm.data.heldout``); a hit drops the ARC task whole, or the composite pair (and its task below 3 pairs).
External pools were filtered at ingest against a superset: the build checks that their ``heldout_ids`` cover every
dev and test task.

Layout of ``out/`` (augmentation and translation happen in the loader, so each base grid is stored once):

- ``meta.json``, ``identifiers.json`` (id -> ``"{task}|||t{tid}|||{perm}"``, id 0 = ``"<blank>"``),
  ``versions_tid.npy`` uint8 [num_ids], ``versions_perm.npy`` uint8 [num_ids, 10],
  ``test_tasks.json`` = {split: {task_id: {"train": demos, "test": [{"input", "output" | None}]}}} for eval splits.
- ``b{S}/{split}/``: ``tasks.json`` (task ids), ``task_vstart``/``task_vcount`` int32 [T] (identifier range of the
  task's versions, identity first), ``task_source`` uint8 [T] (``SOURCE_CODES``), ``task_ex_start`` int32 [T+1],
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
from frm.data.augment import SEP, Version, make_versions
from frm.data.heldout import HeldOut
from frm.data.sizerule import MAX_SIDE, choose_bucket, fits, smallest_bucket

P = 16
VOCAB = 12
M = 3
SOURCE_CODES = {"arc": 0, "composite": 1, "barc": 2, "rearc": 3, "synth": 4}
SPLIT_ORDER = ("train", "dev", "dev_composite", "test")

# source -> (subsets whose test pairs are training data except dev tasks, subsets whose test pairs are dev/test)
ARC_SOURCES = {
    "arc1": (("training", "concept"), ("evaluation",)),
    "arc2": (("training2", "concept"), ("evaluation2",)),
    "arc2-kaggle": (("training2", "concept"), ()),
}
DEV_SUBSET = "evaluation"  # the dev rule's pool: ARC-AGI-1 evaluation
DEV_MIN_DEMOS = 3
MIN_PAIRS = 3
GENERATOR_SOURCES = ("rearc", "synth")  # external tasks that are instances of a generator: split by bucket


@dataclass
class Record:
    """One base task as it enters the dataset."""

    task_id: str
    source: str  # a key of SOURCE_CODES
    original: Task  # as loaded (test outputs possibly unknown)
    train: list[dict]  # labelled training pairs (demos, plus test pairs when they are training data)
    eval_split: str | None = None
    eval_pairs: list[dict] = field(default_factory=list)
    external: bool = False  # from an ingested pool (--external)
    bucket: int = 0
    vstart: int = 0
    vcount: int = 0


def dev_ids(raw: str, n: int, subset: str = DEV_SUBSET) -> list[str]:
    """The ``n`` tasks of ``subset`` (default ARC-AGI-1 evaluation) with >= 3 demos with the smallest md5(task id) hex
    digests, sorted."""
    with open(os.path.join(raw, f"arc-agi_{subset}_challenges.json")) as f:
        tasks = json.load(f)
    ok = sorted((k for k, t in tasks.items() if len(t["train"]) >= DEV_MIN_DEMOS),
                key=lambda k: hashlib.md5(k.encode()).hexdigest())
    return sorted(ok[:n])


def _check_grid(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    a = np.asarray(grid, dtype=np.int64)
    if a.ndim != 2 or not (1 <= a.shape[0] <= MAX_SIDE and 1 <= a.shape[1] <= MAX_SIDE) or a.min() < 0 or a.max() > 9:
        raise ValueError(f"invalid ARC grid of shape {a.shape}")
    return a.astype(np.uint8)


def arc_records(tasks: dict[str, Task], role: str, dev: set[str]) -> list[Record]:
    """role "train": test pairs are training data unless the task is a dev task; role "test": test pairs are eval
    pairs, of `dev` for dev tasks and of `test` otherwise."""
    recs = []
    for tid, t in tasks.items():
        if tid in dev:
            recs.append(Record(tid, "arc", t, t["train"], "dev", t["test"]))
        elif role == "test":
            recs.append(Record(tid, "arc", t, t["train"], "test", t["test"]))
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
    """Smallest bucket for demos (with their sub-goals), held-out inputs and the size rule; never held-out targets."""
    S = choose_bucket({"train": rec.train, "test": [{"input": ex["input"]} for ex in rec.eval_pairs]})
    demo_sub = [g for ex in rec.train for g in ex.get("subgoals", [])]
    if demo_sub:
        S = max(S, smallest_bucket(max(max(len(g), len(g[0])) for g in demo_sub)))
    return S


def bucket_groups(pairs: list[dict]) -> list[tuple[int, list[dict]]]:
    """A generator task's pairs grouped by the smallest bucket fitting each pair, ascending, in their original order
    inside a group. A group of fewer than MIN_PAIRS pairs merges into the next larger group (the largest into the one
    below it); a group's bucket is the largest of its pairs'."""
    by: dict[int, list[dict]] = {}
    for ex in pairs:
        side = max(*np.shape(ex["input"]), *np.shape(ex["output"]))
        by.setdefault(smallest_bucket(side), []).append(ex)
    groups: list[tuple[int, list[dict]]] = []
    carry: list[dict] = []
    for S in sorted(by):
        carry = carry + by[S]
        if len(carry) >= MIN_PAIRS:
            groups.append((S, carry))
            carry = []
    if carry:
        if groups:
            groups[-1] = (max(by), groups[-1][1] + carry)
        else:
            groups.append((max(by), carry))
    return groups


def external_records(source: str, path: str, need: set[str]) -> list[Record]:
    """An ingested pool (``scripts/ingest_external.py``): every pair a training pair. ``need``: the build's dev, test
    and excluded task ids, which the pool's ``heldout_ids`` must cover. Generator tasks (``GENERATOR_SOURCES``: many
    independent instances of one rule) are split into per-bucket sub-tasks ``{source}:{id}@{S}`` (``bucket_groups``),
    so small instances do not pay for the generator's largest grid; other tasks stay whole."""
    if source not in SOURCE_CODES:
        raise ValueError(f"--external {source}={path}: source must be one of {list(SOURCE_CODES)}")
    z = np.load(path, allow_pickle=False)
    if str(z["source"]) != source:
        raise ValueError(f"--external {source}={path}: the file was ingested as source {str(z['source'])!r}")
    missing = sorted(need - {str(x) for x in z["heldout_ids"]})
    if missing:
        raise ValueError(f"--external {source}={path} was not filtered against {len(missing)} dev/test/excluded tasks "
                         f"of this build (e.g. {missing[:5]}): re-run scripts/ingest_external.py against them")
    cells, off, hw, pair, pool_off = z["data"].view(np.uint8), z["off"], z["hw"], z["pair"], z["pool_off"]

    def grid(g: int) -> np.ndarray:
        return cells[off[g] : off[g + 1]].reshape(int(hw[g, 0]), int(hw[g, 1]))

    recs = []
    for t, tid in enumerate(str(x) for x in z["ids"]):
        pairs = [{"input": grid(pair[p, 0]), "output": grid(pair[p, 1])} for p in range(pool_off[t], pool_off[t + 1])]
        name = tid if source == "arc" else f"{source}:{tid}"
        if source in GENERATOR_SOURCES:
            for S, group in bucket_groups(pairs):
                recs.append(Record(f"{name}@{S}", source, {"train": group, "test": []}, group, external=True))
        else:
            recs.append(Record(name, source, {"train": pairs, "test": []}, pairs, external=True))
    return recs


def _contaminated(held: HeldOut, ex: dict) -> bool:
    a, b = np.asarray(ex["input"], np.uint8), np.asarray(ex["output"], np.uint8)
    return held.hit(a, b) is not None or any(held.grid_hit(np.asarray(g, np.uint8)) for g in ex.get("subgoals", []))


def content_filter(recs: list[Record]) -> tuple[list[Record], dict]:
    """Drop ARC/ConceptARC training tasks with any pair, and composite pairs (and their tasks below MIN_PAIRS), that
    match a dev or test task's test pair or informative test output (``frm.data.heldout``, outputs only: test
    inputs are visible at inference; demos of dev/test tasks are training data by design)."""
    held = HeldOut({r.task_id: {"train": [], "test": r.eval_pairs} for r in recs
                    if r.eval_split in ("dev", "test") and r.eval_pairs}, demos=False, inputs=False)
    kept: list[Record] = []
    info = {"arc_tasks_checked": 0, "composite_pairs_checked": 0, "arc_tasks_dropped": [],
            "composite_pairs_dropped": 0, "composite_tasks_dropped": []}
    for r in recs:
        if r.source == "arc" and r.eval_split is None and not r.external:
            info["arc_tasks_checked"] += 1
            if any(_contaminated(held, ex) for ex in r.train):
                info["arc_tasks_dropped"].append(r.task_id)
                continue
        elif r.source == "composite":
            info["composite_pairs_checked"] += len(r.train) + len(r.eval_pairs)
            train = [ex for ex in r.train if not _contaminated(held, ex)]
            held_pairs = [ex for ex in r.eval_pairs if not _contaminated(held, ex)]
            n_drop = len(r.train) + len(r.eval_pairs) - len(train) - len(held_pairs)
            if n_drop:
                info["composite_pairs_dropped"] += n_drop
                if not train or len(train) + len(held_pairs) < MIN_PAIRS:
                    info["composite_tasks_dropped"].append(r.task_id)
                    continue
                r = Record(r.task_id, r.source, {"train": train, "test": held_pairs}, train, r.eval_split, held_pairs)
        kept.append(r)
    return kept, info


def collect(args: argparse.Namespace) -> tuple[list[Record], list[str], dict]:
    """Records of every source, the dev rule's task ids and the content filter's report."""
    recs: list[Record] = []
    dev = dev_ids(args.raw, args.dev_n, args.dev_subset) if any(s in ARC_SOURCES for s in args.sources) else []
    for src in args.sources:
        if src == "kaggle":
            if not args.challenges:
                raise SystemExit("--sources kaggle needs --challenges")
            recs += arc_records(load_tasks(args.challenges, args.solutions), "test", set())
            continue
        train_subsets, test_subsets = ARC_SOURCES[src]
        for sub in train_subsets:
            recs += arc_records(load_subset(args.raw, sub), "train", set(dev))
        for sub in test_subsets:  # the dev pool's tasks give dev pairs; every other held-out task gives test pairs
            recs += arc_records(load_subset(args.raw, sub), "test", set(dev) if sub == args.dev_subset else set())
    for path in args.composite:
        recs += composite_records(path)
    recs, info = content_filter(recs)
    exclude: set[str] = set()
    if args.exclude_ids:
        with open(args.exclude_ids) as f:
            exclude = set(json.load(f))
    # external pools must have been filtered against every dev/test task and every excluded task
    need = {r.task_id for r in recs if r.eval_split in ("dev", "test") and r.eval_pairs} | exclude
    for source, path in args.external:
        recs += external_records(source, path, need)
    # A training task dropped by the content filter is a copy of a held-out task's rule; its RE-ARC generator teaches the
    # same rule with fresh instances that no pair-level filter can catch, so it goes too.
    copied = set(info["arc_tasks_dropped"])
    info["generator_tasks_dropped"] = sorted(r.task_id for r in recs if r.source in GENERATOR_SOURCES and _base_id(r) in copied)
    recs = [r for r in recs if not (r.source in GENERATOR_SOURCES and _base_id(r) in copied)]
    seen: set[str] = set()
    for r in recs:
        if r.task_id in seen:
            raise SystemExit(f"task id {r.task_id} appears twice (overlapping sources?)")
        seen.add(r.task_id)
    hit = sorted(r.task_id for r in recs if r.task_id in exclude or _base_id(r) in exclude)
    if hit:
        raise ValueError(f"--exclude_ids {args.exclude_ids}: {len(hit)} excluded tasks would enter the build "
                         f"(e.g. {hit[:5]})")
    return recs, dev, info


def _base_id(r: Record) -> str:
    """The task id without the build's ``{source}:`` prefix and generator ``@{S}`` suffix."""
    tid = r.task_id.removeprefix(f"{r.source}:")
    return tid.rsplit("@", 1)[0] if r.source in GENERATOR_SOURCES else tid


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
            if not fits(max(g.shape), S):
                if split == "train":
                    raise AssertionError(f"train sub-goal of {r.task_id} overflows its bucket {S}")
                overflow.append([split, r.task_id, int(i), *g.shape, S])
                g = g[:S, :S]
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


def trm_versions(path: str) -> dict[str, list[Version]]:
    """Version table of a TRM dataset's ``identifiers.json`` (``"<blank>"``, then per task its plain id = identity,
    followed by ``"task|||t{tid}|||{perm}"``; frm's names use the same dihedral numbering and colour-permutation
    semantics, ``augment.Version``). Reusing it lets an imported TRM checkpoint keep every trained embedding row."""
    with open(path) as f:
        names = json.load(f)
    table: dict[str, list[Version]] = {}
    for name in names[1:]:
        if SEP in name:
            task_id, v = Version.parse(name)
        else:
            task_id, v = name, Version.identity()
        table.setdefault(task_id, []).append(v)
    for task_id, vs in table.items():
        if vs[0] != Version.identity():
            raise SystemExit(f"{path}: the first version of {task_id} is not the identity")
    return table


def build(args: argparse.Namespace) -> dict:
    recs, dev, content = collect(args)
    identifiers = ["<blank>"]
    tids, perms = [0], [list(range(10))]
    fixed = trm_versions(args.versions_from) if args.versions_from else {}
    for r in recs:
        lift = not args.min_bucket_sources or r.source in args.min_bucket_sources
        r.bucket = max(_bucket(r), args.min_bucket) if lift else _bucket(r)
        n = args.n_aug_external if r.external else args.n_aug_composite if r.source == "composite" else args.n_aug
        fixed_r = fixed.get(r.task_id) if r.source == "arc" and not r.external else None
        versions = fixed_r or make_versions(r.original, n, _task_seed(args.seed, r.task_id))
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
        "n_aug_external": args.n_aug_external,
        "dev_n": args.dev_n,
        "dev_subset": args.dev_subset,
        "dev_ids": dev,
        "seed": args.seed,
        "sources": list(args.sources),
        "composite": list(args.composite),
        "external": [[s, p] for s, p in args.external],
        "exclude_ids": args.exclude_ids,
        "versions_from": args.versions_from,
        "min_bucket": args.min_bucket,
        "min_bucket_sources": args.min_bucket_sources,
        "content_filter": content,
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
    cf = meta["content_filter"]
    lines.append(f"content filter: arc tasks dropped {len(cf['arc_tasks_dropped'])}/{cf['arc_tasks_checked']} "
                 f"{cf['arc_tasks_dropped']}, composite pairs dropped {cf['composite_pairs_dropped']}/"
                 f"{cf['composite_pairs_checked']} (tasks {cf['composite_tasks_dropped']}), generator sub-tasks of "
                 f"dropped tasks {len(cf.get('generator_tasks_dropped', []))}")
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
    ap.add_argument("--external", nargs="+", action="extend", default=[], metavar="SOURCE=PATH",
                    help=f"ingested pool npz (scripts/ingest_external.py); SOURCE in {list(SOURCE_CODES)}")
    ap.add_argument("--n_aug_external", type=int, default=8)
    ap.add_argument("--dev_n", type=int, default=100,
                    help="dev = this many tasks of --dev_subset with >= 3 demos, by md5 of the task id")
    ap.add_argument("--dev_subset", default=DEV_SUBSET, choices=("evaluation", "evaluation2"),
                    help="the dev pool: ARC-AGI-1 evaluation (default) or ARC-AGI-2 evaluation, for models whose "
                    "pretraining saw ARC-AGI-2 training test outputs (an imported arc_v2_public TRM); the rest of the "
                    "pool's tasks give test pairs")
    ap.add_argument("--exclude_ids", help="JSON list of task ids that must not enter the build through any source "
                    "(e.g. data/raw/adev_ids.json, scripts/make_adapt_dev.py): the build fails if one would")
    ap.add_argument("--versions_from", help="a TRM dataset's identifiers.json: ARC/ConceptARC tasks listed there get "
                    "exactly its versions (in its order), so an imported TRM checkpoint's embedding rows carry over "
                    "(scripts/import_trm.py); other tasks get --n_aug random versions")
    ap.add_argument("--min_bucket", type=int, default=0,
                    help="place tasks in a bucket of at least this size (30 = TRM's single 30x30 canvas)")
    ap.add_argument("--min_bucket_sources", nargs="*", default=[], choices=list(SOURCE_CODES),
                    help="sources --min_bucket applies to (default: all); e.g. arc keeps an imported TRM on its "
                    "native canvas while the new sources stay bucketed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not args.sources and not args.composite and not args.external:
        ap.error("nothing to build: give --sources, --composite and/or --external")
    ext = []
    for spec in args.external:
        source, sep, path = spec.partition("=")
        if not sep or source not in SOURCE_CODES or not path:
            ap.error(f"--external {spec}: expected SOURCE=PATH with SOURCE in {list(SOURCE_CODES)}")
        ext.append((source, path))
    args.external = ext
    return args


def main(argv: list[str] | None = None) -> None:
    meta = build(parse_args(argv))
    print(format_counts(meta))


if __name__ == "__main__":
    main()
