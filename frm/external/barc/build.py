"""Build the BARC ARC-Heavy pool: ``{pools}/barc.npz`` (+ barc_val.npz, barc{,_val}.jsonl, barc_build.json,
barc_rule_hits.json) and ``{dir}/sources.jsonl``.

    python -m frm.external.barc.build STAGE [--dir data/external/barc] [--raw data/raw] [--pools data/pools] \
        [--workers 8]

Stages, in order (``frm/external/barc/__init__.py`` lists the whole sequence): parse (rows -> pairs, static
checks), execute (every stored pair re-run, sandboxed), filter (content filter, canonical forms, readability),
select (>= N_MIN pairs, one task per program, the shape mix), then ``frm.external.barc.audit``, then write
(audited tasks only -> pools, reports).  Every stage writes ``{dir}/work/`` and is resumable.

Per pair, in order: cells integers 0-9, sides 1-30; input != output; inputs distinct within a task; the program
passes `policy.check`, loads in the sandbox, and reproduces the stored output from the input under two numpy /
random states (twice, equal); no held-out hit (``frm.external.corpus.heldout``: the instance is dropped); not a
duplicate (canonical form, ``frm.data.heldout.canon``) of an earlier BARC pair or of any pair of DEDUP_POOLS.
Per task: >= N_MIN surviving pairs; one task per normalised program (``policy.normalised``); then same-shape tasks
are thinned by a fixed hash of their id until the pool's same-shape share of pairs is SAME_SHAPE.  Readability
(`structured`) is counted, not filtered: BARC inputs are written as ARC inputs, and denoising tasks are noise by
design.  Validation: a task whose normalised description hashes into 1 of VAL_MOD buckets goes to barc_val, so a
recurring description never sits on both sides; programs are deduplicated above.

DEDUP_POOLS are the pools this repository builds: rearc and a2_new (``{pools}``, from ``frm.external.corpus``),
ARC-AGI-1 training and ConceptARC (from ``{raw}``, with a2_new's rule: a task with a held-out hit is dropped, >= 2
pairs).  The published pool was also deduplicated against two synthetic pools that are not part of this repository,
so a rebuild can keep BARC pairs the published pool dropped as their duplicates.

Deterministic given the inputs, except execute: a call that runs near the sandbox's CALL_S limit can pass on one
machine and time out on another.  ``code_hash`` (``policy.normalised``) is the same on Python 3.12+.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from frm.data.heldout import HeldOut, canon
from frm.external import corpus
from frm.external.barc import fetch, policy, sandbox
from frm.external.flatpool import FlatPool, Pair, build_flat

SHARD = 2000  # rows per shard
SAME_SHAPE = 0.65  # ARC-AGI-1 evaluation's same-shape share of pairs
N_MIN = 14  # pairs a task needs
VAL_MOD = 50
NOISE_AREA = 36  # grids of at most this many cells are motifs: a 3x3 of scattered colours is ARC
DEDUP_POOLS = ("rearc", "a2_new")  # FlatPools in --pools
DEDUP_SUBSETS = ("training", "concept")  # ARC subsets in --raw, made pools here
(DATA_FILE,) = fetch.FILES  # one file (fetch.py says why); row ids h000000...
ID_PREFIX = "h"


def background(g: np.ndarray) -> int:
    """Most common colour; ties go to the smallest."""
    return int(np.argmax(np.bincount(g.reshape(-1).astype(np.int64), minlength=10)))


def noise_share(g: np.ndarray, bg: int) -> float:
    """Share of the foreground cells with no 4-neighbour of their own colour: near 1 on scattered noise."""
    fg = g != bg
    n = int(fg.sum())
    if n == 0:
        return 0.0
    p = np.pad(g.astype(np.int16), 1, constant_values=-1)
    h, w = g.shape
    same = np.zeros(g.shape, bool)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        same |= p[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w] == g
    return float((fg & ~same).sum()) / n


def structured(g: np.ndarray) -> bool:
    """An input a whole-grid rule can be read from: >= 3 foreground cells and >= 6% of the grid, and not scattered
    noise past NOISE_AREA cells."""
    bg = background(g)
    fg = int((g != bg).sum())
    if fg < 3 or fg < 0.06 * g.size:
        return False
    return g.size <= NOISE_AREA or fg < 0.15 * g.size or noise_share(g, bg) <= 0.5


def is_val(key: str) -> bool:
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % VAL_MOD == 0


def work(root: Path) -> Path:
    return root / "work"


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def policy_hash() -> str:
    """The static policy's version: parsed shards record the verdicts of one policy."""
    return fetch.sha256(Path(policy.__file__))[:16]


# --------------------------------------------------------------------------
# parse: rows -> pairs, static checks
# --------------------------------------------------------------------------


def _index(path: Path) -> list[int]:
    """Byte offset of every line start."""
    offs, pos = [0], 0
    with open(path, "rb") as f:
        while block := f.read(64 << 20):
            i = block.find(b"\n")
            while i >= 0:
                offs.append(pos + i + 1)
                i = block.find(b"\n", i + 1)
            pos += len(block)
    if offs[-1] == pos:
        offs.pop()
    return offs


def grid(x) -> np.ndarray | None:
    """An ARC grid from a stored example (lists of numbers, floats included), or None."""
    try:
        a = np.asarray(x, dtype=np.float64)
    except (ValueError, TypeError):
        return None
    if a.ndim != 2 or not (1 <= a.shape[0] <= 30 and 1 <= a.shape[1] <= 30):
        return None
    if not bool(np.all(a == np.round(a))) or a.min() < 0 or a.max() > 9:
        return None
    return np.ascontiguousarray(a.astype(np.int8))


def parse_row(rid: str, row: dict) -> tuple[dict, list[Pair]]:
    src = row.get("source") or ""
    meta = {"id": rid, "seeds": row.get("seeds"), "drops": {}}
    why = policy.check(src)
    meta["policy"] = why
    meta["source"] = src
    if why is None:
        h = policy.header(src)
        code = hashlib.sha1(policy.normalised(src).encode()).hexdigest()[:16]
        desc = policy.norm_text(h["description"]) or f"program:{code}"  # no description: the program is its own
        meta |= {"concepts": h["concepts"], "description": h["description"], "code_hash": code,
                 "desc_hash": hashlib.sha1(desc.encode()).hexdigest()[:16], "colours": policy.rule_colours(src)}
    pairs, seen, drops = [], set(), Counter()
    ex = row.get("examples") or []
    meta["n_raw"] = len(ex)
    for e in ex:
        if not (isinstance(e, list) and len(e) == 2):
            drops["format"] += 1
            continue
        a, b = grid(e[0]), grid(e[1])
        if a is None or b is None:
            drops["cells_or_sides"] += 1
            continue
        if a.shape == b.shape and bool((a == b).all()):
            drops["identity"] += 1
            continue
        k = (a.shape, a.tobytes())
        if k in seen:
            drops["duplicate_input"] += 1
            continue
        seen.add(k)
        pairs.append((a, b))
    meta["drops"] = dict(drops)
    meta["n_static"] = len(pairs)
    return meta, pairs


def _parse_shard(args: tuple) -> dict:
    root, start, offs, k = args
    metas, tasks = [], []
    with open(root / DATA_FILE, "rb") as f:
        for j, off in enumerate(offs):
            f.seek(off)
            line = f.readline()
            rid = f"{ID_PREFIX}{start + j:06d}"
            try:
                row = json.loads(line)
            except ValueError:
                metas.append({"id": rid, "policy": "json", "drops": {}, "n_raw": 0, "n_static": 0})
                tasks.append((rid, []))
                continue
            m, prs = parse_row(rid, row)
            metas.append(m)
            tasks.append((rid, prs))
    out = work(root) / "parsed"
    out.mkdir(parents=True, exist_ok=True)
    build_flat(tasks).save(out / f"{k:03d}.npz")
    with open(out / f"{k:03d}.jsonl.partial", "w") as f:
        f.writelines(json.dumps(m) + "\n" for m in metas)
    (out / f"{k:03d}.jsonl.partial").replace(out / f"{k:03d}.jsonl")
    return {"shard": k, "rows": len(metas)}


def parse(root: Path, workers: int) -> dict:
    """Shards of SHARD rows in file order.  The policy that judged them is stamped before the first shard
    (work/parsed/POLICY): a shard set of another policy, or one with no stamp, is refused (remove work/ and parse
    again), and select refuses the same."""
    if not (root / DATA_FILE).exists():
        raise SystemExit(f"no {root / DATA_FILE} (python -m frm.external.barc.fetch)")
    stamp = work(root) / "parsed" / "POLICY"
    have = parsed_shards(root) if (work(root) / "parsed").exists() else []
    if have and (not stamp.exists() or stamp.read_text().strip() != policy_hash()):
        raise SystemExit(f"{work(root) / 'parsed'} holds another policy's verdicts: remove {work(root)} and rerun")
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(policy_hash() + "\n")
    offs = _index(root / DATA_FILE)
    print(f"{DATA_FILE}: {len(offs)} rows", flush=True)
    jobs = [(root, s, offs[s : s + SHARD], k) for k, s in enumerate(range(0, len(offs), SHARD))
            if not (work(root) / "parsed" / f"{k:03d}.jsonl").exists()]
    t0 = time.time()
    with Pool(workers) as pool:
        for r in pool.imap_unordered(_parse_shard, jobs):
            print(f"  parsed shard {r['shard']} ({r['rows']} rows), {time.time() - t0:.0f}s", flush=True)
    shards = (len(offs) + SHARD - 1) // SHARD
    _write_json(work(root) / "parsed_all.json", {"rows": len(offs), "shards": shards, "policy": policy_hash()})
    return {"rows": len(offs), "shards": shards}


def parsed_shards(root: Path) -> list[int]:
    return sorted(int(p.stem.split(".")[0]) for p in (work(root) / "parsed").glob("*.jsonl"))


def load_shard(root: Path, k: int) -> tuple[list[dict], FlatPool]:
    d = work(root) / "parsed"
    return _jsonl(d / f"{k:03d}.jsonl"), FlatPool.load(d / f"{k:03d}.npz")


# --------------------------------------------------------------------------
# execute: every stored pair re-run in the sandbox
# --------------------------------------------------------------------------


def _exec_job(w: sandbox.Worker, item: tuple) -> dict:
    _, source, pairs = item
    return sandbox.check_and_run(w, source, pairs)


def exec_items(metas: list[dict], fp: FlatPool) -> list[tuple]:
    """(row index, source, stored pairs) of every row execute runs."""
    return [(t, m["source"], [[a.tolist(), b.tolist()] for a, b in fp.pairs(t)])
            for t, m in enumerate(metas) if m["policy"] is None and m["n_static"] >= N_MIN]


def execute(root: Path, workers: int) -> dict:
    sb = sandbox.Sandbox(workers, root)
    t0, n = time.time(), 0
    try:
        for k in parsed_shards(root):
            path = work(root) / "executed" / f"{k:03d}.json"
            if path.exists():
                continue
            metas, fp = load_shard(root, k)
            items = exec_items(metas, fp)
            res = {}
            for (t, _, _), r in sb.map(_exec_job, items):
                res[metas[t]["id"]] = r
            _write_json(path, res)
            n += len(items)
            print(f"  executed shard {k}: {len(items)} programs, {n / (time.time() - t0):.1f}/s", flush=True)
    finally:
        sb.close()
    return {"programs": n, "rss_kills": sb.killed_rss}


# --------------------------------------------------------------------------
# filter: content filter, canonical forms, readability
# --------------------------------------------------------------------------

_HELD: HeldOut | None = None


def _init_filter(held: HeldOut) -> None:
    global _HELD
    _HELD = held


def _filter_shard(args: tuple) -> tuple[int, dict]:
    root, k = args
    assert _HELD is not None
    metas, fp = load_shard(root, k)
    ex = json.loads((work(root) / "executed" / f"{k:03d}.json").read_text())
    out = {}
    for t, m in enumerate(metas):
        r = ex.get(m["id"])
        if r is None or "status" not in r:
            continue
        rows = []
        for (a, b), st in zip(fp.pairs(t), r["status"], strict=True):
            if st != "ok":
                rows.append([st, None, None])
                continue
            hit = _HELD.hit(a, b)
            rows.append([hit or "ok", canon(a, b).hex(), bool(structured(a))])
        out[m["id"]] = rows
    return k, out


def filter_(root: Path, raw: Path, pools: Path, workers: int) -> dict:
    todo = [k for k in parsed_shards(root) if (work(root) / "executed" / f"{k:03d}.json").exists()
            and not (work(root) / "filtered" / f"{k:03d}.json").exists()]
    held = corpus.heldout(raw)
    with Pool(workers, initializer=_init_filter, initargs=(held,)) as pool:
        for k, out in pool.imap_unordered(_filter_shard, [(root, k) for k in todo]):
            _write_json(work(root) / "filtered" / f"{k:03d}.json", out)
            print(f"  filtered shard {k}", flush=True)
    existing_canons(root, raw, pools, workers, held)
    return {"shards": len(todo)}


def _canons(prs: list[Pair]) -> list[str]:
    return [canon(a, b).hex() for a, b in prs]


def _dedup_inputs(raw: Path, pools: Path) -> dict[str, Path]:
    files = {f"{n}.npz": pools / f"{n}.npz" for n in DEDUP_POOLS}
    for sub in (*DEDUP_SUBSETS, *corpus.HELDOUT_SUBSETS):
        for kind in ("challenges", "solutions"):
            files[f"arc-agi_{sub}_{kind}.json"] = raw / f"arc-agi_{sub}_{kind}.json"
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit(f"dedup inputs missing (fetch_data.py, frm.external.corpus): {missing}")
    return files


def existing_canons(root: Path, raw: Path, pools: Path, workers: int, held: HeldOut | None = None) -> set[str]:
    """Canonical pairs of DEDUP_POOLS and DEDUP_SUBSETS, cached in work/existing_canons.json beside the sha256 of
    every input: a changed input is read again."""
    path = work(root) / "existing_canons.json"
    sums = {n: fetch.sha256(p) for n, p in _dedup_inputs(raw, pools).items()}
    cached = json.loads(path.read_text()) if path.exists() else {}
    if cached.get("sha256") != sums:
        held = held or corpus.heldout(raw)
        groups: dict[str, list[list[Pair]]] = {}
        for n in DEDUP_POOLS:
            fp = FlatPool.load(pools / f"{n}.npz")
            groups[n] = [fp.pairs(t) for t in range(fp.n_tasks())]
        for sub in DEDUP_SUBSETS:
            keep, _ = corpus.fixed_pool(corpus.arc_tasks(raw, sub), held, workers)
            groups[sub] = [prs for _, prs in keep]
        canons = {}
        with Pool(workers) as pool:
            for n, gs in groups.items():
                canons[n] = sorted({c for cs in pool.map(_canons, gs, chunksize=8) for c in cs})
        cached = {"sha256": sums, "canons": canons}
        _write_json(path, cached)
    return {c for v in cached["canons"].values() for c in v}


# --------------------------------------------------------------------------
# select: >= N_MIN pairs, one task per program, the shape mix
# --------------------------------------------------------------------------


def _unit(key: str) -> float:
    return int(hashlib.md5(key.encode()).hexdigest()[:12], 16) / float(1 << 48)


def _sandbox_reason(r: dict) -> str:
    if "died" in r:
        return "died"
    if r.get("load") != "ok":
        return str(r.get("load", "missing")).split(":")[0]
    return str(r.get("run", "run")).split(":")[0]


def select(root: Path, raw: Path, pools: Path, workers: int = 8, existing: set[str] | None = None) -> dict:
    """Pair-level verdicts in row order: a canonical pair is kept by the first task that keeps it.  Writes
    work/selected.json: the tasks to audit, in order, with their kept pairs, and every drop count.  ``existing``
    (the dedup digests) defaults to `existing_canons`."""
    w = work(root)
    full = json.loads((w / "parsed_all.json").read_text()) if (w / "parsed_all.json").exists() else None
    if full is None or parsed_shards(root) != list(range(full["shards"])):
        raise SystemExit("select needs every row parsed (build parse)")
    stamp = w / "parsed" / "POLICY"
    if full.get("policy") != policy_hash() or not stamp.exists() or stamp.read_text().strip() != policy_hash():
        raise SystemExit(f"policy.py changed since parse: remove {w} and rebuild")
    late = [k for k in parsed_shards(root) if not ((w / "executed" / f"{k:03d}.json").exists()
                                                  and (w / "filtered" / f"{k:03d}.json").exists())]
    if late:
        raise SystemExit(f"select needs every shard executed and filtered: {late[:5]}...")
    if existing is None:
        existing = existing_canons(root, raw, pools, workers)
    seen: set = set()
    codes: set = set()
    pair_drops, task_drops = Counter(), Counter()
    rows_total = unread = 0
    chosen = []
    for k in parsed_shards(root):
        metas, fp = load_shard(root, k)
        ex = json.loads((w / "executed" / f"{k:03d}.json").read_text())
        fl = json.loads((w / "filtered" / f"{k:03d}.json").read_text())
        for t, m in enumerate(metas):
            rows_total += 1
            for why, n in m["drops"].items():
                pair_drops[f"static:{why}"] += n
            if m["policy"] is not None:
                task_drops[f"policy:{m['policy'].split(':')[0]}"] += 1
                continue
            if m["n_static"] < N_MIN:
                task_drops["static_pairs_below_min"] += 1
                continue
            r = ex.get(m["id"], {})
            if "status" not in r:
                task_drops[f"sandbox:{_sandbox_reason(r)}"] += 1
                continue
            keep, mine = [], set()
            for i, (st, cn, readable) in enumerate(fl[m["id"]]):
                if st != "ok":
                    pair_drops[f"{'content' if st in ('pair', 'grid') else 'sandbox'}:{st}"] += 1
                elif cn in existing:
                    pair_drops["dedup:existing_pool"] += 1
                elif cn in seen or cn in mine:
                    pair_drops["dedup:barc"] += 1
                else:
                    mine.add(cn)
                    keep.append((i, readable))
            if len(keep) < N_MIN:
                task_drops["pairs_below_min"] += 1
                continue
            if m["code_hash"] in codes:
                task_drops["duplicate_program"] += 1
                continue
            codes.add(m["code_hash"])
            seen |= mine
            unread += sum(not rd for _, rd in keep)
            prs = [fp.pair_at(t, i) for i, _ in keep]
            bg = Counter(background(a) for a, _ in prs).most_common(1)[0][0]
            chosen.append({"id": m["id"], "shard": k, "keep": [i for i, _ in keep], "desc_hash": m["desc_hash"],
                           "code_hash": m["code_hash"], "colours": m["colours"], "bg": int(bg), "n": len(prs),
                           "same": sum(a.shape == b.shape for a, b in prs)})
    # the shape mix: whole same-shape tasks thinned by a fixed hash of their id until the kept
    # pairs are SAME_SHAPE same-shape: p * S_full + S_mixed = T * (p * S_full + S_mixed + other)
    s_full = sum(c["n"] for c in chosen if c["same"] == c["n"])
    s_mixed = sum(c["same"] for c in chosen if c["same"] < c["n"])
    other = sum(c["n"] - c["same"] for c in chosen)
    pairs = s_full + s_mixed + other
    p = min(1.0, max(0.0, (SAME_SHAPE * other - (1 - SAME_SHAPE) * s_mixed) / max(1.0, (1 - SAME_SHAPE) * s_full)))
    kept = [c for c in chosen if c["same"] < c["n"] or _unit("shape:" + c["id"]) < p]
    task_drops["shape_rebalance"] = len(chosen) - len(kept)
    report = {"rows": rows_total, "pair_drops": dict(pair_drops), "task_drops": dict(task_drops),
              "candidates": len(chosen), "candidate_pairs": pairs,
              "same_shape_share_before": round((s_full + s_mixed) / max(1, pairs), 3),
              "same_shape_keep_p": round(p, 4), "selected": len(kept),
              "selected_pairs": sum(c["n"] for c in kept),
              "same_shape_share_after": round(sum(c["same"] for c in kept) / max(1, sum(c["n"] for c in kept)), 3),
              "unreadable_inputs_of_candidates": unread}
    _write_json(w / "selected.json", {"report": report, "tasks": kept})
    return report


# --------------------------------------------------------------------------
# write: audited tasks only -> barc, barc_val
# --------------------------------------------------------------------------


def write(root: Path, raw: Path, pools: Path) -> dict:
    from frm.external.barc import audit

    sel = json.loads((work(root) / "selected.json").read_text())
    results = audit.results(root)
    hits = {i: r for i, r in results.items() if r["status"] == "hit"}
    bad_code = {r["code_hash"] for r in hits.values()}
    bad_desc = {r["desc_hash"] for r in hits.values()}
    drops = Counter()
    keep = []
    for c in sel["tasks"]:
        r = results.get(c["id"])
        if r is None:
            drops["not_audited"] += 1
        elif r["status"] == "hit":
            drops["audit_hit"] += 1
        elif r["status"] != "clean":
            drops[f"audit_{r['status']}"] += 1
        elif c["code_hash"] in bad_code or c["desc_hash"] in bad_desc:
            drops["audit_same_program_or_description_as_a_hit"] += 1
        else:
            keep.append(c)
    by_shard: dict[int, list] = {}
    for c in keep:
        by_shard.setdefault(c["shard"], []).append(c)
    out = {"barc": ([], []), "barc_val": ([], [])}
    sources = []
    for k in sorted(by_shard):
        metas, fp = load_shard(root, k)
        idx = {m["id"]: t for t, m in enumerate(metas)}
        for c in by_shard[k]:
            t = idx[c["id"]]
            m = metas[t]
            prs = [fp.pair_at(t, i) for i in c["keep"]]
            name = "barc_val" if is_val(c["desc_hash"]) else "barc"
            out[name][0].append((f"barc/{c['id']}", prs))
            out[name][1].append({"id": f"barc/{c['id']}", "row": c["id"], "concepts": m["concepts"],
                                 "description": m["description"], "seeds": m["seeds"],
                                 "code_hash": m["code_hash"], "desc_hash": m["desc_hash"], "colours": m["colours"],
                                 "n": len(prs), "same_shape": all(a.shape == b.shape for a, b in prs)})
            sources.append({"id": f"barc/{c['id']}", "code_hash": m["code_hash"], "source": m["source"]})
    pools.mkdir(parents=True, exist_ok=True)
    for name, (tasks, metas) in out.items():
        build_flat(tasks).save(pools / f"{name}.npz")
        (pools / f"{name}.jsonl").write_text("".join(json.dumps(m, sort_keys=True) + "\n" for m in metas))
    (root / "sources.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sources))
    n_held = len(audit.heldout(raw))
    manifest = root / "MANIFEST.json"
    report = {"select": sel["report"], "write_drops": dict(drops),
              "barc": {"tasks": len(out["barc"][0]), "pairs": sum(len(p) for _, p in out["barc"][0]),
                       "programs": len({m["code_hash"] for m in out["barc"][1]}),
                       "descriptions": len({m["desc_hash"] for m in out["barc"][1]})},
              "barc_val": {"tasks": len(out["barc_val"][0]), "pairs": sum(len(p) for _, p in out["barc_val"][0]),
                           "programs": len({m["code_hash"] for m in out["barc_val"][1]}),
                           "descriptions": len({m["desc_hash"] for m in out["barc_val"][1]})},
              "audit": audit.summary(results, len(sel["tasks"]), n_held),
              "manifest": json.loads(manifest.read_text()) if manifest.exists() else None}
    (pools / "barc_build.json").write_text(json.dumps(report, indent=1))
    audit.write_report(pools, results, hits, len(sel["tasks"]), n_held)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("parse", "execute", "filter", "select", "write"))
    ap.add_argument("--dir", type=Path, default=fetch.DIR, help="BARC work directory")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"), help="ARC subsets (fetch_data.py)")
    ap.add_argument("--pools", type=Path, default=Path("data/pools"), help="rearc/a2_new in, barc out")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    fn = {"parse": lambda: parse(a.dir, a.workers), "execute": lambda: execute(a.dir, a.workers),
          "filter": lambda: filter_(a.dir, a.raw, a.pools, a.workers),
          "select": lambda: select(a.dir, a.raw, a.pools, a.workers),
          "write": lambda: write(a.dir, a.raw, a.pools)}[a.stage]
    print(json.dumps(fn(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
