"""Rule-level audit of the selected BARC programs against the held-out tasks, sandboxed (untrusted LLM-written code).

    python -m frm.external.barc.audit [--dir data/external/barc] [--raw data/raw] [--workers 8] \
        [--limit N] [--subset N]

Runs after ``build select`` and before ``build write``.  ``--limit N`` audits N more programs and stops (a
throughput measurement); ``--subset N`` audits only the first N of the fixed order (`order`).

Every selected program (work/selected.json) runs on the demonstrations of every held-out task (ARC-AGI-1
evaluation then ARC-AGI-2 evaluation, each sorted by id; demonstration pairs only, never test outputs).  A
program's colours are hard-coded, and training's colour augmentation makes a rule r equal to every
sigma . r . sigma^-1, so the task's colours are rebound instead: a HIT is a colour bijection tau over the held-out
task's colours with main(tau(x)) == tau(y) on every demonstration.  tau is enumerated by `bindings`: the task's
background to the rule's (the mode of its stored inputs' backgrounds), the rule's other colours (the `Color.X`
names of its code outside generate_input) injectively onto the task's other colours, all bindings up to BINDINGS,
else BINDINGS random ones; the task's remaining colours go to colours the rule does not name.  Training also
applies the 8 dihedral transforms to a task, so each is tried too (main(tau(d(x))) == tau(d(y))), once per
distinct transformed demonstration set, and once per class of the program's own symmetries: a program that
commutes with g on its first SYM_PAIRS stored pairs gives the same verdict on d and on g . d.  Demonstration 0
first; a (task, transform) case ends early only when three bindings spread over the table all run and agree on
one output shape that is not the demonstration's.  A program is CLEAN only if every case was answered: any
timeout, more than MAX_TIMEOUTS timeouts, BUDGET_S of CPU, a memory stop or a dead worker make it INCONCLUSIVE, a
load or protocol failure an ERROR; both are dropped with the hits.  ``build write`` drops every hit, and every task
whose normalised program or description is a hit's.

Results: work/audit/*.jsonl (one line per program; resumable).  Not reproducible run to run: INCONCLUSIVE depends
on CPU speed (BUDGET_S, CALL_S).  Report: {pools}/barc_rule_hits.json, written by ``build write``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from frm.external import corpus
from frm.external.barc import build, fetch, sandbox
from frm.external.barc.build import background

BUDGET_S = 60.0  # CPU seconds per program (past it: inconclusive, dropped)
MAX_TIMEOUTS = 3
SYM_PAIRS = 4  # stored pairs a program's dihedral symmetries are read from
BINDINGS = 48


def audit_dir(root: Path) -> Path:
    return build.work(root) / "audit"


def dihedral(g: np.ndarray, k: int) -> np.ndarray:
    """The worker's numbering (k >= 4 flips columns first): case transforms are sent to it as indices."""
    if k >= 4:
        g = g[:, ::-1]
    return np.ascontiguousarray(np.rot90(g, k % 4))


def bindings(rule_cols: list[int], rule_bg: int | None, task_bg: int, task_cols: list[int],
             rng: np.random.Generator) -> list[dict]:
    """Rule colour -> task colour maps: rule_bg -> task_bg, the rule's other colours injectively onto the task's."""
    others = [c for c in rule_cols if c != rule_bg]
    cand = [c for c in task_cols if c != task_bg]
    if len(cand) < len(others):
        cand += [c for c in rng.permutation(10).tolist() if c not in cand and c != task_bg][:len(others) - len(cand)]
    base = {rule_bg: task_bg} if rule_bg is not None else {}
    n = 1
    for i in range(len(others)):
        n *= len(cand) - i
    if n <= BINDINGS:
        perms = list(itertools.permutations(cand, len(others)))
    else:
        perms = [tuple(rng.permutation(cand)[:len(others)].tolist()) for _ in range(BINDINGS)]
    return [{**base, **dict(zip(others, pm, strict=True))} for pm in perms]


def heldout(raw: Path) -> list[tuple]:
    """(subset, tid, demonstrations, background, colours, distinct dihedral transforms) per held-out task."""
    out = []
    for sub in corpus.HELDOUT_SUBSETS:
        for tid, demos, _ in corpus.arc_tasks(raw, sub):
            cols = sorted({int(c) for a, b in demos for g in (a, b) for c in np.unique(g)})
            seen, ds = set(), []
            for d in range(8):
                key = b"".join(dihedral(g, d).tobytes() + bytes(dihedral(g, d).shape) for a, b in demos for g in (a, b))
                if key not in seen:
                    seen.add(key)
                    ds.append(d)
            out.append((sub, tid, demos, background(demos[0][0]), cols, ds))
    return out


def luts(rule_cols: list[int], rule_bg: int, task_bg: int, task_cols: list[int], seed: str) -> list[list[int]]:
    """Every tau (task colour -> rule colour) as a 10-entry table, from `bindings` (rule colour -> task colour),
    inverted; the task's unbound colours go to colours the rule does not name, smallest first.  Deterministic in
    `seed`."""
    rng = np.random.default_rng(int(hashlib.md5(seed.encode()).hexdigest()[:8], 16))
    named = set(rule_cols) | {rule_bg}
    out, seen = [], set()
    for bd in bindings(sorted(named), rule_bg, task_bg, task_cols, rng):
        tau = {tc: rc for rc, tc in bd.items() if tc in task_cols}
        free = [c for c in range(10) if c not in named and c not in tau.values()]
        for tc in task_cols:
            if tc not in tau:
                tau[tc] = free.pop(0)
        rest = [c for c in range(10) if c not in tau.values()]
        lut = [tau[c] if c in tau else rest.pop(0) for c in range(10)]
        key = tuple(lut[c] for c in task_cols)
        if key not in seen:
            seen.add(key)
            out.append(lut)
    return out


def _cases(held: list, rule_cols: list[int], rule_bg: int) -> list:
    """[held-out task index, its distinct dihedral transforms, its tau tables] for one rule's colours."""
    return [[t, ds, luts(rule_cols, rule_bg, tbg, tcols, f"{tid}:{rule_cols}:{rule_bg}")]
            for t, (_, tid, _, tbg, tcols, ds) in enumerate(held)]


def results(root: Path) -> dict[str, dict]:
    return {(r := json.loads(x))["id"]: r for p in sorted(audit_dir(root).glob("*.jsonl"))
            for x in p.read_text().splitlines() if x}


def summary(res: dict, n_selected: int, n_heldout: int) -> dict:
    st = Counter(r["status"] for r in res.values())
    return {"selected": n_selected, "audited": len(res), "fraction_audited": round(len(res) / max(1, n_selected), 4),
            "status": dict(st), "calls": sum(r.get("calls", 0) for r in res.values()),
            "worker_seconds": round(sum(r.get("seconds", 0.0) for r in res.values())),
            "heldout_tasks": n_heldout, "bindings_per_task": BINDINGS, "dihedral_transforms": 8,
            "budget_s": BUDGET_S, "max_timeouts": MAX_TIMEOUTS}


def write_report(pools: Path, res: dict, hits: dict, n_selected: int, n_heldout: int) -> None:
    (pools / "barc_rule_hits.json").write_text(json.dumps(
        {**summary(res, n_selected, n_heldout), "hit_programs": len(hits),
         "heldout_tasks_hit": sorted({(h["hit"]["split"], h["hit"]["tid"]) for h in hits.values()}),
         "hits": sorted(({"id": i, **h["hit"], "code_hash": h["code_hash"], "desc_hash": h["desc_hash"]}
                         for i, h in hits.items()), key=lambda h: (h["tid"], h["id"])),
         "audited_clean": sorted(i for i, r in res.items() if r["status"] == "clean")}, indent=1))


def order(tasks: list[dict]) -> list[dict]:
    """The deterministic audit order (and subset): by a fixed hash of the row id."""
    return sorted(tasks, key=lambda c: build._unit("audit:" + c["id"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=fetch.DIR, help="BARC work directory")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"), help="ARC subsets (fetch_data.py)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="audit this many more programs, then stop")
    ap.add_argument("--subset", type=int, default=0, help="audit only the first N of the fixed order")
    a = ap.parse_args(argv)
    root = a.dir
    sel = json.loads((build.work(root) / "selected.json").read_text())["tasks"]
    todo = order(sel)
    if a.subset:
        todo = todo[:a.subset]
    have = set(results(root)) if audit_dir(root).exists() else set()
    todo = [c for c in todo if c["id"] not in have]
    if a.limit:
        todo = todo[:a.limit]
    held = heldout(a.raw)
    init = [{"op": "heldout", "tasks": [[[x.tolist(), y.tolist()] for x, y in h[2]] for h in held]}]
    sb = sandbox.Sandbox(a.workers, root, init=init)
    sources: dict[str, str] = {}
    own: dict[str, list] = {}
    by_shard: dict[int, list] = {}
    for c in todo:
        by_shard.setdefault(c["shard"], []).append(c)
    for k, cs in by_shard.items():
        metas, fp = build.load_shard(root, k)
        index = {m["id"]: t for t, m in enumerate(metas)}
        for c in cs:
            t = index[c["id"]]
            sources[c["id"]] = metas[t]["source"]
            own[c["id"]] = [[x.tolist(), y.tolist()] for x, y in (fp.pair_at(t, i) for i in c["keep"][:SYM_PAIRS])]
    cache: dict[tuple, bytes] = {}

    def job(w: sandbox.Worker, c: dict) -> dict:
        key = (tuple(c["colours"]), c["bg"])
        if key not in cache:
            cache[key] = (json.dumps({"op": "audit", "cases": _cases(held, list(key[0]), key[1]),
                                      "budget_s": BUDGET_S, "max_timeouts": MAX_TIMEOUTS},
                                     separators=(",", ":")) + "\n").encode()
        t0 = time.monotonic()
        ld = sandbox.load(w, sources[c["id"]])
        if not ld.get("ok"):
            return {"error": f"load:{ld.get('why', ld.get('error', '?'))}"}
        sym = w.request({"op": "symmetry", "pairs": own[c["id"]]}, 8 * SYM_PAIRS * sandbox.CALL_S + 10)
        if "sym" not in sym:
            return {"error": f"symmetry:{sym.get('error', '?')}"}
        r = w.request(cache[key], BUDGET_S * 2 + 60)
        r["seconds"] = round(time.monotonic() - t0, 3)
        r["symmetries"] = sym["sym"]
        return r

    out_dir = audit_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, n, st = time.time(), 0, Counter()
    with open(out_dir / f"{len(list(out_dir.glob('*.jsonl'))):04d}.jsonl", "a") as out:
        try:
            for c, r in sb.map(job, todo):
                if r.get("hit"):
                    status = "hit"
                elif "died" in r or r.get("budget") or r.get("memory") or r.get("timeouts", 0) > 0:
                    status = "inconclusive"  # a binding or a case went untested
                elif "error" in r or "hit" not in r:
                    status = "error"
                else:
                    status = "clean"  # every case answered, no hit
                rec = {"id": c["id"], "status": status, "code_hash": c["code_hash"], "desc_hash": c["desc_hash"],
                       "calls": r.get("calls", 0), "seconds": r.get("seconds", 0.0),
                       "symmetries": r.get("symmetries", [])}
                if status == "hit":
                    t, d, lut = r["hit"]
                    rec["hit"] = {"split": held[t][0], "tid": held[t][1], "dihedral": d, "lut": lut}
                elif status != "clean":
                    rec["why"] = (r.get("died") or r.get("why") or r.get("error")
                                  or ("budget" if r.get("budget") else "memory" if r.get("memory") else "timeout"))
                out.write(json.dumps(rec) + "\n")
                out.flush()
                n += 1
                st[status] += 1
                if n % 100 == 0 or n == len(todo):
                    rate = n / (time.time() - t0)
                    print(f"  {n}/{len(todo)} programs, {rate:.2f}/s, {dict(st)}, "
                          f"projected {len(sel) / rate / 3600:.1f} h for all {len(sel)} selected", flush=True)
        finally:
            sb.close()
    print(json.dumps({"audited_now": n, "status": dict(st), "seconds": round(time.time() - t0),
                      "programs_per_s": round(n / max(1e-9, time.time() - t0), 3), "rss_kills": sb.killed_rss}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
