"""Two arms' learning curves side by side (the readouts of EXPERIMENTS.md §4b/§4c).

    kelvin/k2 pull frmc_a1tn && kelvin/k2 pull trmc_a1tn
    python scripts/arm_curves.py runs/frmc_a1tn runs/trmc_a1tn [--windows 1000-12000 13000-24000 25000-36000]

Reads RUN/eval.jsonl and RUN/train.jsonl of arm A and arm B (the control) and prints, for A − B:
- per evaluation (matched PF): dev / composite pass@2 and row exact;
- pooled windows: each arm's per-task pass@2 averaged over its evaluations inside the window, then the paired
  difference over tasks with a 90% bootstrap CI (smooths the ±0.05 noise of single 100-task dev evaluations);
- slopes of the eval metrics against log(PF) over the last window (± standard error);
- training CE / exact at matched optimizer steps and at matched PF (±5% windows): the per-step and per-compute views.
Only records scored with the trained-versions vote are used (they carry ``versions``).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def evals(run: Path) -> dict[int, dict]:
    out = {}
    for line in (run / "eval.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "versions" in r.get("dev", {}):
            out[int(round(r["pflops"] / 1000)) * 1000] = r  # a final record duplicating the last one overwrites it
    return out


def train_rows(run: Path) -> list[dict]:
    keep = ("step", "pflops", "ce", "exact")
    return [{k: r[k] for k in keep} for r in map(json.loads, (run / "train.jsonl").read_text().splitlines())]


def boot(d: np.ndarray, rng, n: int = 10000) -> tuple[float, float]:
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))


def window(rows: list[dict], key: str, at: float, rel: float = 0.05):
    xs = [r for r in rows if abs(r[key] - at) <= rel * at]
    if not xs:
        return None
    return np.mean([r["ce"] for r in xs]), np.mean([r["exact"] for r in xs])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--windows", nargs="*", default=None, help="PF windows LO-HI for pooled comparisons")
    args = ap.parse_args()
    A, B = evals(args.a), evals(args.b)
    pfs = sorted(set(A) & set(B))
    if not pfs:
        print("no common evaluation points")
        return 1
    rng = np.random.default_rng(0)
    ex = lambda r, s: r[s]["by_round"]["row_exact"][-1]  # noqa: E731
    print(f"A = {args.a.name}, B = {args.b.name}; common evaluations at {pfs[0]}..{pfs[-1]} PF")
    print("   PF  | dev p2 A/B      | dev rowex A/B     | comp p2 A/B     | comp rowex A/B")
    for p in pfs:
        a, b = A[p], B[p]
        print(f"{p:6d} | {a['dev']['pass@2']:.3f} {b['dev']['pass@2']:.3f} | {ex(a, 'dev'):.4f} {ex(b, 'dev'):.4f} | "
              f"{a['dev_composite']['pass@2']:.3f} {b['dev_composite']['pass@2']:.3f} | "
              f"{ex(a, 'dev_composite'):.4f} {ex(b, 'dev_composite'):.4f}")
    wins = args.windows or [f"{pfs[0]}-{pfs[len(pfs) // 3]}", f"{pfs[len(pfs) // 3 + 1]}-{pfs[-1]}"]
    print("pooled per-task pass@2, A − B, 90% paired bootstrap CI")
    for w in wins:
        lo, hi = map(int, w.split("-"))
        sel = [p for p in pfs if lo <= p <= hi]
        if not sel:
            continue
        for split in ("dev", "dev_composite"):
            tasks = sorted(A[sel[0]][split]["per_task"]["pass@2"])
            pa = np.array([[A[p][split]["per_task"]["pass@2"][t] for t in tasks] for p in sel]).mean(0)
            pb = np.array([[B[p][split]["per_task"]["pass@2"][t] for t in tasks] for p in sel]).mean(0)
            ci = boot(pa - pb, rng)
            print(f"  {split:14s} {lo:6d}-{hi:<6d} ({len(sel):2d} evals) A {pa.mean():.3f} B {pb.mean():.3f} "
                  f"d {(pa - pb).mean():+.4f} CI ({ci[0]:+.4f}, {ci[1]:+.4f})")
    lo, hi = map(int, wins[-1].split("-"))
    sel = [p for p in pfs if lo <= p <= hi]
    if len(sel) >= 3:
        print(f"slopes per e-fold of PF over {lo}-{hi}")
        x = np.log(sel)
        for name, fn in (("dev pass@2", lambda r: r["dev"]["pass@2"]), ("dev rowex", lambda r: ex(r, "dev")),
                         ("comp pass@2", lambda r: r["dev_composite"]["pass@2"]),
                         ("comp rowex", lambda r: ex(r, "dev_composite"))):
            fits = []
            for R in (A, B):
                y = np.array([fn(R[p]) for p in sel])
                b, a = np.polyfit(x, y, 1)
                se = math.sqrt(((y - a - b * x) ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
                fits.append((b, se))
            (ba, sa), (bb, sb) = fits
            print(f"  {name:12s} A {ba:+.4f} ± {sa:.4f}  B {bb:+.4f} ± {sb:.4f}  A−B {ba - bb:+.4f} ± {math.hypot(sa, sb):.4f}")
    ta, tb = train_rows(args.a), train_rows(args.b)
    print("training CE / exact at matched steps, then at matched PF (±5% windows)")
    for key, points in (("step", (1000, 4000, 16000, 64000, 128000)), ("pflops", (1000, 4000, 12000, 24000, 36000))):
        for at in points:
            va, vb = window(ta, key, at), window(tb, key, at)
            if va and vb:
                print(f"  {key:6s} {at:7d}: A {va[0]:.3f}/{va[1]:.3f}  B {vb[0]:.3f}/{vb[1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
