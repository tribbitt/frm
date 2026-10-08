"""Evaluation: per-round voting over augmented versions, ARC pass@k, fork and depth instruments (DESIGN.md §7).

    python -m frm.evaluate --run_dir runs/X --split dev [--ckpt best|last|PATH] [--final] [--max_versions N]

Voting follows TRM (trm-samsung/evaluators/arc.py:107-177): each version's prediction is cropped, de-augmented and
hashed; candidates rank by (votes, mean sigmoid(q)); the top two are the attempts. Score = mean over tasks of the
fraction of their test inputs with a correct attempt (official ARC metric). Reading split ``test`` needs ``--final``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from frm.config import Cfg, EvalCfg
from frm.data.augment import Version
from frm.data.encode import crop
from frm.data.loader import Dataset, eval_batches
from frm.model.refiner import Refiner
from frm.objective import row_exact


def _hash(g: np.ndarray) -> str:
    return hashlib.sha1(bytes(g.shape) + np.ascontiguousarray(g, dtype=np.uint8).tobytes()).hexdigest()


class Votes:
    """Per (task, test_index): hash -> [count, sum_q, grid]."""

    def __init__(self) -> None:
        self.table: dict[tuple[str, int], dict[str, list]] = defaultdict(dict)

    def add(self, key: tuple[str, int], grid: np.ndarray, q: float, weight: float = 1.0) -> None:
        h = _hash(grid)
        slot = self.table[key].setdefault(h, [0.0, 0.0, grid])
        slot[0] += weight
        slot[1] += weight * q

    def ranked(self, key: tuple[str, int]) -> list[np.ndarray]:
        cands = self.table.get(key, {})
        order = sorted(cands.values(), key=lambda c: (c[0], c[1] / max(c[0], 1e-9)), reverse=True)
        return [c[2] for c in order]

    def merged(self, other: "Votes", weight: float) -> "Votes":
        out = Votes()
        for src, w in ((self, 1.0), (other, weight)):
            for key, cands in src.table.items():
                for h, (cnt, qs, g) in cands.items():
                    slot = out.table[key].setdefault(h, [0.0, 0.0, g])
                    slot[0] += w * cnt
                    slot[1] += w * qs
        return out


def score(votes: Votes, tasks: dict[str, dict]) -> dict[str, float]:
    """Official ARC scoring for tasks with known outputs."""
    p1, p2 = [], []
    for tid, task in tasks.items():
        s1 = s2 = 0.0
        tests = task["test"]
        if any(t.get("output") is None for t in tests):
            continue
        for i, t in enumerate(tests):
            sol = np.asarray(t["output"], dtype=np.uint8)
            att = votes.ranked((tid, i))[:2]
            ok = [a.shape == sol.shape and bool((a == sol).all()) for a in att]
            s1 += float(bool(ok[:1] and ok[0]))
            s2 += float(any(ok))
        p1.append(s1 / len(tests))
        p2.append(s2 / len(tests))
    return {"pass@1": float(np.mean(p1)) if p1 else float("nan"), "pass@2": float(np.mean(p2)) if p2 else float("nan"),
            "n_tasks": len(p1)}


def attempts(votes: Votes, tasks: dict[str, dict]) -> dict[str, list[dict]]:
    """Kaggle submission dict: one {attempt_1, attempt_2} per test input, padded by repeating attempt 1."""
    sub = {}
    for tid, task in tasks.items():
        entries = []
        for i in range(len(task["test"])):
            ranked = [a.tolist() for a in votes.ranked((tid, i))[:2]] or [[[0]]]
            ranked = ranked + [ranked[0]] * (2 - len(ranked))
            entries.append({"attempt_1": ranked[0], "attempt_2": ranked[1]})
        sub[tid] = entries
    return sub


def _rows_per_batch(model: Refiner, S: int, ecfg: EvalCfg) -> int:
    L = model.cfg.prefix_len + S * S
    width = 1 + model.cfg.K
    return max(1, min(ecfg.max_rows, ecfg.tokens_per_batch // (L * width)))


@torch.no_grad()
def run_rounds(model: Refiner, inputs, ids, rounds: int, dtype, cf: str | None = None, reset_at: int | None = None):
    """Run ``rounds`` rounds from the init carry. Returns per-round parent logits argmax/q and the final RoundOut."""
    S = int(round(inputs.shape[1] ** 0.5))
    carry = model.init_carry(inputs.shape[0], S, inputs.device, dtype)
    preds, qs, logits = [], [], []
    out = None
    for r in range(1, rounds + 1):
        if reset_at is not None and r == reset_at:
            carry = model.init_carry(inputs.shape[0], S, inputs.device, dtype)
        out = model.round(carry, inputs, ids, cf=cf)
        carry = out.carry
        preds.append(out.logits.argmax(-1))
        qs.append(out.q)
        logits.append(out.logits)
    return preds, qs, logits, out


@torch.no_grad()
def evaluate_split(
    model: Refiner,
    ds: Dataset,
    split: str,
    ecfg: EvalCfg,
    device: torch.device,
    dtype: torch.dtype,
    counterfactuals: bool = False,
    keep_attempts: bool = False,
) -> dict:
    model.eval()
    R, K = ecfg.rounds, model.cfg.K
    tasks = ds.tasks(split)
    versions: dict[int, Version] = {}
    per_round = [Votes() for _ in range(R)]
    child_votes = Votes()
    row_exact_r = np.zeros(R)
    n_rows = n_labeled = 0
    fork = defaultdict(float)
    cf_acc = defaultdict(float)
    cf_rows = 0
    for S in ds.buckets:
        for batch in eval_batches(ds, split, S, _rows_per_batch(model, S, ecfg), ecfg.max_versions):
            keys = batch["keys"]
            inputs = torch.from_numpy(batch["inputs"]).long().to(device)
            ids = torch.from_numpy(batch["puzzle_ids"]).long().to(device)
            labels = torch.from_numpy(batch["labels"]).long().to(device)
            labeled = (labels != -100).any(-1)
            preds, qs, logits, out = run_rounds(model, inputs, ids, R, dtype)
            n_rows += len(keys)
            n_labeled += int(labeled.sum())
            for r in range(R):
                row_exact_r[r] += float((row_exact(logits[r], labels) & labeled).sum())
            preds_np = [p.to(torch.uint8).cpu().numpy() for p in preds]
            qs_np = [torch.sigmoid(q).float().cpu().numpy() for q in qs]
            child_np = cq_np = None
            if out.child_logits is not None:
                child_np = out.child_logits.argmax(-1).to(torch.uint8).cpu().numpy()
                cq_np = torch.sigmoid(out.child_q).float().cpu().numpy()
                fork["parent_mass"] += float(out.join_alpha[:, 0].sum())
                fork["join_entropy"] += float(out.join_entropy.sum())
                ce = row_exact(out.child_logits, labels[:, None])
                fork["child_exact_any"] += float((ce.any(-1) & labeled).sum())
                fork["oracle_gain"] += float(((ce.any(-1) | row_exact(logits[-1], labels)) & labeled).sum())
                preds_k = out.child_logits.argmax(-1)
                same = (preds_k[:, :, None] == preds_k[:, None]).all(-1)
                fork["distinct"] += float((~torch.tril(same, diagonal=-1).any(-1)).sum())
            for b, (tid, ti, pid) in enumerate(keys):
                v = versions.get(pid)
                if v is None:
                    v = versions[pid] = Version.parse(ds.identifiers[pid])[1]
                for r in range(R):
                    per_round[r].add((tid, ti), v.invert(crop(preds_np[r][b], S)), float(qs_np[r][b]))
                if child_np is not None:
                    for k in range(K):
                        child_votes.add((tid, ti), v.invert(crop(child_np[b, k], S)), float(cq_np[b, k]))
            if counterfactuals and K > 0 and cf_rows < ecfg.cf_rows and bool(labeled.any()):
                m = labeled
                base = row_exact(logits[-1], labels)[m].float().sum()
                cf_rows += int(m.sum())
                cf_acc["base"] += float(base)
                for name, kw in (("nofork", {"cf": "nofork"}), ("sham", {"cf": "sham"}), ("reset@4", {"reset_at": 4})):
                    _, _, lg, _ = run_rounds(model, inputs, ids, R, dtype, **kw)
                    cf_acc[name] += float(row_exact(lg[-1], labels)[m].float().sum())

    res: dict = {"split": split, "rows": n_rows, "rounds": R}
    by_round = [score(v, tasks) for v in per_round]
    res.update(by_round[-1])
    res["by_round"] = {
        "pass@1": [s["pass@1"] for s in by_round],
        "pass@2": [s["pass@2"] for s in by_round],
        "row_exact": (row_exact_r / max(n_labeled, 1)).round(5).tolist(),
    }
    if R >= 4:
        res["late_gain"] = by_round[-1]["pass@2"] - by_round[3]["pass@2"]
        res["late_gain_rows"] = float(row_exact_r[-1] - row_exact_r[3]) / max(n_labeled, 1)
    if K > 0:
        res["fork"] = {
            "parent_mass": fork["parent_mass"] / max(n_rows, 1),
            "join_entropy": fork["join_entropy"] / max(n_rows, 1),
            "distinct": fork["distinct"] / max(n_rows, 1),
            "child_exact_any": fork["child_exact_any"] / max(n_labeled, 1),
            "oracle_gain": fork["oracle_gain"] / max(n_labeled, 1) - row_exact_r[-1] / max(n_labeled, 1),
            "children_only": score(child_votes, tasks),
        }
        if ecfg.child_vote_weight > 0:
            res["fork"]["with_child_votes"] = score(per_round[-1].merged(child_votes, ecfg.child_vote_weight), tasks)
        if cf_rows:
            res["cf"] = {k: (cf_acc[k] - cf_acc["base"]) / cf_rows for k in ("nofork", "sham", "reset@4")}
            res["cf"]["rows"] = cf_rows
            res["cf"]["base_exact"] = cf_acc["base"] / cf_rows
    if keep_attempts:
        final = per_round[-1]
        if K > 0 and ecfg.child_vote_weight > 0:
            final = final.merged(child_votes, ecfg.child_vote_weight)
        res["attempts"] = attempts(final, tasks)
    return res


def load_for_eval(ckpt_path: Path, data_root: str | None, device: torch.device, use_ema: bool):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Cfg.from_dict(ck["cfg"])
    if data_root:
        cfg.data.root = data_root
    model = Refiner(cfg.model, ck["num_ids"]).to(device)
    model.load_state_dict(ck["model"])
    if use_ema and "ema" in ck:
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(ck["ema"][n])
    return cfg, model, ck


def main(argv: list[str] | None = None) -> int:
    from frm.train import pick_device, pick_dtype

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--ckpt", default="best", help="best | last | path")
    ap.add_argument("--data", default=None, help="override dataset root")
    ap.add_argument("--final", action="store_true", help="required to read split 'test'")
    ap.add_argument("--max_versions", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--submission", default=None, help="write Kaggle submission.json here")
    args = ap.parse_args(argv)
    if args.split == "test" and not args.final:
        print("refusing to read split 'test' without --final (decisions use dev)", file=sys.stderr)
        return 2
    run_dir = Path(args.run_dir)
    ckpt = run_dir / f"{args.ckpt}.pt" if args.ckpt in ("best", "last") else Path(args.ckpt)
    device = pick_device(args.device)
    cfg, model, ck = load_for_eval(ckpt, args.data, device, use_ema=args.ckpt == "last")
    dtype = pick_dtype(cfg.train.precision, device)
    if cfg.train.compile:
        model.compile_cycles()
    if args.max_versions:
        cfg.eval.max_versions = args.max_versions
    if args.rounds:
        cfg.eval.rounds = args.rounds
    ds = Dataset(cfg.data.root)
    res = evaluate_split(model, ds, args.split, cfg.eval, device, dtype, cfg.eval.counterfactuals, bool(args.submission))
    if args.submission:
        Path(args.submission).write_text(json.dumps(res.pop("attempts")))
    res["ckpt"], res["step"] = str(ckpt), ck.get("step")
    out = run_dir / "eval" / f"{args.split}_step{ck.get('step')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "by_round"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
