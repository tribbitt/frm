"""Evaluation: per-round voting over augmented versions, ARC pass@k, fork and depth instruments (DESIGN.md §7).

    python -m frm.evaluate --run_dir runs/X --split dev [--ckpt best|last|PATH] [--final] [--max_versions N]

Voting follows TRM (trm-samsung/evaluators/arc.py:107-177): each version's prediction is cropped, de-augmented and
hashed; candidates rank by (votes, mean sigmoid(q)); the top two are the attempts. Score = mean over tasks of the
fraction of their test inputs with a correct attempt (official ARC metric). Reading split ``test`` needs ``--final``.
Per-task outcomes are stored so arms can be compared with paired statistics (``scripts/compare.py``).
"""

from __future__ import annotations

import argparse
import dataclasses
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
from frm.model.refiner import Refiner, RoundOut
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


def score(votes: Votes, tasks: dict[str, dict]) -> dict:
    """Official ARC scoring for tasks with known outputs, plus per-task pass@2 for paired comparisons."""
    p1, per_task = [], {}
    for tid, task in tasks.items():
        s1 = s2 = 0.0
        tests = task["test"]
        if any(t.get("output") is None for t in tests):
            continue
        for i, t in enumerate(tests):
            sol = np.asarray(t["output"], dtype=np.uint8)
            att = votes.ranked((tid, i))[:2]
            ok = [a.shape == sol.shape and bool((a == sol).all()) for a in att]
            s1 += float(bool(ok) and ok[0])
            s2 += float(any(ok))
        p1.append(s1 / len(tests))
        per_task[tid] = s2 / len(tests)
    return {
        "pass@1": float(np.mean(p1)) if p1 else float("nan"),
        "pass@2": float(np.mean(list(per_task.values()))) if per_task else float("nan"),
        "n_tasks": len(per_task),
        "per_task": per_task,
    }


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


def _pad(t: torch.Tensor, rows: int) -> torch.Tensor:
    """Repeat row 0 up to ``rows`` so every batch of a bucket has one shape (one compiled graph)."""
    return t if t.shape[0] == rows else torch.cat([t, t[:1].expand(rows - t.shape[0], *t.shape[1:])])


def _trim(out: RoundOut, n: int) -> RoundOut:
    """Drop the padding rows added by :func:`_pad`."""
    if out.logits.shape[0] == n:
        return out

    def cut(t):
        return None if t is None else t[:n]

    return RoundOut(
        logits=out.logits[:n],
        q=out.q[:n],
        child_logits=cut(out.child_logits),
        child_q=cut(out.child_q),
        join_alpha=cut(out.join_alpha),
        join_entropy=cut(out.join_entropy),
        carry=None,
        pre_logits=cut(out.pre_logits),
    )


def sham_perm(tasks: list[str], device: torch.device) -> torch.Tensor:
    """perm[b] = a row of a *different* task (searching from half a batch away); the next row if the batch has one task.

    Neighbouring eval rows are other versions of the same test input, so rolling by one would hand a row children that
    solve its own task in another orientation."""
    B = len(tasks)
    perm = []
    for b in range(B):
        offsets = ((b + s) % B for s in range(B // 2, B // 2 + B))
        perm.append(next((j for j in offsets if tasks[j] != tasks[b]), (b + 1) % B))
    return torch.tensor(perm, device=device)


@torch.no_grad()
def run_rounds(
    model: Refiner, inputs, ids, rounds: int, dtype, cf=None, perm=None, reset_at=None, seed: int = 0
) -> list[RoundOut]:
    """Run ``rounds`` rounds from the init carry; one RoundOut per round, without its carry (only the next round needs
    it, and 16 kept carries would cost ~16x the state memory). Children's noise comes from a generator seeded with
    ``seed``, so a batch evaluates identically every time and its counterfactual reruns see the same noise."""
    S = int(round(inputs.shape[1] ** 0.5))
    carry = model.init_carry(inputs.shape[0], S, inputs.device, dtype)
    gen = torch.Generator(device=inputs.device).manual_seed(seed)
    outs = []
    for r in range(1, rounds + 1):
        if reset_at is not None and r == reset_at:
            carry = model.init_carry(inputs.shape[0], S, inputs.device, dtype)
        out = model.round(carry, inputs, ids, cf=cf, sham_perm=perm, gen=gen)
        carry = out.carry
        outs.append(dataclasses.replace(out, carry=None))
    return outs


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
    """Per-round voted pass@k plus the fork and depth instruments of DESIGN.md §7 on one split."""
    model.eval()
    R, K = ecfg.rounds, model.cfg.K
    tasks = ds.tasks(split)
    if split != "test" and ecfg.max_tasks and len(tasks) > ecfg.max_tasks:  # test (and Kaggle) is always complete
        keep = sorted(tasks, key=lambda t: hashlib.sha1(t.encode()).hexdigest())[: ecfg.max_tasks]
        tasks = {t: tasks[t] for t in keep}
    versions: dict[int, Version] = {}
    per_round = [Votes() for _ in range(R)]
    child_votes = Votes()
    exact_r = np.zeros(R)  # labelled rows exact at round r
    first_solve = np.zeros(R)  # labelled rows exact for the first time at round r
    regress = np.zeros(R)  # labelled rows exact at r-1 and wrong at r
    by_k: dict[int, np.ndarray] = {}  # composites: exact rows by round, per composition depth k
    n_k: dict[int, int] = defaultdict(int)
    n_rows = n_labeled = 0
    fork: dict[str, float] = defaultdict(float)
    cf_acc: dict[str, float] = defaultdict(float)
    cf_rows = 0
    buckets = [S for S in ds.buckets if ds.has_split(S, split)]
    cf_budget = ecfg.cf_rows // max(1, len(buckets))  # counterfactual rows per bucket: every canvas size is represented
    for S in buckets:
        rows = _rows_per_batch(model, S, ecfg)
        cf_done = 0
        for bi, batch in enumerate(eval_batches(ds, split, S, rows, ecfg.max_versions, set(tasks))):
            keys = batch["keys"]
            n = len(keys)
            seed = S * 1_000_003 + bi  # reproducible noise per batch, shared by its counterfactual reruns
            inputs = _pad(torch.from_numpy(batch["inputs"]).long().to(device), rows)
            ids = _pad(torch.from_numpy(batch["puzzle_ids"]).long().to(device), rows)
            labels = torch.from_numpy(batch["labels"]).long().to(device)
            labeled = (labels != -100).any(-1)
            outs = [_trim(o, n) for o in run_rounds(model, inputs, ids, R, dtype, seed=seed)]
            n_rows += n
            n_labeled += int(labeled.sum())
            exact = torch.stack([row_exact(o.logits, labels) & labeled for o in outs])  # [R, n]
            exact_np = exact.cpu().numpy()
            exact_r += exact_np.sum(1)
            seen = np.zeros(n, dtype=bool)
            for r in range(R):
                first_solve[r] += (exact_np[r] & ~seen).sum()
                if r:
                    regress[r] += (exact_np[r - 1] & ~exact_np[r]).sum()
                seen |= exact_np[r]
            comp = np.array([k[0].startswith("comp:") for k in keys])
            if comp.any():  # composition depth k = 1 + number of intermediates (buckets without any store none)
                ks = batch["subgoal_mask"].sum(-1) + 1 if "subgoal_mask" in batch else np.ones(n, dtype=np.int64)
                for k in np.unique(ks[comp]):
                    sel = comp & (ks == k)
                    by_k.setdefault(int(k), np.zeros(R))
                    by_k[int(k)] += exact_np[:, sel].sum(1)
                    n_k[int(k)] += int(sel.sum())
            preds_np = [o.logits.argmax(-1).to(torch.uint8).cpu().numpy() for o in outs]
            qs_np = [torch.sigmoid(o.q).float().cpu().numpy() for o in outs]
            last = outs[-1]
            child_np = cq_np = None
            if K > 0:
                child_np = last.child_logits.argmax(-1).to(torch.uint8).cpu().numpy()
                cq_np = torch.sigmoid(last.child_q).float().cpu().numpy()
                fork["parent_mass"] += float(last.join_alpha[:, 0].sum())
                fork["join_entropy"] += float(last.join_entropy.sum())
                child_exact = row_exact(last.child_logits, labels[:, None]).any(-1) & labeled
                fork["child_exact_any"] += float(child_exact.sum())
                fork["oracle"] += float((child_exact | exact[-1]).sum())
                for o in outs:  # adoption: pre-fork parent wrong and some child right -> is the joined parent right?
                    chance = labeled & ~row_exact(o.pre_logits, labels)
                    chance &= row_exact(o.child_logits, labels[:, None]).any(-1)
                    fork["adopt_den"] += float(chance.sum())
                    fork["adopt_num"] += float((chance & row_exact(o.logits, labels)).sum())
            for b, (tid, ti, pid) in enumerate(keys):
                v = versions.get(pid)
                if v is None:
                    v = versions[pid] = Version.parse(ds.identifiers[pid])[1]
                for r in range(R):
                    per_round[r].add((tid, ti), v.invert(crop(preds_np[r][b], S)), float(qs_np[r][b]))
                if child_np is not None:
                    answers = [v.invert(crop(child_np[b, k], S)) for k in range(K)]
                    fork["distinct"] += len({_hash(a) for a in answers})  # distinct cropped answers (what is voted)
                    for k in range(K):
                        child_votes.add((tid, ti), answers[k], float(cq_np[b, k]))
            if counterfactuals and K > 0 and cf_done < cf_budget and bool(labeled.any()):
                cf_done += n
                cf_rows += int(labeled.sum())
                cf_acc["base"] += float(exact[-1].sum())
                perm = sham_perm([k[0] for k in keys] + [keys[0][0]] * (rows - n), device)
                variants = (("nofork", {"cf": "nofork"}), ("sham", {"cf": "sham", "perm": perm}), ("reset@4", {"reset_at": 4}))
                for name, kw in variants:
                    o = _trim(run_rounds(model, inputs, ids, R, dtype, seed=seed, **kw)[-1], n)
                    cf_acc[name] += float((row_exact(o.logits, labels) & labeled).sum())

    res: dict = {"split": split, "rows": n_rows, "rounds": R}
    scored = [score(v, tasks) for v in per_round]
    final = scored[-1]
    res.update({k: final[k] for k in ("pass@1", "pass@2", "n_tasks")})
    res["per_task"] = {"pass@2": final["per_task"]}
    lab = max(n_labeled, 1)
    res["by_round"] = {
        "pass@1": [s["pass@1"] for s in scored],
        "pass@2": [s["pass@2"] for s in scored],
        "row_exact": (exact_r / lab).round(5).tolist(),
        "first_solve": (first_solve / lab).round(5).tolist(),
        "regress": (regress / lab).round(5).tolist(),
    }
    if R >= 4:
        res["per_task"]["pass@2_r4"] = scored[3]["per_task"]
        res["late_gain"] = final["pass@2"] - scored[3]["pass@2"]
        res["late_gain_rows"] = float(exact_r[-1] - exact_r[3]) / lab
    if by_k:
        res["by_k"] = {k: {"rows": n_k[k], "row_exact": (by_k[k] / n_k[k]).round(5).tolist()} for k in sorted(by_k)}
    if K > 0:
        children = score(child_votes, tasks)
        res["fork"] = {
            "parent_mass": fork["parent_mass"] / max(n_rows, 1),
            "join_entropy": fork["join_entropy"] / max(n_rows, 1),
            "distinct": fork["distinct"] / max(n_rows, 1),
            "child_exact_any": fork["child_exact_any"] / lab,
            "oracle_gain": (fork["oracle"] - exact_r[-1]) / lab,
            "adopt": fork["adopt_num"] / fork["adopt_den"] if fork["adopt_den"] else None,
            "adopt_chances": int(fork["adopt_den"]),
            "children_only": {k: children[k] for k in ("pass@1", "pass@2", "n_tasks")},
        }
        if ecfg.child_vote_weight > 0:
            merged = score(per_round[-1].merged(child_votes, ecfg.child_vote_weight), tasks)
            res["fork"]["with_child_votes"] = {k: merged[k] for k in ("pass@1", "pass@2", "n_tasks")}
        if cf_rows:
            res["cf"] = {k: (cf_acc[k] - cf_acc["base"]) / cf_rows for k in ("nofork", "sham", "reset@4")}
            res["cf"]["rows"] = cf_rows
            res["cf"]["base_exact"] = cf_acc["base"] / cf_rows
    if keep_attempts:
        votes = per_round[-1]
        if K > 0 and ecfg.child_vote_weight > 0:
            votes = votes.merged(child_votes, ecfg.child_vote_weight)
        res["attempts"] = attempts(votes, tasks)
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
    ap.add_argument("--max_tasks", type=int, default=None)
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
    for name in ("max_versions", "max_tasks", "rounds"):
        if getattr(args, name) is not None:
            setattr(cfg.eval, name, getattr(args, name))
    ds = Dataset(cfg.data.root)
    res = evaluate_split(model, ds, args.split, cfg.eval, device, dtype, cfg.eval.counterfactuals, bool(args.submission))
    if args.submission:
        Path(args.submission).write_text(json.dumps(res.pop("attempts")))
    res["ckpt"], res["step"] = str(ckpt), ck.get("step")
    out = run_dir / "eval" / f"{args.split}_step{ck.get('step')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("by_round", "per_task")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
