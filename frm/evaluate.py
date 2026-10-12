"""Evaluation: per-round voting over augmented versions, ARC pass@k, fork and depth instruments (DESIGN.md §7).

    python -m frm.evaluate --run_dir runs/X --split dev [--ckpt best|last|PATH] [--final] [--max_versions N]

Voting follows TRM (trm-samsung/evaluators/arc.py:107-177): each version's prediction is cropped, de-augmented and
hashed; candidates rank by (votes, mean sigmoid(q)); the top two are the attempts. Score = mean over tasks of the
fraction of their test inputs with a correct attempt (official ARC metric). Reading split ``test`` needs ``--final``.
Only versions whose puzzle-embedding row was trained vote (``loader.eval_versions``): TRM trains every version it votes
with, while a budgeted run trains a fraction of them, and the untrained rest would vote blind (the result's
``versions`` block reports coverage and voters). Per-task outcomes are stored so arms can be compared with paired
statistics (``scripts/compare.py``).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from frm.config import Cfg, EvalCfg
from frm.data.augment import Version
from frm.data.encode import crop
from frm.data.loader import Dataset, eval_batches, eval_versions
from frm.model.refiner import Refiner, RoundOut
from frm.objective import row_ce, row_exact


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
        fork=cut(out.fork),
        fork_logit=cut(out.fork_logit),
        cf_logits=cut(out.cf_logits),
        gate_rows=cut(out.gate_rows),
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



def _auc(score: np.ndarray, pos: np.ndarray) -> float | None:
    """P(a positive outscores a negative), ties counted half (Mann-Whitney U / (n_pos * n_neg))."""
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if not n_pos or not n_neg:
        return None
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])  # tied scores share their mean rank
    ranks = (first + (counts + 1) / 2)[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def gate_skill(records: list[tuple[int, int, np.ndarray, np.ndarray]], rhos=(0.1, 0.25, 0.4, 0.6)) -> dict:
    """The gate's held-out ranking from (batch, round, gate logits, benefits) of runs where every row forked.

    ``auc``: gate logit vs "benefit > 0" (0.5 = no skill). ``top{rho}``: forking only the top rho rows by logit in each
    batch and round (what ``fork_top`` does) keeps ``share`` of the summed per-round benefit (None if the sum is not
    positive; random choice keeps rho), with ``precision`` and mean ``benefit`` of the kept rows. ``by_round``: the
    same base quantities at round 1, 2-4 and 5+ (round 1 starts from the init carry)."""

    def summary(lg, b):
        return {"rows": int(len(b)), "benefit": float(b.mean()), "base_rate": float((b > 0).mean()),
                "auc": _auc(lg, b > 0)}

    logit = np.concatenate([r[2] for r in records])
    ben = np.concatenate([r[3] for r in records])
    rnd = np.concatenate([np.full(len(r[2]), r[1]) for r in records])
    out = summary(logit, ben)
    total = float(ben.sum())
    for rho in rhos:
        kept = pos = n_top = 0.0
        for _, _, lg, b in records:
            idx = np.argsort(-lg, kind="stable")[: min(len(lg), math.ceil(rho * len(lg)))]
            kept, pos, n_top = kept + float(b[idx].sum()), pos + float((b[idx] > 0).sum()), n_top + len(idx)
        out[f"top{rho:g}"] = {"share": kept / total if total > 0 else None, "precision": pos / n_top,
                              "benefit": kept / n_top}
    out["by_round"] = {name: summary(logit[sel], ben[sel])
                       for name, sel in (("r1", rnd == 0), ("r2-4", (rnd >= 1) & (rnd <= 3)), ("r5+", rnd >= 4))
                       if sel.any()}
    return out


@torch.no_grad()
def run_rounds(
    model: Refiner, inputs, ids, rounds: int, dtype, cf=None, perm=None, reset_at=None, seed: int = 0, fork_mode="gate",
    state_noise: float = 0.0, fork_top: float = 0.0, n_valid: int | None = None, gate_labels: bool = True,
) -> list[RoundOut]:
    """Run ``rounds`` rounds from the init carry; one RoundOut per round, without its carry (only the next round needs
    it, and 16 kept carries would cost ~16x the state memory). Children's noise comes from a generator seeded with
    ``seed``, so a batch evaluates identically every time and its counterfactual reruns see the same noise.
    ``fork_mode`` as in ``Refiner.round`` (no exploration at evaluation); ``fork_top`` > 0 forks the batch's top
    fraction by gate logit instead of logit > 0. ``state_noise`` > 0 adds Gaussian noise of that many RMS units to the
    latent z between rounds (PTRM's test-time noisy trajectories, arXiv 2605.19943)."""
    S = int(round(inputs.shape[1] ** 0.5))
    carry = model.init_carry(inputs.shape[0], S, inputs.device, dtype)
    gen = torch.Generator(device=inputs.device).manual_seed(seed)
    outs = []
    for r in range(1, rounds + 1):
        if reset_at is not None and r == reset_at:
            carry = model.init_carry(inputs.shape[0], S, inputs.device, dtype)
        out = model.round(carry, inputs, ids, cf=cf, sham_perm=perm, gen=gen, fork_mode=fork_mode, fork_top=fork_top,
                          n_valid=n_valid, gate_labels=gate_labels)
        carry = out.carry
        if state_noise > 0:
            z = carry.z.float()
            eps = torch.randn(z.shape, device=z.device, generator=gen)
            z = z + state_noise * z.square().mean(-1, keepdim=True).sqrt() * eps
            carry = dataclasses.replace(carry, z=z.to(carry.z.dtype))
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
    gate_labels: bool = True,
) -> dict:
    """Per-round voted pass@k plus the fork and depth instruments of DESIGN.md §7 on one split. ``gate_labels=False``
    skips the no-fork continuation behind the gate instruments (answers only, e.g. the submission).

    With ``ecfg.fork_top >= 1`` every row forks at every round and carries a gate label, so ``fork.skill`` measures the
    gate's held-out ranking without selection bias: AUC of the gate logit for "the fork beat the TRM round", and for a
    fork rate rho the share of the total per-round CE benefit that the top rho rows by logit (per batch and round)
    would keep (random choice keeps rho)."""
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
    fork_r = np.zeros(R)  # rows that forked at round r
    fork_b: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])  # bucket -> [forked row-rounds, row-rounds]
    cf_acc: dict[str, float] = defaultdict(float)
    cf_rows = 0
    task_rows: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # task -> [labelled rows, exact at the last round]
    skill: list[tuple[int, int, np.ndarray, np.ndarray]] = []  # (batch, round, gate logits, benefits): fork_top >= 1
    skill_batch = 0
    buckets = [S for S in ds.buckets if ds.has_split(S, split)]
    cf_budget = ecfg.cf_rows // max(1, len(buckets))  # counterfactual rows per bucket: every canvas size is represented
    trained = model.puzzle_emb.trained().cpu().numpy()
    coverage, voters, blind = [], [], 0  # per evaluated task: fraction of its versions trained, versions that vote
    for S in buckets:
        d = ds.split_data(S, split)
        for t, tid in enumerate(d.tasks):
            if tid in tasks:
                v0, nv = int(d.vstart[t]), int(d.vcount[t])
                coverage.append(float(trained[v0 : v0 + nv].mean()))
                voters.append(len(eval_versions(d, t, ecfg.max_versions, trained)))
                blind += not trained[v0 : v0 + nv].any()
    for S in buckets:
        rows = _rows_per_batch(model, S, ecfg)
        cf_done = 0
        for bi, batch in enumerate(eval_batches(ds, split, S, rows, ecfg.max_versions, set(tasks), trained)):
            keys = batch["keys"]
            n = len(keys)
            seed = S * 1_000_003 + bi  # reproducible noise per batch, shared by its counterfactual reruns
            inputs = _pad(torch.from_numpy(batch["inputs"]).long().to(device), rows)
            ids = _pad(torch.from_numpy(batch["puzzle_ids"]).long().to(device), rows)
            labels = torch.from_numpy(batch["labels"]).long().to(device)
            labeled = (labels != -100).any(-1)
            top = ecfg.fork_top
            outs = [_trim(o, n) for o in run_rounds(model, inputs, ids, R, dtype, seed=seed, fork_top=top,
                                                    n_valid=n, gate_labels=gate_labels)]
            extra = [  # PTRM-style noisy trajectories: extra voters only, instruments stay on the clean run
                [_trim(o, n) for o in run_rounds(model, inputs, ids, R, dtype, seed=seed + 7919 * s,
                                                 state_noise=ecfg.state_noise, fork_top=top, n_valid=n,
                                                 gate_labels=False)]
                for s in range(1, ecfg.noise_samples)
            ]
            n_rows += n
            n_labeled += int(labeled.sum())
            exact = torch.stack([row_exact(o.logits, labels) & labeled for o in outs])  # [R, n]
            exact_np = exact.cpu().numpy()
            exact_r += exact_np.sum(1)
            lab_np = labeled.cpu().numpy()
            for b, key in enumerate(keys):  # per task: labelled rows (versions x test inputs) exact at the last round
                if lab_np[b]:
                    task_rows[key[0]][0] += 1
                    task_rows[key[0]][1] += int(exact_np[-1, b])
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
            extra_np = [([o.logits.argmax(-1).to(torch.uint8).cpu().numpy() for o in e],
                         [torch.sigmoid(o.q).float().cpu().numpy() for o in e]) for e in extra]
            last = outs[-1]
            child_np = cq_np = forked_np = None
            if K > 0:  # children instruments over the rows that forked at the round in question
                forks = torch.stack([o.fork for o in outs]).sum(1).cpu().numpy()
                fork_r += forks
                fork_b[S][0] += float(forks.sum())
                fork_b[S][1] += n * R
                f = last.fork
                forked_np = f.cpu().numpy()
                child_np = last.child_logits.argmax(-1).to(torch.uint8).cpu().numpy()
                cq_np = torch.sigmoid(last.child_q).float().cpu().numpy()
                fork["forked"] += float(f.sum())
                fork["forked_labeled"] += float((f & labeled).sum())
                fork["parent_mass"] += float(last.join_alpha[f, 0].sum())
                fork["join_entropy"] += float(last.join_entropy[f].sum())
                child_exact = row_exact(last.child_logits, labels[:, None]).any(-1) & labeled & f
                fork["child_exact_any"] += float(child_exact.sum())
                fork["oracle"] += float((child_exact | (exact[-1] & f)).sum())
                fork["oracle_base"] += float((exact[-1] & f).sum())
                for r, o in enumerate(outs):  # adoption: pre-fork parent wrong and some child right -> joined parent right?
                    chance = labeled & o.fork & ~row_exact(o.pre_logits, labels)
                    chance &= row_exact(o.child_logits, labels[:, None]).any(-1)
                    fork["adopt_den"] += float(chance.sum())
                    fork["adopt_num"] += float((chance & row_exact(o.logits, labels)).sum())
                    if o.fork_logit is not None:  # the gate's own rule, whatever decided this run
                        fork["gate_says"] += float((o.fork_logit > 0).sum())
                    if o.gate_rows is not None:  # did the forks the gate chose beat the TRM round on held-out rows?
                        g = o.gate_rows & labeled
                        if bool(g.any()):
                            b = row_ce(o.cf_logits[g], labels[g]) - row_ce(o.logits[g], labels[g])
                            fork["gate_n"] += float(g.sum())
                            fork["gate_hit"] += float((b > 0).sum())
                            fork["benefit"] += float(b.sum())
                            if top >= 1:  # every row forked: an unbiased (logit, benefit) sample of this batch-round
                                skill.append((skill_batch, r, o.fork_logit[g].cpu().numpy(), b.cpu().numpy()))
                skill_batch += 1
            for b, (tid, ti, pid) in enumerate(keys):
                v = versions.get(pid)
                if v is None:
                    v = versions[pid] = Version.parse(ds.identifiers[pid])[1]
                for r in range(R):
                    per_round[r].add((tid, ti), v.invert(crop(preds_np[r][b], S)), float(qs_np[r][b]))
                    for ep, eq in extra_np:
                        per_round[r].add((tid, ti), v.invert(crop(ep[r][b], S)), float(eq[r][b]))
                if child_np is not None and forked_np[b]:
                    answers = [v.invert(crop(child_np[b, k], S)) for k in range(K)]
                    fork["distinct"] += len({_hash(a) for a in answers})  # distinct cropped answers (what is voted)
                    for k in range(K):
                        child_votes.add((tid, ti), answers[k], float(cq_np[b, k]))
            if counterfactuals and K > 0 and cf_done < cf_budget and bool(labeled.any()):
                cf_done += n
                cf_rows += int(labeled.sum())
                cf_acc["base"] += float(exact[-1].sum())
                perm = sham_perm([k[0] for k in keys] + [keys[0][0]] * (rows - n), device)
                variants = [
                    ("nofork", {"cf": "nofork"}), ("sham", {"cf": "sham", "perm": perm}), ("reset@4", {"reset_at": 4})
                ]
                if model.gate is not None:  # every row forks / no row forks, against the gate's own choice
                    variants += [("allfork", {"fork_mode": "all"}), ("gateoff", {"fork_mode": "none"})]
                for name, kw in variants:
                    o = _trim(run_rounds(model, inputs, ids, R, dtype, seed=seed, fork_top=top, n_valid=n,
                                         gate_labels=False, **kw)[-1], n)
                    cf_acc[name] += float((row_exact(o.logits, labels) & labeled).sum())

    res: dict = {"split": split, "rows": n_rows, "rounds": R}
    scored = [score(v, tasks) for v in per_round]
    final = scored[-1]
    res.update({k: final[k] for k in ("pass@1", "pass@2", "n_tasks")})
    res["per_task"] = {"pass@2": final["per_task"],  # row_exact: unvoted, every voting version's rows
                       "row_exact": {t: round(e / n, 5) for t, (n, e) in sorted(task_rows.items())}}
    res["versions"] = {  # the vote rule's inputs: compare arms only on records that carry this block
        "rule": "trained",
        "max": ecfg.max_versions,
        "coverage": float(np.mean(coverage)) if coverage else float("nan"),
        "voters": float(np.mean(voters)) if voters else float("nan"),
        "blind_tasks": int(blind),
    }
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
        forked, forked_lab = fork["forked"], fork["forked_labeled"]

        def per(key: str, den: float) -> float | None:
            return fork[key] / den if den else None

        # Denominators: rows that forked at the final round (forked_rows; forked_labeled for the labelled-row rates).
        res["fork"] = {
            "rate": float(fork_r.sum() / max(n_rows * R, 1)),
            "top": ecfg.fork_top,  # 0: the gate's own decisions (logit > 0); > 0: the top fraction by gate logit
            "rate_by_round": (fork_r / max(n_rows, 1)).round(5).tolist(),
            "rate_by_bucket": {str(S): round(f / max(t, 1), 5) for S, (f, t) in sorted(fork_b.items())},
            "forked_rows": int(forked),
            "forked_labeled": int(forked_lab),
            "parent_mass": per("parent_mass", forked),
            "join_entropy": per("join_entropy", forked),
            "distinct": per("distinct", forked),
            "child_exact_any": per("child_exact_any", forked_lab),
            "oracle_gain": (fork["oracle"] - fork["oracle_base"]) / forked_lab if forked_lab else None,
            "adopt": fork["adopt_num"] / fork["adopt_den"] if fork["adopt_den"] else None,
            "adopt_chances": int(fork["adopt_den"]),
            "children_only": {k: children[k] for k in ("pass@1", "pass@2", "n_tasks")},
        }
        if model.gate is not None:  # the gate's own rule (logit > 0), also when fork_top decided the run
            res["fork"]["gate_rate"] = fork["gate_says"] / max(n_rows * R, 1)
        if fork["gate_n"]:  # the gate's chosen forks, all rounds, labelled rows: benefit = CE(TRM round) - CE(fork)
            res["fork"]["gate_precision"] = fork["gate_hit"] / fork["gate_n"]
            res["fork"]["benefit"] = fork["benefit"] / fork["gate_n"]
            res["fork"]["gate_rows"] = int(fork["gate_n"])
        if skill:
            res["fork"]["skill"] = gate_skill(skill)
        if ecfg.child_vote_weight > 0:
            merged = score(per_round[-1].merged(child_votes, ecfg.child_vote_weight), tasks)
            res["fork"]["with_child_votes"] = {k: merged[k] for k in ("pass@1", "pass@2", "n_tasks")}
        if cf_rows:
            names = [k for k in ("nofork", "sham", "reset@4", "allfork", "gateoff") if k in cf_acc]
            res["cf"] = {k: (cf_acc[k] - cf_acc["base"]) / cf_rows for k in names}
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
    ap.add_argument("--noise_samples", type=int, default=None, help="extra noisy trajectories as voters (PTRM)")
    ap.add_argument("--state_noise", type=float, default=None, help="their z noise in RMS units")
    ap.add_argument("--child_vote_weight", type=float, default=None, help="children's answers vote with this weight")
    ap.add_argument("--fork_top", type=float, default=None,
                    help="gated models: fork the top fraction of each batch by gate logit instead of logit > 0")
    ap.add_argument("--tag", default="", help="suffix for the output file (several settings on one checkpoint)")
    ap.add_argument("--no_cf", action="store_true", help="skip the counterfactual reruns (faster diagnostics)")
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
    for name in ("max_versions", "max_tasks", "rounds", "noise_samples", "state_noise", "child_vote_weight",
                 "fork_top"):
        if getattr(args, name) is not None:
            setattr(cfg.eval, name, getattr(args, name))
    ds = Dataset(cfg.data.root)
    cf = cfg.eval.counterfactuals and not args.no_cf
    res = evaluate_split(model, ds, args.split, cfg.eval, device, dtype, cf, bool(args.submission))
    if args.submission:
        Path(args.submission).write_text(json.dumps(res.pop("attempts")))
    res["ckpt"], res["step"], res["row_rounds"] = str(ckpt), ck.get("step"), ck.get("row_rounds")
    res["pflops"] = ck["flops"] / 1e15 if "flops" in ck else None
    out = run_dir / "eval" / f"{args.split}_step{ck.get('step')}{'_' + args.tag if args.tag else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("by_round", "per_task")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
