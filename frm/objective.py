"""The frm objective (docs/OBJECTIVE.md §3). Per round, per row b, averaged over the rows of the step:

    L_b = CE(parent, y*)                                                       deep supervision (post-join readout)
        + lambda_wta * [(1 - eps) * sum_pi w_pi * cost_pi + eps * mean_k CE(child_k, y*)]   children: set matching
        + lambda_q * [BCE(q, exact) + mean_k BCE(q_k, exact_k)]                stop / score heads

    cost_pi = CE(child_pi(final), y*) + (lambda_sub / lambda_wta) * mean_{t in G_pi} CE(child_pi(t), g_t)
    w = softmax_pi(-cost / tau) over the valid assignments, without gradient

An assignment pi maps the targets {final answer y*} + {intermediate grids g_t of a composite row} one-to-one onto
distinct children; it must match y* and as many valid intermediates as there are spare children (min(|G|, K-1)).
Rows without intermediates have K assignments (y* -> child k), and the term is annealed relaxed winner-takes-all.
On composite rows the one-to-one constraint forces *different* children to produce the different grids.
tau anneals log-linearly from tau_start to tau_end over the first tau_anneal_frac of training. CE is TRM's stablemax
cross-entropy, token-mean per row over non-ignored labels.

With the fork gate the child terms apply to the rows that forked; a declined row's loss is exactly the K = 0 loss.
The gate is trained on its label rows (forked, own join, parent slot kept) by a separate mean over those rows:

    + lambda_gate * mean_{gate rows} BCE(gate_logit, [CE(no-fork continuation, y*) - CE(parent, y*) > price])
"""

from __future__ import annotations

import itertools
import math
from functools import cache

import torch
import torch.nn.functional as F

from frm.config import ObjectiveCfg
from frm.model.refiner import RoundOut

IGNORE = -100


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Only the 1/(1-x) branch is clamped: unclamped it is inf/NaN at x >= 1, which where() would pass into the gradient.
    # The x+1 branch must stay unclamped, or the gradient at exactly x = 0 (clamp's boundary) would be 0.
    s = torch.where(x < 0, 1.0 / (1.0 - x.clamp(max=0)), x + 1.0)
    return torch.log(s / s.sum(dim=dim, keepdim=True))


def token_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Stablemax CE per token (0 where ignored). logits [..., L, V], labels broadcastable to [..., L]."""
    dtype = torch.float32 if logits.device.type == "mps" else torch.float64
    shape = torch.broadcast_shapes(logits.shape[:-1], labels.shape)
    logp = log_stablemax(logits.to(dtype)).expand(*shape, logits.shape[-1])
    labels = labels.expand(shape)
    valid = labels != IGNORE
    picked = logp.gather(-1, torch.where(valid, labels, 0).long().unsqueeze(-1)).squeeze(-1)
    return torch.where(valid, -picked, 0.0)


def row_ce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Token-mean stablemax CE per row: [..., L, V] x [..., L] (broadcast) -> [...] (float32). ``weights``
    (broadcastable to [..., L]) turn it into a weighted mean over the supervised positions."""
    valid = labels != IGNORE
    if weights is None:
        return (token_ce(logits, labels).sum(-1) / valid.sum(-1).clamp_min(1)).float()
    ce = token_ce(logits, labels)
    w = weights.to(ce.dtype) * valid
    return ((ce * w).sum(-1) / w.sum(-1).clamp_min(1e-9)).float()


def row_exact(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Whole-row exact match over supervised positions (False for rows with no supervised position)."""
    valid = labels != IGNORE
    hit = (logits.argmax(-1) == labels) | ~valid
    return hit.all(-1) & valid.any(-1)


def tau_at(cfg: ObjectiveCfg, progress: float) -> float:
    frac = min(1.0, max(0.0, progress) / max(cfg.tau_anneal_frac, 1e-9))
    return math.exp(math.log(cfg.tau_start) + frac * (math.log(cfg.tau_end) - math.log(cfg.tau_start)))


@cache
def _assignment_table(K: int, M: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """[P, 1+M] child index of each target (0 = final, 1..M = intermediates; -1 = unmatched) for every one-to-one map
    of the final target plus any subset of at most K-1 intermediates onto distinct children. Cached per device, so a
    training step never copies it host -> device (which would sync the stream between forward and backward)."""
    rows = []
    for j in range(min(M, K - 1) + 1):
        for inter in itertools.combinations(range(1, M + 1), j):
            targets = (0, *inter)
            for kids in itertools.permutations(range(K), len(targets)):
                row = [-1] * (M + 1)
                for t, k in zip(targets, kids):
                    row[t] = k
                rows.append(row)
    return torch.tensor(rows, dtype=torch.long, device=device)


def set_match(
    ce_final: torch.Tensor,
    ce_sub: torch.Tensor | None,
    mask: torch.Tensor | None,
    tau: float,
    sub_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Annealed one-to-one matching of targets to children.

    ce_final [B, K]: CE of each child against y*; ce_sub [B, K, M]: against each intermediate; mask [B, M] valid
    intermediates. Returns (sum_pi w_pi cost_pi [B], best assignment [B, 1+M] child per target, -1 = unmatched)."""
    B, K = ce_final.shape
    M = 0 if ce_sub is None else ce_sub.shape[-1]
    A = _assignment_table(K, M, ce_final.device)  # [P, 1+M]
    cost = ce_final[:, A[:, 0]]  # [B, P]
    valid = torch.ones_like(cost, dtype=torch.bool)
    if M:
        used = A[:, 1:] >= 0  # [P, M]
        n_used = used.sum(-1)  # [P]
        want = mask.sum(-1).clamp(max=K - 1)  # [B] intermediates that must be matched
        valid = (n_used[None] == want[:, None]) & ~(used[None] & ~mask[:, None, :]).any(-1)
        g = ce_sub[:, A[:, 1:].clamp(min=0), torch.arange(M, device=ce_sub.device)[None, :]]  # [B, P, M]
        cost = cost + sub_ratio * (g * used[None]).sum(-1) / n_used.clamp(min=1)
    w = torch.softmax((-cost.detach() / tau).masked_fill(~valid, -math.inf), dim=-1)
    return (w * cost).sum(-1), A[w.argmax(-1)]


class Objective:
    def __init__(self, cfg: ObjectiveCfg):
        if cfg.lambda_wta <= 0:
            raise ValueError("lambda_wta must be > 0: it scales the children's set term")
        self.cfg = cfg

    def __call__(
        self,
        out: RoundOut,
        batch: dict[str, torch.Tensor],
        progress: float,
        has_subgoals: bool | None = None,
        price: float | None = None,
    ):
        """``has_subgoals``: host-side flag for whether any row carries sub-goals (None: read it from the batch,
        which costs a device sync). ``price``: the fork price in nats per supervised token (None: cfg.fork_price);
        the trainer passes its controller's current value."""
        cfg = self.cfg
        labels = batch["labels"]
        nan = float("nan")
        ce_parent = row_ce(out.logits, labels)
        exact_parent = row_exact(out.logits, labels)
        q_loss = F.binary_cross_entropy_with_logits(out.q, exact_parent.float(), reduction="none")
        per_row = ce_parent + cfg.lambda_q * q_loss
        m: dict[str, torch.Tensor] = {
            "ce": ce_parent.detach(),
            "exact": exact_parent,
            "q_acc": ((out.q >= 0) == exact_parent),
            "q_loss": q_loss.detach(),
        }

        if out.child_logits is not None:
            fork = out.fork if out.fork is not None else torch.ones_like(exact_parent)

            def forked(v: torch.Tensor) -> torch.Tensor:  # a child metric is undefined on rows that did not fork
                return torch.where(fork, v.float(), nan)

            tau = tau_at(cfg, progress)
            ce_plain = row_ce(out.child_logits, labels[:, None])  # [B, K]
            ce_final = ce_plain
            if cfg.child_focus > 0 and out.gate_rows is not None:
                # Boosting-style specialisation: on label rows, cells TRM's own round already gets right weigh
                # (1 - child_focus), so the children's capacity goes where the parent is wrong. No gradient.
                right = (out.cf_logits.argmax(-1) == labels) & out.gate_rows[:, None]
                w = 1.0 - cfg.child_focus * right.float()
                ce_final = row_ce(out.child_logits, labels[:, None], w[:, None])
            exact_child = row_exact(out.child_logits, labels[:, None])  # [B, K]
            if has_subgoals is None:
                has_subgoals = "subgoals" in batch and bool(batch["subgoal_mask"].any())
            ce_sub = mask = None
            if has_subgoals:
                subgoals, mask = batch["subgoals"], batch["subgoal_mask"]
                ce_sub = row_ce(out.child_logits[:, :, None], subgoals[:, None])  # [B, K, M]
            match, best = set_match(ce_final, ce_sub, mask, tau, cfg.lambda_sub / cfg.lambda_wta)
            set_term = cfg.lambda_wta * ((1.0 - cfg.wta_eps) * match + cfg.wta_eps * ce_final.mean(-1))
            q_child = F.binary_cross_entropy_with_logits(out.child_q, exact_child.float(), reduction="none").mean(-1)
            per_row = per_row + torch.where(fork, set_term + cfg.lambda_q * q_child, 0.0)
            m.update(
                {
                    "set": forked(set_term.detach()),
                    "child_ce_min": forked(ce_plain.detach().min(-1).values),
                    "child_exact_any": forked(exact_child.any(-1)),
                    "winner": forked(best[:, 0]),
                    "distinct": forked(_distinct(out.child_logits.argmax(-1), labels)),
                    "parent_mass": out.join_alpha[:, 0],
                    "join_entropy": out.join_entropy,
                    "fork_rate": fork.float(),
                }
            )
            if has_subgoals:
                m["sub_coverage"] = forked(_coverage(out.child_logits, batch["subgoals"], mask, best[:, 1:]))

        loss = per_row.mean()
        if out.fork_logit is not None:  # the gate's label: did this fork beat the no-fork continuation by the price?
            g = out.gate_rows
            benefit = (row_ce(out.cf_logits, labels) - ce_parent).detach()
            target = (benefit > (cfg.fork_price if price is None else price)).float()
            bce = F.binary_cross_entropy_with_logits(out.fork_logit, target, reduction="none")
            loss = loss + cfg.lambda_gate * torch.where(g, bce, 0.0).sum() / g.sum().clamp_min(1)
            m.update(
                {
                    "gate_acc": torch.where(g, ((out.fork_logit > 0).float() == target).float(), nan),
                    "gate_pos": torch.where(g, target, nan),
                    "benefit": torch.where(g, benefit, nan),
                }
            )
        return loss, m


def _coverage(child_logits, subgoals, mask, assigned) -> torch.Tensor:
    """Fraction of a row's intermediates reproduced exactly by the child they are matched to (NaN without any)."""
    hit = row_exact(child_logits[:, :, None], subgoals[:, None])  # [B, K, M]
    got = hit.gather(1, assigned.clamp(min=0)[:, None]).squeeze(1) & (assigned >= 0) & mask  # [B, M]
    n = mask.sum(-1)
    return torch.where(n > 0, got.sum(-1).float() / n.clamp(min=1), torch.nan)


def _distinct(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Number of distinct child answers per row, compared on supervised positions. preds [B,K,L]."""
    valid = (labels != IGNORE)[:, None, None, :]
    same = ((preds[:, :, None, :] == preds[:, None, :, :]) | ~valid).all(-1)  # [B, K, K]
    earlier = torch.tril(same, diagonal=-1).any(-1)  # child k equals some child j < k
    return (~earlier).sum(-1)
