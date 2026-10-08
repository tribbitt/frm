"""The frm objective (docs/OBJECTIVE.md). Per round, per row, averaged over the rows of the step:

    L = CE(parent)                                         deep supervision on the post-join readout
      + lambda_wta * sum_k w_k CE(child_k)                 annealed relaxed winner-takes-all over children
      + lambda_sub * mean_t sum_k v_tk CE(child_k, g_t)    composites: every intermediate grid claimed by some child
      + lambda_q * [BCE(q, exact) + mean_k BCE(q_k, exact_k)]

w = (1 - eps) * softmin_tau(CE_child) + eps / K and v_t = softmin_tau(CE(child, g_t)) are computed without gradient;
tau anneals log-linearly from tau_start to tau_end over the first tau_anneal_frac of training. CE is TRM's stablemax
cross-entropy, token-mean per row over non-ignored labels.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from frm.config import ObjectiveCfg
from frm.model.refiner import RoundOut

IGNORE = -100


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Each branch sees only its own domain, so the unused branch cannot produce inf/NaN gradients through where().
    s = torch.where(x < 0, 1.0 / (1.0 - x.clamp(max=0)), x.clamp(min=0) + 1.0)
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


def row_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Token-mean stablemax CE per row: [..., L, V] x [..., L] (broadcast) -> [...] (float32)."""
    valid = labels != IGNORE
    return (token_ce(logits, labels).sum(-1) / valid.sum(-1).clamp_min(1)).float()


def row_exact(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Whole-row exact match over supervised positions (False for rows with no supervised position)."""
    valid = labels != IGNORE
    hit = (logits.argmax(-1) == labels) | ~valid
    return hit.all(-1) & valid.any(-1)


def tau_at(cfg: ObjectiveCfg, progress: float) -> float:
    frac = min(1.0, max(0.0, progress) / max(cfg.tau_anneal_frac, 1e-9))
    return math.exp(math.log(cfg.tau_start) + frac * (math.log(cfg.tau_end) - math.log(cfg.tau_start)))


def wta_weights(ce: torch.Tensor, tau: float, eps: float) -> torch.Tensor:
    """Relaxed annealed WTA weights over the last dim (children). No gradient."""
    K = ce.shape[-1]
    soft = torch.softmax(-ce.detach() / tau, dim=-1)
    return (1.0 - eps) * soft + eps / K


class Objective:
    def __init__(self, cfg: ObjectiveCfg):
        self.cfg = cfg

    def __call__(self, out: RoundOut, batch: dict[str, torch.Tensor], progress: float):
        cfg = self.cfg
        labels = batch["labels"]
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
            B, K = out.child_logits.shape[:2]
            tau = tau_at(cfg, progress)
            ce_child = row_ce(out.child_logits, labels[:, None])  # [B, K]
            exact_child = row_exact(out.child_logits, labels[:, None])  # [B, K]
            w = wta_weights(ce_child, tau, cfg.wta_eps)
            wta = (w * ce_child).sum(-1)
            q_child = F.binary_cross_entropy_with_logits(out.child_q, exact_child.float(), reduction="none").mean(-1)
            per_row = per_row + cfg.lambda_wta * wta + cfg.lambda_q * q_child
            m.update(
                {
                    "wta": wta.detach(),
                    "tau": torch.full_like(wta, tau),
                    "child_ce_min": ce_child.detach().min(-1).values,
                    "child_exact_any": exact_child.any(-1),
                    "winner": ce_child.detach().argmin(-1),
                    "distinct": _distinct(out.child_logits.argmax(-1), labels),
                    "parent_mass": out.join_alpha[:, 0],
                    "join_entropy": out.join_entropy,
                }
            )
            if "subgoals" in batch and batch["subgoal_mask"].any():
                sub, sub_cov = self._subgoal_term(out.child_logits, batch["subgoals"], batch["subgoal_mask"], tau)
                per_row = per_row + cfg.lambda_sub * sub
                m["sub"] = sub.detach()
                m["sub_coverage"] = sub_cov

        loss = per_row.mean()
        return loss, m

    @staticmethod
    def _subgoal_term(child_logits, subgoals, mask, tau):
        """child_logits [B,K,L,V], subgoals [B,M,L] (IGNORE where absent), mask [B,M] -> per-row loss [B], coverage [B]."""
        ce = row_ce(child_logits[:, :, None], subgoals[:, None])  # [B, K, M]
        v = torch.softmax(-ce.detach() / tau, dim=1)
        per_target = (v * ce).sum(1)  # [B, M]
        maskf = mask.float()
        loss = (per_target * maskf).sum(-1) / maskf.sum(-1).clamp_min(1.0)
        hit = row_exact(child_logits[:, :, None], subgoals[:, None]).any(1)  # [B, M]
        coverage = torch.where(mask.any(-1), (hit & mask).sum(-1).float() / maskf.sum(-1).clamp_min(1.0), torch.nan)
        return loss, coverage


def _distinct(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Number of distinct child answers per row, compared on supervised positions. preds [B,K,L]."""
    valid = (labels != IGNORE)[:, None, None, :]
    same = ((preds[:, :, None, :] == preds[:, None, :, :]) | ~valid).all(-1)  # [B, K, K]
    earlier = torch.tril(same, diagonal=-1).any(-1)  # child k equals some child j < k
    return (~earlier).sum(-1)
