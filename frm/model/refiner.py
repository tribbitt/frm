"""The forking refiner: TRM's tied 2-layer recursion with a fork-join in every round (docs/DESIGN.md §4).

One round (= one deep-supervision step) with H_cycles=3:

    K = 0 (``trm`` control):  2 parent cycles without grad, 1 with grad            (TRM exactly)
    K > 0 (FRM):              1 parent cycle without grad
                              K children cycle(y, z, x, c_k) with grad            (fork)
                              per-cell attention join over {parent, children}     (join, with grad)
                              1 parent cycle with grad -> graded readout

A cycle is n z-updates z = f(z, y + x + c) then one y-update y = f(y, z); f is the same two blocks everywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from frm.config import ModelCfg
from frm.model.embedding import PuzzleEmbedding
from frm.model.layers import Block, CastedLinear, Rope2D, block_params, rms_norm, trunc_normal_

CF_MODES = (None, "nofork", "sham")


@dataclass
class Carry:
    y: torch.Tensor  # answer state (TRM z_H)   [B, P+S*S, D]
    z: torch.Tensor  # latent state (TRM z_L)   [B, P+S*S, D]


@dataclass
class RoundOut:
    logits: torch.Tensor  # [B, S*S, V] float32, parent's post-join readout
    q: torch.Tensor  # [B] float32 halting logit of the parent
    child_logits: torch.Tensor | None  # [B, K, S*S, V] float32
    child_q: torch.Tensor | None  # [B, K] float32
    join_alpha: torch.Tensor | None  # [B, K+1] join weights averaged over grid cells (slot 0 = parent)
    join_entropy: torch.Tensor | None  # [B] mean per-cell entropy of the join weights
    carry: Carry  # detached


class Refiner(nn.Module):
    def __init__(self, cfg: ModelCfg, num_ids: int):
        super().__init__()
        self.cfg = cfg
        D, P = cfg.hidden, cfg.prefix_len
        self.scale = math.sqrt(D)

        self.tok_emb = nn.Parameter(trunc_normal_(torch.empty(cfg.vocab, D), std=1.0 / self.scale))
        self.puzzle_emb = PuzzleEmbedding(num_ids, D)
        self.rope = Rope2D(D // cfg.heads, P, cfg.rope_theta)
        self.blocks = nn.ModuleList(Block(D, cfg.heads, cfg.expansion, cfg.rms_eps) for _ in range(cfg.layers))
        self.lm_head = CastedLinear(D, cfg.vocab)
        self.q_head = CastedLinear(D, 1, bias=True, std=0.0)
        with torch.no_grad():
            self.q_head.bias.fill_(-5.0)
        self.register_buffer("y_init", trunc_normal_(torch.empty(D), std=1.0))
        self.register_buffer("z_init", trunc_normal_(torch.empty(D), std=1.0))

        if cfg.K > 0:
            self.codes = nn.Parameter(torch.randn(cfg.K, D) * cfg.code_init_std)
            self.code_down = CastedLinear(D, cfg.code_rank)
            self.code_up = nn.Parameter(torch.zeros(cfg.K, cfg.code_rank, D))
            self.join_q = CastedLinear(D, cfg.join_dim)
            self.join_k = CastedLinear(D, cfg.join_dim)
            self.join_beta = nn.Parameter(torch.zeros(()))
        self._cycle_fn = self._cycle

    # ------------------------------------------------------------------ basics
    def compile_cycles(self, **kwargs) -> None:
        """torch.compile the cycle (one graph per canvas size); call before training/eval on GPU."""
        self._cycle_fn = torch.compile(self._cycle, dynamic=False, **kwargs)

    def init_carry(self, B: int, S: int, device: torch.device, dtype: torch.dtype) -> Carry:
        L = self.cfg.prefix_len + S * S
        return Carry(
            y=self.y_init.to(device, dtype).expand(B, L, -1).clone(),
            z=self.z_init.to(device, dtype).expand(B, L, -1).clone(),
        )

    def reset_carry(self, carry: Carry, reset: torch.Tensor) -> Carry:
        m = reset.view(-1, 1, 1)
        return Carry(
            y=torch.where(m, self.y_init.to(carry.y.dtype), carry.y),
            z=torch.where(m, self.z_init.to(carry.z.dtype), carry.z),
        )

    def embed(self, inputs: torch.Tensor, puzzle_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        B, P = inputs.shape[0], self.cfg.prefix_len
        tok = F.embedding(inputs.long(), self.tok_emb)
        pz = self.puzzle_emb(puzzle_ids)  # [B, D]
        prefix = torch.cat([pz[:, None, :], pz.new_zeros(B, P - 1, pz.shape[-1])], dim=1)
        return (self.scale * torch.cat([prefix, tok], dim=1)).to(dtype)

    def _f(self, h: torch.Tensor, u: torch.Tensor, cos_sin: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        h = h + u
        for blk in self.blocks:
            h = blk(h, cos_sin)
        return h

    def _cycle(self, y, z, x, c, cos, sin, n: int):
        inj = x if c is None else x + c
        for _ in range(n):
            z = self._f(z, y + inj, (cos, sin))
        y = self._f(y, z, (cos, sin))
        return y, z

    # ------------------------------------------------------------------ fork / join
    def _child_codes(self, z: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        """[B, L, D] -> codes [B, K, L, D]: static per-child code + parent-conditioned per-cell code."""
        u = self.code_down(rms_norm(z, self.cfg.rms_eps))  # [B, L, r]
        codes = torch.einsum("blr,krd->bkld", u, self.code_up.to(u.dtype)) + self.codes.to(u.dtype)[None, :, None, :]
        if c is not None:
            codes = codes + c[:, None]
        return codes

    def _join(self, y, z, ys, zs, q_parent, q_children, cf):
        """Per-cell softmax attention of the parent over {parent, children}; returns new (y, z) and stats."""
        B, K = ys.shape[0], ys.shape[1]
        if cf == "sham":  # children of another row: tests whether the parent uses child *content*
            ys, zs, q_children = ys.roll(1, dims=0), zs.roll(1, dims=0), q_children.roll(1, dims=0)
        Y = torch.cat([y[:, None], ys], dim=1)  # [B, K+1, L, D]
        Z = torch.cat([z[:, None], zs], dim=1)
        qv = self.join_q(rms_norm(z, self.cfg.rms_eps))  # [B, L, d]
        kv = self.join_k(rms_norm(Z, self.cfg.rms_eps))  # [B, K+1, L, d]
        q_all = torch.cat([q_parent[:, None], q_children], dim=1).detach()  # [B, K+1]
        s = torch.einsum("bld,bkld->bkl", qv, kv).float() / math.sqrt(qv.shape[-1])
        s = s + self.join_beta * q_all[:, :, None]
        alpha = s.softmax(dim=1)  # [B, K+1, L]
        if cf == "nofork":
            alpha = torch.zeros_like(alpha)
            alpha[:, 0] = 1.0
        a = alpha.to(Y.dtype)[..., None]
        y_new, z_new = (a * Y).sum(1), (a * Z).sum(1)
        P = self.cfg.prefix_len
        grid_alpha = alpha[:, :, P:].detach()
        ent = -(grid_alpha.clamp_min(1e-9).log() * grid_alpha).sum(1).mean(-1)
        return y_new, z_new, grid_alpha.mean(-1), ent

    def _node(self, y, z, x, c, depth: int, cos, sin, cf):
        """Fork K children (each a node one level deeper), join them, run the post cycle. Returns children too."""
        cfg = self.cfg
        B, L, D = y.shape
        K = cfg.K
        codes = self._child_codes(z, c).reshape(B * K, L, D)
        yk = y[:, None].expand(B, K, L, D).reshape(B * K, L, D)
        zk = z[:, None].expand(B, K, L, D).reshape(B * K, L, D)
        xk = x[:, None].expand(B, K, L, D).reshape(B * K, L, D)
        if depth + 1 < cfg.fork_depth:
            yk, zk, _ = self._node(yk, zk, xk, codes, depth + 1, cos, sin, cf)
        else:
            n_child = cfg.child_L_cycles or cfg.L_cycles
            if cfg.grad_ckpt_children and torch.is_grad_enabled():
                yk, zk = checkpoint(self._cycle_fn, yk, zk, xk, codes, cos, sin, n_child, use_reentrant=False)
            else:
                yk, zk = self._cycle_fn(yk, zk, xk, codes, cos, sin, n_child)
        ys, zs = yk.view(B, K, L, D), zk.view(B, K, L, D)
        q_parent = self._q(y)
        q_children = self._q(yk).view(B, K)
        y, z, alpha, ent = self._join(y, z, ys, zs, q_parent, q_children, cf)
        y, z = self._cycle_fn(y, z, x, c, cos, sin, cfg.L_cycles)
        return y, z, (ys, q_children, alpha, ent)

    # ------------------------------------------------------------------ heads / round
    def _q(self, y: torch.Tensor) -> torch.Tensor:
        return self.q_head(y[:, 0]).squeeze(-1).float()

    def _logits(self, y: torch.Tensor) -> torch.Tensor:
        return self.lm_head(y[..., self.cfg.prefix_len :, :]).float()

    def round(self, carry: Carry, inputs: torch.Tensor, puzzle_ids: torch.Tensor, cf: str | None = None) -> RoundOut:
        """One deep-supervision round. Gradient flows only if grad mode is enabled by the caller."""
        assert cf in CF_MODES
        cfg = self.cfg
        S = int(round(math.sqrt(inputs.shape[1])))
        y, z = carry.y, carry.z
        x = self.embed(inputs, puzzle_ids, y.dtype)
        cos, sin = self.rope(S, y.device)
        n = cfg.L_cycles
        child_logits = child_q = alpha = ent = None
        if cfg.K == 0:
            with torch.no_grad():
                for _ in range(cfg.H_cycles - cfg.grad_cycles):
                    y, z = self._cycle_fn(y, z, x, None, cos, sin, n)
            for _ in range(cfg.grad_cycles):
                y, z = self._cycle_fn(y, z, x, None, cos, sin, n)
        else:
            with torch.no_grad():
                for _ in range(cfg.H_cycles - 2):
                    y, z = self._cycle_fn(y, z, x, None, cos, sin, n)
            y, z, (ys, child_q, alpha, ent) = self._node(y, z, x, None, 0, cos, sin, cf)
            child_logits = self._logits(ys)
        return RoundOut(
            logits=self._logits(y),
            q=self._q(y),
            child_logits=child_logits,
            child_q=child_q,
            join_alpha=alpha,
            join_entropy=ent,
            carry=Carry(y.detach(), z.detach()),
        )

    # ------------------------------------------------------------------ accounting
    def passes_per_round(self) -> tuple[int, int, int]:
        """(no-grad passes, grad passes, checkpoint-recomputed passes) per parent row per training round.

        A pass is one application of f (all blocks) to one row; children count once per child.
        """
        cfg = self.cfg
        n, nc = cfg.L_cycles + 1, (cfg.child_L_cycles or cfg.L_cycles) + 1
        if cfg.K == 0:
            return (cfg.H_cycles - cfg.grad_cycles) * n, cfg.grad_cycles * n, 0

        def node(depth: int) -> tuple[int, int]:  # (grad passes, leaf passes) below and including this node
            if depth + 1 < cfg.fork_depth:
                g, leaf = node(depth + 1)
                return cfg.K * g + n, cfg.K * leaf
            return cfg.K * nc + n, cfg.K * nc

        grad, leaf = node(0)
        return (cfg.H_cycles - 2) * n, grad, leaf if cfg.grad_ckpt_children else 0

    def flops_per_row_pass(self, S: int) -> float:
        cfg = self.cfg
        L = cfg.prefix_len + S * S
        return cfg.layers * L * (2 * block_params(cfg.hidden, cfg.expansion) + 4 * L * cfg.hidden)

    def train_flops_per_row(self, S: int) -> float:
        nograd, grad, recompute = self.passes_per_round()
        return (nograd + 3 * grad + recompute) * self.flops_per_row_pass(S)

    def eval_flops_per_row(self, S: int) -> float:
        nograd, grad, _ = self.passes_per_round()
        return (nograd + grad) * self.flops_per_row_pass(S)
