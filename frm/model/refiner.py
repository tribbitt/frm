"""Forking refiner: TRM's tied 2-layer recursion with a fork-join per round (DESIGN.md §4).

Round (H_cycles=3): K=0 is TRM (2 no-grad cycles, 1 grad). K>0: 1 no-grad cycle, K children with grad, per-cell
attention join, 1 grad cycle. With ``fork_gate``, declined rows run TRM's remaining round instead.
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
from frm.model.layers import Block, CastedLinear, Rope2D, RopeTRM1D, block_params, rms_norm, trunc_normal_

CF_MODES = (None, "nofork", "sham")
FORK_MODES = ("gate", "all", "none")


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
    carry: Carry | None  # detached; None in outputs kept for instruments
    pre_logits: torch.Tensor | None = None  # [B, S*S, V] pre-fork parent readout (detached)
    # Child fields keep full-B shapes; on rows with fork=False join stats are NaN and child logits/q zeros: mask by fork.
    fork: torch.Tensor | None = None  # [B] bool rows that forked this round (None for K = 0)
    fork_logit: torch.Tensor | None = None  # [B] float32 gate logit (None without fork_gate)
    cf_logits: torch.Tensor | None = None  # [B, S*S, V] no-fork continuation readout (detached); zeros off gate_rows
    gate_rows: torch.Tensor | None = None  # [B] bool rows carrying a gate label


class Refiner(nn.Module):
    def __init__(self, cfg: ModelCfg, num_ids: int):
        super().__init__()
        self.cfg = cfg
        D, P = cfg.hidden, cfg.prefix_len
        self.scale = math.sqrt(D)

        self.tok_emb = nn.Parameter(trunc_normal_(torch.empty(cfg.vocab, D), std=1.0 / self.scale))
        self.puzzle_emb = PuzzleEmbedding(num_ids, D)
        rope = {"2d": Rope2D, "trm1d": RopeTRM1D}[cfg.pos_encoding]
        self.rope = rope(D // cfg.heads, P, cfg.rope_theta)
        self.blocks = nn.ModuleList(
            Block(D, cfg.heads, cfg.expansion, cfg.rms_eps, cfg.pos_encoding) for _ in range(cfg.layers)
        )
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
            if cfg.noise_dim:
                self.code_noise = CastedLinear(cfg.noise_dim, D, std=cfg.noise_init / math.sqrt(cfg.noise_dim))
        # Gate head (DESIGN.md §4.8), built last so other parameters initialise as in an ungated model.
        self.gate: nn.Module | None = None
        if cfg.fork_gate and cfg.K > 0:
            if cfg.fork_depth != 1 or cfg.grad_cycles > 2:
                raise ValueError("fork_gate needs fork_depth=1 and grad_cycles <= 2")
            self.gate = nn.Sequential(
                CastedLinear(2 * D, cfg.gate_hidden), nn.SiLU(), CastedLinear(cfg.gate_hidden, 1, bias=True)
            )
        self._cycle_fn = self._cycle

    # ------------------------------------------------------------------ basics
    def compile_cycles(self, **kwargs) -> None:
        """torch.compile the cycle. Raises dynamo's recompile limits (many graphs per bucket; default 8 silently falls
        back to eager). With the gate, batch dim is dynamic since fork/decline subset sizes vary per step."""
        import torch._dynamo

        for name in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit", "accumulated_cache_size_limit"):
            if hasattr(torch._dynamo.config, name):
                setattr(torch._dynamo.config, name, max(getattr(torch._dynamo.config, name), 512))
        if self.gate is None:
            self._cycle_fn = torch.compile(self._cycle, dynamic=False, **kwargs)
            return
        compiled = torch.compile(self._cycle, **kwargs)

        def cycle(y, z, x, c, cos, sin, n: int, ckpt: bool = False):
            for t in (y, z, x) if c is None else (y, z, x, c):
                torch._dynamo.maybe_mark_dynamic(t, 0)
            return compiled(y, z, x, c, cos, sin, n, ckpt)

        self._cycle_fn = cycle

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

    def _f_ckpt(self, h: torch.Tensor, u: torch.Tensor, cos_sin: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """_f with activation checkpointing."""
        return checkpoint(self._f, h, u, cos_sin, use_reentrant=False)

    def _cycle(self, y, z, x, c, cos, sin, n: int, ckpt: bool = False):
        inj = x if c is None else x + c
        f = self._f_ckpt if ckpt else self._f
        for _ in range(n):
            z = f(z, y + inj, (cos, sin))
        y = f(y, z, (cos, sin))
        return y, z

    # ------------------------------------------------------------------ fork / join
    def _child_codes(self, z: torch.Tensor, c: torch.Tensor | None, gen: torch.Generator | None) -> torch.Tensor:
        """[B, L, D] -> [B, K, L, D]: static code + parent-conditioned per-cell code + per-row noise (new proposals
        even at a fixed point)."""
        cfg = self.cfg
        u = self.code_down(rms_norm(z, cfg.rms_eps))  # [B, L, r]
        codes = torch.einsum("blr,krd->bkld", u, self.code_up.to(u.dtype)) + self.codes.to(u.dtype)[None, :, None, :]
        if cfg.noise_dim:
            xi = torch.randn(z.shape[0], cfg.K, cfg.noise_dim, device=z.device, generator=gen)
            codes = codes + self.code_noise(xi.to(u.dtype))[:, :, None, :]
        if c is not None:
            codes = codes + c[:, None]
        return codes

    def _join(self, y, z, ys, zs, q_parent, q_children, cf, sham_perm):
        """Per-cell softmax join over {parent, children}; returns (y, z, mass, entropy, dropped-parent rows)."""
        if cf == "sham":  # another row's children: tests whether the join uses child content
            perm = sham_perm if sham_perm is not None else torch.roll(torch.arange(ys.shape[0], device=ys.device), 1)
            ys, zs, q_children = ys[perm], zs[perm], q_children[perm]
        Y = torch.cat([y[:, None], ys], dim=1)  # [B, K+1, L, D]
        Z = torch.cat([z[:, None], zs], dim=1)
        qv = self.join_q(rms_norm(z, self.cfg.rms_eps))  # [B, L, d]
        kv = self.join_k(rms_norm(Z, self.cfg.rms_eps))  # [B, K+1, L, d]
        q_all = torch.cat([q_parent[:, None], q_children], dim=1).detach()  # [B, K+1]
        s = torch.einsum("bld,bkld->bkl", qv, kv).float() / math.sqrt(qv.shape[-1])
        s = s + self.join_beta * q_all[:, :, None]
        drop = None
        if self.training and self.cfg.join_drop_parent > 0 and cf is None:
            drop = torch.rand(s.shape[0], device=s.device) < self.cfg.join_drop_parent  # rows that must use children
            s = torch.cat([s[:, :1].masked_fill(drop[:, None, None], float("-inf")), s[:, 1:]], dim=1)
        alpha = s.softmax(dim=1)  # [B, K+1, L]
        if cf == "nofork":
            alpha = torch.zeros_like(alpha)
            alpha[:, 0] = 1.0
        a = alpha.to(Y.dtype)[..., None]
        y_new, z_new = (a * Y).sum(1), (a * Z).sum(1)
        P = self.cfg.prefix_len
        grid_alpha = alpha[:, :, P:].detach()
        ent = -(grid_alpha.clamp_min(1e-9).log() * grid_alpha).sum(1).mean(-1)
        mass = grid_alpha.mean(-1)
        if drop is not None:  # forced rows say nothing about the join's choice
            mass = mass.masked_fill(drop[:, None], float("nan"))
            ent = ent.masked_fill(drop, float("nan"))
        return y_new, z_new, mass, ent, drop

    def _node(self, y, z, x, c, depth: int, cos, sin, cf, sham_perm=None, gen=None):
        """Fork K children (recursively to fork_depth), join, run the post cycle. cf applies to the root join only."""
        cfg = self.cfg
        B, L, D = y.shape
        K = cfg.K
        codes = self._child_codes(z, c, gen).reshape(B * K, L, D)
        yk = y[:, None].expand(B, K, L, D).reshape(B * K, L, D)
        zk = z[:, None].expand(B, K, L, D).reshape(B * K, L, D)
        xk = x[:, None].expand(B, K, L, D).reshape(B * K, L, D)
        if depth + 1 < cfg.fork_depth:
            yk, zk, _ = self._node(yk, zk, xk, codes, depth + 1, cos, sin, None, gen=gen)
        else:
            n_child = cfg.child_L_cycles or cfg.L_cycles
            ckpt = cfg.grad_ckpt_children and torch.is_grad_enabled()
            yk, zk = self._cycle_fn(yk, zk, xk, codes, cos, sin, n_child, ckpt)
        ys, zs = yk.view(B, K, L, D), zk.view(B, K, L, D)
        q_children = self._q(yk).view(B, K)
        y, z, alpha, ent, drop = self._join(y, z, ys, zs, self._q(y), q_children, cf, sham_perm)
        y, z = self._cycle_fn(y, z, x, c, cos, sin, cfg.L_cycles)
        return y, z, (ys, q_children, alpha, ent, drop)

    # ------------------------------------------------------------------ heads / round
    def _q(self, y: torch.Tensor) -> torch.Tensor:
        return self.q_head(y[:, 0]).squeeze(-1).float()

    def _logits(self, y: torch.Tensor) -> torch.Tensor:
        return self.lm_head(y[..., self.cfg.prefix_len :, :]).float()

    def _trm_tail(self, y, z, x, cos, sin, nograd: int):
        """``nograd`` no-grad cycles, then ``grad_cycles`` cycles in the caller's grad mode."""
        n = self.cfg.L_cycles
        with torch.no_grad():
            for _ in range(nograd):
                y, z = self._cycle_fn(y, z, x, None, cos, sin, n)
        for _ in range(self.cfg.grad_cycles):
            y, z = self._cycle_fn(y, z, x, None, cos, sin, n)
        return y, z

    def _decide(
        self, fork_logit, fork_mode: str, explore: float, B: int, device, top: float = 0.0, n_valid: int | None = None
    ) -> torch.Tensor:
        """[B] bool fork mask. ``top`` > 0: top ceil(top * n_valid) rows by gate logit fork; padding never forks."""
        if fork_mode == "none":
            return torch.zeros(B, dtype=torch.bool, device=device)
        if fork_mode == "all" or fork_logit is None:
            return torch.ones(B, dtype=torch.bool, device=device)
        if top > 0:
            n = B if n_valid is None else n_valid
            score = fork_logit.masked_fill(torch.arange(B, device=device) >= n, float("-inf"))
            fork = torch.zeros(B, dtype=torch.bool, device=device)
            fork[score.topk(min(n, math.ceil(top * n))).indices] = True
        else:
            fork = fork_logit > 0
        if explore > 0:  # keeps gate labels arriving for states the gate declines
            fork = fork | (torch.rand(B, device=device) < explore)
        return fork

    def round(
        self,
        carry: Carry,
        inputs: torch.Tensor,
        puzzle_ids: torch.Tensor,
        cf: str | None = None,
        sham_perm: torch.Tensor | None = None,
        gen: torch.Generator | None = None,
        fork_mode: str = "gate",
        explore: float = 0.0,
        fork_top: float = 0.0,
        n_valid: int | None = None,
        gate_labels: bool = True,
    ) -> RoundOut:
        """One deep-supervision round; gradient flows only if the caller enables grad.

        ``cf``: "nofork" forces the root join onto the parent; "sham" joins row ``sham_perm[b]``'s children (default:
        next row). ``gen`` seeds child noise. ``fork_mode``: "gate" (logit > 0, plus ``explore`` rate), "all", "none".
        Forked and declined subsets run separately. Gated rows in ``gate_rows`` also get ``cf_logits`` (no grad)
        unless ``gate_labels`` is False."""
        assert cf in CF_MODES and fork_mode in FORK_MODES
        cfg = self.cfg
        S = int(round(math.sqrt(inputs.shape[1])))
        y, z = carry.y, carry.z
        x = self.embed(inputs, puzzle_ids, y.dtype)
        cos, sin = self.rope(S, y.device)
        if cfg.K == 0:
            y, z = self._trm_tail(y, z, x, cos, sin, cfg.H_cycles - cfg.grad_cycles)
            return RoundOut(
                logits=self._logits(y), q=self._q(y), child_logits=None, child_q=None, join_alpha=None,
                join_entropy=None, carry=Carry(y.detach(), z.detach()),
            )
        with torch.no_grad():
            for _ in range(cfg.H_cycles - 2):
                y, z = self._cycle_fn(y, z, x, None, cos, sin, cfg.L_cycles)
            pre_logits = self._logits(y)
        B, K, P = y.shape[0], cfg.K, cfg.prefix_len
        fork_logit = None
        if self.gate is not None:  # reads the no-grad pre-fork state: no gradient into the trunk
            fork_logit = self.gate(torch.cat([y[:, 0], y[:, P:].mean(1)], dim=-1)).squeeze(-1).float()
        fork = self._decide(fork_logit, fork_mode, explore, B, y.device, fork_top, n_valid)
        if fork_mode == "none":
            n_fork = 0
        elif fork_mode == "all" or self.gate is None:
            n_fork = B
        else:
            n_fork = int(fork.sum())  # host sync: the subsets' sizes
        rest = 2 - cfg.grad_cycles  # no-grad cycles a declined row still owes TRM's round
        drop = None
        if n_fork == B:
            y_out, z_out, (ys, child_q, alpha, ent, drop) = self._node(y, z, x, None, 0, cos, sin, cf, sham_perm, gen)
            child_logits = self._logits(ys)
        else:
            f32 = torch.float32
            child_logits = pre_logits.new_zeros(B, K, *pre_logits.shape[1:])
            child_q = pre_logits.new_zeros(B, K)
            alpha = torch.full((B, K + 1), float("nan"), dtype=f32, device=y.device)
            ent = torch.full((B,), float("nan"), dtype=f32, device=y.device)
            if n_fork == 0:
                y_out, z_out = self._trm_tail(y, z, x, cos, sin, rest)
            else:
                order = torch.argsort((~fork).to(torch.uint8), stable=True)
                idx_f, idx_n = order[:n_fork], order[n_fork:]
                if cf == "sham":  # run every row so sham_perm indexes the whole batch
                    yf, zf, stats = self._node(y, z, x, None, 0, cos, sin, cf, sham_perm, gen)
                    yf, zf = yf[idx_f], zf[idx_f]
                    ys, cq, a, e = (t[idx_f] for t in stats[:4])
                else:
                    yf, zf, (ys, cq, a, e, d) = self._node(
                        y[idx_f], z[idx_f], x[idx_f], None, 0, cos, sin, cf, None, gen
                    )
                    if d is not None:
                        drop = torch.zeros_like(fork)
                        drop[idx_f] = d
                yn, zn = self._trm_tail(y[idx_n], z[idx_n], x[idx_n], cos, sin, rest)
                y_out = y.index_copy(0, idx_f, yf).index_copy(0, idx_n, yn)
                z_out = z.index_copy(0, idx_f, zf).index_copy(0, idx_n, zn)
                child_logits = child_logits.index_copy(0, idx_f, self._logits(ys))
                child_q = child_q.index_copy(0, idx_f, cq)
                alpha, ent = alpha.index_copy(0, idx_f, a), ent.index_copy(0, idx_f, e)
        cf_logits = gate_rows = None
        if self.gate is not None:
            # Label rows: forks with the model's own join, compared against the TRM continuation from the same state.
            gate_rows = torch.zeros_like(fork) if cf is not None else fork if drop is None else fork & ~drop
            if not gate_labels:
                gate_rows = torch.zeros_like(fork)
            cf_logits = torch.zeros_like(pre_logits)
            if cf is None and n_fork and gate_labels:
                with torch.no_grad():
                    idx_g = gate_rows.nonzero().squeeze(1)
                    if idx_g.numel():
                        yg, _ = self._trm_tail(y[idx_g], z[idx_g], x[idx_g], cos, sin, rest)
                        cf_logits.index_copy_(0, idx_g, self._logits(yg))
        return RoundOut(
            logits=self._logits(y_out),
            q=self._q(y_out),
            child_logits=child_logits,
            child_q=child_q,
            join_alpha=alpha,
            join_entropy=ent,
            carry=Carry(y_out.detach(), z_out.detach()),
            pre_logits=pre_logits,
            fork=fork,
            fork_logit=fork_logit,
            cf_logits=cf_logits,
            gate_rows=gate_rows,
        )

    # ------------------------------------------------------------------ accounting
    def passes_per_round(self, fork: bool = True) -> tuple[int, int, int]:
        """(no-grad, grad, checkpoint-recomputed) passes of f per parent row per training round; children count once
        each. ``fork=False``: TRM's round."""
        cfg = self.cfg
        n, nc = cfg.L_cycles + 1, (cfg.child_L_cycles or cfg.L_cycles) + 1
        if cfg.K == 0 or not fork:
            return (cfg.H_cycles - cfg.grad_cycles) * n, cfg.grad_cycles * n, 0

        def node(depth: int) -> tuple[int, int]:  # (grad passes, leaf passes) at and below this node
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
        """Training FLOPs per row-round when every row forks."""
        nograd, grad, recompute = self.passes_per_round()
        return (nograd + 3 * grad + recompute) * self.flops_per_row_pass(S)

    def eval_flops_per_row(self, S: int) -> float:
        nograd, grad, _ = self.passes_per_round()
        return (nograd + grad) * self.flops_per_row_pass(S)

    def round_flops(self, S: int, rows: int, fork_rows: int, cf_rows: int, train: bool = True) -> float:
        """FLOPs of one round; ``cf_rows`` also ran the 2-cycle no-fork continuation. Grad passes cost 3x in training;
        gate head not counted."""

        def units(passes: tuple[int, int, int]) -> int:
            nograd, grad, recompute = passes
            return nograd + 3 * grad + recompute if train else nograd + grad

        cont = 2 * (self.cfg.L_cycles + 1)
        total = (rows - fork_rows) * units(self.passes_per_round(fork=False)) + fork_rows * units(self.passes_per_round())
        return (total + cf_rows * cont) * self.flops_per_row_pass(S)
