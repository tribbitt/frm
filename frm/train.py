"""Trainer: TRM's online deep supervision with one carry per canvas bucket (DESIGN.md §6).

    python -m frm.train --config configs/final_frm_a2tn.json --run_dir runs/final_frm_a2tn [--resume auto] [k=v ...]

RUN/last.pt every ``ckpt_minutes``; SIGTERM/SIGUSR1 -> checkpoint, exit 0; budget reached -> checkpoint, final dev
eval, RUN/DONE, exit 0; failure -> nonzero. ``--resume auto`` continues from RUN/last.pt.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from frm.config import Cfg
from frm.data.build import SOURCE_CODES
from frm.data.loader import Dataset, TrainStream
from frm.evaluate import evaluate_split
from frm.model.layers import select_sdpa_backend
from frm.model.refiner import Carry, Refiner
from frm.objective import Objective, tau_at

ROUND_METRICS = (
    "exact", "ce", "q_loss", "q_acc", "set", "child_exact_any", "child_ce_min", "parent_mass", "join_entropy",
    "distinct", "sub_coverage", "fork_rate", "gate_acc", "gate_pos", "benefit", "gate_says", "declined_pos",
)


# ---------------------------------------------------------------------------------------------------- utilities
def pick_device(name: str) -> torch.device:
    """Resolve the device and select the SDPA backend for it."""
    device = torch.device(name if name != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    select_sdpa_backend(device)
    return device


def pick_dtype(precision: str, device: torch.device) -> torch.dtype:
    if precision == "auto":
        if device.type == "cuda":
            # Native bf16 only; emulated bf16 is slower than fp16.
            return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


class EMA:
    """EMA of dense parameters (puzzle-embedding buffer excluded, as in TRM)."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            self.shadow[n].lerp_(p.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def swap(self, model: torch.nn.Module) -> None:
        """Swap model parameters and shadow in place (call twice to restore)."""
        for n, p in model.named_parameters():
            tmp = p.detach().clone()
            p.copy_(self.shadow[n])
            self.shadow[n].copy_(tmp)


def load_pretrained(model: Refiner, path: str, device: torch.device, mode: str = "adapt") -> None:
    """Load weights (EMA when present) from a checkpoint.

    ``adapt``: new dataset; every non-blank embedding row = mean of the checkpoint's trained rows.
    ``continue``: same dataset; table kept; fork modules missing from a K=0 checkpoint keep their fresh init.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    state = dict(ck["model"])
    if "ema" in ck:
        state.update(ck["ema"])
    if mode == "continue":
        if state["puzzle_emb.table"].shape != model.puzzle_emb.table.shape:
            raise ValueError(f"{path}: init_mode=continue needs the checkpoint's dataset ({ck.get('num_ids')} ids), "
                             f"the model has {model.puzzle_emb.num_ids}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        shared = set(Refiner(_forkless(model.cfg), num_ids=1).state_dict())
        if unexpected or any(k in shared for k in missing):
            raise ValueError(f"checkpoint/model mismatch: missing {missing}, unexpected {unexpected}")
        return
    if mode != "adapt":
        raise ValueError(f"init_mode {mode!r} not in ('adapt', 'continue')")
    old = state.pop("puzzle_emb.table")
    old_visits = state.pop("puzzle_emb.visits", None)
    trained = (old_visits > 0) if old_visits is not None else (old != 0).any(-1)
    trained[0] = False
    missing, unexpected = model.load_state_dict(state, strict=False)
    if sorted(missing) != ["puzzle_emb.table", "puzzle_emb.visits"] or unexpected:
        raise ValueError(f"checkpoint/model mismatch: missing {missing}, unexpected {unexpected}")
    if not bool(trained.any()):
        raise ValueError(f"{path}: no trained puzzle-embedding row to initialise the new table from")
    with torch.no_grad():
        model.puzzle_emb.table.copy_(old[trained].mean(0).expand_as(model.puzzle_emb.table))
        model.puzzle_emb.table[0].zero_()
        model.puzzle_emb.visits.zero_()


def _forkless(cfg):
    """K=0 twin of a model config: its state-dict keys are the shared (non-fork) weights."""
    fields = {f.name for f in dataclasses.fields(cfg)}
    return dataclasses.replace(cfg, K=0, **({"fork_gate": False} if "fork_gate" in fields else {}))


@dataclass
class BucketState:
    S: int
    rows: int
    carry: Carry
    batch: dict[str, torch.Tensor]
    steps: torch.Tensor  # [B] int32 rounds completed by each row
    halted: torch.Tensor  # [B] bool
    solved: torch.Tensor  # [B] bool: row has been exact at some earlier round
    prev_exact: torch.Tensor  # [B] bool
    has_sub: np.ndarray  # [B] bool, host copy: row has sub-goals


def to_device(arrays: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in arrays.items():
        if not isinstance(v, np.ndarray):
            continue
        t = torch.from_numpy(np.ascontiguousarray(v))
        if t.dtype in (torch.uint8, torch.int16):
            t = t.long()
        out[k] = t.to(device, non_blocking=True)
    return out


SOURCE_METRICS = ("exact", "ce", "fork_rate", "benefit", "gate_pos", "gate_acc", "gate_says", "declined_pos",
                  "child_ce_min")
SOURCE_NAMES = {code: name for name, code in SOURCE_CODES.items()}


class RoundStats:
    """Per-row metrics by round (1..max_rounds) between log lines: ``{name}``, ``{name}_by_round`` and, for
    ``SOURCE_METRICS``, ``{name}_by_source``."""

    def __init__(self, max_rounds: int):
        self.R = max_rounds
        self.sums: dict[str, np.ndarray] = {}
        self.counts: dict[str, np.ndarray] = {}
        self.src_sums: dict[str, np.ndarray] = {}
        self.src_counts: dict[str, np.ndarray] = {}

    def add_many(self, rounds: torch.Tensor, metrics: dict[str, torch.Tensor], sources: torch.Tensor | None = None):
        """One device->host transfer per step; NaN entries skipped."""
        names = list(metrics)
        head = [rounds.float()] + ([] if sources is None else [sources.float()])
        stacked = torch.stack(head + [metrics[n].detach().float() for n in names]).cpu().numpy()
        r = np.clip(stacked[0].astype(np.int64), 1, self.R)
        src = None if sources is None else stacked[1].astype(np.int64)
        for i, name in enumerate(names):
            v = stacked[i + len(head)]
            keep = ~np.isnan(v)
            s = self.sums.setdefault(name, np.zeros(self.R + 1))
            c = self.counts.setdefault(name, np.zeros(self.R + 1))
            np.add.at(s, r[keep], v[keep])
            np.add.at(c, r[keep], 1)
            if src is not None and name in SOURCE_METRICS:
                n_src = len(SOURCE_NAMES)
                s = self.src_sums.setdefault(name, np.zeros(n_src))
                c = self.src_counts.setdefault(name, np.zeros(n_src))
                np.add.at(s, src[keep], v[keep])
                np.add.at(c, src[keep], 1)

    def summary(self) -> dict:
        out = {}
        for name, s in self.sums.items():
            c = self.counts[name]
            out[f"{name}_by_round"] = [round(float(s[r] / c[r]), 5) if c[r] else None for r in range(1, self.R + 1)]
            out[name] = float(s.sum() / max(c.sum(), 1))
        for name, s in self.src_sums.items():
            c = self.src_counts[name]
            out[f"{name}_by_source"] = {SOURCE_NAMES[k]: round(float(s[k] / c[k]), 5) for k in np.flatnonzero(c)}
        self.sums, self.counts, self.src_sums, self.src_counts = {}, {}, {}, {}
        return out


# ---------------------------------------------------------------------------------------------------- trainer
class Trainer:
    def __init__(self, cfg: Cfg, run_dir: Path, resume: str):
        self.cfg, self.run_dir = cfg, run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        tc = cfg.train
        torch.manual_seed(tc.seed)
        self.device = pick_device(tc.device)
        self.dtype = pick_dtype(tc.precision, self.device)
        self.ds = Dataset(cfg.data.root)
        self.model = Refiner(cfg.model, self.ds.num_ids).to(self.device)
        if tc.init_from:
            load_pretrained(self.model, tc.init_from, self.device, tc.init_mode)
        if tc.compile:
            self.model.compile_cycles()
        self.objective = Objective(cfg.objective)
        fused = self.device.type == "cuda"
        groups = [{"params": list(self.model.parameters()), "lr_mult": 1.0}]
        if tc.fork_lr_mult != 1.0 and cfg.model.K > 0:
            shared = set(Refiner(_forkless(cfg.model), num_ids=1).state_dict())
            named = list(self.model.named_parameters())
            groups = [{"params": [p for n, p in named if n in shared], "lr_mult": 1.0},
                      {"params": [p for n, p in named if n not in shared], "lr_mult": tc.fork_lr_mult}]
        self.opt = torch.optim.AdamW(groups, lr=tc.lr, betas=(tc.beta1, tc.beta2), weight_decay=tc.weight_decay,
                                     fused=fused)
        self.scaler = torch.amp.GradScaler("cuda") if self.dtype == torch.float16 else None
        self.ema = EMA(self.model, tc.ema)
        self.step = 0
        self.flops = 0.0
        self.row_rounds = 0  # training rows x rounds processed
        self.best = -1.0
        self.price = cfg.objective.fork_price  # gate-label fork price (nats/token)
        self.next_eval = tc.eval_every_pflops
        if "test" in cfg.eval.splits:
            raise ValueError("eval.splits must not contain 'test': checkpoints are selected on dev (DESIGN.md D14)")
        P = cfg.model.prefix_len
        buckets = cfg.data.buckets or self.ds.buckets
        self.buckets = [S for S in buckets if self.ds.num_train_examples(S) > 0]
        self.rows = {S: max(1, min(cfg.data.max_rows, cfg.data.tokens_per_step // (P + S * S))) for S in self.buckets}
        self.bucket_p, source_w = self._sampling_weights()
        self.rng = np.random.default_rng(tc.seed)
        self.streams = {
            S: TrainStream(
                self.ds, S, np.random.default_rng([tc.seed, S]), source_w[S], cfg.data.translate_p,
                max_pairs_per_draw=cfg.data.max_pairs_per_draw,
            )
            for S in self.buckets
        }
        self.states: dict[int, BucketState] = {}
        self.stats = RoundStats(tc.max_rounds)
        self.stop_requested = False
        self.t_start = time.time()
        self.hours_before = 0.0  # wall-clock of earlier sessions (max_hours is cumulative)
        if resume != "none":
            self._maybe_resume(resume)
        (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    # ------------------------------------------------------------------ checkpointing
    def _ckpt_state(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "ema": self.ema.shadow,
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "step": self.step,
            "flops": self.flops,
            "row_rounds": self.row_rounds,
            "hours": self.hours(),
            "price": self.price,
            "best": self.best,
            "next_eval": self.next_eval,
            "streams": {S: s.state_dict() for S, s in self.streams.items()},
            "rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
            "cfg": self.cfg.to_dict(),
            "num_ids": self.ds.num_ids,
        }

    def save(self, name: str = "last.pt") -> None:
        path = self.run_dir / name
        tmp = path.with_suffix(".tmp")
        torch.save(self._ckpt_state(), tmp)
        os.replace(tmp, path)

    def _sampling_weights(self) -> tuple[np.ndarray, dict[int, dict[str, float]]]:
        """Bucket probabilities and per-bucket source weights: source s gives fraction w_s of rows, and every task of
        a source is visited at the same rate whatever its bucket (e[s,S] = w_s * n[s,S] / n[s], capped draw sizes)."""
        cap = self.cfg.data.max_pairs_per_draw
        n: dict[int, dict[str, float]] = {}
        for S in self.buckets:
            d = self.ds.split_data(S, "train")
            capped = np.minimum(np.diff(d.ex_start), cap)
            n[S] = {s: float(capped[d.source == c].sum()) for s, c in SOURCE_CODES.items() if np.any(d.source == c)}
        sources = sorted({s for S in self.buckets for s in n[S]})
        total = {s: sum(n[S].get(s, 0.0) for S in self.buckets) for s in sources}
        w = self.cfg.data.source_weights or {s: total[s] for s in sources}
        w = {s: w.get(s, 0.0) for s in sources if total[s] > 0}
        z = sum(w.values())
        if z <= 0:
            raise ValueError(f"data.source_weights {self.cfg.data.source_weights} give no weight to sources {sources}")
        e = {S: {s: w[s] / z * n[S].get(s, 0.0) / total[s] for s in w} for S in self.buckets}
        p = np.array([sum(e[S].values()) / self.rows[S] for S in self.buckets], dtype=np.float64)
        source_w = {S: {s: v for s, v in e[S].items() if v > 0} for S in self.buckets}
        keep = p > 0
        self.buckets = [S for S, k in zip(self.buckets, keep) if k]
        return p[keep] / p[keep].sum(), {S: source_w[S] for S in self.buckets}

    def _maybe_resume(self, resume: str) -> None:
        """``auto``: RUN/last.pt if present, else ``train.resume_from`` (a branch; best score reset)."""
        path = self.run_dir / "last.pt" if resume == "auto" else Path(resume)
        branch = False
        if not path.exists():
            if resume != "auto":
                raise FileNotFoundError(path)
            if not self.cfg.train.resume_from:
                return
            path, branch = Path(self.cfg.train.resume_from), True
            if not path.exists():
                raise FileNotFoundError(f"train.resume_from: {path}")
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.ema.shadow = {k: v.to(self.device) for k, v in ck["ema"].items()}
        self.opt.load_state_dict(ck["opt"])
        if self.scaler and ck["scaler"]:
            self.scaler.load_state_dict(ck["scaler"])
        self.step, self.flops, self.best, self.next_eval = ck["step"], ck["flops"], ck["best"], ck["next_eval"]
        if branch:
            self.best = -1.0
        self.row_rounds = ck["row_rounds"]
        self.hours_before = ck.get("hours", 0.0)
        self.price = ck.get("price", self.price)
        for S, st in ck["streams"].items():
            if S in self.streams:
                self.streams[S].load_state_dict(st)
        self.rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        if ck.get("cuda_rng") is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        print(f"resumed {path} at step {self.step}, {self.flops / 1e15:.1f} PFLOP", flush=True)

    # ------------------------------------------------------------------ one step
    def _state(self, S: int) -> BucketState:
        if S not in self.states:
            B = self.rows[S]
            raw = self.streams[S].take(B)
            self.states[S] = BucketState(
                S=S,
                rows=B,
                carry=self.model.init_carry(B, S, self.device, self.dtype),
                batch=to_device(raw, self.device),
                steps=torch.zeros(B, dtype=torch.int32, device=self.device),
                halted=torch.zeros(B, dtype=torch.bool, device=self.device),
                solved=torch.zeros(B, dtype=torch.bool, device=self.device),
                prev_exact=torch.zeros(B, dtype=torch.bool, device=self.device),
                has_sub=raw["subgoal_mask"].any(-1),
            )
        return self.states[S]

    def _refill(self, st: BucketState) -> None:
        """Replace halted rows with fresh rows from the bucket's stream."""
        idx_np = st.halted.nonzero().squeeze(-1).cpu().numpy()
        if idx_np.size == 0:
            return
        raw = self.streams[st.S].take(int(idx_np.size))
        st.has_sub[idx_np] = raw["subgoal_mask"].any(-1)
        idx = torch.from_numpy(idx_np).to(self.device)
        for k, v in to_device(raw, self.device).items():
            st.batch[k][idx] = v
        st.carry = self.model.reset_carry(st.carry, st.halted)
        st.steps[idx] = 0
        st.solved[idx] = False
        st.prev_exact[idx] = False

    def lr_factor(self) -> float:
        """Linear warm-up, constant, then optional (1-sqrt) cooldown to ``decay_to`` (arXiv 2405.18392)."""
        tc = self.cfg.train
        f = min(1.0, (self.step + 1) / max(1, tc.warmup_steps))
        if tc.decay_frac > 0:
            u = min(1.0, max(0.0, 1.0 - (1.0 - self.progress()) / tc.decay_frac))
            f *= tc.decay_to + (1.0 - tc.decay_to) * (1.0 - math.sqrt(u))
        return f

    def progress(self) -> float:
        """Budget fraction spent: max of compute, steps and wall-clock fractions."""
        tc = self.cfg.train
        p = self.flops / (tc.budget_pflops * 1e15)
        if tc.max_steps:
            p = max(p, self.step / tc.max_steps)
        if tc.max_hours:
            p = max(p, self.hours() / tc.max_hours)
        return min(1.0, p)

    def hours(self) -> float:
        """Cumulative training wall-clock."""
        return self.hours_before + (time.time() - self.t_start) / 3600

    def train_step(self) -> dict:
        tc = self.cfg.train
        S = int(self.rng.choice(self.buckets, p=self.bucket_p))
        st = self._state(S)
        self._refill(st)
        self.model.train()
        f = self.lr_factor()
        for g in self.opt.param_groups:
            g["lr"] = tc.lr * f * g["lr_mult"]
        progress = self.progress()
        # Forced forks early, then the gate decides with an exploration floor (ungated models always fork).
        forced = progress < tc.fork_force_frac
        fork_mode, explore = ("all", 0.0) if forced else ("gate", tc.fork_explore)
        out = self.model.round(
            st.carry, st.batch["inputs"], st.batch["puzzle_ids"], fork_mode=fork_mode, explore=explore
        )
        price = self.price
        loss, m = self.objective(out, st.batch, progress, has_subgoals=bool(st.has_sub.any()), price=price)
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
        else:
            loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
        host = [loss.detach().float(), gnorm.float()]
        if out.fork is not None:
            host.append(out.fork.sum().float())
        if out.gate_rows is not None:
            host.append(out.gate_rows.sum().float())
        loss_val, gnorm_val, *counts = torch.stack(host).tolist()
        n_fork = int(counts[0]) if out.fork is not None else st.rows
        n_cf = int(counts[1]) if out.gate_rows is not None else 0
        if not math.isfinite(loss_val):
            raise FloatingPointError(f"non-finite loss at step {self.step}")
        if self.scaler:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.model.puzzle_emb.step(tc.emb_lr * f, tc.emb_weight_decay)
        self.ema.update(self.model)

        with torch.no_grad():
            st.carry = out.carry
            st.steps += 1
            exact = m["exact"]
            per_row = {name: m[name] for name in ROUND_METRICS if name in m}
            per_row["first_solve"] = exact & ~st.solved
            per_row["regress"] = st.prev_exact & ~exact
            if "winner" in m:
                w = m["winner"]
                per_row.update(
                    {f"win{k}": (w == k).float().masked_fill(w.isnan(), math.nan) for k in range(self.cfg.model.K)}
                )
            self.stats.add_many(st.steps, per_row, sources=st.batch["source"])
            st.solved |= exact
            st.prev_exact = exact
            halted = st.steps >= tc.max_rounds
            if tc.max_rounds > 1:
                halted |= out.q > 0
                explore = torch.rand(st.rows, device=self.device) < tc.halt_explore
                min_rounds = torch.randint(2, tc.max_rounds + 1, (st.rows,), device=self.device)
                halted &= st.steps >= torch.where(explore, min_rounds, 0)
            st.halted = halted
        self.flops += self.model.round_flops(S, st.rows, n_fork, n_cf)
        self.row_rounds += st.rows
        self.step += 1
        if self.model.gate is not None and not forced and tc.fork_rate_target > 0:
            self.price = max(0.0, self.price + tc.fork_price_lr * (n_fork / st.rows - tc.fork_rate_target))
        tau = tau_at(self.cfg.objective, progress)
        info = {"loss": loss_val, "gnorm": gnorm_val, "S": S, "rows": st.rows, "tau": tau}
        if self.model.gate is not None:
            info["price"] = price
        return info

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, final: bool = False) -> dict:
        ec = self.cfg.eval
        if ec.use_ema:
            self.ema.swap(self.model)
        try:
            res = {"step": self.step, "pflops": self.flops / 1e15, "row_rounds": self.row_rounds}
            for split in ec.splits:
                if split in self.ds.splits():
                    res[split] = evaluate_split(
                        self.model, self.ds, split, ec, self.device, self.dtype, counterfactuals=ec.counterfactuals
                    )
            # best.pt is selected on the first listed split.
            primary = next((res[s] for s in ec.splits if s in res), {})
            score = primary.get("pass@2", 0.0)
            score = 0.0 if score != score else score
            res["score"] = float(score)
            if score >= self.best:
                self.best = float(score)
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "step": self.step,
                        "flops": self.flops,
                        "row_rounds": self.row_rounds,
                        "price": self.price,
                        "cfg": self.cfg.to_dict(),
                        "num_ids": self.ds.num_ids,
                        "score": score,
                    },
                    self.run_dir / "best.pt",
                )
        finally:
            if ec.use_ema:
                self.ema.swap(self.model)
        res["final"] = final
        with open(self.run_dir / "eval.jsonl", "a") as fh:
            fh.write(json.dumps(res) + "\n")
        return res

    # ------------------------------------------------------------------ loop
    def done(self) -> bool:
        """Budget spent (compute, steps or cumulative hours) or ``stop_pflops`` reached."""
        tc = self.cfg.train
        over_time = tc.max_hours > 0 and self.hours() > tc.max_hours
        stop = tc.stop_pflops > 0 and self.flops >= tc.stop_pflops * 1e15
        return over_time or stop or self.flops >= tc.budget_pflops * 1e15 or (0 < tc.max_steps <= self.step)

    def run(self) -> int:
        tc = self.cfg.train
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, lambda *_: setattr(self, "stop_requested", True))
        self.t_start = last_ckpt = time.time()
        t0, f0, rows_acc, log = time.time(), self.flops, 0, open(self.run_dir / "train.jsonl", "a")
        while not self.done():
            if self.stop_requested:
                self.save()
                print(f"stop requested: saved at step {self.step}", flush=True)
                return 0
            info = self.train_step()
            rows_acc += info["rows"]
            if self.step % tc.log_every == 0:
                dt = time.time() - t0
                rec = {
                    "step": self.step,
                    "pflops": round(self.flops / 1e15, 4),
                    "row_rounds": self.row_rounds,
                    "lr": tc.lr * self.lr_factor(),
                    "tflops_per_s": round((self.flops - f0) / max(dt, 1e-9) / 1e12, 2),
                    "rows_per_s": round(rows_acc / max(dt, 1e-9), 1),
                    **info,
                    **self.stats.summary(),
                }
                if self.device.type == "cuda":
                    rec["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(
                    f"step {self.step} {rec['pflops']:.3f}PF loss {info['loss']:.4f} exact {rec.get('exact', 0):.3f} "
                    f"S{info['S']} {rec['tflops_per_s']}TF/s",
                    flush=True,
                )
                t0, f0, rows_acc = time.time(), self.flops, 0
            if self.flops / 1e15 >= self.next_eval:
                self.next_eval += tc.eval_every_pflops
                res = self.evaluate()
                print(f"eval step {self.step}: score {res['score']:.4f} (best {self.best:.4f})", flush=True)
            if time.time() - last_ckpt > tc.ckpt_minutes * 60:
                self.save()
                last_ckpt = time.time()
        self.save()
        res = self.evaluate(final=True)
        print(f"final eval: score {res['score']:.4f}", flush=True)
        (self.run_dir / "DONE").write_text(json.dumps({"step": self.step, "pflops": self.flops / 1e15}))
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--resume", default="auto", help="auto | none | path to checkpoint")
    ap.add_argument("overrides", nargs="*", help="dot-path overrides, e.g. train.lr=3e-4")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    if (run_dir / "DONE").exists():
        print(f"{run_dir} is DONE; nothing to do", flush=True)
        return 0
    cfg = Cfg.load(args.config, args.overrides)
    trainer = Trainer(cfg, run_dir, args.resume)
    return trainer.run()


if __name__ == "__main__":
    sys.exit(main())
