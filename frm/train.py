"""Trainer: TRM's online deep supervision with one carry per canvas bucket (docs/DESIGN.md §6).

    python -m frm.train --config configs/frm_a1.json --run_dir runs/frm_a1 [--resume auto] [key=value ...]

Contract (kelvin/jobs/train.sbatch relies on it): checkpoints to RUN/last.pt every ``ckpt_minutes``; SIGTERM/SIGUSR1
-> checkpoint and exit 0; when the compute budget (``train.budget_pflops``) or ``train.max_steps`` is reached ->
final checkpoint, final dev evaluation, RUN/DONE, exit 0; any failure -> nonzero exit.
"""

from __future__ import annotations

import argparse
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
from frm.data.loader import Dataset, TrainStream
from frm.evaluate import evaluate_split
from frm.model.refiner import Carry, Refiner
from frm.objective import Objective

ROUND_METRICS = ("exact", "ce", "child_exact_any", "parent_mass", "distinct", "sub_coverage")


# ---------------------------------------------------------------------------------------------------- utilities
def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(precision: str, device: torch.device) -> torch.dtype:
    if precision == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


class EMA:
    """Exponential moving average of dense parameters (the puzzle-embedding buffer is excluded, as in TRM)."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            self.shadow[n].lerp_(p.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def swap(self, model: torch.nn.Module) -> None:
        """Exchange model parameters and shadow in place (call twice to restore)."""
        for n, p in model.named_parameters():
            tmp = p.detach().clone()
            p.copy_(self.shadow[n])
            self.shadow[n].copy_(tmp)


def load_pretrained(model: Refiner, path: str, device: torch.device) -> None:
    """Load dense weights and fixed buffers from a checkpoint (EMA weights when present).

    The puzzle-embedding table belongs to the new dataset's identifiers: every row except the blank becomes the mean
    trained row of the checkpoint (TRM's reset-to-mean, applied only to the new table).
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    state = dict(ck["model"])
    if "ema" in ck:
        state.update(ck["ema"])
    old = state.pop("puzzle_emb.table")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing != ["puzzle_emb.table"] or unexpected:
        raise ValueError(f"checkpoint/model mismatch: missing {missing}, unexpected {unexpected}")
    with torch.no_grad():
        model.puzzle_emb.table.copy_(old[1:].mean(0).expand_as(model.puzzle_emb.table))
        model.puzzle_emb.table[0].zero_()


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


class RoundStats:
    """Accumulates per-row metrics by round index (1..max_rounds) between log lines."""

    def __init__(self, max_rounds: int):
        self.R = max_rounds
        self.sums: dict[str, np.ndarray] = {}
        self.counts: dict[str, np.ndarray] = {}
        self.first_solve = np.zeros(max_rounds + 1)
        self.regress = np.zeros(max_rounds + 1)

    def add(self, name: str, rounds: torch.Tensor, values: torch.Tensor) -> None:
        v = values.detach().float()
        keep = ~torch.isnan(v)
        r = rounds[keep].clamp(1, self.R).cpu().numpy()
        v = v[keep].cpu().numpy()
        s = self.sums.setdefault(name, np.zeros(self.R + 1))
        c = self.counts.setdefault(name, np.zeros(self.R + 1))
        np.add.at(s, r, v)
        np.add.at(c, r, 1)

    def summary(self) -> dict:
        out = {}
        for name, s in self.sums.items():
            c = self.counts[name]
            out[f"{name}_by_round"] = [round(float(s[r] / c[r]), 5) if c[r] else None for r in range(1, self.R + 1)]
            out[name] = float(s.sum() / max(c.sum(), 1))
        out["first_solve_by_round"] = self.first_solve[1:].astype(int).tolist()
        out["regress_by_round"] = self.regress[1:].astype(int).tolist()
        self.__init__(self.R)
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
            load_pretrained(self.model, tc.init_from, self.device)
        if tc.compile:
            self.model.compile_cycles()
        self.objective = Objective(cfg.objective)
        fused = self.device.type == "cuda"
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=tc.lr, betas=(tc.beta1, tc.beta2), weight_decay=tc.weight_decay, fused=fused
        )
        self.scaler = torch.amp.GradScaler("cuda") if self.dtype == torch.float16 else None
        self.ema = EMA(self.model, tc.ema)
        self.step = 0
        self.flops = 0.0
        self.best = -1.0
        self.next_eval = tc.eval_every_pflops
        P = cfg.model.prefix_len
        buckets = cfg.data.buckets or self.ds.buckets
        self.buckets = [S for S in buckets if self.ds.num_train_examples(S) > 0]
        self.rows = {S: max(1, min(cfg.data.max_rows, cfg.data.tokens_per_step // (P + S * S))) for S in self.buckets}
        w = np.array([self.ds.num_train_examples(S) / self.rows[S] for S in self.buckets], dtype=np.float64)
        self.bucket_p = w / w.sum()
        self.rng = np.random.default_rng(tc.seed)
        self.streams = {
            S: TrainStream(
                self.ds, S, np.random.default_rng([tc.seed, S]), cfg.data.source_weights, cfg.data.translate_p
            )
            for S in self.buckets
        }
        self.states: dict[int, BucketState] = {}
        self.stats = RoundStats(tc.max_rounds)
        self.stop_requested = False
        self.t_start = time.time()
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
            "best": self.best,
            "next_eval": self.next_eval,
            "streams": {S: s.state_dict() for S, s in self.streams.items()},
            "rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cfg": self.cfg.to_dict(),
            "num_ids": self.ds.num_ids,
        }

    def save(self, name: str = "last.pt") -> None:
        path = self.run_dir / name
        tmp = path.with_suffix(".tmp")
        torch.save(self._ckpt_state(), tmp)
        os.replace(tmp, path)

    def _maybe_resume(self, resume: str) -> None:
        path = self.run_dir / "last.pt" if resume == "auto" else Path(resume)
        if not path.exists():
            if resume != "auto":
                raise FileNotFoundError(path)
            return
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.ema.shadow = {k: v.to(self.device) for k, v in ck["ema"].items()}
        self.opt.load_state_dict(ck["opt"])
        if self.scaler and ck["scaler"]:
            self.scaler.load_state_dict(ck["scaler"])
        self.step, self.flops, self.best, self.next_eval = ck["step"], ck["flops"], ck["best"], ck["next_eval"]
        for S, st in ck["streams"].items():
            if S in self.streams:
                self.streams[S].load_state_dict(st)
        self.rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        print(f"resumed {path} at step {self.step}, {self.flops / 1e15:.1f} PFLOP", flush=True)

    # ------------------------------------------------------------------ one step
    def _state(self, S: int) -> BucketState:
        if S not in self.states:
            B = self.rows[S]
            empty = to_device(self.streams[S].take(B), self.device)
            self.states[S] = BucketState(
                S=S,
                rows=B,
                carry=self.model.init_carry(B, S, self.device, self.dtype),
                batch=empty,
                steps=torch.zeros(B, dtype=torch.int32, device=self.device),
                halted=torch.zeros(B, dtype=torch.bool, device=self.device),
                solved=torch.zeros(B, dtype=torch.bool, device=self.device),
                prev_exact=torch.zeros(B, dtype=torch.bool, device=self.device),
            )
        return self.states[S]

    def _refill(self, st: BucketState) -> None:
        idx = st.halted.nonzero().squeeze(-1)
        if idx.numel() == 0:
            return
        new = to_device(self.streams[st.S].take(int(idx.numel())), self.device)
        for k, v in new.items():
            st.batch[k][idx] = v
        st.carry = self.model.reset_carry(st.carry, st.halted)
        st.steps[idx] = 0
        st.solved[idx] = False
        st.prev_exact[idx] = False

    def lr_factor(self) -> float:
        return min(1.0, (self.step + 1) / max(1, self.cfg.train.warmup_steps))

    def progress(self) -> float:
        tc = self.cfg.train
        p = self.flops / (tc.budget_pflops * 1e15)
        if tc.max_steps:
            p = max(p, self.step / tc.max_steps)
        return min(1.0, p)

    def train_step(self) -> dict:
        tc = self.cfg.train
        S = int(self.rng.choice(self.buckets, p=self.bucket_p))
        st = self._state(S)
        self._refill(st)
        self.model.train()
        f = self.lr_factor()
        for g in self.opt.param_groups:
            g["lr"] = tc.lr * f
        out = self.model.round(st.carry, st.batch["inputs"], st.batch["puzzle_ids"])
        loss, m = self.objective(out, st.batch, self.progress())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {self.step}")
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
        else:
            loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
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
            rounds = st.steps.long()
            for name in ROUND_METRICS:
                if name in m:
                    self.stats.add(name, rounds, m[name].float())
            exact = m["exact"]
            r = rounds.clamp(1, tc.max_rounds).cpu().numpy()
            np.add.at(self.stats.first_solve, r, (exact & ~st.solved).cpu().numpy())
            np.add.at(self.stats.regress, r, (st.prev_exact & ~exact).cpu().numpy())
            st.solved |= exact
            st.prev_exact = exact
            halted = st.steps >= tc.max_rounds
            if tc.max_rounds > 1:
                halted |= out.q > 0
                explore = torch.rand(st.rows, device=self.device) < tc.halt_explore
                min_rounds = torch.randint(2, tc.max_rounds + 1, (st.rows,), device=self.device)
                halted &= st.steps >= torch.where(explore, min_rounds, 0)
            st.halted = halted
        self.flops += st.rows * self.model.train_flops_per_row(S)
        self.step += 1
        info = {"loss": loss.item(), "gnorm": gnorm.item(), "S": S, "rows": st.rows}
        for k in ("wta", "sub", "q_loss", "q_acc", "tau", "winner"):
            if k in m:
                v = m[k].float()
                v = v[~torch.isnan(v)]
                if v.numel():
                    info[k] = v.mean().item() if k != "winner" else np.bincount(
                        m[k].cpu().numpy(), minlength=self.cfg.model.K
                    ).tolist()
        return info

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, final: bool = False) -> dict:
        ec = self.cfg.eval
        if ec.use_ema:
            self.ema.swap(self.model)
        try:
            res = {"step": self.step, "pflops": self.flops / 1e15}
            for split in ec.splits:
                if split in self.ds.splits():
                    res[split] = evaluate_split(
                        self.model, self.ds, split, ec, self.device, self.dtype, counterfactuals=ec.counterfactuals
                    )
            score = np.mean([res[s]["pass@2"] for s in ec.splits if s in res and "pass@2" in res[s]] or [0.0])
            res["score"] = float(score)
            if score > self.best:
                self.best = float(score)
                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "step": self.step,
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
        tc = self.cfg.train
        over_time = tc.max_hours > 0 and time.time() - self.t_start > tc.max_hours * 3600
        return over_time or self.flops >= tc.budget_pflops * 1e15 or (tc.max_steps > 0 and self.step >= tc.max_steps)

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
