"""Configuration: dataclasses loaded from JSON with dot-path overrides (``train.lr=3e-4``)."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelCfg:
    hidden: int = 512
    heads: int = 8
    expansion: float = 4.0
    layers: int = 2
    H_cycles: int = 3
    L_cycles: int = 4
    prefix_len: int = 16
    vocab: int = 12
    rope_theta: float = 10000.0
    # "2d": row/column RoPE, prefix unrotated. "trm1d": TRM's 1-D RoPE at 30x30-canvas indices (imported TRM).
    pos_encoding: str = "2d"
    rms_eps: float = 1e-5
    # Forking. K=0 is the forkless TRM control.
    K: int = 4
    fork_depth: int = 1
    child_L_cycles: int | None = None  # None = L_cycles
    code_rank: int = 64
    code_init_std: float = 0.5
    join_dim: int = 64
    # Per-round, per-child latent noise dim mapped into the child code; 0 disables. noise_init: initial output RMS.
    noise_dim: int = 16
    noise_init: float = 0.25
    # Training only: per row/round probability of masking the join's parent slot (OBJECTIVE.md §7); 0 disables.
    join_drop_parent: float = 0.0
    # Adaptive forking (DESIGN.md §4.8); off = every round forks. Needs fork_depth = 1.
    fork_gate: bool = False
    gate_hidden: int = 128
    # Gradient window (cycles) of the K=0 control and of gate-declined rows; TRM = 1.
    grad_cycles: int = 1
    grad_ckpt_children: bool = False


@dataclass
class ObjectiveCfg:
    lambda_wta: float = 0.5
    lambda_sub: float = 0.5
    lambda_q: float = 0.5
    wta_eps: float = 0.05
    tau_start: float = 1.0
    tau_end: float = 0.02
    tau_anneal_frac: float = 0.3
    # Fork gate BCE (OBJECTIVE.md §1.6); fork_price in nats per supervised token.
    lambda_gate: float = 0.1
    fork_price: float = 0.0
    # Gated models: weight 1 - child_focus on cells the no-fork continuation gets right in child CE; 0 = uniform.
    child_focus: float = 0.0


@dataclass
class DataCfg:
    root: str = "data/built/a1"
    # Row fractions per source; absent sources dropped, rest renormalised.
    source_weights: dict[str, float] | None = field(
        default_factory=lambda: {"arc": 0.5, "composite": 0.15, "barc": 0.25, "rearc": 0.10}
    )
    translate_p: float = 0.75
    tokens_per_step: int = 120_000  # parent tokens (prefix included) per optimizer step
    max_rows: int = 4096
    max_pairs_per_draw: int = 16  # max pairs of one task per draw
    buckets: list[int] | None = None  # None = every bucket in the dataset


@dataclass
class TrainCfg:
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    # Warmup-stable-decay: (1-sqrt) cooldown over the last decay_frac of the budget to decay_to x lr; 0 = constant.
    decay_frac: float = 0.0
    decay_to: float = 0.0
    emb_lr: float = 1e-2
    emb_weight_decay: float = 0.1
    grad_clip: float = 1.0
    ema: float = 0.999
    max_rounds: int = 16
    halt_explore: float = 0.1
    # Gate schedule: forced forks for the first fork_force_frac, then declined rows fork w.p. fork_explore.
    # fork_rate_target > 0: price = max(0, price + fork_price_lr * (fork rate - target)) per gated step.
    fork_force_frac: float = 0.3
    fork_explore: float = 0.1
    fork_rate_target: float = 0.0
    fork_price_lr: float = 0.01
    # lr multiplier for fork modules (codes, join, noise map, gate); 1 = single parameter group.
    fork_lr_mult: float = 1.0
    budget_pflops: float = 1000.0  # training stops when this much compute is spent
    # Early stop at this compute; schedule still follows budget_pflops. 0 = off.
    stop_pflops: float = 0.0
    max_steps: int = 0  # 0 = budget only
    max_hours: float = 0.0  # cumulative wall-clock limit; 0 = none
    init_from: str | None = None  # checkpoint whose weights initialise the model (see init_mode)
    # "adapt": new tasks, embedding rows start at the mean trained row. "continue": same dataset, table kept.
    init_mode: str = "adapt"
    # With --resume auto and no RUN/last.pt: resume the full training state from this checkpoint.
    resume_from: str | None = None
    eval_every_pflops: float = 100.0
    log_every: int = 50
    ckpt_minutes: float = 30.0
    seed: int = 0
    precision: str = "auto"  # auto | bf16 | fp16 | fp32
    compile: bool = False
    device: str = "auto"


@dataclass
class EvalCfg:
    splits: list[str] = field(default_factory=lambda: ["dev", "dev_composite"])
    max_versions: int = 32
    max_tasks: int = 256  # per split, hash-ordered subset; 0 = all
    tokens_per_batch: int = 400_000
    max_rows: int = 2048
    rounds: int = 16
    child_vote_weight: float = 0.0
    # Gated models: 0 = gate decides (logit > 0); rho > 0 = top rho of rows by gate logit fork per batch and round.
    fork_top: float = 0.0
    counterfactuals: bool = True
    cf_rows: int = 1024
    use_ema: bool = True
    # Test-time noisy runs: noise_samples - 1 extra voters with state_noise RMS Gaussian noise on z between rounds.
    noise_samples: int = 1
    state_noise: float = 0.0


@dataclass
class Cfg:
    name: str = "frm"
    model: ModelCfg = field(default_factory=ModelCfg)
    objective: ObjectiveCfg = field(default_factory=ObjectiveCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Cfg":
        cfg = Cfg()
        for key, value in d.items():
            sub = getattr(cfg, key)
            if dataclasses.is_dataclass(sub):
                _assign(sub, value, key)
            else:
                setattr(cfg, key, value)
        return cfg

    @staticmethod
    def load(path: str | Path, overrides: list[str] | None = None) -> "Cfg":
        cfg = Cfg.from_dict(json.loads(Path(path).read_text()))
        for item in overrides or []:
            cfg.override(item)
        return cfg

    def override(self, item: str) -> None:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        path, raw = item.split("=", 1)
        *parents, leaf = path.split(".")
        target: Any = self
        for p in parents:
            target = getattr(target, p)
        if not hasattr(target, leaf):
            raise KeyError(f"unknown config key {path!r}")
        setattr(target, leaf, _parse(raw))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _assign(obj: Any, values: dict[str, Any], where: str) -> None:
    names = {f.name for f in dataclasses.fields(obj)}
    for k, v in values.items():
        if k not in names:
            raise KeyError(f"unknown config key {where}.{k}")
        setattr(obj, k, v)


def _parse(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw
