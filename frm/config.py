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
    # "2d": row/column RoPE, prefix unrotated (frm's own models). "trm1d": TRM's 1-D RoPE with every cell at its
    # 30x30-canvas index (prefix i -> i, cell (r, c) -> prefix_len + 30r + c), so a TRM checkpoint sees the relative
    # positions it was trained with on any bucket (scripts/import_trm.py).
    pos_encoding: str = "2d"
    rms_eps: float = 1e-5
    # Forking. K=0 is the forkless TRM control.
    K: int = 4
    fork_depth: int = 1
    child_L_cycles: int | None = None  # None = L_cycles
    code_rank: int = 64
    code_init_std: float = 0.5
    join_dim: int = 64
    # Fresh per-round, per-child latent noise xi ~ N(0, I) mapped into the child code (IMLE-style stochastic children):
    # without it, children are a deterministic function of the parent state and repeat themselves at a parent fixed
    # point. 0 disables. The map is learned; noise_init is the initial RMS of its output.
    noise_dim: int = 16
    noise_init: float = 0.25
    # Training only: probability per row and round that the join's parent slot is masked, so the post-join cycle must
    # build on the children and the parent's loss reaches them at full strength (OBJECTIVE.md §7, response to
    # parent_mass -> 1). Evaluation never drops. 0 disables.
    join_drop_parent: float = 0.0
    # "softmax": the join is a per-cell softmax over {parent pre-fork state, children} feeding the post cycle (§4.4).
    # "residual": function-preserving forks for warm starts (§4.10): the parent runs its own TRM cycle, the children's
    # attention mix is added to it through a per-cell tanh gate initialised at 0, so the model *is* the source TRM at
    # step 0 and every fork contribution has to be learned on top (zero-init branches: ReZero, Flamingo, ControlNet).
    # join_drop_parent then forces the gate to 1 on dropped rows.
    join_mode: str = "softmax"
    # Adaptive forking (DESIGN.md §4.8): a gate head on the pre-fork state decides per row and round whether to fork;
    # declined rows run TRM's round. Off = every round forks. Needs fork_depth = 1.
    fork_gate: bool = False
    gate_hidden: int = 128
    # Gradient window of the K=0 control (TRM = 1 cycle; trm_h2 = 2), and of the rows a fork gate declines.
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
    # Fork gate (OBJECTIVE.md §3.6): BCE of the gate logit against "forking beat the no-fork continuation by more than
    # fork_price nats per supervised token", on rows that forked with their own join.
    lambda_gate: float = 0.1
    fork_price: float = 0.0
    # Error-focused children (gated models): on label rows, cells that TRM's own round (the no-fork continuation)
    # already predicts right weigh 1 - child_focus in the children's final-target CE, so children specialise on the
    # parent's errors (boosting). 0 = uniform. Sub-goal targets are never reweighted.
    child_focus: float = 0.0


@dataclass
class DataCfg:
    root: str = "data/built/a1"
    # Fractions of training rows per source (sources absent from the dataset are dropped and the rest renormalised).
    source_weights: dict[str, float] | None = field(
        default_factory=lambda: {"arc": 0.5, "composite": 0.15, "barc": 0.25, "rearc": 0.10}
    )
    translate_p: float = 0.75
    tokens_per_step: int = 120_000  # parent tokens (prefix included) per optimizer step; T2 bench re-sizes it
    max_rows: int = 4096
    max_pairs_per_draw: int = 16  # a draw yields at most this many pairs of one task (RE-ARC tasks have ~1000)
    buckets: list[int] | None = None  # None = every bucket in the dataset


@dataclass
class TrainCfg:
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    emb_lr: float = 1e-2
    emb_weight_decay: float = 0.1
    grad_clip: float = 1.0
    ema: float = 0.999
    max_rounds: int = 16
    halt_explore: float = 0.1
    # Fork gate schedule: every row-round forks for the first fork_force_frac of training (familiarisation), then the
    # gate decides and a declined row forks anyway with probability fork_explore. fork_rate_target > 0 turns the price
    # into a controller: after each gated step price = max(0, price + fork_price_lr * (fork rate - target)).
    fork_force_frac: float = 0.3
    fork_explore: float = 0.1
    fork_rate_target: float = 0.0
    fork_price_lr: float = 0.01
    # Learning-rate multiplier for the fork modules (the parameters a K=0 model lacks: codes, join, noise map, gate).
    # After a warm start from a trained TRM they start from scratch while the shared weights only fine-tune; > 1 lets
    # them learn at their own pace. 1 keeps a single parameter group (and resumes pre-existing runs unchanged).
    fork_lr_mult: float = 1.0
    budget_pflops: float = 1000.0  # training stops when this much compute is spent
    max_steps: int = 0  # 0 = budget only
    max_hours: float = 0.0  # 0 = no wall-clock limit (used by Kaggle adaptation)
    init_from: str | None = None  # checkpoint whose weights initialise the model (see init_mode)
    # "adapt": new tasks (Kaggle adaptation): every puzzle-embedding row starts at the mean trained row, untrained.
    # "continue": the same dataset (e.g. an imported TRM, scripts/import_trm.py): the table and its visit counts are
    # kept; fork modules absent from the checkpoint (a K=0 source) keep their fresh initialisation.
    init_mode: str = "adapt"
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
    max_tasks: int = 256  # per split, deterministic hash-ordered subset (0 = all); ARC dev has ~140 tasks
    tokens_per_batch: int = 400_000
    max_rows: int = 2048
    rounds: int = 16
    child_vote_weight: float = 0.0
    counterfactuals: bool = True
    cf_rows: int = 1024
    use_ema: bool = True
    # Test-time noisy trajectories (PTRM): noise_samples - 1 extra runs per row with Gaussian noise of state_noise RMS
    # units added to z between rounds; their predictions are extra voters (the clean run keeps the instruments).
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
