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
    rms_eps: float = 1e-5
    # Forking. K=0 is the forkless TRM control.
    K: int = 4
    fork_depth: int = 1
    child_L_cycles: int | None = None  # None = L_cycles
    code_rank: int = 64
    code_init_std: float = 0.5
    join_dim: int = 64
    # Gradient window of the K=0 control (TRM = 1 cycle; trm_h2 = 2).
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


@dataclass
class DataCfg:
    root: str = "data/built/a1"
    source_weights: dict[str, float] | None = field(default_factory=lambda: {"arc": 0.75, "composite": 0.25})
    translate_p: float = 0.75
    tokens_per_step: int = 120_000  # parent tokens (prefix included) per optimizer step; T2 bench re-sizes it
    max_rows: int = 4096
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
    budget_pflops: float = 1000.0  # training stops when this much compute is spent
    max_steps: int = 0  # 0 = budget only
    max_hours: float = 0.0  # 0 = no wall-clock limit (used by Kaggle adaptation)
    init_from: str | None = None  # pretrained checkpoint: weights loaded, puzzle-embedding rows re-initialised
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
    tokens_per_batch: int = 400_000
    max_rows: int = 2048
    rounds: int = 16
    child_vote_weight: float = 0.0
    counterfactuals: bool = True
    cf_rows: int = 1024
    use_ema: bool = True


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
