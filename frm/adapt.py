"""Kaggle adaptation (DESIGN.md §8): a pretrained checkpoint + unseen tasks -> submission.json.

    python -m frm.adapt --ckpt runs/frm_a2_kaggle/best.pt --challenges arc-agi_test_challenges.json \
        --work /kaggle/working/frm --hours 8 [--solutions FILE] [--n_aug 300] [--submission submission.json] \
        [key=value ...]

Hidden tasks have no puzzle identifiers in the checkpoint, so: build them (demos -> train rows, test inputs -> test
rows), give their identifiers fresh embedding rows (mean of the trained rows), fine-tune all weights on the demos for
the time budget (McGovern 2025: embeddings-only adaptation was not enough), then vote over versions and write two
attempts per test input. With ``--solutions`` (the local proxy on ARC-AGI-2 evaluation) the score is printed too.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from frm.config import Cfg
from frm.data import build as build_mod
from frm.data.loader import Dataset
from frm.evaluate import evaluate_split
from frm.train import Trainer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="pretrained checkpoint (best.pt or last.pt)")
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--solutions", default=None)
    ap.add_argument("--work", required=True, help="working directory for the dataset and the adaptation run")
    ap.add_argument("--n_aug", type=int, default=300)
    ap.add_argument("--hours", type=float, default=8.0, help="wall-clock budget for fine-tuning")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--submission", default="submission.json")
    ap.add_argument("overrides", nargs="*", help="dot-path overrides applied to the checkpoint's config")
    args = ap.parse_args(argv)

    work = Path(args.work)
    data_dir = work / "data"
    if not (data_dir / "meta.json").exists():
        build_argv = ["--sources", "kaggle", "--challenges", args.challenges, "--out", str(data_dir)]
        build_argv += ["--n_aug", str(args.n_aug)]
        if args.solutions:
            build_argv += ["--solutions", args.solutions]
        build_mod.main(build_argv)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, ck_price = Cfg.from_dict(ck["cfg"]), ck.get("price")
    del ck  # the table and optimizer state are loaded again by the trainer; do not hold a second copy
    cfg.data.root = str(data_dir)
    cfg.data.buckets = None
    cfg.data.source_weights = None
    cfg.train.init_from = args.ckpt
    cfg.train.lr = args.lr
    cfg.train.warmup_steps = args.warmup_steps
    cfg.train.max_hours = args.hours
    cfg.train.max_steps = args.max_steps
    cfg.train.budget_pflops = float("inf")
    cfg.train.eval_every_pflops = float("inf")
    cfg.eval.splits = []
    # The pretrained children are already specialised: continue at the final WTA temperature, do not re-anneal from 1.
    cfg.objective.tau_start = cfg.objective.tau_end
    # Likewise a pretrained fork gate: no forced-fork familiarisation, and its labels keep the price it ended at.
    cfg.train.fork_force_frac = 0.0
    cfg.objective.fork_price = cfg.objective.fork_price if ck_price is None else ck_price
    for item in args.overrides:
        cfg.override(item)

    trainer = Trainer(cfg, work / "run", resume="auto")  # resumes an interrupted adaptation from last.pt
    if not (work / "run" / "DONE").exists():
        trainer.run()

    ds = Dataset(cfg.data.root)
    trainer.ema.swap(trainer.model)
    res = evaluate_split(trainer.model, ds, "test", cfg.eval, trainer.device, trainer.dtype, keep_attempts=True)
    trainer.ema.swap(trainer.model)
    Path(args.submission).write_text(json.dumps(res.pop("attempts")))
    summary = {"steps": trainer.step, "pflops": trainer.flops / 1e15, "submission": args.submission,
               "versions": res["versions"]}
    if args.solutions:
        summary.update({k: res[k] for k in ("pass@1", "pass@2", "n_tasks", "per_task")})
        summary["pass@2_by_round"] = res["by_round"]["pass@2"]  # per_task feeds scripts/compare.py (paired adev tests)
    (work / "adapt_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
