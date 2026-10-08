# frm — Forking Refinement Model for ARC-AGI

**Trains forks: yes.** Every training round of every non-control configuration forks K children from the parent
state, runs them through the same tied-weight TRM refiner, and joins them back into the parent before the graded readout
(`frm/model/refiner.py`, gated by `tests/test_model.py::test_fork_gate`). The only forkless configuration is the
`trm` control arm.

frm keeps Samsung's TRM (2-layer tied transformer, answer state y and latent z, H/L cycles, online deep supervision,
ACT halting, puzzle-id embeddings, stablemax CE, EMA) and changes three things:

1. **Fork-join inside every round.** The middle cycle of TRM's three is widened into K children with distinct learned
   codes; a per-cell attention join over {parent, children} feeds the parent's last cycle, whose readout is graded. The
   gradient runs through the join into the children: a child's intermediate state (a detour) is paid for by what the parent
   does with it.
2. **An objective for depth and meaningful forks.** Parent deep supervision + annealed relaxed winner-takes-all over
   children + order-free sub-goal matching on synthetic composites + stop/score heads (`docs/OBJECTIVE.md`).
3. **Data that demands depth.** Synthetic composite tasks f_k∘…∘f_1 (k ≤ 4, ≥50% genuine detours) with intermediate grids,
   mixed into ARC training (`frm/data/composite.py`).

Plus efficiency: size-bucketed canvases with 2-D RoPE (0.34–0.49 of TRM's 30×30 token cost on ARC-AGI-1 data),
per-pass activation checkpointing of children, FLOP-budgeted training.

## Documents

| doc | read for |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | the design and the code contract (start here) |
| [`docs/DEPTH.md`](docs/DEPTH.md) | why recursive refiners stop using depth, and frm's answer per mechanism |
| [`docs/OBJECTIVE.md`](docs/OBJECTIVE.md) | loss research and the objective, term by term |
| [`docs/LINEAGE.md`](docs/LINEAGE.md) | audit of frmv1, tiny, tinyv2.2, tinyabstractor, frmv2, TRM, ARM: reuse / avoid |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | arms, pre-registered decision rules, compute budget |
| [`kelvin/README.md`](kelvin/README.md) | running on Kelvin2 |

## Layout

```
frm/config.py          dataclass configs, JSON + dot-path overrides
frm/data/              ARC loading, augmentation, encoding, size rule, build CLI, loader, composite generator
frm/model/             layers (2-D RoPE), sparse puzzle embedding, the forking refiner
frm/objective.py       the objective
frm/train.py           trainer (per-bucket ACT carries, EMA, checkpoints, SIGTERM-safe, FLOP budget)
frm/evaluate.py        per-round voting, pass@k, fork/depth instruments, counterfactuals
frm/adapt.py           Kaggle adaptation -> submission.json
configs/                  arms (trm_a1, trm_h2_a1, frm2_a1, frm_a1, nest_a1, frm_a2_kaggle, frm_a2), smoke, learn
kelvin/                   Kelvin2 job kit
scripts/                  fetch_data.py, bench.py, compare.py
tests/                    84 tests incl. the end-to-end smoke
```

## Quick start (laptop, CPU)

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python torch numpy pytest
.venv/bin/python -m pytest -q                                   # 84 tests, ~25 s
.venv/bin/python scripts/fetch_data.py                          # ARC-AGI-1/2 + ConceptARC into data/raw
.venv/bin/python -m frm.data.composite --n 2000 --seed 0     # data/raw/composite_s0.json
.venv/bin/python -m frm.data.build --sources arc1 --composite data/raw/composite_s0.json \
    --n_aug 300 --dev_frac 0.25 --out data/built/a1
.venv/bin/python -m frm.train --config configs/smoke.json --run_dir runs/smoke train.device=cpu
.venv/bin/python -m frm.evaluate --run_dir runs/smoke --split dev
```

Learnability check on CPU (≈10 min): build `data/built/learn` from 60 small composites
(`python -m frm.data.composite --n 60 --seed 7 --depth_min 1 --depth_max 2 --max_side 6 --out data/raw/composite_learn.json`,
then `python -m frm.data.build --composite data/raw/composite_learn.json --out data/built/learn --n_aug_composite 2`),
then `python -m frm.train --config configs/learn.json --run_dir runs/learn`. Expected: loss ≈3.7 → ≈1.1 in 600
steps, CE decreasing with every round.

## Training and evaluation

```bash
python -m frm.train --config configs/frm_a1.json --run_dir runs/frm_a1 [key=value ...]   # resumable, SIGTERM-safe
python -m frm.evaluate --run_dir runs/frm_a1 --split dev                               # decisions use dev
python -m frm.evaluate --run_dir runs/frm_a1 --split test --final                       # the reported number
python scripts/bench.py --config configs/frm_a1.json --arms "trm:model.K=0" "frm2:model.K=2" "frm4:"
```

## Kaggle (ARC Prize 2026, ARC-AGI-2 track)

```bash
python -m frm.adapt --ckpt runs/frm_a2_kaggle/best.pt --challenges arc-agi_test_challenges.json \
    --work /kaggle/working/frm --hours 8 --submission submission.json
```

Builds the hidden tasks, gives their augmented versions fresh embedding rows, fine-tunes all weights on their demos for
the time budget, votes over versions, and writes two attempts per test input. With `--solutions` the same command is the
local leaderboard proxy (pretrain on `a2_kaggle`, adapt on ARC-AGI-2 evaluation).

## Status

Built and verified on CPU (tests, smoke, learnability). Not yet run on GPU; next steps are the Kelvin tiers T1–T3 in
`docs/EXPERIMENTS.md`, which need the user's approval for anything beyond the default limits.
