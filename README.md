# frm — Forking Refinement Model for ARC-AGI

**Trains forks: yes.** Every non-control configuration forks K children from the parent state in its training rounds
(every round, or with the adaptive gate on the rows it picks: 40% of row-rounds in the ARC-AGI-2 final), runs them
through the same tied-weight TRM refiner, and joins them back into the parent before the graded readout
(`frm/model/refiner.py`, gated by `tests/test_model.py::test_fork_gate` and `tests/test_gate.py`). The only
forkless configuration is the `trm` control arm.

frm keeps Samsung's TRM (2-layer tied transformer, answer state y and latent z, H/L cycles, online deep supervision,
ACT halting, puzzle-id embeddings, stablemax CE, EMA) and changes three things:

1. **Fork-join inside the round.** The middle cycle of TRM's three is widened into K children with distinct learned
   codes plus fresh noise each round; a per-cell attention join over {parent, children} feeds the parent's last cycle,
   whose readout is graded. The gradient runs through the join into the children: a child's intermediate state (a
   detour) is paid for by what the parent does with it, and a parent stuck at a fixed point still gets new proposals.
2. **An objective for depth and meaningful forks.** Parent deep supervision + annealed one-to-one set matching of the
   children to {final answer} ∪ {intermediate grids of synthetic composites} (winner-takes-all on plain rows) +
   stop/score heads (`docs/OBJECTIVE.md` §3.0 writes it out).
3. **Data.** ARC + ConceptARC + synthetic composite tasks f_k∘…∘f_1 (k ≤ 4, ≥50% genuine detours, with intermediate
   grids) + BARC + RE-ARC, all filtered against every held-out set (`docs/DESIGN.md` §3.4–3.6).

Plus efficiency: size-bucketed canvases with 2-D RoPE (0.34–0.49 of TRM's 30×30 token cost on ARC-AGI-1 data; warm
starts from a TRM checkpoint keep ARC on its 30×30 canvas with TRM's 1-D RoPE), RE-ARC split per bucket, adaptive
forking at a held rate with fixed-rate forking at inference, the memory-efficient attention kernel on ROCm, per-pass
activation checkpointing of children, FLOP-budgeted training with a cooldown.

## Documents

| doc | read for |
|---|---|
| [`docs/HANDOFF.md`](docs/HANDOFF.md) | **continuing the project**: state, constraints, results, the running final, commands |
| [`docs/DESIGN.md`](docs/DESIGN.md) | the design and the code contract (start here) |
| [`docs/DEPTH.md`](docs/DEPTH.md) | why recursive refiners stop using depth, and frm's answer per mechanism |
| [`docs/OBJECTIVE.md`](docs/OBJECTIVE.md) | loss research and the objective, term by term |
| [`docs/LINEAGE.md`](docs/LINEAGE.md) | audit of frmv1, tiny, tinyv2.2, tinyabstractor, frmv2, TRM, ARM: reuse / avoid |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | arms, pre-registered decision rules, compute budget |
| [`kelvin/README.md`](kelvin/README.md) | running on Kelvin2 |
| [`scripts/monitor.md`](scripts/monitor.md) | the live monitor: what every number means, good and bad readings |

## Layout

```
frm/config.py          dataclass configs, JSON + dot-path overrides
frm/data/              ARC loading, augmentation, encoding, size rule, build CLI, loader, composite generator
frm/model/             layers (2-D RoPE, TRM's 1-D RoPE, SDPA kernel choice), puzzle embedding, the forking refiner
frm/objective.py       the objective
frm/train.py           trainer (per-bucket ACT carries, EMA, checkpoints, SIGTERM-safe, FLOP budget, cooldown)
frm/evaluate.py        per-round voting, pass@k, fork/depth instruments, counterfactuals, fixed-rate forking
frm/adapt.py           Kaggle adaptation -> submission.json
frm/monitor/           the live monitor's source (Kelvin over the ssh master), run state, charts, pages
configs/                  pilots trm_a1, frm_a1, frm2_a1, frm_a1_arc; optional trm_h2_a1, nest_a1;
                          warm start from TRM: trm_import, trmc_a1tn (TRM continued), frmc_a1tn (FRM continued, gated),
                          frmc40_a1tn / frmcf40_a1tn (R: fork-rate targeting / + error-focused children);
                          the ARC-AGI-2 final: final_frm_a2tn and its TRM control final_trm_a2tn
kelvin/                   Kelvin2 job kit
scripts/                  fetch_data.py, ingest_external.py, make_adapt_dev.py, bench.py, compare.py, monitor.py,
                          import_trm.py (published TRM checkpoint -> frm checkpoint aligned with a frm dataset),
                          arm_curves.py (two arms' curves: pooled paired windows, slopes, matched steps/PF)
tests/                    unit tests + the end-to-end smoke
```

## Quick start (laptop, CPU)

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python torch numpy pytest
.venv/bin/python -m pytest -q                                   # ~25 s
.venv/bin/python scripts/fetch_data.py                          # ARC-AGI-1/2 + ConceptARC into data/raw
.venv/bin/python -m frm.data.composite --n 2000 --seed 0     # data/raw/composite_s0.json
S=../tinyabstractor/data/cache
.venv/bin/python scripts/ingest_external.py --src $S --pools a2_new  # data/raw/external/a2_new.npz
.venv/bin/python scripts/make_adapt_dev.py                       # data/raw/adev_*.json: 120 a2_new tasks
.venv/bin/python scripts/ingest_external.py --src $S --pools barc rearc --heldout_pools data/raw/external/a2_new.npz
.venv/bin/python -m frm.data.build --sources arc1 --composite data/raw/composite_s0.json --n_aug 300 \
    --external barc=data/raw/external/barc.npz rearc=data/raw/external/rearc.npz \
    --exclude_ids data/raw/adev_ids.json --out data/built/a1
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
python -m frm.evaluate --run_dir runs/frm_a1 --split dev                               # checkpoint choice
python -m frm.adapt --ckpt runs/frm_a1/best.pt --challenges data/raw/adev_challenges.json \
    --solutions data/raw/adev_solutions.json --work runs/frm_a1/adev --hours 2.5          # adev: unseen tasks
python -m frm.evaluate --run_dir runs/frm_a1 --split test --final                       # the reported number
python scripts/compare.py runs/frm_a1/eval.jsonl runs/trm_a1/eval.jsonl --split dev [--match samples]
python scripts/bench.py --config configs/frm_a1.json --arms "trm:model.K=0" "frm2:model.K=2" "frm4:"
kelvin/k2 monitor                                       # live: every active run on Kelvin2 (scripts/monitor.md)
```

## Kaggle (ARC Prize 2026, ARC-AGI-2 track)

```bash
python -m frm.adapt --ckpt runs/final_frm_a2tn/last.pt --challenges arc-agi_test_challenges.json \
    --work /kaggle/working/frm --hours 8 --submission submission.json \
    train.fork_rate_target=0.25 train.fork_price_lr=1e-4 eval.fork_top=0.25
```

Builds the hidden tasks, gives their augmented versions fresh embedding rows, fine-tunes all weights on their demos for
the time budget, votes over the versions it trained, and writes two attempts per test input. The three overrides hold
the fork rate at 0.25 during adaptation and at inference (`docs/EXPERIMENTS.md` §4c `A`: adev pass@2 0.235 against
0.190 with dense forks and 0.252 for TRM-continued in the same 2.5 h). Adaptation keeps the pretraining's 160k
tokens/step, which needs 139–145 GB for FRM on an MI300X: on a smaller GPU add a lower `data.tokens_per_step` and
`model.grad_ckpt_children=true` (untested there). With `--solutions` the same command scores itself; for a model
pretrained on `a2tn` the ARC-AGI-2 evaluation demos were in pretraining, so scoring it there is transductive, not an
unseen-task proxy.

## Status

Built and verified on CPU (tests, smoke, learnability) and on Kelvin's MI300X (ROCm venv, smoke incl. SIGTERM/resume;
compiled, with the memory-efficient attention kernel: `trm` ≈266, FRM K=4 at a 0.4 fork rate ≈238 TFLOP/s). P1 (from
scratch): FRM ≥ TRM per step once the join's parent slot is dropped in training, 3–4× the cost per step. W
(`docs/EXPERIMENTS.md` §4b, from the ARC Prize verified TRM, 36,000 PF per arm): FRM ahead per step throughout, equal
to TRM-continued at matched compute on ARC dev, a steeper composite slope, behind on time-limited adaptation. R
(§4c): the gate cannot rank forks, but forking the top 25–40% at evaluation keeps all of the accuracy, and holding
training at a 0.4 fork rate beats dense forking on composites at matched PF while tying TRM-continued on dev;
targeted adaptation closes most of the adev deficit. **The ARC-AGI-2 final** (R1's configuration from
`arc_v2_public`, 100,000 PF, up to 2 × 72 GPU-h) and its TRM-continued control are training, due ≈ 2026-10-17 20:15
(TRM) and ≈ 2026-10-18 12:30 (FRM). To 32.5k PF, FRM leads on held-out composites by a growing margin (+0.067 pass@2
pooled over 15–32.5k, CI > 0; 0.482 vs 0.381 at 32.5k) and ties on the 40-task dev, where both solve 0–2 tasks
(`docs/HANDOFF.md` §8).
