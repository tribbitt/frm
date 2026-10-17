# FRM — Forking Refinement Model for ARC-AGI

FRM is Samsung's [TRM](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) (2-layer tied-weight transformer,
answer state y and latent z, H/L cycles, deep supervision, ACT halting, puzzle-id embeddings, stablemax CE, EMA) with
forking. With `K=0` it is exactly TRM. It changes three things:

1. **Fork-join inside the round.** The middle of TRM's three cycles is widened into K children, each with its own
   learned code and fresh noise every round. A per-cell attention join over {parent, children} feeds the parent's last
   cycle, and that cycle's readout is graded. The gradient flows through the join into the children. An adaptive gate
   decides per row and round whether to fork.
2. **An objective that rewards depth and meaningful forks.** It combines parent deep supervision, annealed one-to-one
   set matching of the children to {final answer} ∪ {intermediate grids of synthetic composite tasks}, stop/score
   heads and the gate's term ([`docs/OBJECTIVE.md`](docs/OBJECTIVE.md) §3.0).
3. **Data.** ARC, ConceptARC, synthetic composites f_k∘…∘f_1 (k ≤ 4, with intermediate grids), BARC and RE-ARC. Every
   source is filtered against every held-out set ([`docs/DESIGN.md`](docs/DESIGN.md) §3).

| doc | contents |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | design, decisions and the code contract |
| [`docs/OBJECTIVE.md`](docs/OBJECTIVE.md) | the objective, term by term |
| [`docs/DEPTH.md`](docs/DEPTH.md) | why recursive refiners stop using depth, and FRM's answer per mechanism |

## Layout

```
frm/config.py        dataclass configs, JSON + dot-path overrides (train.lr=3e-4)
frm/model/           layers (2-D RoPE, TRM's 1-D RoPE), puzzle embedding, the forking refiner
frm/objective.py     the objective
frm/train.py         trainer: per-bucket ACT carries, EMA, FLOP budget, cooldown, resumable, SIGTERM-safe
frm/evaluate.py      per-round voting, pass@k, fork and depth instruments
frm/adapt.py         test-time adaptation on new tasks -> Kaggle submission.json
frm/data/            ARC loading, augmentation, encoding, size rule, composite generator, dataset build, loader
frm/external/        builders of the external pools: RE-ARC and a2_new (corpus.py), BARC ARC-Heavy (barc/)
scripts/                fetch_data, ingest_external (BARC/RE-ARC filter), import_trm (TRM checkpoint import),
                        arm_curves (paired FRM-vs-TRM comparison of two runs)
configs/                final_frm_a2tn.json (FRM) and final_trm_a2tn.json (its K=0 TRM control)
tests/                  unit tests and an end-to-end train/resume/eval/adapt smoke
```

## Setup

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q    # CPU; tests that need ARC data skip until data/raw exists (step 1 below)
```

## Reproducing the final run

The final run continues the ARC Prize verified ARC-AGI-2 TRM checkpoint on dataset `a2tn` with two arms: FRM
(`configs/final_frm_a2tn.json`) and a TRM control (`configs/final_trm_a2tn.json`). Each arm gets 100,000 PFLOP of
training (`train.budget_pflops`), with a cooldown over the last 15%. Both ran on one 192 GB GPU each. The FRM arm
peaks near 160 GB at 160k tokens per step; on smaller GPUs add `model.grad_ckpt_children=true`.

**1. ARC data** (ARC-AGI-1, ARC-AGI-2, ConceptARC into `data/raw`):

```bash
python scripts/fetch_data.py
```

**2. Synthetic composite tasks:**

```bash
python -m frm.data.composite --n 2000 --seed 0          # data/raw/composite_s0.json
```

**3. External pools.** RE-ARC, `a2_new` (ARC-AGI-2 training tasks that are not ARC-AGI-1 tasks) and BARC
(ARC-Heavy) become FlatPool `.npz` files in `data/pools`. The BARC stages run untrusted LLM-written programs in a
sandbox (macOS `sandbox-exec`; on other platforms only with `BARC_UNSANDBOXED=1`). The audit takes hours. Details and
pinned revisions are in the docstrings of `frm/external/corpus.py` and `frm/external/barc/`.

```bash
git clone --depth 1 https://github.com/michaelhodel/re-arc data/external/re-arc
unzip -q data/external/re-arc/re_arc.zip -d data/external
python -m frm.external.corpus                       # data/pools/{rearc,a2_new}.npz
python -m frm.external.barc.fetch                   # pinned BARC dataset, BARC's common.py, sandbox venv (uv)
python -m frm.external.barc.sandbox --selftest
for s in parse execute filter select; do python -m frm.external.barc.build $s; done
python -m frm.external.barc.audit
python -m frm.external.barc.build write             # data/pools/barc.npz
```

Ingest filters the pools against every ARC-AGI-1/2 evaluation task, and filters BARC/RE-ARC against `a2_new` too:

```bash
python scripts/ingest_external.py --src data/pools --pools a2_new
python scripts/ingest_external.py --src data/pools --pools barc rearc --heldout_pools data/raw/external/a2_new.npz
```

A rebuild matches the pools the final run used closely, but not exactly. The audit's time limits depend on the
machine, and the original BARC selection also deduplicated against two synthetic pools that are not in this repo
(`frm/external/barc/build.py`).

**4. TRM checkpoint and its identifiers.** Download `arc_v2_public/step_723914` from the Hugging Face repo
`arcprize/trm_arc_prize_verification` into `CKPT`. The checkpoint does not ship the puzzle identifiers its embedding
table is indexed by, so rebuild them with TRM's own dataset builder (default seed 42) in a TinyRecursiveModels
checkout. `TRM_DATA` is that checkout's `data/` directory:

```bash
python -m dataset.build_arc_dataset --input-file-prefix kaggle/combined/arc-agi \
    --output-dir data/arc2concept-aug-1000 --subsets training2 evaluation2 concept --test-set-name evaluation2
```

**5. Build `a2tn`.** The source is ARC-AGI-2 training + evaluation demos + ConceptARC. ARC tasks keep TRM's exact
versions and its 30×30 canvas, so every imported embedding row transfers. Composites and external pools use size
buckets. Dev is the test pairs of 40 ARC-AGI-2 evaluation tasks; `test` is the other 80.

```bash
python -m frm.data.build --sources arc2 --composite data/raw/composite_s0.json --n_aug 300 \
    --external barc=data/raw/external/barc.npz rearc=data/raw/external/rearc.npz \
    --versions_from TRM_DATA/arc2concept-aug-1000/identifiers.json --min_bucket 30 --min_bucket_sources arc \
    --dev_subset evaluation2 --dev_n 40 --out data/built/a2tn
```

**6. Import the TRM checkpoint** (into a K=0 model with TRM's 1-D RoPE; every ARC row maps by identifier):

```bash
python scripts/import_trm.py --trm CKPT/arc_v2_public/step_723914 \
    --trm_ids TRM_DATA/arc2concept-aug-1000/identifiers.json \
    --config configs/final_trm_a2tn.json --data data/built/a2tn --out runs/trm_v2/import_a2tn.pt
```

**7. Train both arms** (resumable with the same command; SIGTERM/SIGUSR1 checkpoint and exit; dev and dev_composite
are evaluated every 2,500 PFLOP into `RUN/eval.jsonl`):

```bash
python -m frm.train --config configs/final_frm_a2tn.json --run_dir runs/final_frm_a2tn
python -m frm.train --config configs/final_trm_a2tn.json --run_dir runs/final_trm_a2tn
```

**8. Compare and score.** The shipped model is `last.pt` (EMA weights after the cooldown), not `best.pt`: the 40-task
dev set is too small to select on. `--split test` is the held-out read (80 ARC-AGI-2 evaluation tasks), and it
requires `--final`.

```bash
python scripts/arm_curves.py runs/final_frm_a2tn runs/final_trm_a2tn \
    --windows 2500-12500 15000-50000 52500-85000 87500-100000
python -m frm.evaluate --run_dir runs/final_frm_a2tn --split test --final --ckpt last
python -m frm.evaluate --run_dir runs/final_trm_a2tn --split test --final --ckpt last
```

Contamination: `arc_v2_public` trained on all of ARC-AGI-2 training, which contains the test outputs of 376 ARC-AGI-1
evaluation tasks. ARC-AGI-1 evaluation numbers are therefore not clean for this model. Its ARC-AGI-2 evaluation numbers
are transductive: demos were trained, test outputs never were (DESIGN.md §3.4–3.6, §4.9).

## Kaggle (ARC-AGI-2)

```bash
python -m frm.adapt --ckpt runs/final_frm_a2tn/last.pt --challenges arc-agi_test_challenges.json \
    --work /kaggle/working/frm --hours 8 --submission submission.json \
    train.fork_rate_target=0.25 train.fork_price_lr=1e-4 eval.fork_top=0.25
```

This builds the hidden tasks and gives their augmented versions fresh embedding rows. It then fine-tunes all weights
on the tasks' demos for the time budget, votes over the versions it trained, and writes two attempts per test input.
The overrides hold the fork rate at 0.25 during adaptation and at inference. Adaptation keeps the pretraining's 160k
tokens per step; on GPUs with less memory, lower `data.tokens_per_step` and set `model.grad_ckpt_children=true`. With
`--solutions FILE` the same command scores itself.
