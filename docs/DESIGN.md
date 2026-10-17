# frm — design

Forking Refinement Model (FRM): TRM's looped tied-weight refiner plus a fork, trained on ARC-AGI. This document holds
the design decisions and the code contract. Companion documents:

| doc | contents |
|---|---|
| [`DEPTH.md`](DEPTH.md) | why recursive refiners (TRM, HRM, FRM's predecessors) stop using depth; mechanisms ranked by evidence |
| [`OBJECTIVE.md`](OBJECTIVE.md) | loss research and the frm objective, term by term |
| [`../README.md`](../README.md) | setup and the commands that reproduce the final run |

Evidence tags: **[M]** measured in this repo or a predecessor project, **[L]** literature (URL in
`OBJECTIVE.md`/`DEPTH.md`), **[I]** inference.

---

## 0. Core-idea ledger

The ledger states first whether the training forward forks.

| core idea | trained in the forward? | where | test that proves it |
|---|---|---|---|
| tied-weight TRM refiner (y, z, H/L cycles, deep supervision, ACT) | **yes** | `frm/model/refiner.py` `Refiner.cycle`, `Refiner.round` | `tests/test_model.py::test_trm_mode_matches_reference_structure` |
| **forking**: a training round spawns K children from the parent state, runs them through the same tied refiner, and joins them back into the parent before the graded readout | **yes, K≥1 in every non-control config**: every round by default; with `fork_gate` (D5) every round during the forced-fork phase, then on the rows the gate picks plus an exploration floor | `Refiner.round` (`_node`, `_join`) | `tests/test_model.py::test_fork_gate` fails if no child cycle runs or if the parent loss sends no gradient into the fork codes, child cycles and join; `tests/test_gate.py` fails if a gated round's forked rows are not exactly the FRM round |
| children may fork again (nested) | yes when `fork_depth ≥ 2` (off by default for compute; §4.5) | `Refiner._node` | — |
| depth utilisation as a first-class objective | yes: objective + data + instruments (§5, §7.3) | `frm/objective.py`, `frm/data/composite.py` | per-round metrics at every eval |

The **only** forkless configuration is the `trm` control (`K=0`, `configs/final_trm_a2tn.json`), which measures the
fork's value at matched compute.

Words follow mechanisms: "fork", "child", "join" are used only for computation spawned and consumed inside the same
forward. Augmented views are "views"; inference-time voting is "voting".

---

## 1. Goal and thesis

**Goal.** A TRM-derived refiner that forks, trained on ARC-AGI data, ARC-shaped end to end (task JSON in, two attempts
per test input out), submittable to the ARC Prize 2026 ARC-AGI-2 Kaggle track with one adaptation stage (§8).

**Thesis.** Two claims, each falsifiable with the instruments in §7:

1. *Forking makes depth useful.* A single deterministic recursion converges to a fixed point (often the wrong one)
   within a few rounds and then repeats itself; this is what "no progress after depth ~4" is [M: TRM
   38.25→40.38→40.13→40.50% by step, arXiv 2512.11847; L: "Reasoning or Guessing?" 2601.10679 — trajectories plateau
   at spurious fixed points]. A round that forks K children with distinct codes **and fresh noise each round**, trains
   them as competing hypotheses, and lets the parent read them, injects *new information into every round* — even
   when the parent's state has stopped moving (without the noise the fork would be a deterministic function of that
   state and repeat itself). Depth paid off in the predecessors only when new information arrived between segments
   [M: frmv2, a checker variant gained +0.023 from segment 1 to 3, every other variant ≈0].
2. *Forking permits detours.* TRM grades every round's canvas against the final answer through a one-cycle gradient
   window with a detached carry, so a step that makes the canvas worse in order to make the answer reachable is
   penalised and never credited [L: TRM 2510.04871; M: frmv1 final-only supervision produced detours that were never
   rewarded]. In frm, children are never required to be correct: their computation is credited through the join by
   the parent's post-join readout, i.e. a two-cycle credit path that runs through an ungraded intermediate state.

**What would falsify it:** at matched training FLOPs, FRM does not beat the `trm` control on dev pass@2, *and* its
late-round gain (acc@16 − acc@4) is not larger than the control's, *and* the sham-children counterfactual costs
nothing. A negative result means redesign of the fork (§12), not removal.

---

## 2. Decisions at a glance

| # | decision | choice | evidence | rejected alternative |
|---|---|---|---|---|
| D1 | task conditioning | TRM's per-augmented-puzzle embedding, transductive (all demos of all tasks, incl. eval demos, in training; eval *test* pairs never) | in-context predecessors: 2–12% ARC-AGI-1 [M: tiny 25/419, tinyabstractor 11.5%]; TRM-style 40–45% [M: TRM checkpoint 169/190 of 419]; Kaggle-feasible via full fine-tune [L: McGovern 2511.02886, 6.67% ARC-AGI-2 semi-private] | in-context demo encoder (ARM, tinyabstractor): weaker at our compute, longer sequences |
| D2 | backbone | TRM exactly: 2 post-norm blocks, d=512, 8 heads, SwiGLU ×4, one module for y and z updates, fixed init buffers, stablemax CE, q-head on position 0 | TRM `models/recursive_reasoning/trm.py` | HRM two-module, ARM latents (ARM's H-level got no CE gradient [M]) |
| D3 | positions | 2D RoPE (row/col halves of each head), prefix unrotated; **`trm1d`** for TRM warm starts: TRM's 1-D RoPE at 30×30-canvas indices (prefix i → i, cell (r,c) → 16+30r+c) on every bucket (`layers.RopeTRM1D`; import exact at S=30, §4.9) | needed for size buckets; 1D RoPE over a variable stride was a measured bug [M: frmv1] (a fixed stride of 30 is not) | TRM 1D RoPE over 30×30 |
| D4 | canvas | size buckets {8,12,16,20,24,30} chosen from demos, test input and a demo-verified size rule, never the test output; warm-start builds keep ARC at 30 (§4.9) | token cost 0.34–0.49 of 30×30 on ARC-AGI-1, 0.82 on ARC-AGI-2 eval; overflow 2/419 and 1/172 [M: §3.3] | fixed 30×30 (2–3× more tokens) |
| D5 | fork structure | every round forks K children (default K=4) at the middle cycle; **adaptive forking** (`fork_gate`, used by the final): a gate on the pre-fork state decides per row and round; declined rows run TRM's round (§4.8). The gate gets no end-to-end gradient: BCE on the measured counterfactual benefit of each fork, after a forced-fork phase, with an exploration floor and no step-index input | learned gates collapse to schedules [M: frmv1 gate 98.9% cycle-index variance]; discrete decisions add high-variance gradients [L] | REINFORCE/Gumbel decisions, a soft gate trained through the loss, round-index features |
| D6 | child symmetry breaking and diversity | distinct non-zero learned fork codes at **every** child z-update (inside the gradient window), a parent-conditioned per-cell code, **fresh per-round noise** through a learned map | zero-init codes + zero-init arbitration collapsed onto child 0 (w=0.993) [M: tiny]; anything only on a no-grad path never trains [M: tinyv2.2, ARM]; deterministic recursion cannot leave a wrong fixed point [L: PTRM 2605.19943]; IMLE cannot drop modes [L: 1809.09087] | zero-init codes, warm-up-only seeds, deterministic children |
| D7 | child information | children start from the full parent state (y, z) and see the full input x | masked/starved children equalled random shams [M: tinyabstractor, children right 0.13 vs base 0.74 on disagreements] | crop/masked-plane children |
| D8 | join | per-cell softmax attention over {parent, child₁..child_K} on (y, z), with gradient; the parent slot lets the parent ignore children | detached joins train nothing [M: tinyabstractor, frmv2]; merge forms carried the decomposition solver's gains (21-0), branch-only forms were null [M: decomposition-solver-arc]; fresh fork modules joined onto a finished TRM do not damage it (training CE equal to TRM's at first, lower from step ≈ 2,000) [M: §4.10] | detached claim join, inference-only selection |
| D9 | child objective | one-to-one annealed relaxed set matching of {final answer} ∪ {composite intermediates} onto distinct children; on plain rows annealed relaxed WTA | MCL/rWTA/annealed MCL [L: 1606.07839, 1612.00197, 2407.15580]; set matching forces unique predictions [L: DETR 2005.12872]; same-target CE on all children → collapse [M: tiny] | per-child CE to the final target, diversity bonuses, independent per-target softmin |
| D10 | depth demand | synthetic composites f_k∘…∘f_1 (k = 1..4) with intermediate grids, mixed with ARC; per-round metrics by k | corpora without detour tasks cannot test the thesis [M: tinyabstractor's generator refused "second step hides the first"; frmv1 branching factor 1 in 100% of tasks] | ARC only |
| D11 | deep supervision | TRM's online carry: one optimizer step per round, halted rows replaced, per-bucket carries | TRM's regime produced 40%+; frmv1/frmv2 replaced it with unrolled graphs and lost TRM's behaviour [M] | unrolled multi-segment graph per step |
| D12 | optimizer | AdamW (0.9, 0.95), lr 1e-4, wd 0.1, warm-up 2k, constant; puzzle embedding SignSGD lr 1e-2 wd 0.1; EMA 0.999 | TRM `config/cfg_pretrain.yaml` (AdamATan2 → AdamW: no CUDA extension to build) | Muon (misrouted in tiny [M]) |
| D13 | halting | TRM: q_halt = BCE(parent canvas exactly correct), halt at q>0 in training with 10% exploration, 16 rounds at eval | TRM default; q also ranks votes and children | PonderNet/geometric priors (reward early exit [L: Ouro 2510.25741]) |
| D14 | evaluation discipline | checkpoints chosen on **dev**, never `test`; `test` read once with `--final`. The final (`a2tn`, from `arc_v2_public`, which saw ARC-AGI-1 eval test outputs, §4.9): dev = 40 ARC-AGI-2 evaluation tasks, `test` = the other 80, ships `last.pt` (EMA after the cooldown): a dev solving 0–2 tasks cannot select. `arc1` builds: dev = 100 ARC-AGI-1 evaluation tasks (§3.4) | tiny/tinyv2.2 picked best.pt on eval test pairs [M]; a dev from ARC-AGI-1 training tasks is inflated by RE-ARC/BARC relatives | selecting on eval; dev from training tasks |

---

## 3. I/O and data

### 3.1 Tokens and layout (TRM-compatible)

- Vocabulary 12: `PAD=0`, `EOS=1`, colour c ↦ c+2.
- A grid of h×w is written into an S×S canvas at offset (r₀, c₀); `EOS` fills row r₀+h over columns c₀..c₀+w−1 and
  column c₀+w over rows r₀..r₀+h−1 when those lie inside the canvas; everything else is `PAD`. Identical to
  TRM `dataset/build_arc_dataset.py` with 30 replaced by S.
- Labels: `PAD` ↦ ignore (−100); `EOS` and colours are supervised, so the model predicts the output size.
- Decoding: TRM's `_crop` — the largest top-left rectangle of colour tokens (TRM `evaluators/arc.py`).
  Test rows are never translated, so outputs are read top-left.
- Sequence = P=16 prefix tokens + S² grid tokens. Prefix token 0 carries the puzzle embedding; tokens 1–15 are
  zero-input registers (TRM).

### 3.2 Task conditioning: puzzle identifiers (transductive, TRM)

Each augmented version of each task has its own identifier and embedding row (512-d, zero-initialised, SignSGD).
Identifier 0 is the blank. Versions = identity + `n_aug` random (dihedral ∈ 0..7, permutation of colours 1..9, colour 0
fixed), deduplicated by puzzle hash (as TRM's `build_arc_dataset.py`, but hashing demos and test *inputs* only, so the
version table cannot depend on test outputs). Default `n_aug=300` for ARC and ConceptARC (ARC Prize's HRM analysis:
300 ≈ max, 30 within 4pp [L]), 20 for composites, 8 for external tasks (`--n_aug_external`; they are numerous).

Augmentation and translation are applied **on the fly** from the version table (dihedral id + permutation per
identifier), vectorised per batch (`frm/data/encode.py:render`; ≈30 ms for 4,096 rows at S=16 on a laptop CPU [M]),
so the stored dataset holds each base grid once.

### 3.3 Size buckets

Buckets S ∈ {8, 12, 16, 20, 24, 30}. A grid of max side m fits S iff m+1 ≤ S, or S = 30 and m ≤ 30. A task's bucket is
the smallest S that fits every demo input/output, every test input, and the predicted test-output size from a
demo-verified size rule (per axis: constant output, constant ratio, or constant difference across demos; on abstain, a
margin of +4 cells). **The test output is never read.** Measured on the ARC Kaggle-format task files [M]:

| split | test outputs that overflow | rule abstains | token cost vs 30×30 |
|---|---|---|---|
| ARC-AGI-1 training | 1/416 | 52 | 0.34 |
| ARC-AGI-1 evaluation | 2/419 | 50 | 0.49 |
| ARC-AGI-2 training | 3/1076 | 147 | 0.44 |
| ARC-AGI-2 evaluation | 1/172 | 46 | 0.82 |
| ConceptARC | 0/480 | 96 | 0.38 |

Training rows of a task use the task's bucket (for training tasks whose test pairs are training data, those pairs count
as demos, so train rows never overflow). Translation: each training row is shifted by a fresh random offset at load
time with probability 0.75 (input, output and sub-goals share the offset, and the EOS row/column always stays inside the
canvas); dev/test rows are never shifted. A test output too large for its bucket (2 of 419 in `arc1`) is stored cropped
and listed in `meta.json["overflow"]`; such a pair cannot be solved. `--min_bucket 30 --min_bucket_sources arc` keeps
ARC/ConceptARC tasks at 30×30 while other sources still bucket (warm starts, §4.9).

### 3.4 Sources and splits

| source | contents | split use |
|---|---|---|
| `arc2` | ARC-AGI-2 training (1000) + evaluation (120) + ConceptARC | training2 test pairs → train (except dev tasks'); evaluation2 test pairs → test, or with `--dev_subset evaluation2 --dev_n 40` dev = 40 of them, test = the other 80. ARC-AGI-2 training contains 376 ARC-AGI-1 eval tasks [M], so `arc1` and `arc2` never mix |
| `arc1` | ARC-AGI-1 training (400) + evaluation (400) + ConceptARC (160) | demos of all and training/concept test pairs → train; evaluation test pairs → **dev** for 100 tasks, **test** for the other 300 |
| `arc2-kaggle` | `arc2` without evaluation2 | evaluation2 becomes an unseen-task adaptation test (§8) |
| `composite` | synthetic f_k∘…∘f_1 tasks with intermediate grids (§3.5) | demos → train; 2 held-out pairs per task → `dev_composite` |
| `barc` (external) | BARC ARC-Heavy: 28,605 LLM-written generator tasks, 827,130 pairs; programs re-run in a sandbox and audited against the 520 ARC-AGI-1/2 evaluation tasks (623 dropped) by `frm/external/barc/` | train only; re-filtered by frm (§3.6) |
| `rearc` (external) | RE-ARC: generators of the 400 ARC-AGI-1 *training* tasks, 399,503 pairs (`frm/external/corpus.py`) | train only; split into per-bucket sub-tasks `rearc:{id}@{S}` |
| `kaggle` | hidden test challenges (or ARC-AGI-2 evaluation as proxy) | demos → train (adaptation); test inputs → test |

External pools are FlatPool `.npz` files (format in `frm/external/flatpool.py`), built by `frm/external/` and
filtered again by `scripts/ingest_external.py` before a build reads them.

- **`a2tn`** (the final's build; command in `README.md`): `arc2` + composites + BARC + RE-ARC; ARC tasks at 30×30
  (`--min_bucket 30 --min_bucket_sources arc`) with exactly TRM's versions (`--versions_from` TRM's `identifiers.json`,
  §4.9); dev = test pairs of 40 ARC-AGI-2 evaluation tasks (`--dev_subset evaluation2 --dev_n 40`), `test` = the other
  80, plus `dev_composite`. TRM saw the evaluation2 demos but not their test outputs.
- **`arc1` dev** = the first 100 ARC-AGI-1 evaluation tasks with ≥ 3 demos by md5(task id). Training tasks are not
  used for dev: RE-ARC generators are those tasks and BARC seeds are drawn from them.
- **Excluded tasks.** `--exclude_ids FILE` keeps listed tasks (e.g. an unseen-task adaptation set of ARC-AGI-2-only
  training tasks) out of a build; it raises if any source, including RE-ARC sub-tasks, would bring one in.
- Pools designed by looking at evaluation tasks (e.g. ARC-GEN, 328 of whose tasks are ARC-AGI-1 evaluation tasks) are
  never used. A `synth` pool is ingestible but not in the default mix: composites play its role.
- A draw yields at most `max_pairs_per_draw` (16) pairs of one task, so a 1,000-pair generator cannot flood a batch;
  sampling uses the capped counts, so `source_weights` are the actual row fractions.

### 3.5 Composite tasks (depth demand)

`frm/data/composite.py` generates tasks whose rule is a composition of k ∈ {1..4} primitives from a library of grid
transforms (dihedral maps, quadrant/half swaps, gravity in 4 directions, colour maps, recolour-by-size, fill enclosed
regions, crop-to-content, tile, upscale, mirror-concatenate, border, shift, object-wise ops). For every example it
stores the k−1 intermediate grids g₁ = f₁(x), g₂ = f₂(g₁), …. Filters (after decomposition-solver-arc's synthetic tasks,
reimplemented): every step changes the grid on every demo; the composition is not equal on all demos to any single
primitive or shorter composition; at least one intermediate is *not* closer to the output than the input is (a genuine
detour, e.g. "shuffle quadrants then gravity"); grids ≤ 30×30. Each task: 6–10 demo pairs + 2 held-out pairs, its own
augmented versions (`n_aug_composite`, default 20). They are the only data with known compositional depth, so they
double as the depth instrument (§7.3).

Default row mix (`DataCfg.source_weights`): ARC 0.50, composites 0.15, BARC 0.25, RE-ARC 0.10. ARC keeps half the rows
because the evaluation-task embeddings are learned only from ARC rows.

### 3.6 Contamination guarantees

| held-out set | what protects it | test |
|---|---|---|
| dev and test outputs | never written to a train row; buckets, offsets and the size rule read demos and test inputs only | `test_poisoning_leaves_everything_but_test_outputs_unchanged`, `test_no_test_output_in_train_rows` |
| the same, copied inside *training* tasks | every build drops ARC/ConceptARC training tasks (whole) and composite pairs whose canonical pair (8 dihedral × colour relabel) equals a held-out test pair or that contain an informative (≥36 cells, ≥8 off-background) held-out test output; a dropped task's RE-ARC generator goes too (it copies the *rule*). On `arc1`: 6 tasks (e.g. `40853293` contains `070dd51e`'s test pair), 4 RE-ARC generators, 7 composite pairs | `meta.json["content_filter"]`; `test_content_filter_drops_planted` |
| everything held out, against external pools | `scripts/ingest_external.py` filters pools against all ARC-AGI-1/2 evaluation tasks and any held-out pools (pairs and informative grids of demos and tests); the build refuses an external file whose recorded held-out ids do not cover its dev/test/excluded tasks. Measured: **0** contaminated pairs against the evaluation sets (the positive control, ARC-AGI-1 eval tasks inside ARC-AGI-2 training, is fully detected); **176** against 230 held-out ARC-AGI-2-only training tasks, all dropped (BARC 169, incl. `barc/h035065`, a copy of `bd5af378`; RE-ARC 3; synth 4) | `*_report.json`; `tests/test_external.py` |
| dev representativeness | dev tasks are evaluation tasks, which no external source derives from | dev rule test |
| between held-out tasks | not filtered, by design: eval demos are training data (TRM's protocol), and one eval task's demo can equal another's test output (once in `arc1`: `bf32578f` demo 0 output = `6df30ad6` test output up to flip + recolour); competition time gives all hidden demos together too | — |
| excluded tasks | `--exclude_ids` raises if any source (incl. RE-ARC sub-task base ids) would bring one in; externals filtered by content | `tests/test_external.py` |

---

## 4. Model

### 4.1 Retained TRM core

Notation follows the TRM paper: answer state **y** (`z_H` in code) and latent **z** (`z_L`), both [B, P+S², 512].
x = √d · (token embedding ‖ puzzle-embedding prefix). f(h, u) = Block₂(Block₁(h + u)), each block post-norm
`h = rms(h + Attn(h)); h = rms(h + SwiGLU(h))`, no learnable norm gains, bf16 compute with fp32 norms. One module f
serves every update. y₀, z₀ are fixed trunc-normal buffers broadcast over positions (as TRM `trm.py`).

```
cycle(y, z, x, c):            # one H-cycle; c = fork code (0 for the parent)
    for i in 1..n:            # n = L_cycles = 4 (TRM's ARC setting)
        z = f(z, y + x + c)
    y = f(y, z)
    return y, z
```

Heads: `lm_head(y)[:, P:]` → logits over 12 tokens; `q_head(y[:, 0])` → halting logit (weight 0, bias −5 at init).

### 4.2 Round = one supervision step

`H_cycles = 3`. A round maps the detached carry (y, z) to a graded readout and a new detached carry.

**`trm` control (K = 0)**, exactly TRM:
```
no_grad: 2 × cycle(y, z, x, 0)
         1 × cycle(y, z, x, 0)  with grad   → readout, carry
```

**FRM (K ≥ 1)**, same sequential depth (3 cycles), middle cycle widened into K parallel children:
```
no_grad: (H−2) × cycle(y, z, x, 0)                              # pre-fork parent cycle
for k in 1..K (batched):                                         # fork
    c_k   = e_k + A_k · B · rms(z) + W_ξ ξ_k,  ξ_k ~ N(0, I₁₆) fresh   # fork code: per child, per cell, per round
    y_k, z_k = cycle(y, z, x, c_k)            with grad          # children: same tied f
y, z = join(y, z, {y_k, z_k, q_k})             with grad          # §4.4
y, z = cycle(y, z, x, 0)                       with grad          # post-join parent cycle
readout: lm_head(y), q_head(y[:,0]); children readouts lm_head(y_k), q_head(y_k[:,0])
carry = (y.detach(), z.detach())
```

The parent's graded readout depends on the children through the join, so the gradient path is
`L_parent → post cycle → join → child cycles → fork codes, shared f`. This two-cycle credit path through an ungraded
intermediate state is what TRM lacks (§1 claim 2).

### 4.3 Fork

- **Codes.** `e_k ∈ ℝ^512`, k = 1..K, initialised N(0, 0.5²) and distinct; `B: 512→64` shared, `A_k: 64→512` per child,
  zero-initialised (`e_k` already breaks symmetry; `A_k` still gets gradient since `B·rms(z) ≠ 0`). The per-cell term
  lets the parent give each child a different job per region (a soft "argument": region, object, layer).
- **Noise.** `ξ_k ~ N(0, I₁₆)` per row, child and round, mapped by a learned `W_ξ: 16→512` (initial output RMS 0.25,
  `ModelCfg.noise_dim`/`noise_init`), broadcast over cells. Training uses the global RNG; evaluation seeds a generator
  per batch (reproducible; counterfactual reruns see the same noise). IMLE-style min-over-samples training cannot
  collapse modes, and a parent at a fixed point still gets new proposals (`OBJECTIVE.md` §3.2; PTRM 2605.19943).
- **Injection.** c_k is added to the z-update input at **every** child z-update, always inside the gradient window.
- **Batching.** Children are a batch expansion [B, K, L, D] → [B·K, L, D]: no separate buckets or episodes (frmv2's
  failure mode). `grad_ckpt_children` checkpoints every child pass individually (stores pass inputs, recomputes one
  pass at a time in backward); exact (`test_child_checkpointing_is_exact`).
- **Exchangeability.** Children differ only by codes; the join is permutation-equivariant over children.

### 4.4 Join

For every cell p and candidate j ∈ {0 (parent pre-fork state), 1..K}:

```
s_j[p] = ⟨W_q rms(z[p]), W_k rms(z_j[p])⟩ / √64  +  β · sg(q_j)        # q_j = halting logit of candidate j
α[p]   = softmax_j(s[p])
y[p]   = Σ_j α_j[p] · y_j[p]
z[p]   = Σ_j α_j[p] · z_j[p]
```

`W_q, W_k: 512→64` (standard init), β scalar (init 0). One mechanism covers the three uses the thesis needs:
*select* (one-hot α: adopt a child's state), *compose* (α varies by cell: assemble the canvas from children, the
ASSEMBLE/COMBINE forms that carried the decomposition solver [M]), *detour* (the post-join cycle finishes a child's
intermediate state). The parent slot makes "ignore the children" representable, so usage is measured, not forced.
q_j enters detached so selection pressure cannot miscalibrate the halting head.

**Parent-slot dropout** (`ModelCfg.join_drop_parent`, training only; 0.25 in the final): per row and round the parent
slot is masked with probability p, so the post-join cycle must build on the children and the parent's loss reaches
them at full strength. Without it the join ignored the children within the first 3% of training (parent mass 0.39 →
0.95) and their gradient shrank with their weight [M]. Dropped rows are NaN in the join statistics; evaluation never
drops.

### 4.5 Nested forks (`fork_depth ≥ 2`)

`_node(y, z, depth)`: at `depth < fork_depth` the node forks K children, each `_node(·, depth+1)`, joins them and runs a
post cycle; at `depth = fork_depth` a node is a leaf (one cycle). Codes and join are tied across levels (one set of
weights at every node and depth). Default `fork_depth = 1` because cost grows as K^depth (§9); `fork_depth = 2, K = 2`
is the smallest nested setting.

### 4.6 Size and init

Dense parameters ≈ 6.83M (TRM) + fork ≈ K·512 + 512·64 + K·64·512 + join 2·512·64 ≈ 0.2M at K=4. Puzzle-embedding
table: `n_ids × 512` fp32, outside the dense count (≈ 600k rows for an `arc1` build with externals, 1.2 GB on the GPU).

### 4.7 Gradient-path invariants (tested)

1. FRM round: `e_k`, `A_k`, `B`, `W_q`, `W_k`, β and f's weights all receive non-zero gradient from `L_parent` alone.
2. Children's cycles run with grad; the pre-fork cycle runs without.
3. The carry is detached: a round's loss sends no gradient into the previous round.
4. `trm` control reproduces TRM's structure: (H−1) no-grad cycles, one grad cycle, same op order.
5. Inference forward == training forward (same function, no grad), except the training-only parent-slot dropout and
   fork exploration.
6. No parameter is reachable only through a no-grad path (`test_every_param_gets_grad`).
7. With `fork_gate`: a declined row's round equals the K = 0 round with the same weights, a forked row's equals the
   ungated FRM round, each row is independent of how the batch was split, and the gate head gets gradient only from
   its own loss and sends none into the trunk (`tests/test_gate.py`).

### 4.8 Adaptive forking (`fork_gate`)

Off by default (every round forks, as above); on in the final. The model chooses, per row and round, whether to fork,
so forks are spent where they pay. Forking stays in the trained forward; the gate only decides where.

```
no_grad: (H−2) × cycle(y, z, x, 0)                               # pre-fork cycle, every row
g    = MLP([y[:,0] ‖ mean_cells(y)])   2d → gate_hidden → SiLU → 1   # on the detached pre-fork state
fork = g > 0, or (training) with probability fork_explore anyway
forked rows:   children → join (parent-slot dropout) → post cycle → readout      # the FRM round of §4.2
declined rows: no_grad (H−grad_cycles)−(H−2) cycles, then grad_cycles cycles     # = the K = 0 round, same weights
label rows (forked, own join, slot not dropped): no_grad, the declined rows' continuation from the same
         pre-fork state → cf_logits                                             # what not forking would have given
```

- **Execution.** Each subset is gathered, run alone and scattered back (`index_copy`, autograd-safe), so a declined
  row costs a TRM round and no children. Child outputs keep full-batch shapes (join statistics NaN, child logits zero
  on declined rows; `RoundOut.fork` masks them), except that the `sham` counterfactual runs every row's children. With
  `compile` the cycle's batch dimension is dynamic; ungated models keep static graphs.
- **Training signal.** The gate's inputs are detached; it is trained by BCE against "this fork beat the no-fork
  continuation by more than the price" (`OBJECTIVE.md` §3.6) on the label rows. Trunk and gate losses never cross.
- **Collapse mitigations.** frmv1's learned gate became a schedule (98.9% of its variance was the cycle index [M]).
  Here: (1) labels are the *measured* counterfactual benefit, not gradients through the decision; (2) a **forced-fork
  phase**: for the first `train.fork_force_frac` (default 0.3) of training every row-round forks, so children and join
  learn everywhere and every state gets labels; (3) an **exploration floor**: afterwards a declined row forks anyway
  with probability `train.fork_explore` (0.1), so a "no" can be corrected; (4) **no step-index input**.
  `fork.rate_by_round` shows whether it became a schedule anyway.
- **Price.** `objective.fork_price` (nats per supervised token, default 0). With `train.fork_rate_target > 0`, after
  each gated step outside the forced phase: price ← max(0, price + `fork_price_lr` · (batch fork rate − target)). The
  price is trainer state: logged, checkpointed, restored on resume; adaptation inherits it and skips the forced phase.
- **Accounting.** `Refiner.round_flops(S, rows, fork_rows, cf_rows)` charges declined rows at the K = 0 cost, forked
  rows at the FRM cost, label rows plus the continuation's 2 forward cycles; `passes_per_round`/`train_flops_per_row`
  stay the every-round-forks numbers (§9).
- **Evaluation** uses the gate's decisions (no exploration), or with `eval.fork_top` = ρ > 0 forks the top ρ of each
  batch and round by gate logit (padding excluded), fixing the eval fork rate whatever the gate's held-out
  calibration; `fork.gate_rate` logs the gate's own rule. That calibration drifts: a gate trained to a target rate
  forked 0.3% of dev row-rounds and 79% of composite ones with the same weights; in a warm-started ARC-AGI-1
  comparison at 36k PF, forking the top 25% or 40% matched the dev accuracy of forking every row [M]. `fork_top` = 1
  forks and labels every row, so `fork.skill` measures held-out ranking without selection bias; `cf.allfork` /
  `cf.gateoff` rerun with every / no row forking (§7.2). Submission inference skips the label continuation. Requires
  `fork_depth = 1` and `grad_cycles ≤ 2` (else `ValueError`).

### 4.9 Warm start from a published TRM (`scripts/import_trm.py`)

The ARC Prize verified TRM checkpoints (`arcprize/trm_arc_prize_verification`: `arc_v1_public/step_518071`, 40% on
ARC-AGI-1 semi-private; `arc_v2_public`, 6.2% on ARC-AGI-2) load into frm:

- **Weights.** Every dense tensor maps one to one (qkv layout, SwiGLU width 1536, H_init/L_init → y_init/z_init,
  q_head's halt row). With `pos_encoding="trm1d"` (D3) the import reproduces TRM's forward: at S = 30, logits of 4
  rounds of dev rows are bitwise equal in fp32 and halt logits agree to 2·10⁻⁶ [M: against TRM's model code].
- **Puzzle embedding.** TRM's table is indexed by its dataset's identifiers, which share frm's naming and
  augmentation semantics. Building with `--versions_from` TRM's `identifiers.json` gives every ARC/ConceptARC task
  exactly TRM's ≈1000 versions, so every row transfers (871,328 of 871,328 for `arc1`; 5,077 unused rows belong to the
  6 content-filtered tasks). BARC, RE-ARC and composite rows start at the mean TRM row. Imported rows carry one visit,
  so all vote (§7.1).
- **Canvas.** TRM computes on all 900 cells, PAD included, and relies on that space: the import scores dev pass@2
  0.485 at 30×30 and 0.410 on size buckets (row exact 0.336 → 0.203) [M]. Warm-start builds keep ARC at 30
  (`--min_bucket 30 --min_bucket_sources arc`) and bucket only sources TRM never saw.
- **Continuing.** `train.init_from` = the import, `train.init_mode="continue"`: table and visits kept, fresh optimizer
  and EMA. A K > 0 model takes every shared weight from the checkpoint; its fork modules (codes, join, noise map, gate)
  start fresh with a higher learning rate (`train.fork_lr_mult`, a second optimizer group; 4 in the final). FRM is
  thus trained on top of a TRM that has already spent its full compute budget.
- **Contamination.** `arc_v1_public` saw ARC-AGI-1 training + evaluation *demos* + ConceptARC, so an `arc1` dev and
  `test` stay clean. `arc_v2_public` saw ARC-AGI-2 training, which holds the test outputs of 376 ARC-AGI-1 evaluation
  tasks, so the final draws dev from ARC-AGI-2 evaluation test pairs (TRM saw only their demos).

### 4.10 Spending forks where they pay after a warm start

In a warm-started ARC-AGI-1 comparison (FRM and TRM both continued from `arc_v1_public`, price-0 gate) [M]: per
optimizer step FRM is ahead and the gap grows (training CE equal to TRM's for ≈ 1,000 steps, then lower: 0.259 vs
0.307, exact 0.293 vs 0.226 at step 170k), but each step costs ≈ 3.4× because the gate forked on 60% of round-1 and
97% of later row-rounds. The fork modules do not damage the source model: a zero-initialised, function-preserving
residual join tracked the softmax join at matched steps and opened to the children within 300 PF. The per-step price
hurts most where steps are scarce: adapting to unseen ARC-AGI-2 tasks in 2.5 h gave FRM 8,482 steps against TRM's
31,246 (pass@2 0.190 vs 0.252). Two levers act on the price without making children cheaper or fewer:

- **Compute targeting** (`train.fork_rate_target`, §4.8): hold the fork rate at ρ, keep the forks the gate ranks
  highest, run every declined row as a TRM round. Cost per row-round ≈ 25 + 65ρ units at K = 4 without checkpointing
  (§9): ρ = 0.94 → 3.4×, ρ = 0.4 → 2.0× (measured in the final: 547 vs 275 GFLOP), ρ = 0.25 → 1.65×.
- **Error-focused children** (`objective.child_focus`, `OBJECTIVE.md` §3.7): the children's final-target loss is
  concentrated on the cells TRM's own round gets wrong.

Measured in the same comparison [M]: at a held 0.4 fork rate (`fork_top` 0.4) FRM beats dense-forking FRM on
composites at matched PF (+0.017 pass@2, +0.014 row exact, CIs > 0), ties it and the continued TRM on dev, and takes
1.34× its steps per PF, although the gate's held-out ranking is no better than chance. Error-focused children add on
composites (+0.025) but fall behind the continued TRM on dev row exact, so the final holds the 0.4 rate without them.
In adaptation `train.fork_rate_target=0.25`, `eval.fork_top=0.25` reached 0.235 on the unseen tasks in 2.5 h vs 0.190
with dense forks (continued TRM 0.252, CI ∋ 0).

---

## 5. Objective (summary — derivation in `OBJECTIVE.md`)

Per round, per row $b$, averaged over the step's rows. Every symbol, the code behind each term and its measured
behaviour are in `OBJECTIVE.md` §3.

$$
\mathcal{L}_b \;=\; \mathrm{CE}(\hat y_b, y_b) \;+\; \lambda_q\,\mathrm{BCE}(q_b, e_b)
\;+\; f_b\Big[\lambda_{\mathrm{wta}}\Big((1-\varepsilon)\sum_{\pi} w_{b\pi}\,c_{b\pi} + \frac{\varepsilon}{K}\sum_{k=1}^{K}\mathrm{CE}(\hat y_{bk}, y_b)\Big)
\;+\; \frac{\lambda_q}{K}\sum_{k=1}^{K}\mathrm{BCE}(q_{bk}, e_{bk})\Big]
$$

$$
c_{b\pi} \;=\; \mathrm{CE}\big(\hat y_{b,\pi(\star)}, y_b\big) + \frac{\lambda_{\mathrm{sub}}}{\lambda_{\mathrm{wta}}}\,\frac{1}{\lvert T_\pi\rvert}\sum_{t\in T_\pi}\mathrm{CE}\big(\hat y_{b,\pi(t)}, g_{bt}\big),
\qquad
w_{b\pi} \;=\; \operatorname{softmax}_\pi\big(-\operatorname{sg}(c_{b\pi})/\tau\big)
$$

$\hat y_b$ is the parent's readout after the fork, the join and the post-join cycle, so its gradient reaches the
children. $w$ is a soft-min, without gradient, over one-to-one assignments $\pi$ of the final answer $\star$ and the
composite intermediates $g_{bt}$ to distinct children. $f_b = 1$ if row $b$ forked, so a declined row's loss is exactly
the $K = 0$ loss. With `fork_gate` the gate adds
$\lambda_{\mathrm{gate}}\,\frac{1}{\lvert G\rvert}\sum_{b\in G}\mathrm{BCE}\big(\gamma_b,\,\mathbb{1}[\mathrm{CE}(\tilde y_b, y_b) - \mathrm{CE}(\hat y_b, y_b) > p]\big)$
over its label rows $G$, with $\tilde y_b$ TRM's own round from the same pre-fork state (`OBJECTIVE.md` §3.6).

CE is TRM's stablemax cross-entropy, a token mean per row over non-PAD labels. Defaults:
$\lambda_{\mathrm{wta}} = \lambda_{\mathrm{sub}} = \lambda_q = 0.5$, $\varepsilon = 0.05$, $\tau$ annealed log-linearly $1 \to 0.02$ over the first
`tau_anneal_frac` of the budget (0.3; the final: 0.048, its first 4,800 PF), $\lambda_{\mathrm{gate}} = 0.1$, price
0 (the final moves it to hold a 0.4 fork rate). Plain rows (no intermediates) reduce to annealed relaxed
winner-takes-all. Uniform weight per round (no discounting). No term rewards a difference between two
model-controlled quantities: the gate's label is such a difference, but it is detached and trains only the gate, which
cannot change either quantity.

---

## 6. Training loop

- **Online deep supervision (TRM).** One optimizer step = one round for every row of one bucket's carry. Rows halt when
  `q_parent > 0` (training) or at 16 rounds; with p=0.1 a row gets a minimum round count U{2..16}. Only halted rows are
  replaced, by fresh rows from that bucket's stream (TRM `trm.py` discards the rest).
- **Per-bucket carries.** Each step picks a bucket with probability ∝ its share of training examples ÷ its rows per
  step, so every example is visited at the same rate. Rows per step = `tokens_per_step` (default 120k parent tokens,
  equal across compared arms; the final uses 160k) ÷ sequence length, capped at `max_rows`: 120k ≈ 750 rows at S=12,
  ≈ 130 at S=30 (TRM: 768 rows of 916 tokens, mostly padding).
- **Sampling.** Group-balanced per bucket (TRM `puzzle_dataset.py`): epochs permute base tasks; a draw picks a random
  version and yields its train examples; leftover rows carry to the next request. Source weights set row fractions
  (§3.5). Per-row loss is averaged over the step's rows.
- **Optimizers.** Fused AdamW on dense params; SignSGD with decoupled weight decay on touched embedding rows; clip 1.0;
  warm-up 2,000 steps then constant (TRM), optionally a cooldown over the last `decay_frac` of the budget, lr ×
  (1 − √u) down to `decay_to` × lr for dense and embedding lr alike (Hägele et al. 2024, arXiv 2405.18392; the final
  uses 15%). EMA 0.999; evaluation and `best.pt` use EMA weights; `last.pt` holds raw weights + EMA + optimizer.
- **Budget.** Training stops at `budget_pflops` of the model's FLOP accounting (or `max_steps`, or `max_hours`,
  cumulative across resumes); progress is the largest of these fractions and drives the τ anneal, the fork schedule and
  the cooldown, so they are comparable across arms and a wall-clock-bound run still finishes its cooldown.
  `stop_pflops` ends a run early without changing that schedule; `resume_from` = another run's `last.pt` continues its
  full state (weights, EMA, optimizer, step, compute, data streams, RNG, price) under a new config, so two arms can
  branch from one shared stretch.
- **Fork policy (`fork_gate`, §4.8).** Every row-round forks while progress < `fork_force_frac`; then the gate decides
  with exploration `fork_explore`; the price controller runs after each gated step. FLOPs are charged from the step's
  actual fork and label-row counts.
- **Precision/speed.** bf16 compute with fp32 master weights (TRM's "casted" layers), fp16 + GradScaler on V100; SDPA
  attention (on ROCm `layers.select_sdpa_backend` disables flash so the memory-efficient kernel runs: +9–10%
  throughput, −19 GB peak for FRM K=4 [M]); optional `torch.compile` of the cycle per bucket shape; per-pass child
  checkpointing.
- **Fault tolerance.** Atomic `last.pt` (model, EMA, optimizer, embedding table, step, RNG, data-stream state) every
  `ckpt_minutes`; SIGTERM/SIGUSR1 → checkpoint and exit 0; `--resume auto`. Carries are not saved (in-flight rows
  restart). `best.pt` = best dev pass@2 on the first eval split, ties to the later step; the final ships `last.pt`
  (D14).
- **Logs.** `train.jsonl` (losses, per-round metrics, first-solve/regression histograms, WTA winners, fork stats,
  throughput, memory, `{metric}_by_source`), `eval.jsonl`, `config.json`.

---

## 7. Inference, evaluation, instruments

### 7.1 Inference

For each test input and each augmented version: run 16 rounds from the init carry; keep, per round, the parent canvas
and q, and at the final round the K children canvases and q's. De-augment (inverse dihedral, then inverse colour map),
crop, hash. Vote per test input: rank distinct grids by (vote count, mean σ(q)) over versions — TRM's rule
(TRM `evaluators/arc.py`), with children optionally voting at weight `child_vote_weight` (default 0). Top-2 →
`attempt_1`, `attempt_2`. Score = mean over tasks of the fraction of test inputs with a correct attempt (official ARC
metric).

**Only trained versions vote** (`loader.eval_versions`): up to `eval.max_versions` versions whose embedding row was
updated at least once (`PuzzleEmbedding.visits`), identity first; a task with no trained version gets one blind
attempt from its identity version. TRM trains every version it votes with; a budgeted run does not, and an untrained
row carries no task information (in a from-scratch ARC-AGI-1 comparison at 3,400 PF, FRM had trained 15% of its dev
voting versions, `trm` 54%, so blind FRM votes outvoted informed ones [M]). Every eval result records `versions` =
{coverage, voters per task, blind_tasks}. The same rule scores adaptation and the submission. Adaptation's new rows
start at the mean of the checkpoint's *trained* rows (`train.load_pretrained`), not diluted by never-trained zeros.

### 7.2 Fork instruments (each can fail in the direction of the failure)

Training metrics: `train.jsonl` (mean and `_by_round`); evaluation: `eval.jsonl` / `<run_dir>/eval/*.json` under
`fork`, `cf`, `by_round`, `by_k`, `per_task`.

| metric (key) | definition | failure it detects |
|---|---|---|
| `parent_mass` (train, `fork.parent_mass`) | mean α₀ over grid cells | parent ignores children (→1) |
| `join_entropy` | mean per-cell entropy of α | uniform blending (→log(K+1)) |
| `win0..winK-1` (train) | fraction of rows each child wins the WTA | dead children (any ≈0) |
| `distinct` | distinct child answers per row (train: on supervised cells; eval: cropped, de-augmented) | collapse (→1) |
| `child_exact_any`, `child_ce_min`, `fork.oracle_gain` | best child vs parent | children cover nothing the parent lacks |
| `fork.adopt` (+ `adopt_chances`) | P(parent exact after the round ∣ pre-fork parent wrong, some child exact) | join cannot select |
| `cf.sham` | Δ exact at round 16 when every root join reads the children of a row from a **different task** | parent ignores child *content* |
| `cf.nofork` | Δ exact when the root join is forced onto the parent slot | parent does not rely on children |
| `sub_coverage` (train) | composites: fraction of intermediates some child reproduces exactly | no decomposition learned |
| `fork_rate` (train); `fork.rate`, `fork.rate_by_round`, `fork.rate_by_bucket` (eval) | fraction of row-rounds that forked | gate never forks (→0), always forks (→1), or became a round schedule |
| `gate_acc`, `gate_pos`, `benefit` (train) | on label rows: (logit > 0) == label; label rate; CE(no-fork) − CE(fork) in nats/token | gate no better than the base rate; forks that do not pay |
| `gate_says` (train), `fork.gate_rate` (eval) | fraction of row-rounds with gate logit > 0, whatever decided | calibration drifts from the actual fork rate |
| `declined_pos` (train) | label rate on declined rows that forked through exploration | gate declines forks that would pay (≈ `gate_pos`) |
| `{metric}_by_source` (train) | exact, CE, fork rate, gate metrics per data source | forks spent on one source; a source the warm start memorised |
| `fork.gate_precision`, `fork.benefit` (eval) | on held-out labelled forked rows: fraction where the fork beat the no-fork continuation; mean CE gain | forks do not pay on unseen pairs |
| `fork.skill` (eval, `fork_top` = 1) | AUC of the gate logit for benefit > 0; per ρ the share of summed benefit the top ρ keep, their precision and benefit; by round group | gate cannot rank (AUC ≈ 0.5, `share` ≈ ρ) or ranks wrongly (`share` < ρ: a warm-started FRM at 36k PF kept 8% at ρ = 0.4 [M]) |
| `price` (train) | current fork price | controller pinned at 0 or running away |
| `cf.allfork`, `cf.gateoff` (gated) | Δ exact when every / no row forks, vs the run's own decisions | gate declines forks that pay (`allfork` > 0); forking useless (`gateoff` ≈ 0) |

With the gate, children instruments (`parent_mass`, `join_entropy`, `distinct`, `child_exact_any`, `oracle_gain`,
child votes) cover only rows that forked: NaN on declined rows in training, the rows that forked at the final round at
evaluation (`fork.forked_rows`, `fork.forked_labeled`); `adopt` counts only forked rounds. Counterfactual rows are
spread evenly over buckets (`eval.cf_rows` in total).

### 7.3 Depth instruments

| metric (key) | definition |
|---|---|
| `by_round.pass@2`, `by_round.row_exact` | voted pass@2 and unvoted row exact at round r = 1..16 |
| `late_gain`, `late_gain_rows` | round 16 − round 4 (where depth use stops, `DEPTH.md`); `per_task` holds both rounds for paired CIs |
| `by_round.first_solve`, `by_round.regress` | fraction of labelled rows first exact at round r; exact at r−1 but wrong at r |
| `by_k` | composites: row exact by round for each composition depth k |
| `cf.reset@4` | Δ exact at round 16 when the carry is reset to init at round 4 (do later rounds build on the carry?) |
| training `*_by_round`, `first_solve`, `regress` | the same per-round quantities on the training stream, by each row's round index |

`scripts/arm_curves.py` compares two runs' eval curves (pooled paired windows, slopes).

---

## 8. Kaggle path (ARC Prize 2026, ARC-AGI-2 track)

The track requires a Kaggle notebook, no internet, 2 attempts per test input, 12 h GPU run-time [L: arcprize.org
competitions/2026/arc-agi-2; hardware announced at launch; ARC Prize 2025 gave 4 × L4 24 GB]. Hidden tasks have no
puzzle identifiers, so the submission runs one **adaptation stage** (`frm.adapt`):

1. Build the hidden tasks (`arc-agi_test_challenges.json`, `--sources kaggle`): demos → train rows, test inputs → test
   rows, `n_aug` versions.
2. The new identifier space's embedding rows start at the mean of the checkpoint's trained rows
   (`frm.train.load_pretrained`). No pretraining rows are replayed.
3. Fine-tune **all** weights (embeddings-only or LoRA was not enough [L: McGovern]) on the hidden demos for a
   wall-clock budget (`--hours`, optionally `--max_steps`); resumable like any training run.
4. Run §7.1 inference with EMA weights on **every** test task and write `submission.json`
   (`{task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}`, one entry per test input, padded by repeating
   attempt 1).

With `--solutions` the same command scores itself: on `arc2-kaggle` + evaluation2 it measures unseen tasks; for the
final (pretrained on `a2tn`, which holds evaluation2's demos) that proxy is transductive. Packaging: the `frm`
package (pure PyTorch + NumPy, no compiled extensions), the checkpoint, offline wheels if needed. Adaptation is
single-GPU and keeps the pretraining's `tokens_per_step` unless overridden (memory: §9).

---

## 9. Compute model

Unit = one L-level pass (2 blocks over the sequence) = 15.92 GFLOP per sample at 916 tokens [M: SpecTRM]. Per
sample-round with H=3, n=4 (backward ≈ 2× forward on grad passes):

| setting | forward passes | grad passes | total units | × TRM |
|---|---|---|---|---|
| `trm` (K=0) | 15 | 5 | 25 | 1.0 |
| K=0, 2 grad cycles | 15 | 10 | 35 | 1.4 |
| FRM K=2 | 20 | 15 | 50 (+10 with child checkpointing) | 2.0 (2.4) |
| FRM K=4 | 30 | 25 | 80 (+20) | 3.2 (4.0) |
| FRM nested K=2, depth 2 | 40 | 35 | 110 (+20) | 4.4 (5.2) |

`Refiner.passes_per_round` computes these counts (`test_pass_accounting_matches_execution` checks them against
executed passes); `train_flops_per_row` converts them with the exact per-bucket pass cost `layers · L · (2·N_block +
4·L·d)`, which reproduces TRM's 398.1 GFLOP per 30×30 row-round [M]. Weighted by the default row mix over an `arc1`
build's buckets (child checkpointing on): `trm` 152, K=0 with 2 grad cycles 212, FRM K=2 364, FRM K=4 607 GFLOP per
row-round, i.e. FRM K=4 on buckets ≈ 1.5× TRM on 30×30. Warm-start builds keep ARC at 30×30 (§4.9) and cost more: in
the final, `trm` 275 and gated FRM K=4 at fork rate 0.4 without checkpointing 547 GFLOP per row-round. TRM's ARC
schedule is ≈1.59e20 FLOPs (≈3 days on 4×H100) [M: SpecTRM]. Matched compute = the same `budget_pflops` for both
arms (100,000 PF in the final).

With `fork_gate` the trainer charges each step's actual mix (`Refiner.round_flops`, checked against executed passes by
`test_round_flops_match_executed_passes`): a declined row costs the `trm` row, a forked row the FRM row, and a label
row adds its no-fork continuation (2 cycles forward, 10 units at n=4). FRM K=4 at fork rate ρ with parent-slot
dropout p costs ≈ 25 + 55ρ + 10ρ(1−p) units, e.g. 43.8 at ρ=0.3, p=0.25 (vs 80 when every round forks).

Memory [I]: a grad pass keeps ≈29·d values per token per block (q/k upcast for RoPE, SwiGLU intermediates, fp32 norm
inputs) ≈ 60 KB per token per pass in bf16, ≈300 KB per grad cycle. FRM K=4 without checkpointing holds (K+1)× that
per parent token (≈1.5 MB); per-pass child checkpointing ≈0.6 MB (the estimate overstates this case ≈2.3×). Measured
peaks [M]: at 120k tokens/step on buckets with child checkpointing `trm` 23 GB, FRM K=4 31 GB (62 GB at 240k); at 160k
on warm-start builds without checkpointing `trm` 35.6 GB, FRM K=4 ≈158 GB with flash SDPA and 139–145 GB with the
memory-efficient kernel (§6), growing as rarer batches arrive; FRM at 240k without checkpointing exceeded 192 GB. Below
≈150 GB, FRM at 160k needs `grad_ckpt_children` or fewer tokens per step.

---

## 10. Tests

`pytest -q` (CPU; tests that need ARC data skip until `data/raw` exists):

- `test_data.py`: augmentation inverses and dedup; encode/crop round trip per bucket; size rule, buckets and the whole
  built dataset unchanged when test outputs are poisoned; no test output in a train row; dev rules (md5 order,
  evaluation2 pool); untranslated, consistently keyed eval rows; Kaggle tasks without outputs; `take()` content, EOS,
  sub-goal masks, source weights, per-draw cap, exact resume.
- `test_external.py`: canonical digest invariance; ingest drops planted contamination and refuses held-out id overlaps;
  the build refuses unfiltered externals; content filter drops planted copies; per-bucket generator pools.
- `test_composite.py`: determinism, shortcut filters, detour quota, sub-goals = program prefixes, every primitive.
- `test_model.py`: `test_fork_gate` (child cycles run with grad; codes, join, β get gradient from `L_parent` alone);
  `test_trm_mode_matches_reference_structure`; detached carry; `test_every_param_gets_grad`; inference == training
  forward; fresh per-round noise; exact child checkpointing; pass accounting; counterfactual modes; parent-slot dropout
  in training only; embedding visit counts; reset-to-mean over trained rows.
- `test_gate.py` (§4.8): `fork_mode="none"` = the K=0 model, `"all"` = the ungated round; every row of a mixed batch
  equals its own path run alone; `cf_logits` = TRM's round; no label from dropped parent slots; declined rows send no
  gradient into the fork and have exactly the K=0 loss; the gate trains only from its loss; label = benefit > price;
  `round_flops`; gated training (forced phase, price controller, resume); fork lr multiplier; (1−√u) cooldown;
  `evaluate_split` rates, counterfactuals, `fork_top` and `fork.skill`.
- `test_objective.py`: stablemax normalisation and finite, non-zero gradients; PAD ignored; one-to-one assignments; set
  matching = annealed WTA on plain rows, distinct targets to distinct children, masks; child gradient = w_k × own CE
  gradient; mixed composite/plain batches.
- `test_evaluate.py`: official scoring, vote ranking, only trained versions vote, submission shape, cross-task shams.
- `test_train_smoke.py`: build → train → SIGTERM → resume → DONE → evaluate → `--final` guard → adapt → submission.
- `test_trm_import.py` (§4.9): `trm1d` positions, weight copy, embedding rows aligned by identifier, `--versions_from`,
  `init_mode=continue`.

Not covered: the ROCm branch of `layers.select_sdpa_backend`.

---

## 11. Module layout and interfaces (the code contract)

```
frm/
  config.py            dataclasses ModelCfg, ObjectiveCfg, DataCfg, TrainCfg, EvalCfg; Cfg.load(json) + dot-path
                       overrides
  data/
    arc.py             load_tasks(challenges, solutions=None), load_subset(raw_dir, subset)
    augment.py         dihedral / inverse_dihedral, Version(tid, perm).apply/invert/parse/name, make_versions
    encode.py          PAD/EOS/colour layout: encode_grid, labels_from, crop, render (vectorised augment+translate)
    sizerule.py        predict_output_size(task, test_input), choose_bucket(task) — never reads test outputs
    heldout.py         canon (8 dihedral × colour relabel digest), informative, HeldOut(...).hit — the content filter
    build.py           CLI: dataset dir from --sources arc1|arc2|arc2-kaggle|kaggle, --composite JSON, --external
                       SOURCE=PATH, --versions_from, --min_bucket(_sources), --dev_subset/--dev_n, --exclude_ids;
                       dev rule; content filter; generator pools split per bucket
    loader.py          Dataset(root): buckets, identifiers, tasks(split), splits(); TrainStream(ds, S, rng, weights,
                       translate_p, max_pairs_per_draw).take(n) + state_dict; eval_batches(ds, split, S, rows, ...)
    composite.py       33 primitives, generate(n_tasks, depth_range, seed, ...) with sub-goals; CLI writes JSON
  model/
    layers.py          rms_norm, CastedLinear, SwiGLU, Rope2D, RopeTRM1D, Attention (SDPA), select_sdpa_backend,
                       Block, block_params
    embedding.py       PuzzleEmbedding (buffer table, gather-to-leaf, SignSGD step)
    refiner.py         Refiner(ModelCfg, num_ids): init_carry, reset_carry, round(carry, inputs, ids, cf, sham_perm,
                       gen, fork_mode, explore, fork_top, n_valid, gate_labels) -> RoundOut, passes_per_round(fork),
                       train_flops_per_row, eval_flops_per_row, round_flops(S, rows, fork_rows, cf_rows, train),
                       compile_cycles
  objective.py         set_match, Objective(ObjectiveCfg)(RoundOut, batch, progress, has_subgoals, price)
                       -> (loss, per-row metrics)
  train.py             CLI: python -m frm.train --config cfg.json --run_dir RUN_DIR [--resume auto] [key=value ...]
  evaluate.py          CLI + evaluate_split: per-round voting, pass@k, instruments, counterfactuals, submission
  adapt.py             CLI: Kaggle adaptation (build, re-initialised ids, fine-tune, predict, write submission)
  external/
    flatpool.py        FlatPool .npz format (data, off, hw, pair, pool_off, ids)
    corpus.py          CLI: RE-ARC and a2_new pools from re_arc/tasks and data/raw
    barc/              CLIs: BARC fetch (pinned), sandbox (+ --selftest), build parse|execute|filter|select|write,
                       audit against the held-out tasks
scripts/
  fetch_data.py        download ARC-AGI-1/2 and ConceptARC into data/raw
  ingest_external.py   filter BARC/RE-ARC/a2_new/synth FlatPool .npz pools against held-out tasks
  import_trm.py        published TRM checkpoint → frm checkpoint (§4.9)
  arm_curves.py        two runs' eval curves: pooled paired windows, slopes
configs/
  final_frm_a2tn.json  the final: FRM K=4, gated at a 0.4 fork rate, warm-started from the imported TRM
  final_trm_a2tn.json  its K=0 TRM control at matched compute
tests/                 test_composite, test_data, test_evaluate, test_external, test_gate, test_model, test_objective,
                       test_train_smoke, test_trm_import
```

`Batch` (all torch tensors on device unless noted): `inputs` int [B, S²], `labels` int [B, S²] (−100 = ignore),
`puzzle_ids` int [B], `subgoals` int [B, M, S²] (−100 rows where absent), `subgoal_mask` bool [B, M], plus numpy-side
`keys` for eval rows (task id, test index, identifier).

`RoundOut`: `logits` [B, S², 12] float32, `q` [B], `child_logits` [B, K, S², 12] or None, `child_q` [B, K] or None,
`join_alpha` [B, K+1] (cell-mean, for stats), `join_entropy` [B], `carry` (y, z detached), `pre_logits`; with K > 0
`fork` bool [B] (rows that forked; child fields are zeros/NaN elsewhere); with `fork_gate` also `fork_logit` [B],
`gate_rows` bool [B] (label rows) and `cf_logits` [B, S², 12] (no-fork continuation, zeros off the label rows).

---

## 12. Risks and planned responses (redesign, never removal)

| risk | detection | response | available |
|---|---|---|---|
| parent ignores children | `parent_mass → 1`, `cf/sham ≈ 0` | raise `objective.lambda_wta`; parent-slot dropout (`model.join_drop_parent`, §4.4); more pre-fork cycles (`H_cycles=4`) so children start from a better state | config |
| children collapse | `distinct → 1`, win rates degenerate | raise `objective.wta_eps`, slow the τ anneal (`tau_anneal_frac`), stronger per-round fork noise (`model.noise_init`, §4.3) | config |
| fork costs more than it returns | matched-FLOP dev pass@2 below `trm` | K=2; cheaper children (`child_L_cycles=2`); adaptive forking (`model.fork_gate`, §4.8: fork where the gate predicts it pays); fork only on alternate rounds | K, child_L_cycles, fork_gate: config. alternate rounds: a code change |
| fork gate collapses | `fork_rate` → 0 or 1 after the forced phase, `rate_by_round` a step in r, `gate_acc` ≈ max(`gate_pos`, 1−`gate_pos`), `cf.allfork` > 0 | raise `fork_explore`, lengthen `fork_force_frac`, set `fork_rate_target` (price controller) | config |
| depth still unused | `late_gain ≈ 0` on dev and composites | raise composite fraction/depth (`data.source_weights`, generator `--depth_max`); inspect `cf/reset@4`; more rounds trained (`halt_explore`) | config |
| memory | OOM (FRM K=4 at 240k tokens without checkpointing did, §9) | `grad_ckpt_children`; smaller `tokens_per_step` | config |
| Kaggle hardware/time | adaptation does not fit 12 h | fewer versions (`--n_aug`), fewer inference rounds (`eval.rounds`; per-round curves say where accuracy saturates) | config |

A response that removes forking from the trained model is outside this design: FRM without forks is the K=0 control.

## 13. Non-goals

In-context demonstration encoders; symbolic solvers or DSL search at inference; multi-GPU DDP (every arm fits one
192 GB GPU, or a smaller one with `grad_ckpt_children`); ARC-AGI-3.
