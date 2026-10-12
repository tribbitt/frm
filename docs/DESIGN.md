# frm — design

Forking Refinement Model (FRM), iteration 3 of the lineage `frmv1 → tiny → tinyv2.2 → tinyabstractor → frmv2`.
Built fresh; nothing is imported from predecessors. Companion documents:

| doc | contents |
|---|---|
| [`DEPTH.md`](DEPTH.md) | why recursive refiners (TRM, HRM, every predecessor) stop using depth; mechanisms ranked by evidence |
| [`OBJECTIVE.md`](OBJECTIVE.md) | loss research and the frm objective, term by term |
| [`LINEAGE.md`](LINEAGE.md) | audit of the five predecessors + TRM + ARM: what to reuse, what to avoid, with paths |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | arms, decision rules, compute budget, Kelvin plan |
| [`../kelvin/README.md`](../kelvin/README.md) | how jobs run on Kelvin2 |

Evidence tags: **[M]** measured in this repo or a predecessor's files, **[L]** literature (URL in `OBJECTIVE.md`/`DEPTH.md`),
**[I]** inference.

---

## 0. Core-idea ledger (read first; keep current)

Rule inherited from `frmv2/project-drift-failure.md` §7: the ledger states, in its first lines, whether the training
forward forks. Only the user may retire, defer, or make conditional any row.

| core idea | trained in the forward? | where | gate that proves it |
|---|---|---|---|
| tied-weight TRM refiner (y, z, H/L cycles, deep supervision, ACT) | **yes** | `frm/model/refiner.py` `Refiner.cycle`, `Refiner.round` | `tests/test_model.py::test_trm_mode_matches_reference_structure` |
| **forking**: a training round spawns K children from the parent state, runs them through the same tied refiner, and joins them back into the parent before the graded readout | **yes, K≥1 in every non-control config: every round by default; with `fork_gate` (user decision, D5) every round during the forced-fork phase, then on the rows the gate picks plus an exploration floor** | `Refiner.round` (`_node`, `_join`) | `tests/test_model.py::test_fork_gate` fails if no child cycle runs or if the parent loss sends no gradient into the fork codes, child cycles and join; `tests/test_gate.py` fails if a gated round's forked rows are not exactly the FRM round |
| children may fork again (nested) | yes when `fork_depth ≥ 2` (off by default for compute; see §4.5) | `Refiner._node` | `test_nested_fork_gate` |
| depth utilisation as a first-class objective | yes: objective + data + instruments (§5, §7.3) | `frm/objective.py`, `frm/data/composite.py` | per-round metrics logged every eval; `EXPERIMENTS.md` decision rules |

The **only** forkless configuration is the control arm `trm` (`K=0`), which exists to measure the fork's value at
matched compute. It is never the shipped model unless the user decides so.

Words follow mechanisms: "fork", "child", "join" are used only for computation spawned and consumed inside the same
forward. Augmented views are "views"; inference-time voting is "voting".

---

## 1. Goal and thesis

**Goal.** A TRM-derived refiner that forks, trained on ARC-AGI data, ARC-shaped end to end (task JSON in, two attempts per
test input out), submittable to the ARC Prize 2026 ARC-AGI-2 Kaggle track with one adaptation stage (§8).

**Thesis.** Two claims, each falsifiable with the instruments in §7:

1. *Forking makes depth useful.* A single deterministic recursion converges to a fixed point (often the wrong one) within
   a few rounds and then repeats itself; this is what "no progress after depth ~4" is [M: TRM 38.25→40.38→40.13→40.50% by
   step, arXiv 2512.11847; L: "Reasoning or Guessing?" 2601.10679 — trajectories plateau at spurious fixed points]. A round
   that forks K children with distinct codes **and fresh noise each round**, trains them as competing hypotheses, and
   lets the parent read them, injects *new information into every round* — even when the parent's state has stopped
   moving (without the noise the fork would be a deterministic function of that state and repeat itself). Depth paid
   off in the lineage only when new information arrived between segments [M: frmv2 `p2_c` checker arm, s1→s3 gain
   +0.023 vs ≈0 in every other arm].
2. *Forking permits detours.* TRM grades every round's canvas against the final answer through a one-cycle gradient
   window with a detached carry, so a step that makes the canvas worse in order to make the answer reachable is
   penalised and never credited [L: TRM 2510.04871; M: frmv1 final-only supervision produced detours that were never
   rewarded]. In frm, children are never required to be correct: their computation is credited through the join by
   the parent's post-join readout, i.e. a two-cycle credit path that runs through an ungraded intermediate state.

**What would falsify it** (pre-registered in `EXPERIMENTS.md`): at matched training FLOPs, the FRM arm does not beat the
`trm` control on dev pass@2, *and* its late-round gain (acc@16 − acc@4) is not larger than the control's, *and* the
sham-children counterfactual costs nothing. A negative result means redesign of the fork (see §12), not removal.

---

## 2. Decisions at a glance

| # | decision | choice | evidence | rejected alternative |
|---|---|---|---|---|
| D1 | task conditioning | TRM's per-augmented-puzzle embedding, transductive (all demos of all tasks, incl. eval demos, in training; eval *test* pairs never) | in-context lineage models: 2–12% ARC-AGI-1 [M: tiny 25/419, tinyabstractor 11.5% a1_test]; TRM-style 40–45% [M: TRM ckpt 169/190 of 419 on tiny's harness]; Kaggle-feasible via full fine-tune [L: McGovern 2511.02886, 6.67% ARC-AGI-2 semi-private] | in-context demo encoder (ARM support-TTT, tinyabstractor): weaker at our compute, larger sequences |
| D2 | backbone | TRM exactly: 2 post-norm blocks, d=512, 8 heads, SwiGLU ×4, one module for both y and z updates, fixed (non-learned) init buffers, stablemax CE, q-head on position 0 | `trm-samsung/models/recursive_reasoning/trm.py:65-222` | HRM two-module, ARM latents (ARM's H-level got no CE gradient [M]) |
| D3 | positions | 2D RoPE (row/col halves of each head), prefix tokens unrotated; **`trm1d`** for models started from a published TRM checkpoint: TRM's 1-D RoPE with every cell at its 30×30-canvas index (prefix i → i, cell (r,c) → 16+30r+c), so the checkpoint keeps its relative positions on every bucket (`layers.RopeTRM1D`; import is numerically exact at S=30, §4.9) | needed for size buckets; 1D RoPE over a variable stride was a measured frmv1 bug [M: `frmv1/models/layers.py:44-52`] (a fixed stride of 30 is not that bug) | TRM 1D RoPE over 30×30 |
| D4 | canvas | size buckets {8,12,16,20,24,30}, chosen from demo grids + test input + a demo-verified size rule; never from the test output | token cost 0.32–0.49 of 30×30 on ARC-AGI-1 sets, 0.82 on ARC-AGI-2 eval; overflow 2/419 a1 eval, 1/172 a2 eval [M: computed from `trm-samsung/kaggle/combined`, see §3.3] | fixed 30×30 (2–3× more tokens) |
| D5 | fork structure | **fixed structure by default**: every round forks K children (default K=4) at the middle cycle. **Adaptive forking (`fork_gate`, the user's decision):** a gate head on the pre-fork state decides per row and round whether to fork; declined rows run TRM's round (§4.8) | learned gates collapse to schedules [M: frmv1 gate 98.9% cycle-index variance]; discrete decisions add high-variance gradients [L: research C10]. The gate therefore gets **no end-to-end gradient**: it is trained by BCE on labels from the *measured* counterfactual benefit of each fork (fork CE vs the no-fork continuation's CE, from the same state), after a **forced-fork familiarisation phase** (`fork_force_frac`), with an **exploration floor** (`fork_explore`) so declined states keep getting labels, and **no step-index input** (it reads only the state) | REINFORCE/Gumbel fork decisions, a soft gate trained through the loss, round-index features |
| D6 | child symmetry breaking and diversity | non-zero, distinct learned fork codes injected at **every** child z-update (inside the gradient window), a parent-conditioned per-cell code, and **fresh per-round noise** through a learned map | tiny's zero-init codes + zero-init arbitration collapsed onto child 0 (w=0.993) [M: `tiny/results/fork_probe.json`]; anything only on a no-grad path never trains [M: tinyv2.2 `init_state`, ARM latents]; deterministic recursion has no escape from a wrong fixed point [L: PTRM 2605.19943]; IMLE cannot drop modes [L: 1809.09087] | zero-init codes, seeds only in warm-up, deterministic children |
| D7 | child information | children start from the full parent state (y, z) and see the full input x; nothing is masked | masked/starved children equalled random shams [M: tinyabstractor I-9, children right 0.13 vs base 0.74 on disagreements] | crop/masked-plane children |
| D8 | join | per-cell softmax attention over {parent, child₁..child_K} on (y, z), with gradient; parent slot lets the parent ignore children | detached joins train nothing [M: tinyabstractor `fork_in` detaches all; frmv2 "nothing learned through the merge" removed forks]; merge forms carried the decomposition solver's gains (21-0) while branch-only forms were null [M: decomposition-solver-arc ladder]; fresh fork modules joined onto a finished TRM do not damage it: at matched optimizer steps FRM's training CE equals TRM's from the first steps and is lower from step ≈ 2,000 on [M: W, `EXPERIMENTS.md` §4b] | detached claim join, inference-only selection |
| D9 | child objective | one-to-one set matching of {final answer} ∪ {composite intermediates} onto distinct children, annealed and relaxed; on plain rows this is annealed relaxed winner-takes-all | MCL/rWTA/annealed MCL [L: 1606.07839, 1612.00197, 2407.15580]; bipartite set matching forces unique predictions [L: DETR 2005.12872]; same-target CE on all children → collapse [M: tiny] | per-child CE to the final target, diversity bonuses, per-target independent softmin (one child can claim every target) |
| D10 | depth demand | synthetic composite tasks f_k∘…∘f_1 (k = 1..4) with intermediate grids, mixed with ARC; per-round metrics by k | corpora without detour tasks cannot test the thesis [M: tinyabstractor `synth/compose.py:29-32` refused "second step hides the first"; frmv1 branching factor 1 in 100% of tasks] | ARC only |
| D11 | deep supervision regime | TRM's online carry: one optimizer step per round, halted rows replaced, per-bucket carries | TRM's regime is what produced 40%+; frmv1/frmv2 replaced it with unrolled graphs and lost TRM's behaviour [M] | unrolled multi-segment graph per step |
| D12 | optimizer | AdamW (0.9, 0.95), lr 1e-4, wd 0.1, warm-up 2k, constant; puzzle embedding SignSGD lr 1e-2 wd 0.1; EMA 0.999 | TRM `cfg_pretrain.yaml` (AdamATan2 replaced by AdamW: no CUDA extension to build on Kelvin) | Muon (misrouted in tiny [M]) |
| D13 | halting | TRM: q_halt = BCE(current parent canvas exactly correct), halt when q>0 in training with 10% exploration, always 16 rounds at eval | TRM default; q also ranks votes and children | PonderNet/geometric priors (reward early exit [L: Ouro 2510.25741]) |
| D14 | evaluation discipline | checkpoints chosen on **dev** (test pairs of 100 ARC-AGI-1 *evaluation* tasks, demos trained; no external source derives from them); pilots additionally scored on **adev** (120 ARC-AGI-2 training tasks absent from pretraining, via adaptation); a1 `test` and ARC-AGI-2 eval read once with `--final` | tiny/tinyv2.2 picked best.pt on eval test pairs [M]; a dev drawn from ARC-AGI-1 training tasks would be inflated by RE-ARC/BARC relatives | selecting on eval; dev from training tasks |

---

## 3. I/O and data

### 3.1 Tokens and layout (TRM-compatible)

- Vocabulary 12: `PAD=0`, `EOS=1`, colour c ↦ c+2.
- A grid of h×w is written into an S×S canvas at offset (r₀, c₀); `EOS` fills row r₀+h over columns c₀..c₀+w−1 and
  column c₀+w over rows r₀..r₀+h−1 when those lie inside the canvas; everything else is `PAD`. Identical to
  `trm-samsung/dataset/build_arc_dataset.py:50-74` with 30 replaced by S.
- Labels: `PAD` ↦ ignore (−100); `EOS` and colours are supervised, so the model predicts the output size.
- Decoding: TRM's `_crop` — the largest top-left rectangle of colour tokens (`trm-samsung/evaluators/arc.py:13-36`).
  Test rows are never translated, so outputs are read top-left.
- Sequence = P=16 prefix tokens + S² grid tokens. Prefix token 0 carries the puzzle embedding; tokens 1–15 are zero-input
  registers (TRM).

### 3.2 Task conditioning: puzzle identifiers (transductive, TRM)

Each augmented version of each task has its own identifier and embedding row (512-d, zero-initialised, SignSGD).
Identifier 0 is the blank. Versions = identity + `n_aug` random (dihedral ∈ 0..7, permutation of colours 1..9, colour 0
fixed), deduplicated by puzzle hash (TRM `build_arc_dataset.py:126-164`, but hashing demos and test *inputs* only, so the
version table cannot depend on test outputs). Default `n_aug=300` for ARC and ConceptARC (ARC Prize's HRM analysis:
300 ≈ max, 30 within 4pp [L]), 20 for composites, 8 for external tasks (`--n_aug_external`; they are numerous).

Augmentation and translation are applied **on the fly** from the version table (dihedral id + permutation per
identifier), vectorised per batch (`frm/data/encode.py:render`; ≈30 ms for 4,096 rows at S=16 on a laptop CPU [M]),
so the stored dataset holds each base grid once.

### 3.3 Size buckets

Buckets S ∈ {8, 12, 16, 20, 24, 30}. A grid of max side m fits S iff m+1 ≤ S, or S = 30 and m ≤ 30. A task's bucket is the
smallest S that fits every demo input/output, every test input, and the predicted test-output size from a
demo-verified size rule (per axis: constant output, constant ratio, or constant difference across demos; on abstain, a
margin of +4 cells). **The test output is never read.** Measured on `trm-samsung/kaggle/combined` [M]:

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
and listed in `meta.json["overflow"]`; such a pair cannot be solved.

### 3.4 Sources and splits

| source | contents | split use |
|---|---|---|
| `arc1` | ARC-AGI-1 training (400) + evaluation (400) + ConceptARC (160) | demos of all → train; training/concept test pairs → train; evaluation test pairs → **dev** for the 100 *dev* tasks, **test** for the other 300 |
| `arc2` | ARC-AGI-2 training (1000) + evaluation (120) + ConceptARC | training2 test pairs → train, except the dev tasks' (ARC-AGI-1 eval tasks are inside training2); evaluation2 test pairs → test. ARC-AGI-2 training contains 376 ARC-AGI-1 eval tasks [M], so `arc1` and `arc2` never mix |
| `arc2-kaggle` | as `arc2` but **without** evaluation2 | the final pretraining source; evaluation2 is then the adaptation test (§8) |
| `composite` | synthetic f_k∘…∘f_1 tasks with intermediate grids (§3.5) | demos → train; 2 held-out pairs per task → `dev_composite` |
| `barc` (external) | BARC ARC-Heavy: 28,605 LLM-written generator tasks, 827,130 pairs, sandbox-run and program-audited against the 520 ARC-AGI-1/2 evaluation tasks by tinyabstractor (623 programs dropped) | train only; ingested with frm's own filter (§3.6) |
| `rearc` (external) | RE-ARC: generators of the 400 ARC-AGI-1 *training* tasks, 399,503 pairs | train only; each generator is split into per-bucket sub-tasks `rearc:{id}@{S}` so small examples are not trained on 30×30 canvases |
| `kaggle` | hidden test challenges at competition time (or ARC-AGI-2 evaluation for the proxy) | demos → train (adaptation); test inputs → test |

- **dev** = 100 ARC-AGI-1 evaluation tasks with ≥ 3 demos, first 100 by md5(task id) — tinyabstractor's rule, so the a1
  `test` split (the other 300) is the exact set its 11.50% pass@2 was measured on. ARC-AGI-1 *training* tasks are not
  used for dev: RE-ARC and BARC contain relatives of them (RE-ARC generators are those tasks; BARC seeds are drawn from
  them), which would make a training-task dev easier than evaluation.
- **adev** = 120 ARC-AGI-2 training tasks that are neither ARC-AGI-1 tasks nor ARC-AGI-2 evaluation tasks, first 120 by
  md5 (`scripts/make_adapt_dev.py`). They are excluded from every pilot build (`--exclude_ids`, which raises if any source
  would bring one in) and from the externals' ingest filter's targets, and are scored by `frm.adapt` on a finished
  pilot checkpoint: the pilots' measure of solving tasks never seen in pretraining (`EXPERIMENTS.md` §2).
- The tinyabstractor pools `synth_gap` (designed by looking at ARC-AGI-1 evaluation tasks) and ARC-GEN (328 of its tasks
  are ARC-AGI-1 evaluation tasks) are never used. tinyabstractor's `synth` is ingestible (`--with-synth`) but not in the
  default mix: frm's composites play its role.
- Each draw yields at most `max_pairs_per_draw` (16) pairs of one task, so a 1,000-pair RE-ARC generator cannot flood a
  batch; bucket and source sampling use the capped counts, so `source_weights` are the actual row fractions.

### 3.5 Composite tasks (depth demand)

`frm/data/composite.py` generates tasks whose rule is a composition of k ∈ {1..4} primitives from a library of grid
transforms (dihedral maps, quadrant/half swaps, gravity in 4 directions, colour maps, recolour-by-size, fill enclosed
regions, crop-to-content, tile, upscale, mirror-concatenate, border, shift, object-wise ops). For every example it stores
the k−1 intermediate grids g₁ = f₁(x), g₂ = f₂(g₁), …. Filters (from `decomposition-solver-arc/decomp/synth.py:make_task`,
reimplemented): every step changes the grid on every demo; the composition is not equal on all demos to any single
primitive or shorter composition; at least one intermediate is *not* closer to the output than the input is (a genuine
detour, e.g. "shuffle quadrants then gravity"); grids ≤ 30×30. Each task: 6–10 demo pairs + 2 held-out pairs, its own
augmented versions (`n_aug_composite`, default 20). They are the only data with known compositional depth, so they double
as the depth instrument (§7.3).

Default row mix (`DataCfg.source_weights`): ARC 0.50, composites 0.15, BARC 0.25, RE-ARC 0.10. ARC keeps half the rows
because the evaluation-task embeddings are learned only from ARC rows; the pilots test whether the external half pays
(`EXPERIMENTS.md` P2).

### 3.6 Contamination guarantees

| held-out set | what protects it | gate |
|---|---|---|
| eval test outputs (a1 `test`, ARC-AGI-2 eval) and dev test outputs | never written to a train row; buckets, offsets and the size rule read demos and test inputs only | `test_poisoning_leaves_everything_but_test_outputs_unchanged`, `test_no_test_output_in_train_rows` |
| the same, against copies inside *training* tasks | every build drops ARC/ConceptARC training tasks (whole) and composite pairs whose canonical pair (8 dihedral × colour relabel) equals a held-out test pair, or that contain an informative (≥36 cells, ≥8 off-background) held-out test output. A dropped task is a copy of a held-out task's *rule*, so its RE-ARC generator (fresh instances no pair filter can catch) is dropped too. Measured on `arc1`: 6 training/concept tasks dropped (e.g. `40853293` contains `070dd51e`'s test pair), their 4 RE-ARC generators' sub-tasks, and 7 composite pairs | `meta.json["content_filter"]`; `test_content_filter_drops_planted` |
| everything held out, against external pools | `scripts/ingest_external.py` filters BARC/RE-ARC/synth against all ARC-AGI-1/2 evaluation tasks and all 230 adev candidates (demos and tests, pairs and informative grids) and drops contaminated instances; the build refuses an external file whose recorded held-out ids do not cover the build's dev/test/excluded tasks. Measured: **0** contaminated pairs against the evaluation sets (tinyabstractor's filter held; the positive control — ARC-AGI-1 eval tasks inside ARC-AGI-2 training — is fully detected) and **176** against the adev candidates, all dropped: BARC 169 (94 tasks; `barc/h035065` was a copy of `bd5af378` and went whole), RE-ARC 3, synth 4 | `*_report.json`; `tests/test_external.py` (planted contamination is removed) |
| dev representativeness | dev tasks are ARC-AGI-1 evaluation tasks, which no external source derives from | dev rule test |
| between held-out tasks | not filtered, by design: eval tasks' demos are training data (TRM's protocol), and one eval task's demo can equal another's test output (measured once in `arc1`: dev `bf32578f` demo 0 output = dev `6df30ad6` test output up to flip + recolour). The same information is available at competition time, where every hidden task's demos are given together | documented |
| adev | excluded from pilot builds by id (`--exclude_ids` raises, including RE-ARC sub-task base ids), filtered out of externals by content. No ARC/ConceptARC training task contains an adev test pair or test output (one, `31aa019c`, shares a grid with an adev *demo*, which adaptation shows the model anyway) | `test_exclude_ids`, `test_make_adapt_dev` |

---

## 4. Model

### 4.1 Retained TRM core

Notation follows the TRM paper: answer state **y** (`z_H` in code) and latent **z** (`z_L`), both [B, P+S², 512].
x = √d · (token embedding ‖ puzzle-embedding prefix). f(h, u) = Block₂(Block₁(h + u)), each block post-norm
`h = rms(h + Attn(h)); h = rms(h + SwiGLU(h))`, no learnable norm gains, bf16 compute with fp32 norms. One module f
serves every update. y₀, z₀ are fixed trunc-normal buffers broadcast over positions (`trm.py:152-154`).

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
  `A_k` zero-initialised (the static `e_k` already breaks symmetry; `A_k` still receives gradient because `B·rms(z) ≠ 0`).
  The per-cell term lets the parent hand each child a different job per region (a soft "argument": region, object, layer),
  learned end to end.
- **Noise.** `ξ_k ~ N(0, I₁₆)` is drawn per row, per child, per round and mapped by a learned `W_ξ: 16→512` (initial
  output RMS 0.25, `ModelCfg.noise_dim`/`noise_init`), broadcast over cells. Training uses the global RNG; evaluation
  seeds a generator per batch (reproducible, and counterfactual reruns see the same noise). Rationale: IMLE-style
  min-over-samples training cannot collapse modes, and a parent at a fixed point still gets new proposals
  (`OBJECTIVE.md` §3.2; PTRM 2605.19943). Gated by `test_children_get_fresh_noise_each_round`.
- **Injection.** c_k is added to the z-update input at **every** child z-update, so codes are always inside the gradient
  window.
- **Batching.** Children are a batch expansion [B, K, L, D] → [B·K, L, D]. No separate buckets, no separate episodes
  (the frmv2 failure mode). With `grad_ckpt_children`, every pass of a child cycle is checkpointed individually (only the
  pass inputs are stored; one pass is recomputed at a time in backward) — exact, gated by
  `test_child_checkpointing_is_exact`.
- **Exchangeability.** Children are distinguished only by codes; the join is permutation-equivariant over children.

### 4.4 Join

For every cell p and candidate j ∈ {0 (parent pre-fork state), 1..K}:

```
s_j[p] = ⟨W_q rms(z[p]), W_k rms(z_j[p])⟩ / √64  +  β · sg(q_j)        # q_j = halting logit of candidate j
α[p]   = softmax_j(s[p])
y[p]   = Σ_j α_j[p] · y_j[p]
z[p]   = Σ_j α_j[p] · z_j[p]
```

`W_q, W_k: 512→64` (standard init), β scalar (init 0). One mechanism covers the three uses the thesis needs:
*select* (one-hot α: adopt a child's whole state), *compose* (α varies by cell: assemble the canvas from children —
the ASSEMBLE/COMBINE forms that carried the decomposition solver [M]), *detour* (the post-join cycle reads a child's
intermediate state and finishes it). The parent slot makes "ignore the children" representable, so usage is
measurable instead of forced. q_j enters detached so selection pressure cannot miscalibrate the halting head.

**Parent-slot dropout** (`ModelCfg.join_drop_parent`, training only; 0.25 in the FRM pilot configs since P1's first run,
`EXPERIMENTS.md` §4): per row and round, the parent slot is masked with probability p, so the post-join cycle must build
on the children and the parent's loss reaches them at full strength. Without it the join learned to ignore the
children within the first 3% of training (parent mass 0.39 → 0.95) and the gradient into them shrank with their weight.
Dropped rows are excluded from the training join statistics (NaN); evaluation never drops.

### 4.5 Nested forks (`fork_depth ≥ 2`)

`_node(y, z, depth)`: at `depth < fork_depth` the node forks K children, each `_node(·, depth+1)`, joins them and runs a
post cycle; at `depth = fork_depth` a node is a leaf (one cycle). Codes and join are tied across levels (one set of
weights at every node and depth). Default `fork_depth = 1` because cost grows as K^depth (§9); `fork_depth = 2, K = 2` is
the configured nested arm.

### 4.6 Size and init

Dense parameters ≈ 6.83M (TRM) + fork ≈ K·512 + 512·64 + K·64·512 + join 2·512·64 ≈ 0.2M at K=4. Puzzle-embedding
table: `n_ids × 512` fp32, outside the dense count (≈ 600k rows for the `a1` build with externals, 1.2 GB on the GPU).

### 4.7 Gradient-path invariants (gated)

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

Off by default (every round forks, as above). The user decided (D5) that the model should choose, per row and round,
whether to fork, so forks are spent where they pay. Forking stays in the trained forward; the gate only decides where.

```
no_grad: (H−2) × cycle(y, z, x, 0)                               # pre-fork cycle, every row
g    = MLP([y[:,0] ‖ mean_cells(y)])   2d → gate_hidden → SiLU → 1   # on the detached pre-fork state
fork = g > 0, or (training) with probability fork_explore anyway
forked rows:   children → join (parent-slot dropout) → post cycle → readout      # the FRM round of §4.2
declined rows: no_grad (H−grad_cycles)−(H−2) cycles, then grad_cycles cycles     # = the K = 0 round, same weights
label rows (forked, own join, slot not dropped): no_grad, the declined rows' continuation from the same
         pre-fork state → cf_logits                                             # what not forking would have given
```

- **Execution.** Each subset is gathered, run on its own and scattered back (`index_copy`, autograd-safe), so a declined
  row costs a TRM round and no children. Child outputs keep full-batch shapes: join statistics are NaN and child
  logits zeros on declined rows (`RoundOut.fork` masks them). Exception: the `sham` counterfactual runs the children
  of every row, so the permutation keeps indexing the whole batch. With `compile`, the cycle's batch dimension is
  marked dynamic (subset sizes change every step); ungated models keep static graphs.
- **Training signal.** No end-to-end gradient: the gate's inputs are detached, and it is trained by BCE against the
  label "this fork beat the no-fork continuation by more than the price" (`OBJECTIVE.md` §3.6), measured on the label
  rows. The trunk's losses never reach the gate, and the gate's loss never reaches the trunk.
- **Collapse risk and mitigations.** frmv1's learned gate became a schedule (98.9% of its variance was the cycle index
  [M]). Here: (1) labels come from the *measured* counterfactual benefit of each fork, not from gradients through the
  decision; (2) a **forced-fork familiarisation phase**: for the first `train.fork_force_frac` (0.3) of training every
  row-round forks, so children and join learn everywhere and the gate gets labels on every state; (3) an
  **exploration floor**: afterwards a declined row forks anyway with probability `train.fork_explore` (0.1), so the
  states the gate declines keep producing labels (otherwise a "no" can never be corrected); (4) **no step-index
  input**: the gate reads only the state. `fork.rate_by_round` shows whether it nevertheless became a schedule.
- **Price.** `objective.fork_price` (nats per supervised token, default 0: fork whenever it helps at all). With
  `train.fork_rate_target > 0` the trainer moves it after each gated step:
  price ← max(0, price + `fork_price_lr` · (batch fork rate − target)); not during the forced phase, whose rate is 1
  by construction. The price is trainer state: logged (`price`), checkpointed in `last.pt`/`best.pt`, restored on
  resume; adaptation inherits it and skips the forced phase.
- **Accounting.** The trainer charges `Refiner.round_flops(S, rows, fork_rows, cf_rows)` from the step's actual counts
  (declined rows at the K = 0 cost, forked rows at the FRM cost, label rows plus the continuation's 2 forward
  cycles); `passes_per_round`/`train_flops_per_row` remain the every-round-forks numbers (§9, bench).
- **Evaluation** uses the gate's decisions (no exploration), or with `eval.fork_top` = ρ > 0 forks the top ρ of each
  batch and round by gate logit (padding rows excluded), so the eval fork rate is fixed whatever the gate's calibration
  on held-out states; `fork.gate_rate` logs the gate's own rule beside it. The handoff's "eval fork rate collapses to
  ≈ 0" was that calibration: a 550-PF targeted gate forked 0.3% of dev row-rounds and 79% of composite ones with the
  same weights, and W's price-0 gate already split dev 0.39 / composites 0.95 at 1k PF. On `F` (W, 36k) forking the
  top 25% or 40% gives the dev accuracy of forking every row (`EXPERIMENTS.md` §4c G). `fork_top` = 1 forks and labels
  every row, so `fork.skill` measures the gate's held-out ranking without selection bias (§7.2). `cf.allfork` /
  `cf.gateoff` rerun with every row / no row forking (§7.2). Submission inference (`frm.adapt` without
  `--solutions`) skips the gate labels' no-fork continuation. Requires `fork_depth = 1` and `grad_cycles ≤ 2` (else
  `ValueError`).

### 4.9 Warm start from a published TRM (`scripts/import_trm.py`)

The ARC Prize verified TRM checkpoints (`arcprize/trm_arc_prize_verification`: `arc_v1_public/step_518071`, trained
for TRM's full recipe, 40% on ARC-AGI-1 semi-private; `arc_v2_public`, 6.2% on ARC-AGI-2) load into frm:

- **Weights.** frm's refiner is TRM's (D2): every dense tensor maps one to one (qkv layout, SwiGLU width 1536,
  H_init/L_init → y_init/z_init, q_head's halt row). With `pos_encoding="trm1d"` (D3) the imported model reproduces
  TRM's own forward exactly: on S = 30 the logits of 4 rounds of dev rows are bitwise equal in fp32 and the halt logits
  agree to 2·10⁻⁶ [M: run against `trm-samsung`'s model code, 2026-10-09].
- **Puzzle embedding.** TRM's table is indexed by its dataset's identifiers, which use frm's naming and augmentation
  semantics (identity = the bare task id). A dataset built with `--versions_from` TRM's `identifiers.json` gives every
  ARC/ConceptARC task exactly TRM's ≈1000 versions, so every row transfers (871,328 of 871,328 for `arc1`; the 5,077
  unused rows belong to the 6 tasks the content filter drops). BARC, RE-ARC and composite rows start at the mean TRM
  row, untrained. Imported rows carry one visit, so all of them vote (§7.1).
- **Canvas.** TRM computes on all 900 cells, PAD included, and relies on that space: the imported model scores dev
  pass@2 0.485 on its 30×30 canvas and 0.410 on size buckets (row exact 0.336 → 0.203) [M]. Builds for warm starts
  keep ARC tasks at 30 (`--min_bucket 30 --min_bucket_sources arc`) and bucket only the sources TRM never saw.
- **Continuing.** `train.init_from` = the import, `train.init_mode="continue"`: table and visits kept, fresh optimizer
  and EMA from the imported weights. A K > 0 model takes every shared weight from the checkpoint; its fork modules
  (codes, join, noise map, gate) start fresh and can learn faster than the fine-tuned shared weights
  (`train.fork_lr_mult`, a second optimizer group). This is how FRM is trained on top of a TRM that has already spent
  its full compute budget (`EXPERIMENTS.md` §4b).
- **Contamination.** `arc_v1_public` trained on ARC-AGI-1 training + evaluation *demos* + ConceptARC: the frm dev
  and a1 `test` (evaluation test pairs) and adev (ARC-AGI-2-only tasks) stay clean. `arc_v2_public` trained on
  ARC-AGI-2 training, which holds the test outputs of 376 ARC-AGI-1 evaluation tasks: with it, frm's a1 dev is
  contaminated and a final run needs a dev drawn from ARC-AGI-2 evaluation test pairs (TRM saw only their demos).

### 4.10 Spending forks where they pay after a warm start

What W (`EXPERIMENTS.md` §4b) showed about FRM on top of a finished TRM: per optimizer step FRM is ahead from the
start and the gap grows (training CE equal to TRM's for the first ≈ 1,000 steps, then lower: 0.259 vs 0.307 and exact
0.293 vs 0.226 at step 170k); per unit of compute it pays ≈ 3.4× per step because the gate forks on 60% of round-1 and
97% of later row-rounds (price 0). The fork modules do not damage the source model — a zero-initialised,
function-preserving residual join was built, smoke-tested on MI300X and removed again: at matched steps it tracked the
softmax join (CE 1.000 vs 1.030 at step 200, 0.663 vs 0.649 at step 1,000) and its gate opened to the children within
300 PF, so it removed nothing that was costing compute. The per-PF disadvantage is the per-step price, and it costs most
where steps are scarce: adev adaptation in a fixed 2.5 h gave FRM 8,482 steps against TRM's 31,246 (pass@2 0.190 vs
0.252). Two levers act on that price without making children cheaper or fewer:

- **Compute targeting** (`train.fork_rate_target`, §4.8). The gate's held-out precision rose to 0.64 while it forked
  almost everywhere; holding the fork rate at ρ keeps the forks it ranks highest and makes every declined row a plain
  TRM round. Cost per row-round 25 + 65ρ units at K = 4 without checkpointing (a forked row 80 plus its 10-unit label
  continuation, a TRM round 25): ρ = 0.94 (W) → 3.4×, ρ = 0.4 → 2.0×, ρ = 0.25 → 1.65×.
- **Error-focused children** (`objective.child_focus`, `OBJECTIVE.md` §3.7): the children's final-target loss is
  concentrated on the cells TRM's own round gets wrong, so a fork's benefit (and the gate's precision) can rise
  instead of the children re-deriving what TRM already knows.

Adaptation (Kaggle) uses the same levers: `frm.adapt` overrides such as `train.fork_rate_target=0.25` trade forks
for adaptation steps inside the fixed time budget.

---

## 5. Objective (summary — derivation in `OBJECTIVE.md`)

Per round, per row (mean over rows); written out symbol by symbol in `OBJECTIVE.md` §3.0:

```
L = CE(parent, y*)                                                      # deep supervision on the post-join readout
  + λ_wta · [(1−ε) · Σ_π w_π · c_π  +  ε · mean_k CE(child_k, y*)]       # children: annealed one-to-one set matching
  + λ_q   · [BCE(q_parent, exact_parent) + mean_k BCE(q_k, exact_k)]     # stop / score heads

c_π = CE(child_π(final), y*) + (λ_sub/λ_wta) · mean_{t matched} CE(child_π(t), g_t)    # g_t: composite intermediates
w   = softmax_π(−c / τ) over one-to-one assignments of {y*} ∪ intermediates to distinct children, no gradient
```

With `fork_gate` the children's terms apply to the rows that forked (a declined row's loss is exactly the K = 0
loss) and the gate adds `λ_gate · mean_{label rows} BCE(g, [CE(no-fork continuation) − CE(parent) > price])`
(`OBJECTIVE.md` §3.6).

CE is TRM's stablemax cross-entropy, token-mean per row over non-PAD labels. Defaults: λ_wta = 0.5, λ_sub = 0.5,
λ_q = 0.5, ε = 0.05, τ annealed log-linearly 1.0 → 0.02 over the first 30% of the budget, λ_gate = 0.1, price 0.
Plain rows (no intermediates) reduce to annealed relaxed winner-takes-all. Uniform weight per round (no discounting).
No term rewards a difference between two model-controlled quantities: the gate's label is such a difference, but it is
detached and trains only the gate, which cannot change either quantity.

---

## 6. Training loop

- **Online deep supervision (TRM).** One optimizer step = one round for every row of one bucket's carry. Rows halt when
  `q_parent > 0` (training) or at 16 rounds; exploration: with p=0.1 a row gets a minimum round count U{2..16}. Halted rows
  are replaced by fresh rows pulled from that bucket's stream — only as many as halted (fixes TRM's discarded rows,
  `trm.py:256`).
- **Per-bucket carries.** One carry per bucket; each step picks a bucket with probability ∝ its share of training
  examples ÷ its rows-per-step, so every example is visited at the same rate. Rows per step: a token budget
  (`tokens_per_step`, default 120k parent tokens — the same for every arm, so batch size is not a confound; the A100
  benchmark re-sizes it) divided by sequence length, capped at `max_rows`. Bucketed, 120k tokens is ≈750 rows at S=12 and
  ≈130 at S=30 (TRM: 768 rows of 916 tokens, mostly padding).
- **Sampling.** Group-balanced per bucket (TRM `puzzle_dataset.py`): epochs are permutations of base tasks; a draw picks a
  random version of a task and yields its train examples; rows of a draw left over are carried to the next request, not
  discarded. Source weights mix `arc` and `composite` (default 0.75 / 0.25 of rows).
- **Loss scaling.** Per-row loss averaged over the rows of the step, so step gradients are comparable across buckets.
- **Optimizers.** AdamW (fused) on dense params; SignSGD with decoupled weight decay on the touched puzzle-embedding rows;
  gradient clip 1.0 on dense params; warm-up 2,000 steps then constant (TRM), optionally followed by a cooldown over the
  last `decay_frac` of the budget, lr × (1 − √u) at the fraction u of the window elapsed, down to `decay_to` × lr, for
  the dense and the embedding lr alike (Hägele et al. 2024, arXiv 2405.18392: (1−√u) beats a linear cooldown, and a
  weight average at constant lr leaves a gap to a cooldown; the ARC-AGI-2 final uses 15%). EMA 0.999 of dense params;
  evaluation and `best.pt` use EMA weights; `last.pt` holds raw weights + EMA + optimizer.
- **Budget.** Training stops when the model's FLOP accounting reaches `budget_pflops` (or `max_steps`, or `max_hours`,
  which counts the training time of every link of the run); progress is the largest of those fractions, and the
  objective's τ anneal, the fork schedule and the cooldown follow it, so they are comparable across arms and a
  wall-clock-bound link ends with its cooldown done. `stop_pflops` ends a run early (checkpoint, final eval, `DONE`)
  without changing that schedule; another run with `resume_from` = its `last.pt` continues the full training state
  (weights, EMA, optimizer, step, compute, data streams, RNG, price) under its own config and its own `best.pt`. That
  is how two arms branch from one shared stretch (`EXPERIMENTS.md` §4c).
- **Fork policy (`fork_gate` only, §4.8).** Every row-round forks while progress < `fork_force_frac`; then the gate
  decides with exploration `fork_explore`; the price controller (if `fork_rate_target > 0`) runs after each gated step.
  FLOPs are charged from the step's actual fork and label-row counts, read in the step's existing device→host sync
  (the gated round itself syncs once for the subset sizes and once for the label rows).
- **Precision/speed.** bf16 compute with fp32 master weights (TRM's "casted" layers), fp16 + GradScaler on V100, SDPA
  attention (flash), optional `torch.compile` of the cycle per bucket shape, per-pass checkpointing of child cycles.
- **Fault tolerance.** Atomic checkpoints (`last.pt`: model, EMA, optimizer, embedding table, step, RNG, data-stream
  state) every `ckpt_minutes`; SIGTERM/SIGUSR1 → checkpoint and exit 0; `--resume auto`. Carries are not saved (in-flight
  rows restart; harmless). `best.pt` = best dev pass@2 on the first listed eval split (ARC dev), ties to the later step.
- **Logs.** `train.jsonl` (losses, per-round metrics by round index, first-solve and regression histograms, WTA winner
  counts, fork stats, throughput, memory; `{metric}_by_source` for exact, CE, fork rate, the gate's metrics and
  `child_ce_min`, from each row's data source), `eval.jsonl`, `config.json`.

---

## 7. Inference, evaluation, instruments

### 7.1 Inference

For each test input and each augmented version: run 16 rounds from the init carry; keep, per round, the parent canvas and
q, and at the final round the K children canvases and q's. De-augment (inverse dihedral then inverse colour map), crop,
hash. Vote per test input: rank distinct grids by (vote count, mean σ(q)) over versions — TRM's rule
(`trm-samsung/evaluators/arc.py:107-177`), with children optionally contributing votes at weight `child_vote_weight`
(default 0; a pre-registered variant in `EXPERIMENTS.md`). Top-2 → `attempt_1`, `attempt_2`. Score = mean over tasks of the
fraction of test inputs with a correct attempt (official ARC metric).

**Only trained versions vote** (`loader.eval_versions`): up to `eval.max_versions` versions of a task whose
puzzle-embedding row has been updated at least once (`PuzzleEmbedding.visits`), identity first, then version order; a
task with no trained version gets one blind attempt from its identity version. TRM trains every version it votes with
(≈400 row-rounds per identifier); a budgeted run does not, and an untrained row (zero in pretraining, the reset-to-mean
row in adaptation) carries no task information. Measured in P1 at 3,400 PF: FRM had trained 15% of its dev voting
versions, `trm` 54%, so most FRM votes were blind and outvoted the informed ones. Every eval result carries
`versions` = {coverage (fraction of the evaluated tasks' versions trained), voters (mean versions voting per task),
blind_tasks}; `scripts/compare.py` refuses records without it. The same rule scores adaptation and the Kaggle
submission. Adaptation's new rows start at the mean of the checkpoint's *trained* rows (`train.load_pretrained`);
averaging in the never-trained zero rows would shrink them by the checkpoint's coverage.

### 7.2 Fork instruments (each can fail in the direction of the failure)

Where each lives: training metrics in `train.jsonl` (mean and `_by_round`), evaluation metrics in `eval.jsonl` /
`runs/X/eval/*.json` under `fork`, `cf`, `by_round`, `by_k`, `per_task`.

| metric (key) | definition | failure it detects |
|---|---|---|
| `parent_mass` (train, `fork.parent_mass`) | mean α₀ over grid cells | parent ignores children (→1) |
| `join_entropy` | mean per-cell entropy of α | degenerate uniform blending (→log(K+1)) |
| `win0..winK-1` (train) | fraction of rows each child wins the WTA | dead children (any ≈0) |
| `distinct` (train: answers on supervised cells; eval: distinct cropped, de-augmented answers) | distinct child answers per row | collapse (→1) |
| `child_exact_any`, `child_ce_min`, `fork.oracle_gain` | best child vs parent | children cover nothing the parent lacks |
| `fork.adopt` (+ `adopt_chances`) | P(parent exact after the round ∣ pre-fork parent wrong, some child exact), over all rounds | join cannot select |
| `cf.sham` | Δ exact at round 16 when every round's root join reads the children of a row from a **different task** | parent does not use child *content* |
| `cf.nofork` | Δ exact when the root join is forced onto the parent slot | trained parent does not rely on children |
| `sub_coverage` (train) | composites: fraction of intermediates some child reproduces exactly | no decomposition learned |
| `fork_rate` (train), `fork.rate`, `fork.rate_by_round`, `fork.rate_by_bucket` (eval) | fraction of row-rounds that forked (overall; per round index; per canvas bucket) | gate never forks (→0, the model became TRM), always forks (→1, no compute saved), or became a round schedule (`rate_by_round` a step function of r) |
| `gate_acc`, `gate_pos`, `benefit` (train) | on label rows: (gate logit > 0) == label; label rate; CE(no-fork continuation) − CE(fork) in nats/token | gate no better than the label base rate; forks that do not pay (`benefit` ≤ price) |
| `gate_says` (train), `fork.gate_rate` (eval) | fraction of row-rounds with gate logit > 0 (the gate's own rule, whatever decided: forced phase, exploration, `fork_top`) | the gate's calibration drifts from the rate the run forks at (an eval rate ≈ 0 under logit > 0) |
| `declined_pos` (train) | label rate on rows the gate declined that forked through exploration: the unbiased check of its "no" | the gate declines forks that would pay (`declined_pos` ≈ `gate_pos`) |
| `{metric}_by_source` (train) | exact, CE, fork rate and the gate metrics per data source | forks spent on one source only; a source the warm start has memorised |
| `fork.gate_precision`, `fork.benefit` (eval) | on held-out labelled rows the gate chose to fork (all rounds): fraction where the fork beat the no-fork continuation in CE; mean CE gain | the gate's forks do not pay on unseen pairs (precision ≈ 0.5, benefit ≤ 0) |
| `fork.skill` (eval, `fork_top` = 1) | every row forks and is labelled: AUC of the gate logit for benefit > 0; per ρ the share of the summed benefit the top ρ by logit keep (per batch and round), their precision and benefit; the same by round group | the gate cannot rank forks (AUC ≈ 0.5, `share` ≈ ρ) or ranks the wrong thing (`share` < ρ: `F` at 36k keeps 8% at ρ = 0.4) |
| `price` (train) | the current fork price (moves only with `train.fork_rate_target > 0`) | controller pinned at 0 or running away |
| `cf.allfork`, `cf.gateoff` (gated models) | Δ exact when every row / no row forks, against the run's own decisions | gate declines forks that pay (`allfork` > 0); forking is useless (`gateoff` ≈ 0) |

With the gate, the children instruments (`parent_mass`, `join_entropy`, `distinct`, `child_exact_any`,
`oracle_gain`, child votes) are computed over the rows that forked at the round in question: in training they are
NaN (skipped) on declined rows; at evaluation over the rows that forked at the final round (`fork.forked_rows`;
`fork.forked_labeled` for the labelled-row rates), and `adopt` counts only rounds at which the row forked. Without the
gate every row forks and the numbers are as before.

Counterfactual rows are spread evenly over buckets (`eval.cf_rows` in total).

### 7.3 Depth instruments

| metric (key) | definition |
|---|---|
| `by_round.pass@2`, `by_round.row_exact` | voted pass@2 and unvoted row exact at round r = 1..16 |
| `late_gain`, `late_gain_rows` | round 16 − round 4 (the user's stated failure point); `per_task` holds both rounds for paired CIs (`scripts/compare.py`) |
| `by_round.first_solve`, `by_round.regress` | fraction of labelled rows first exact at round r; exact at r−1 but wrong at r |
| `by_k` | composites: row exact by round for each composition depth k |
| `cf.reset@4` | Δ exact at round 16 when the carry is reset to init at round 4 (do later rounds build on the carry?) |
| training `*_by_round`, `first_solve`, `regress` | the same per-round quantities on the training stream, by each row's round index |

---

## 8. Kaggle path (ARC Prize 2026, ARC-AGI-2 track)

The track requires a Kaggle notebook, no internet, 2 attempts per test input, 12 h GPU run-time [L: arcprize.org
competitions/2026/arc-agi-2; hardware "announced at launch", Kaggle overview cites 96 GB GPU memory — re-check before
packaging]. Puzzle identifiers do not exist for hidden tasks, so the submission runs one **adaptation stage**:

1. Build the hidden tasks (`arc-agi_test_challenges.json`) with `frm.data.build --sources kaggle --challenges …`:
   demos → train rows, test inputs → test rows, `n_aug` versions.
2. The new dataset has its own identifier space; every row of its embedding table starts at the mean of the checkpoint's
   trained rows (`frm.train.load_pretrained`). No pretraining rows are replayed.
3. Fine-tune **all** weights (McGovern: embeddings-only or LoRA was not enough [L]) on the hidden demos for a wall-clock
   budget (`frm.adapt --hours`, optionally `--max_steps`); the adaptation run is resumable like any training run.
4. Run §7.1 inference with EMA weights on **every** test task (the `test` split is never subsampled) and write
   `submission.json` (`{task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}`, one entry per test input, padded by
   repeating attempt 1).

The same script runs locally on `arc2-kaggle` + evaluation2 (with `--solutions`) as the leaderboard proxy. Packaging: the
`frm` package (pure PyTorch + NumPy, no compiled extensions), the checkpoint, and offline wheels if Kaggle's image
lacks a dependency. Adaptation and inference are single-GPU.

---

## 9. Compute model

Unit = one L-level pass (2 blocks over the sequence) = 15.92 GFLOP per sample at 916 tokens [M: SpecTRM]. Per
sample-round with H=3, n=4 (backward ≈ 2× forward on grad passes):

| arm | forward passes | grad passes | total units | × TRM |
|---|---|---|---|---|
| `trm` (K=0) | 15 | 5 | 25 | 1.0 |
| `trm_h2` (K=0, 2 grad cycles) | 15 | 10 | 35 | 1.4 |
| FRM K=2 | 20 | 15 | 50 (+10 with child checkpointing) | 2.0 (2.4) |
| FRM K=4 | 30 | 25 | 80 (+20) | 3.2 (4.0) |
| FRM nested K=2, depth 2 | 40 | 35 | 110 (+20) | 4.4 (5.2) |

`Refiner.passes_per_round` computes these counts and `test_pass_accounting_matches_execution` checks them against the
passes actually executed; `train_flops_per_row` converts them with the exact per-bucket pass cost
`layers · L · (2·N_block + 4·L·d)`, which reproduces TRM's 398.1 GFLOP per 30×30 row-round [M]. Weighted by the default
row mix over the `a1` build's buckets (RE-ARC split per bucket): `trm` 152, `trm_h2` 212, FRM K=2 364, FRM K=4 607 GFLOP
per row-round (child checkpointing on; the ARC-AGI-2 mix is within 1%) — i.e. FRM K=4 on buckets costs ≈1.5× TRM on
30×30. How this lines up with TRM's whole schedule is in `EXPERIMENTS.md` §1. TRM's ARC schedule is ≈1.59e20 FLOPs (≈3 days on 4×H100)
[M: SpecTRM]. Matched compute is enforced in `EXPERIMENTS.md` by giving every arm the same `budget_pflops`.

With `fork_gate` the trainer charges each step's actual mix (`Refiner.round_flops`, checked against executed passes by
`test_round_flops_match_executed_passes`): a declined row costs the `trm` row, a forked row the FRM row, and a label
row adds its no-fork continuation (2 cycles forward, 10 units at n=4). FRM K=4 at fork rate ρ with parent-slot
dropout p costs ≈ 25 + 55ρ + 10ρ(1−p) units, e.g. 43.8 at ρ=0.3, p=0.25 (vs 80 when every round forks).

Memory [I, to be measured in T2]: a grad pass keeps ≈29·d values per token per block (q/k upcast for RoPE, SwiGLU
intermediates, fp32 norm inputs) ≈ 60 KB per token per pass in bf16, so ≈300 KB per token per grad cycle. FRM K=4 without
checkpointing holds (K+1)× that per parent token (≈1.5 MB); per-pass child checkpointing stores ≈2 KB per child pass plus
one pass of transients, ≈0.6 MB per parent token in total. At 80 GB that bounds `tokens_per_step` near 130k for FRM K=4
(hence the 120k default for every arm) and near 250k for `trm`.

---

## 10. Gates (must pass before any GPU job)

`pytest -q` (148 tests, ≈30 s on a laptop CPU):

- `test_data.py`: dihedral inverse for all 8 ids; version apply/invert round trip; dedup; encode/crop round trip in every
  bucket; size rule and buckets unchanged when test outputs are poisoned (whole built dataset byte-identical except
  dev/test outputs); composite buckets ignore held-out targets; no test output in any train row; dev rule (md5 order);
  dev/test rows untranslated and keyed consistently; `take()` shapes, translation keeps content and EOS, sub-goal masks,
  source weights and each row's source code, per-draw cap, exact resume.
- `test_external.py`: canonical digest invariant under dihedral × recolouring; ingest removes planted contamination and
  refuses held-out id overlaps; build refuses externals not filtered against its held-out ids; content filter drops a
  planted training copy of a held-out test output; generator pools split per bucket; adev files; `--exclude_ids`.
- `test_composite.py`: determinism, filters (no single primitive or shorter program reproduces a task), detour quota,
  sub-goals equal program prefixes, every primitive exercised.
- `test_model.py`: `test_fork_gate` (child cycles run with grad; codes, join and β get gradient from `L_parent` alone),
  `test_nested_fork_gate`, `test_trm_mode_matches_reference_structure` (K=0 equals an independent restatement of TRM's
  round), carry detached and reset, `test_every_param_gets_grad`, inference == training forward,
  `test_children_get_fresh_noise_each_round`, `test_child_checkpointing_is_exact`, pass accounting == executed passes,
  counterfactual modes, parent-slot dropout (children only in training, never at eval), embedding visit counts (and
  their derivation for checkpoints written before the counter), reset-to-mean over trained rows only.
- `test_gate.py` (adaptive forking, §4.8): `fork_mode="none"` equals the K=0 model with the same weights;
  `fork_mode="all"` equals the ungated round (incl. parent-slot dropout); in a mixed batch every row equals its own
  path run alone; `cf_logits` equal an independent restatement of TRM's round; dropped parent slots give no label;
  declined rows send no gradient into the fork; the gate trains only from its loss and sends nothing into the trunk;
  decisions and exploration; declined rows' loss is exactly the K=0 loss and child terms add per forked row; gate label
  = benefit > price, plain mean over label rows; error-focused children (`child_focus` = 1 ignores cells TRM's round
  gets right); `round_flops` boundaries and executed passes; gated training steps (forced phase, price controller,
  resume restores the price), per-source training logs, the fork learning-rate multiplier, the (1−√u) cooldown, a
  branch from another run's state (`resume_from`, own `best.pt`) and `stop_pflops`/cumulative `max_hours`, and
  `evaluate_split` (`fork.rate`, `cf.allfork/gateoff`; `fork_top` forks ceil(ρ·n) real rows per batch and round,
  padding excluded; at `fork_top` 1 the `fork.skill` block covers every labelled row-round; `gate_skill`'s AUC and
  per-ρ benefit shares).
- `test_objective.py`: stablemax normalised, NaN-free gradients and non-zero gradients at zero logits; PAD ignored;
  assignment table one-to-one; set matching reduces to annealed WTA, sends distinct targets to distinct children and
  respects masks/spare children; each child's gradient = w_k × its own CE gradient; mixed composite/plain batches.
- `test_evaluate.py`: official scoring and per-task outcomes, vote ranking, only trained versions vote (identity first,
  capped, blind fallback), submission shape, sham children from other tasks.
- `test_train_smoke.py`: build → train → SIGTERM → resume (step count continues) → DONE → evaluate → `--final` guard →
  adapt → valid `submission.json`.

---

## 11. Module layout and interfaces (the code contract)

```
frm/
  config.py            dataclasses ModelCfg, ObjectiveCfg, DataCfg, TrainCfg, EvalCfg; Cfg.load(json) + dot-path overrides
  data/
    arc.py             load_tasks(challenges, solutions=None), load_subset(raw_dir, subset)
    augment.py         dihedral / inverse_dihedral, Version(tid, perm).apply/invert/parse/name, make_versions
    encode.py          PAD/EOS/colour layout: encode_grid, labels_from, crop, render (vectorised augment+translate)
    sizerule.py        predict_output_size(task, test_input), choose_bucket(task) — never reads test outputs
    heldout.py         canon (8 dihedral × colour relabel digest), informative, HeldOut(...).hit — the content filter
    build.py           CLI: dataset dir from --sources arc1|arc2|arc2-kaggle|kaggle, --composite JSON, --external
                       SOURCE=PATH, --exclude_ids; dev rule; content filter; generator pools split per bucket
    loader.py          Dataset(root): buckets, identifiers, tasks(split), splits(); TrainStream(ds, S, rng, weights,
                       translate_p, max_pairs_per_draw).take(n) + state_dict; eval_batches(ds, split, S, rows, ...)
    composite.py       33 primitives, generate(n_tasks, depth_range, seed, ...) with sub-goals; CLI writes JSON
  model/
    layers.py          rms_norm, CastedLinear, SwiGLU, Rope2D, Attention (SDPA), Block, block_params
    embedding.py       PuzzleEmbedding (buffer table, gather-to-leaf, SignSGD step)
    refiner.py         Refiner(ModelCfg, num_ids): init_carry, reset_carry, round(carry, inputs, ids, cf, sham_perm, gen,
                       fork_mode, explore) -> RoundOut, passes_per_round(fork), train_flops_per_row, eval_flops_per_row,
                       round_flops(S, rows, fork_rows, cf_rows, train), compile_cycles
  objective.py         set_match, Objective(ObjectiveCfg)(RoundOut, batch, progress, has_subgoals, price)
                       -> (loss, per-row metrics)
  train.py             CLI: python -m frm.train --config cfg.json --run_dir runs/X [--resume auto] [key=value ...]
  evaluate.py          CLI + evaluate_split: per-round voting, pass@k, instruments, counterfactuals, submission
  adapt.py             CLI: Kaggle adaptation (build, re-initialised ids, fine-tune, predict, write submission)
  monitor/             the live monitor (laptop side, no torch): source.py (Kelvin2 over the ssh master, incremental
                       reads; local dirs), run.py (live pass, rates, eta, liveness, alerts), charts.py, view.py
configs/               trm_a1, frm_a1, frm2_a1, frm_a1_arc (pilots); trm_h2_a1, nest_a1 (optional); warm starts from TRM:
                       trm_import, trmc_a1tn / frmc_a1tn (W), frmc40_a1tn / frmcf40_a1tn (R); the ARC-AGI-2 final:
                       final_frm_a2tn, final_trm_a2tn (control); smoke; learn (CPU check)
kelvin/                k2 helper (incl. push-external, monitor), submit.sh (approval thresholds), jobs/{env.sh, setup_env,
                       data, smoke, bench, train, eval}
scripts/               fetch_data.py, ingest_external.py, make_adapt_dev.py, bench.py, compare.py (paired bootstrap + McNemar),
                       monitor.py (live monitor; every reading explained in monitor.md)
tests/                 test_data, test_external, test_composite, test_model, test_gate, test_objective, test_evaluate,
                       test_train_smoke, test_monitor
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

| risk | detection | response | available today? |
|---|---|---|---|
| parent ignores children | `parent_mass → 1`, `cf/sham ≈ 0` | raise `objective.lambda_wta`; join-dropout (mask the parent slot with p=0.25 in training); more pre-fork cycles (`H_cycles=4`) so children start from a better state | all config (`model.join_drop_parent`, applied from P1's rerun) |
| children collapse | `distinct → 1`, win rates degenerate | raise `objective.wta_eps`, slow the τ anneal (`tau_anneal_frac`), per-child noise on codes | ε, τ: config. noise: a code change |
| fork costs more than it returns | matched-FLOP dev pass@2 below `trm` | K=2; cheaper children (`child_L_cycles=2`); adaptive forking (`model.fork_gate`, §4.8: fork where the gate predicts it pays); fork only on alternate rounds | K, child_L_cycles, fork_gate: config. alternate rounds: a code change |
| fork gate collapses | `fork_rate` → 0 or 1 after the forced phase, `rate_by_round` a step in r, `gate_acc` ≈ max(`gate_pos`, 1−`gate_pos`), `cf.allfork` > 0 | raise `fork_explore`, lengthen `fork_force_frac`, set `fork_rate_target` (price controller) | config |
| depth still unused | `late_gain ≈ 0` on dev and composites | raise composite fraction/depth (`data.source_weights`, generator `--depth_max`); inspect `cf/reset@4`; more rounds trained (`halt_explore`) | config |
| memory | OOM in T2 | `grad_ckpt_children`; smaller `tokens_per_step` | config |
| Kaggle hardware/time | adaptation does not fit 12 h | fewer versions (`--n_aug`), fewer inference rounds (`eval.rounds`; per-round curves say where accuracy saturates) | config |

Any response that removes forking from the trained model is out of scope for the agent; it goes to the user as its own
question with the sentence "after this, the model does not train forks".

## 13. Non-goals

In-context demonstration encoders; symbolic solvers or DSL search at inference; multi-GPU DDP (single-GPU jobs schedule
fastest on Kelvin and every arm fits one A100; DDP is a later change if a final run needs wall-clock); ARC-AGI-3.
