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
| **forking**: every training round spawns K children from the parent state, runs them through the same tied refiner, and joins them back into the parent before the graded readout | **yes, every round, K≥1 in every non-control config** | `Refiner.round` (`_fork`, `_join`) | `tests/test_model.py::test_fork_gate` fails if no child cycle runs or if the parent loss sends no gradient into the fork codes, child cycles and join |
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
   that forks K children with distinct codes, trains them as competing hypotheses, and lets the parent read them,
   injects *new information into every round*. Depth paid off in the lineage only when new information arrived between
   segments [M: frmv2 `p2_c` checker arm, s1→s3 gain +0.023 vs ≈0 in every other arm].
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
| D3 | positions | 2D RoPE (row/col halves of each head), prefix tokens unrotated | needed for size buckets; 1D RoPE over a variable stride was a measured frmv1 bug [M: `frmv1/models/layers.py:44-52`] | TRM 1D RoPE over 30×30 |
| D4 | canvas | size buckets {8,12,16,20,24,30}, chosen from demo grids + test input + a demo-verified size rule; never from the test output | token cost 0.32–0.49 of 30×30 on ARC-AGI-1 sets, 0.82 on ARC-AGI-2 eval; overflow 2/419 a1 eval, 1/172 a2 eval [M: computed from `trm-samsung/kaggle/combined`, see §3.3] | fixed 30×30 (2–3× more tokens) |
| D5 | fork structure | **fixed-structure**: every round forks K children (default K=4) at the middle cycle; no learned "whether to fork" gate | learned gates collapse to schedules [M: frmv1 gate 98.9% cycle-index variance]; discrete decisions add high-variance gradients [L: research C10] | learned fork gate, REINFORCE/Gumbel fork decisions |
| D6 | child symmetry breaking | non-zero, distinct learned fork codes injected at **every** child z-update (inside the gradient window), plus a parent-conditioned per-cell code | tiny's zero-init codes + zero-init arbitration collapsed onto child 0 (w=0.993) [M: `tiny/results/fork_probe.json`]; anything only on a no-grad path never trains [M: tinyv2.2 `init_state`, ARM latents] | zero-init codes, seeds only in warm-up |
| D7 | child information | children start from the full parent state (y, z) and see the full input x; nothing is masked | masked/starved children equalled random shams [M: tinyabstractor I-9, children right 0.13 vs base 0.74 on disagreements] | crop/masked-plane children |
| D8 | join | per-cell softmax attention over {parent, child₁..child_K} on (y, z), with gradient; parent slot lets the parent ignore children | detached joins train nothing [M: tinyabstractor `fork_in` detaches all; frmv2 "nothing learned through the merge" removed forks]; merge forms carried the decomposition solver's gains (21-0) while branch-only forms were null [M: decomposition-solver-arc ladder] | detached claim join, inference-only selection |
| D9 | child objective | annealed, relaxed winner-takes-all over children (+ sub-goal set matching on synthetic composites) | MCL/rWTA/annealed MCL [L: 1606.07839, 1612.00197, 2407.15580]; same-target CE on all children → collapse [M: tiny] | per-child CE to the final target, diversity bonuses |
| D10 | depth demand | synthetic composite tasks f_k∘…∘f_1 (k = 1..4) with intermediate grids, mixed with ARC; per-round metrics by k | corpora without detour tasks cannot test the thesis [M: tinyabstractor `synth/compose.py:29-32` refused "second step hides the first"; frmv1 branching factor 1 in 100% of tasks] | ARC only |
| D11 | deep supervision regime | TRM's online carry: one optimizer step per round, halted rows replaced, per-bucket carries | TRM's regime is what produced 40%+; frmv1/frmv2 replaced it with unrolled graphs and lost TRM's behaviour [M] | unrolled multi-segment graph per step |
| D12 | optimizer | AdamW (0.9, 0.95), lr 1e-4, wd 0.1, warm-up 2k, constant; puzzle embedding SignSGD lr 1e-2 wd 0.1; EMA 0.999 | TRM `cfg_pretrain.yaml` (AdamATan2 replaced by AdamW: no CUDA extension to build on Kelvin) | Muon (misrouted in tiny [M]) |
| D13 | halting | TRM: q_halt = BCE(current parent canvas exactly correct), halt when q>0 in training with 10% exploration, always 16 rounds at eval | TRM default; q also ranks votes and children | PonderNet/geometric priors (reward early exit [L: Ouro 2510.25741]) |
| D14 | evaluation discipline | checkpoints chosen on **dev** (held-out test pairs of 25% of training and concept tasks; composites are reported, not selected on); ARC eval test pairs read only for reported numbers | tiny/tinyv2.2 picked best.pt on eval test pairs [M] | selecting on eval |

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
version table cannot depend on test outputs). Default `n_aug=300` (ARC Prize's HRM analysis: 300 ≈ max, 30 within
4pp [L]); `n_aug=1000` for a final run if dev says it matters. Measured: `arc1` + 2000 composites at `n_aug=300` →
324,368 identifiers, 71 MB on disk, built in ≈30 s [M].

Augmentation and translation are applied **on the fly** from the version table (dihedral id + permutation per
identifier), vectorised per batch (`frm/data/encode.py:render`; 23 ms for 4,096 rows at S=16 on a laptop CPU [M]), so
the stored dataset holds each base grid once.

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
| `arc1` | ARC-AGI-1 training (400) + evaluation (400) + ConceptARC (160) | demos of all → train; test pairs of training/concept → train, except the **dev** hold-out (`--dev_frac`, default 25% of training and concept tasks by sha256 of task id: 141 tasks); evaluation test pairs → test |
| `arc2` | ARC-AGI-2 training (1000) + evaluation (120) + ConceptARC | same pattern with evaluation2 as test. ARC-AGI-2 training contains 376 ARC-AGI-1 eval tasks [M: `decomposition-solver-arc/decomp/harvest.py:72`], so `arc1` and `arc2` never mix |
| `arc2-kaggle` | as `arc2` but **without** evaluation2 demos | Kaggle proxy: pretrain on this, then run the §8 adaptation on evaluation2 |
| `composite` | synthetic f_k∘…∘f_1 tasks with intermediate grids (§3.5) | demos → train; 2 held-out pairs per task → `dev_composite` |
| `kaggle` | hidden test challenges at competition time | demos → train (adaptation); test inputs → test |

Leakage rules (gated in `tests/test_data.py`): no eval test output enters any train row; buckets, offsets and size rules
read inputs and demos only; dev/test rows are never translated; a poisoning test perturbs test outputs and asserts the
train arrays are unchanged.

### 3.5 Composite tasks (depth demand)

`frm/data/composite.py` generates tasks whose rule is a composition of k ∈ {1..4} primitives from a library of grid
transforms (dihedral maps, quadrant/half swaps, gravity in 4 directions, colour maps, recolour-by-size, fill enclosed
regions, crop-to-content, tile, upscale, mirror-concatenate, border, shift, object-wise ops). For every example it stores
the k−1 intermediate grids g₁ = f₁(x), g₂ = f₂(g₁), …. Filters (from `decomposition-solver-arc/decomp/synth.py:make_task`,
reimplemented): every step changes the grid on every demo; the composition is not equal on all demos to any single
primitive or shorter composition; at least one intermediate is *not* closer to the output than the input is (a genuine
detour, e.g. "shuffle quadrants then gravity"); grids ≤ 30×30. Each task: 6–10 demo pairs + 2 held-out pairs, its own
augmented versions (`n_aug_composite`, default 20).

Mix: composites are a fixed fraction of training rows (`composite_frac`, default 0.25). They are the only data with
known compositional depth, so they double as the depth instrument (§7.3).

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
    c_k   = e_k + A_k · B · rms(z)                               # fork code, per child, per cell
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

### 4.5 Nested forks (`fork_depth ≥ 2`)

`_node(y, z, depth)`: at `depth < fork_depth` the node forks K children, each `_node(·, depth+1)`, joins them and runs a
post cycle; at `depth = fork_depth` a node is a leaf (one cycle). Codes and join are tied across levels (one set of
weights at every node and depth). Default `fork_depth = 1` because cost grows as K^depth (§9); `fork_depth = 2, K = 2` is
the configured nested arm.

### 4.6 Size and init

Dense parameters ≈ 6.83M (TRM) + fork ≈ K·512 + 512·64 + K·64·512 + join 2·512·64 ≈ 0.2M at K=4. Puzzle-embedding
table: `n_ids × 512` fp32, outside the dense count (≈ 289k rows for `arc1` at `n_aug=300`, 0.59 GB).

### 4.7 Gradient-path invariants (gated)

1. FRM round: `e_k`, `A_k`, `B`, `W_q`, `W_k`, β and f's weights all receive non-zero gradient from `L_parent` alone.
2. Children's cycles run with grad; the pre-fork cycle runs without.
3. The carry is detached: a round's loss sends no gradient into the previous round.
4. `trm` control reproduces TRM's structure: (H−1) no-grad cycles, one grad cycle, same op order.
5. Inference forward == training forward (same function, no grad).
6. No parameter is reachable only through a no-grad path (`test_every_param_gets_grad`).

---

## 5. Objective (summary — derivation in `OBJECTIVE.md`)

Per round, per row (mean over rows):

```
L = CE(parent)                                      # deep supervision on the post-join readout
  + λ_wta · Σ_k w_k · CE(child_k)                    # annealed relaxed WTA, w = (1−ε)·softmin_τ(CE) + ε/K, sg(w)
  + λ_sub · mean_t softmin-match_k CE(child_k, g_t)  # composites only: each intermediate grid claimed by some child
  + λ_q   · [BCE(q_parent, exact_parent) + mean_k BCE(q_k, exact_k)]
```

CE is TRM's stablemax cross-entropy, token-mean per row over non-PAD labels. Defaults: λ_wta = 0.5, λ_sub = 0.5,
λ_q = 0.5, ε = 0.05, τ annealed log-linearly 1.0 → 0.02 over the first 30% of steps. Uniform weight per round (no
discounting). No term rewards a difference between two model-controlled quantities.

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
  gradient clip 1.0 on dense params; warm-up 2,000 steps then constant (TRM). EMA 0.999 of dense params; evaluation and
  `best.pt` use EMA weights; `last.pt` holds raw weights + EMA + optimizer.
- **Budget.** Training stops when the model's FLOP accounting reaches `budget_pflops` (or `max_steps`, or `max_hours`);
  the objective's τ anneal is scheduled on the same progress fraction, so it is comparable across arms.
- **Precision/speed.** bf16 compute with fp32 master weights (TRM's "casted" layers), fp16 + GradScaler on V100, SDPA
  attention (flash), optional `torch.compile` of the cycle per bucket shape, per-pass checkpointing of child cycles.
- **Fault tolerance.** Atomic checkpoints (`last.pt`: model, EMA, optimizer, embedding table, step, RNG, data-stream
  state) every `ckpt_minutes`; SIGTERM/SIGUSR1 → checkpoint and exit 0; `--resume auto`. Carries are not saved (in-flight
  rows restart; harmless). `best.pt` = best dev pass@2 on the first listed eval split (ARC dev), ties to the later step.
- **Logs.** `train.jsonl` (losses, per-round metrics by round index, first-solve and regression histograms, WTA winner
  counts, fork stats, throughput, memory), `eval.jsonl`, `config.json`.

---

## 7. Inference, evaluation, instruments

### 7.1 Inference

For each test input and each augmented version: run 16 rounds from the init carry; keep, per round, the parent canvas and
q, and at the final round the K children canvases and q's. De-augment (inverse dihedral then inverse colour map), crop,
hash. Vote per test input: rank distinct grids by (vote count, mean σ(q)) over versions — TRM's rule
(`trm-samsung/evaluators/arc.py:107-177`), with children optionally contributing votes at weight `child_vote_weight`
(default 0; a pre-registered variant in `EXPERIMENTS.md`). Top-2 → `attempt_1`, `attempt_2`. Score = mean over tasks of the
fraction of test inputs with a correct attempt (official ARC metric).

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
`layers · L · (2·N_block + 4·L·d)`, which reproduces TRM's 398.1 GFLOP per 30×30 row-round [M]. Example-weighted over
`arc1` training buckets: `trm` 142, `trm_h2` 199, FRM K=2 342, FRM K=4 570 GFLOP per row-round (child checkpointing on)
— i.e. FRM K=4 on buckets costs ≈1.4× TRM on 30×30. TRM's ARC schedule is ≈1.59e20 FLOPs (≈3 days on 4×H100)
[M: SpecTRM]. Matched compute is enforced in `EXPERIMENTS.md` by giving every arm the same `budget_pflops`.

Memory [I, to be measured in T2]: a grad pass keeps ≈29·d values per token per block (q/k upcast for RoPE, SwiGLU
intermediates, fp32 norm inputs) ≈ 60 KB per token per pass in bf16, so ≈300 KB per token per grad cycle. FRM K=4 without
checkpointing holds (K+1)× that per parent token (≈1.5 MB); per-pass child checkpointing stores ≈2 KB per child pass plus
one pass of transients, ≈0.6 MB per parent token in total. At 80 GB that bounds `tokens_per_step` near 130k for FRM K=4
(hence the 120k default for every arm) and near 250k for `trm`.

---

## 10. Gates (must pass before any GPU job)

`pytest -q` (84 tests, ≈25 s on a laptop CPU):

- `test_data.py`: dihedral inverse for all 8 ids; version apply/invert round trip; dedup; encode/crop round trip in every
  bucket; size rule and buckets unchanged when test outputs are poisoned (whole built dataset byte-identical except test
  outputs); composite buckets ignore held-out targets; no test output in any train row; dev/test rows untranslated and
  keyed consistently; `take()` shapes, translation keeps content and EOS, sub-goal masks, source weights, exact resume.
- `test_composite.py`: determinism, filters (no single primitive or shorter program reproduces a task), detour quota,
  sub-goals equal program prefixes, every primitive exercised.
- `test_model.py`: `test_fork_gate` (child cycles run with grad; codes, join and β get gradient from `L_parent` alone),
  `test_nested_fork_gate`, `test_trm_mode_matches_reference_structure` (K=0 equals an independent restatement of TRM's
  round), carry detached and reset, `test_every_param_gets_grad`, inference == training forward,
  `test_child_checkpointing_is_exact`, pass accounting == executed passes, counterfactual modes.
- `test_objective.py`: stablemax normalised and NaN-free gradients, PAD ignored, WTA weights at τ→0/∞, each child's
  gradient = w_k × its own CE gradient, sub-goal matching, distinct-answer count.
- `test_evaluate.py`: official scoring and per-task outcomes, vote ranking, submission shape, sham children from other
  tasks.
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
    build.py           CLI: dataset dir from --sources arc1|arc2|arc2-kaggle|kaggle and --composite JSON
    loader.py          Dataset(root): buckets, identifiers, tasks(split), splits(); TrainStream(ds, S, rng, weights,
                       translate_p).take(n) + state_dict; eval_batches(ds, split, S, rows, max_versions)
    composite.py       33 primitives, generate(n_tasks, depth_range, seed, ...) with sub-goals; CLI writes JSON
  model/
    layers.py          rms_norm, CastedLinear, SwiGLU, Rope2D, Attention (SDPA), Block, block_params
    embedding.py       PuzzleEmbedding (buffer table, gather-to-leaf, SignSGD step)
    refiner.py         Refiner(ModelCfg, num_ids): init_carry, reset_carry, round(carry, inputs, ids, cf) -> RoundOut,
                       passes_per_round, train_flops_per_row, eval_flops_per_row, compile_cycles
  objective.py         Objective(ObjectiveCfg)(RoundOut, batch, progress) -> (loss, per-row metrics)
  train.py             CLI: python -m frm.train --config cfg.json --run_dir runs/X [--resume auto] [key=value ...]
  evaluate.py          CLI + evaluate_split: per-round voting, pass@k, instruments, counterfactuals, submission
  adapt.py             CLI: Kaggle adaptation (build, re-initialised ids, fine-tune, predict, write submission)
configs/               trm_a1, trm_h2_a1, frm2_a1, frm_a1, nest_a1, frm_a2_kaggle, frm_a2; smoke; learn (CPU check)
kelvin/                k2 helper, submit.sh (approval thresholds), jobs/{env.sh, setup_env, data, smoke, bench, train, eval}
scripts/               fetch_data.py, bench.py, compare.py (paired bootstrap + McNemar between arms)
tests/                 test_data, test_composite, test_model, test_objective, test_evaluate, test_train_smoke
```

`Batch` (all torch tensors on device unless noted): `inputs` int [B, S²], `labels` int [B, S²] (−100 = ignore),
`puzzle_ids` int [B], `subgoals` int [B, M, S²] (−100 rows where absent), `subgoal_mask` bool [B, M], plus numpy-side
`keys` for eval rows (task id, test index, identifier).

`RoundOut`: `logits` [B, S², 12] float32, `q` [B], `child_logits` [B, K, S², 12] or None, `child_q` [B, K] or None,
`join_alpha` [B, K+1] (cell-mean, for stats), `join_entropy` [B], `carry` (y, z detached).

---

## 12. Risks and planned responses (redesign, never removal)

| risk | detection | response | available today? |
|---|---|---|---|
| parent ignores children | `parent_mass → 1`, `cf/sham ≈ 0` | raise `objective.lambda_wta`; join-dropout (mask the parent slot with p=0.25 in training); more pre-fork cycles (`H_cycles=4`) so children start from a better state | λ_wta, H_cycles: config. join-dropout: a code change |
| children collapse | `distinct → 1`, win rates degenerate | raise `objective.wta_eps`, slow the τ anneal (`tau_anneal_frac`), per-child noise on codes | ε, τ: config. noise: a code change |
| fork costs more than it returns | matched-FLOP dev pass@2 below `trm` | K=2; cheaper children (`child_L_cycles=2`); fork only on alternate rounds | K, child_L_cycles: config. alternate rounds: a code change |
| depth still unused | `late_gain ≈ 0` on dev and composites | raise composite fraction/depth (`data.source_weights`, generator `--depth_max`); inspect `cf/reset@4`; more rounds trained (`halt_explore`) | config |
| memory | OOM in T2 | `grad_ckpt_children`; smaller `tokens_per_step` | config |
| Kaggle hardware/time | adaptation does not fit 12 h | fewer versions (`--n_aug`), fewer inference rounds (`eval.rounds`; per-round curves say where accuracy saturates) | config |

Any response that removes forking from the trained model is out of scope for the agent; it goes to the user as its own
question with the sentence "after this, the model does not train forks".

## 13. Non-goals

In-context demonstration encoders; symbolic solvers or DSL search at inference; multi-GPU DDP (single-GPU jobs schedule
fastest on Kelvin and every arm fits one A100; DDP is a later change if a final run needs wall-clock); ARC-AGI-3.
