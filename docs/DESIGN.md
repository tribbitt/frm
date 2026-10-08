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
| D14 | evaluation discipline | checkpoints chosen on **dev** (held-out test pairs of 10% of training tasks + held-out composite pairs); ARC eval test pairs read only for reported numbers | tiny/tinyv2.2 picked best.pt on eval test pairs [M] | selecting on eval |

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
fixed), deduplicated by puzzle hash (TRM `build_arc_dataset.py:126-164`). Default `n_aug=300` (ARC Prize's HRM analysis:
300 ≈ max, 30 within 4pp [L]); `n_aug=1000` for a final run if dev says it matters.

Augmentation is applied **on the fly** from the version table (dihedral id + permutation per identifier), so the stored
dataset holds each base grid once.

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

Training rows of a task use the task's bucket. Translation: each training row is shifted by a fresh random offset at
load time with probability 0.75 (input, output and sub-goals share the offset); test rows are never shifted.

### 3.4 Sources and splits

| source | contents | split use |
|---|---|---|
| `arc1` | ARC-AGI-1 training (400) + evaluation (400) + ConceptARC (160) | demos of all → train; test pairs of training/concept → train, except the **dev** hold-out (10% of training tasks by sha256 of task id); evaluation test pairs → test |
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
- **Batching.** Children are a batch expansion [B, K, L, D] → [B·K, L, D]; x is expanded as a view. No separate buckets,
  no separate episodes (the frmv2 failure mode). Child cycles may use activation checkpointing (`grad_ckpt_children`).
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
  (`tokens_per_step`, default 350k parent tokens, TRM-mode) divided by sequence length, capped at `max_rows`.
- **Sampling.** Group-balanced per bucket (TRM `puzzle_dataset.py`): epochs are permutations of base tasks; a draw picks a
  random version of a task and yields its train examples. Source weights mix `arc*` and `composite`.
- **Loss scaling.** Per-row loss averaged over the rows of the step, so step gradients are comparable across buckets.
- **Optimizers.** AdamW (fused) on dense params; SignSGD with decoupled weight decay on the touched puzzle-embedding rows;
  gradient clip 1.0 on dense params; warm-up 2,000 steps then constant (TRM). EMA 0.999 of dense params, evaluated and
  saved alongside raw weights.
- **Precision/speed.** bf16 autocast (fp16 + GradScaler path for V100), SDPA attention (flash), optional
  `torch.compile` per bucket shape, optional activation checkpointing of child cycles.
- **Fault tolerance.** Atomic checkpoints (`last.pt`: model, EMA, optimizer, embedding table, step, RNG, data-stream
  state) every `ckpt_minutes`; SIGTERM/SIGUSR1 → checkpoint and exit 0; `--resume auto`. Carries are not saved (in-flight
  rows restart; harmless).
- **Logs.** `train.jsonl` (losses, per-round metrics by round index, fork stats, throughput), `eval.jsonl`, `config.json`.

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

| metric | definition | failure it detects |
|---|---|---|
| `fork/parent_mass` | mean α₀ over cells | parent ignores children (→1) |
| `fork/join_entropy` | mean per-cell entropy of α | degenerate uniform blending (→log(K+1)) |
| `fork/win_rate[k]` | fraction of rows child k wins the WTA | dead children (any ≈0) |
| `fork/distinct` | mean number of distinct child canvases per row | collapse (→1) |
| `fork/oracle_gain` | exact(best child) − exact(parent, same round) | children cover nothing the parent lacks |
| `fork/adopt` | P(parent correct after join ∣ some child correct, pre-fork parent wrong) | join cannot select |
| `cf/sham` | Δ exact when children are rolled across rows before the join (eval) | parent does not use child *content* |
| `cf/nofork` | Δ exact when α is forced onto the parent slot (eval) | trained parent does not rely on children |
| `sub/coverage` | composites: fraction of rows where some child reproduces an intermediate grid exactly | no decomposition learned |

### 7.3 Depth instruments

| metric | definition |
|---|---|
| `depth/acc@r` | parent exact (and pass@2 after voting) at round r = 1..16 on dev, eval, `dev_composite` |
| `depth/late_gain` | acc@16 − acc@4 (the user's stated failure point) with paired-bootstrap CI |
| `depth/first_solve` | histogram of the first round at which a row is correct; `depth/regress` = rows correct at r−1, wrong at r |
| `depth/by_k` | acc@r on composites by composition depth k |
| `cf/reset@r` | Δ acc when the carry is reset to init at round r (does round r+1 use the carry?) |
| training | the same per-round exact/CE aggregated by each row's round index on the training stream |

---

## 8. Kaggle path (ARC Prize 2026, ARC-AGI-2 track)

The track requires a Kaggle notebook, no internet, 2 attempts per test input, 12 h GPU run-time [L: arcprize.org
competitions/2026/arc-agi-2; hardware "announced at launch", Kaggle overview cites 96 GB GPU memory — re-check before
packaging]. Puzzle identifiers do not exist for hidden tasks, so the submission runs one **adaptation stage**:

1. Build the hidden tasks (`arc-agi_test_challenges.json`) with `frm.data.build --source kaggle`: demos → train rows,
   test inputs → test rows, `n_aug` versions.
2. Append their identifiers to the checkpoint's embedding table (new rows = mean of trained rows; old rows kept).
3. Fine-tune **all** weights (McGovern: embeddings-only or LoRA was not enough [L]) on the hidden demos for a fixed
   step budget sized to the time limit (`frm.adapt --time_budget_h`), optionally replaying a fraction of pretraining
   rows.
4. Run §7.1 inference and write `submission.json` (`{task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}`, one entry
   per test input, padded by repeating attempt 1).

The same script runs locally on `arc2-kaggle` + evaluation2 as the leaderboard proxy. Packaging: the `frm` package
(pure PyTorch + NumPy, no compiled extensions), the checkpoint, and offline wheels if Kaggle's image lacks a dependency.

---

## 9. Compute model

Unit = one L-level pass (2 blocks over the sequence) = 15.92 GFLOP per sample at 916 tokens [M: SpecTRM]. Per
sample-round with H=3, n=4 (backward ≈ 2× forward on grad passes):

| arm | forward passes | grad passes | total units | × TRM |
|---|---|---|---|---|
| `trm` (K=0) | 15 | 5 | 25 | 1.0 |
| `trm_h2` (K=0, 2 grad cycles) | 15 | 10 | 35 | 1.4 |
| FRM K=2 | 20 | 15 | 50 | 2.0 |
| FRM K=4 | 30 | 25 | 80 | 3.2 |
| FRM nested K=2, depth 2 | 40 | 35 | 110 | 4.4 |

Buckets cut the per-pass cost to ≈0.35–0.45 of a 916-token pass on ARC-AGI-1/2 training data (§3.3; attention more than
that), so FRM K=4 on buckets costs ≈1.3× TRM on 30×30 per sample-round. TRM's ARC schedule is ≈1.59e20 FLOPs (≈3 days
on 4×H100) [M: SpecTRM]. One A100 at a realistic 40% of bf16 peak ≈ 1.1e19 FLOP/day. Matched compute is enforced in
`EXPERIMENTS.md` by giving every arm the same FLOP budget (`budget_pflops`), not the same step count.

Memory: a grad cycle stores ≈155 KB of activations per token [I: per-block tensor count × bf16]; FRM K=4 stores
(K+1)× the parent's. Child-cycle activation checkpointing trades +1 forward of the child cycles for ≈5× less child
memory; the A100 benchmark (`EXPERIMENTS.md` T2) decides `grad_ckpt_children` and `tokens_per_step`.

---

## 10. Gates (must pass before any GPU job)

`pytest -q` in `tests/`:

- `test_data.py`: augmentation inverse round-trip for all 8 dihedral × random permutations; encode/crop round-trip;
  bucket choice never reads test outputs (poisoning test); dev/test rows untranslated; version dedup.
- `test_model.py`: `test_fork_gate` (child cycles executed; non-zero grads into codes, join, children from `L_parent`
  alone), `test_trm_mode_matches_reference_structure`, `test_carry_detached`, `test_every_param_gets_grad`,
  `test_inference_equals_training_forward`, `test_nested_fork_gate`.
- `test_objective.py`: WTA weights sum to 1, winner gets ≥ 1−ε+ε/K at τ→0, uniform at τ→∞; sub-goal matching picks the
  closest child; stablemax CE ignores PAD; q targets are exact-match.
- `test_train_smoke.py`: a CPU run on a tiny dataset trains, checkpoints, is killed with SIGTERM, resumes, evaluates,
  and writes a valid `submission.json`.

---

## 11. Module layout and interfaces (the code contract)

```
frm/
  config.py            dataclasses ModelCfg, ObjectiveCfg, DataCfg, TrainCfg, EvalCfg; Cfg.load(json) + dot-path CLI overrides
  data/
    arc.py             load_tasks(path_or_name) -> {task_id: {"train": [...], "test": [...]}} ; solutions merged when present
    augment.py         dihedral(grid, tid), inverse_dihedral(grid, tid), Version(tid, perm), apply/invert, make_versions(task, n, rng)
    encode.py          PAD/EOS/colour layout: encode_grid(grid, S, offset) -> uint8[S*S]; crop(seq, S) -> grid
    sizerule.py        predict_output_size(task, test_input) -> (h, w) | None ; choose_bucket(task) -> S
    build.py           CLI: build a dataset dir from sources (arc1, arc2, arc2-kaggle, composite, kaggle)
    loader.py          Dataset(dir): buckets, identifiers, version table; TrainStream(dataset, bucket, weights, rng).take(n) -> Batch;
                       eval_batches(dataset, split, bucket, batch_rows) -> iterator of Batch with keys
    composite.py       primitives, compose, generate(n_tasks, depth_range, seed) -> tasks with "subgoals"; CLI writes JSON
  model/
    layers.py          rms_norm, CastedLinear, SwiGLU, Attention (SDPA, 2D RoPE), Block
    embedding.py       PuzzleEmbedding (buffer table, gather-to-leaf, SignSGD step, extend(n_new))
    refiner.py         Refiner(ModelCfg): init_carry(B, S), round(carry, batch, mode) -> RoundOut
  objective.py         Objective(ObjectiveCfg): __call__(RoundOut, batch, step) -> (loss, metrics)
  train.py             CLI: python -m frm.train --config cfg.json --run_dir runs/X [key=value ...]
  evaluate.py          CLI: per-round voting, pass@k, instruments, counterfactuals; writes eval json + submission.json
  adapt.py             CLI: Kaggle adaptation (extend ids, fine-tune, predict, write submission)
configs/               smoke.json, trm_a1.json, frm_a1.json, frm2_a1.json, frm_a2.json, frm_a2_kaggle.json
kelvin/                env.sh, setup_env.sbatch, train.sbatch (3-day chained links), smoke.sbatch, k2 helper, README.md
scripts/               fetch_data.py, bench.py
tests/
```

`Batch` (all torch tensors on device unless noted): `inputs` int [B, S²], `labels` int [B, S²] (−100 = ignore),
`puzzle_ids` int [B], `subgoals` int [B, M, S²] (−100 rows where absent), `subgoal_mask` bool [B, M], plus numpy-side
`keys` for eval rows (task id, test index, identifier).

`RoundOut`: `logits` [B, S², 12] float32, `q` [B], `child_logits` [B, K, S², 12] or None, `child_q` [B, K] or None,
`join_alpha` [B, K+1] (cell-mean, for stats), `join_entropy` [B], `carry` (y, z detached).

---

## 12. Risks and planned responses (redesign, never removal)

| risk | detection | response |
|---|---|---|
| parent ignores children | `parent_mass → 1`, `cf/sham ≈ 0` | raise λ_wta; join-dropout (randomly mask the parent slot with p=0.25 in training); more pre-fork cycles so children start from a better state |
| children collapse | `distinct → 1`, win rates degenerate | raise ε, slow τ anneal, per-child noise σ on codes |
| fork costs more than it returns | matched-FLOP dev pass@2 below `trm` | K=2; cheaper children (`child_L_cycles=2`); fork every other round (`fork_every=2`) |
| depth still unused | `late_gain ≈ 0` on dev and composites | raise composite fraction/depth; inspect `cf/reset@r`; increase rounds trained (exploration p) |
| memory | OOM in T2 | child activation checkpointing; smaller `tokens_per_step` |
| Kaggle hardware/time | adaptation does not fit 12 h | fewer versions at adaptation, fewer rounds at inference (per-round curves say where accuracy saturates) |

Any response that removes forking from the trained model is out of scope for the agent; it goes to the user as its own
question with the sentence "after this, the model does not train forks".

## 13. Non-goals

In-context demonstration encoders; symbolic solvers or DSL search at inference; multi-GPU DDP (single-GPU jobs schedule
fastest on Kelvin and every arm fits one A100; DDP is a later change if a final run needs wall-clock); ARC-AGI-3.
