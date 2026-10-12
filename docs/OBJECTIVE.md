# The frm objective

Deliverable 3: research on loss design for recursive refiners with branches, and the objective frm trains with
(`frm/objective.py`). Read `DEPTH.md` first — every term below answers a mechanism listed there.

Tags: **[M]** measured, **[L]** literature, **[I]** inference. All arXiv IDs below were checked against the arXiv API.

---

## 1. Requirements

An objective for frm has to:

| # | requirement | why (evidence) |
|---|---|---|
| R1 | match the metric: whole-grid exact match, two attempts | ARC scores exact pass@2; cell metrics are dominated by copy cells [M: tinyabstractor DATA.md 10] |
| R2 | stay dense: grade every round | deep supervision is what makes recursion pay *in training* (+13pp 1→2 loops [L: ARC Prize HRM analysis]); frmv1's final-only grading lost [M] |
| R3 | pay for detours | the greedy optimum of per-round CE is an idempotent solver (DEPTH.md M1) |
| R4 | make forks useful *and* different | forks collapsed (tiny, w₀=0.993), became irrelevant (tinyabstractor), or learned a schedule (frmv1) [M] |
| R5 | contain no term that rewards a difference between two model-controlled quantities | such terms are gamed by making the subtrahend worse (frmv2's P1 "gain" was segment 1 getting worse [M]) |
| R6 | add no forward passes | compute is the binding constraint (`DESIGN.md` §9) |
| R7 | not reward early exit | geometric/decaying priors starve deep steps of credit [L: Ouro 2510.25741; CGAR 2511.08653] |
| R8 | leave no parameter reachable only through a no-grad path | tinyv2.2's init states and warm-up step embeddings, ARM's latent tokens and H-level never trained [M] |

## 2. Research

### 2.1 Deep supervision in recursive models

- **TRM** ([2510.04871](https://arxiv.org/abs/2510.04871)): stablemax CE on every supervision step against y*, one optimizer
  step per supervision step, detached carry, gradient through the last full cycle; halting = BCE(q, current answer
  exactly correct), weight 0.5; the Q-learning "continue" loss was dropped (86.1 → 87.4%). Ablations: 1-step gradient
  56.5% vs 87.4%; no EMA 79.9%. → Keep all of it as the backbone of the objective; change only what depth and forks need.
- **Progressive loss** (Bansal et al., [2202.05826](https://arxiv.org/abs/2202.05826)): run n ~ U{0..m−1} iterations
  without gradient, then k with gradient, grade at the end; with input re-injection ("recall") this stops "overthinking"
  and lets nets extrapolate to far more iterations. → TRM's carried, detached state is already a progressive loss with
  k = one cycle: it buys path independence, not multi-step strategies [I].
- **Random depth with truncated backprop** (Huginn, [2502.05171](https://arxiv.org/abs/2502.05171)): sample the recurrence
  count per batch, backprop through the last 8; step-index conditioning breaks extrapolation. → No step inputs.
- **Intermediate gradient horizon** (Shakhvalieva et al., [2609.39967](https://arxiv.org/abs/2609.39967)): neither 1-step nor
  full BPTT; plus large physical batches and controlled state updates; 59.5% pass@2 ARC-AGI-1. → Extend the horizon
  modestly; keep physical batches large.
- **Deep Improvement Supervision** ([2511.16886](https://arxiv.org/abs/2511.16886)): give each step its own target that
  contracts toward y* (corruption schedule of y*): 24% ARC-AGI-1 at 0.8M params, 18× fewer forward passes. → Per-step
  targets make every step work, but corruption targets *force monotone* progress — the opposite of detours. Use only
  targets that come from real decompositions (§3.3).
- **Continuous Thought Machines** ([2505.05522](https://arxiv.org/abs/2505.05522)): loss = ½(L at the min-loss tick + L at the
  max-certainty tick); intermediate ticks are free. → WTA over time; incompatible with TRM's one-optimizer-step-per-round
  regime, but the principle (grade the best, not every) transfers to WTA over children.

### 2.2 Multi-hypothesis objectives

- **Multiple Choice Learning** (Guzman-Rivera et al., NeurIPS 2012): loss = min_k L(ŷ_k, y). Only the best hypothesis is
  trained; the set learns to cover the modes. **sMCL** (Lee et al., [1606.07839](https://arxiv.org/abs/1606.07839)): the same
  with SGD and shared parameters — hypotheses may share everything except a head or a code.
- **Relaxed WTA** (Rupprecht et al., [1612.00197](https://arxiv.org/abs/1612.00197)): winner weight 1−ε, others share ε;
  prevents hypotheses that never win from never training. **Evolving WTA** (Makansi et al.,
  [1906.03631](https://arxiv.org/abs/1906.03631)): top-k winners with k shrinking. **Annealed MCL** (Perera et al.,
  [2407.15580](https://arxiv.org/abs/2407.15580)): softmin assignment with a temperature schedule — starts as "all
  hypotheses equal", ends as WTA, escapes WTA's bad local minima.
- **Resilient MCL** (Letzelter et al., [2311.01052](https://arxiv.org/abs/2311.01052)): a score head per hypothesis predicts
  whether it wins; gives a probabilistic reading and a ranking. → frm's per-child q head plays this role.
- **Pass@k training** (Chen et al., [2508.10751](https://arxiv.org/abs/2508.10751)): optimising pass@k instead of pass@1
  improves exploration. ARC scores pass@2, so a set objective is metric-aligned.
- **Collapse** is the default without these: tiny's two children with zero-init codes, zero-init arbitration and the same
  target collapsed onto child 0 [M: `tiny/results/fork_probe.json`]. A diversity bonus would be R5-gameable (produce
  garbage to differ); MCL-family objectives create diversity only where it lowers the best loss.
- **WTA's blind spot.** With one deterministic target per row, min-over-children pulls the winner to y* and leaves the
  losers where they are; nothing moves them *apart*. Two literature answers apply. **IMLE** (Li & Malik,
  [1809.09087](https://arxiv.org/abs/1809.09087)): draw fresh noise for every sample at every step and pull only the
  sample nearest the target — equivalent to maximum likelihood under conditions, and it cannot drop modes, because every
  data point keeps pulling its nearest sample. **Set prediction with bipartite matching** (DETR, Carion et al.,
  [2005.12872](https://arxiv.org/abs/2005.12872)): when there are several distinct targets, match them one-to-one to the
  predictions, which "forces unique predictions". frm uses both (§3.2).
- **Noise as escape from fixed points.** PTRM (Sghaier, Parviz & Jolicoeur-Martineau — TRM's author —
  [2605.19943](https://arxiv.org/abs/2605.19943)): TRM's "deterministic recursion can lead to convergence at suboptimal
  solutions, without escape mechanism"; Gaussian noise at each recursion step, parallel trajectories and selection by the
  Q head lift Sudoku-Extreme 87.4% → 98.75% *without retraining*. frm trains the noise in rather than adding it at
  test time.

### 2.3 Credit for detours

- **Potential-based shaping** (Ng, Harada, Russell 1999): F = γΦ(s′) − Φ(s) leaves the optimum unchanged; with Φ = −CE of
  the canvas the per-round terms telescope to final − initial loss [I]. Improvement rewards on the canvas therefore add no
  signal (and, without a stop-gradient, invite R5 gaming).
- **Learned values** (TD-trained V(state) ≈ P(eventual solve)) can credit detours, but backpropagating a critic into the
  state it scores is DDPG-like and unstable at this scale; the lineage's rule "outcome-trained heads read detached inputs"
  was right for halting and selection [I].
- **Gradient paths through ungraded intermediates**: the cleanest credit for a detour is a gradient that reaches the
  detour's computation from a later graded readout *without* grading the detour itself. This is what fork → join → post
  cycle provides inside one round (§3.1).
- **Intermediate targets in a fixed order** failed in frmv1 (−2.9σ: 66.6% of tasks admit other orderings) [M]; an
  order-free *set* of sub-goals does not have that problem (§3.3).

### 2.4 Halting

ACT ponder cost ([1603.08983](https://arxiv.org/abs/1603.08983)) and PonderNet's KL to a geometric prior
([2107.05407](https://arxiv.org/abs/2107.05407)) both reward stopping early; Ouro found a uniform prior better and an
improvement-gated exit (trained on detached I_t = max(0, L_{t−1} − L_t)) +2–3pp [L: 2510.25741]. TRM's correctness BCE
is simple and, with evaluation always running 16 rounds, only controls *training-time row turnover* — it never truncates
inference. → Keep TRM's halting; feed q into the join detached.

### 2.5 Stability and ARC practice

Stablemax CE (TRM) avoids softmax collapse on long training; EMA 0.999 (+7.5pp on Sudoku without it [L: TRM]); large
physical batches (ARM's TRM baseline at batch 80 reached 20.75% vs the paper's 44.6% [M:
`arm-marcelmensch/experiments/abl-c3-trmopt`]); augmentation voting adds ~11pp to TRM [L: 2512.11847]; test-time
training was the only neural lever with measured gains in the lineage [M: frmv1 7c, decomposition-solver 12 vs 0].

## 3. The objective

### 3.0 Written out

For a step on bucket S (B rows, the carry of each row at its own round r_b; one optimizer step per round, carry detached
between rounds, as in TRM), the loss is the mean over rows:

$$
\mathcal{L} \;=\; \frac{1}{B}\sum_{b=1}^{B}\Big[\;
\underbrace{\mathrm{CE}(\hat y_b, y_b)}_{\text{(1) parent}}
\;+\;\underbrace{\lambda_{\mathrm{wta}}\Big((1-\varepsilon)\sum_{\pi\in\Pi_b} w_{b\pi}\,c_{b\pi}
\;+\;\frac{\varepsilon}{K}\sum_{k=1}^{K}\mathrm{CE}(\hat y_{bk}, y_b)\Big)}_{\text{(2) children: annealed set matching}}
\;+\;\underbrace{\lambda_q\Big(\mathrm{BCE}(q_b, e_b) + \frac{1}{K}\sum_{k=1}^{K}\mathrm{BCE}(q_{bk}, e_{bk})\Big)}_{\text{(3) stop / score heads}}
\;\Big]
$$

with, for an assignment π (a one-to-one map of row b's targets onto distinct children),

$$
c_{b\pi} \;=\; \mathrm{CE}\big(\hat y_{b,\pi(\star)},\, y_b\big) \;+\; \frac{\lambda_{\mathrm{sub}}}{\lambda_{\mathrm{wta}}}
\cdot\frac{1}{|T_\pi|}\sum_{t\in T_\pi}\mathrm{CE}\big(\hat y_{b,\pi(t)},\, g_{bt}\big),
\qquad
w_{b\pi} \;=\; \frac{\exp(-c_{b\pi}/\tau)}{\sum_{\pi'\in\Pi_b}\exp(-c_{b\pi'}/\tau)}\quad(\text{no gradient})
$$

Symbols:

| symbol | meaning |
|---|---|
| y_b | the row's target canvas (colour tokens + EOS border; PAD ignored) |
| ŷ_b | parent readout after fork → join → post-join cycle; ŷ_bk: child k's readout after its cycle |
| CE(ŷ, y) | TRM stablemax cross-entropy, mean over the row's supervised tokens: −(1/N)Σ_i log s(ŷ_i)[y_i] / Σ_c s(ŷ_i)[c], s(x) = x+1 for x ≥ 0, 1/(1−x) for x < 0 |
| g_bt | intermediate grid t of a composite row (t = 1..k−1 for a depth-k program); none on ARC/BARC/RE-ARC rows |
| Π_b | assignments that match the final target ⋆ to one child and min(#intermediates, K−1) intermediates T_π to other, distinct children. Plain rows: Π_b = {⋆ → k : k = 1..K} |
| τ | temperature, log-linear from 1.0 to 0.02 nats/token over the first 30% of the run's budget, then constant |
| e_b, e_bk | 1 if the parent / child k canvas is exactly right (every supervised token), else 0 |
| q_b, q_bk | stop/score logits read from position 0 of y (parent / child k) |

Defaults: λ_wta = 0.5, λ_sub = 0.5, λ_q = 0.5 (TRM's value), ε = 0.05, K = 4. With no intermediates, term (2) reduces
exactly to annealed relaxed winner-takes-all: child k's gradient is λ_wta·[(1−ε)·softmin_τ(CE)_k + ε/K]·∇CE_k
(`tests/test_objective.py::test_children_gradient_is_weighted_ce_gradient`).

Implementation: `frm/objective.py` (`set_match`, `Objective.__call__`); Π is enumerated once per (K, M) — 136
assignments at K = 4 with up to 3 intermediates — and evaluated as one gather per step.

With the fork gate (`model.fork_gate`, `DESIGN.md` §4.8), let f_b = 1 if row b forked this round. Terms (2) and the
children's half of (3) are multiplied by f_b, so a declined row's loss is exactly the K = 0 loss, and the gate adds a
fourth term, a mean over its label rows G (forked, own join, parent slot not dropped; §3.6):

$$
\mathcal{L} \;\mathrel{+}=\; \lambda_{\mathrm{gate}}\cdot\frac{1}{|G|}\sum_{b\in G}
\mathrm{BCE}\big(g_b,\; \mathbb{1}[\,\mathrm{CE}(\tilde y_b, y_b) - \mathrm{CE}(\hat y_b, y_b) > p\,]\big)
\qquad (0 \text{ if } G = \varnothing)
$$

with g_b the gate logit, ỹ_b the readout of the no-fork continuation (TRM's round from the same pre-fork state, no
grad) and p the fork price in nats per supervised token.

### 3.1 Term 1 — parent deep supervision (R1, R2, R3)

Unchanged from TRM in form, changed in what it trains. Because the graded readout comes *after* the join and a full
post-join cycle, its gradient runs `post cycle → join → child cycles → fork codes / shared weights`. A child that turns
the canvas into something the parent can finish — a detour — is paid for by this term even though nothing grades the
detour itself. The credit horizon is two cycles along the fork path (TRM: one), an intermediate horizon (§2.1).

Uniform weight across rounds: reweighting changed nothing in three lineage attempts and decaying weights push toward early
exit (DEPTH.md §4). No term multiplies the CE by a round-dependent factor.

### 3.2 Term 2 — children: annealed set matching (R3, R4, R7)

What makes children *non-identical* — five mechanisms, each with a different job:

1. **Distinct static codes** e_k (N(0, 0.5²), never zero): breaks the exact symmetry that collapsed tiny's children.
2. **Parent-conditioned per-cell codes** A_k·B·rms(z): lets the parent give each child a different job per region.
3. **Fresh noise every round** (model, not loss): c_k += W_ξ ξ_k, ξ_k ~ N(0, I₁₆) drawn per row, per child, per round
   (`ModelCfg.noise_dim`, learned W_ξ with initial output RMS 0.25; seeded at evaluation for reproducibility). Children
   become samples of a learned proposal distribution instead of fixed functions of the parent state. Two consequences:
   min-over-children training is IMLE, which cannot collapse modes (§2.2); and a parent stuck at a fixed point still
   receives *new* proposals each round — without noise the whole round, fork included, is a deterministic map with the
   same fixed point (DEPTH.md M2; PTRM).
4. **One-to-one matching on composite rows**: the final answer and the intermediates go to *different* children, so
   two near-identical children are pushed apart, one toward each grid
   (`test_set_match_sends_distinct_targets_to_distinct_children`). This is diversity pressure with ground truth — it
   cannot be gamed by producing garbage, because every matched child is graded against a real grid. The matching is over
   a set, so the program's step order does not matter (frmv1's ordered trace supervision lost −2.9σ because orderings are
   not unique); the parent is never graded on intermediates, so "produce the intermediate, let the parent finish" is
   what minimises terms 1 + 2 together.
5. **Annealed relaxed WTA on plain rows**: only the best child (in the soft sense) is pulled to y*; where the model is
   uncertain, different children win on different rows. τ = 1 starts with every child learning the task (no early
   monopoly); τ → 0.02 specialises (annealed MCL); ε/K per child keeps every code on a gradient (no dead children).

Why it drives depth: when the parent is already right, children agree and the term costs nothing extra. When the parent is
stuck, the child whose noise sample moved it closest to y* takes the gradient and term 1 trains the join to adopt it — so
the gradient concentrates on the stuck rows at later rounds without any round-dependent weight.

### 3.3 Term 3 — stop/score heads (R7, R1)

BCE(q, exact) for the parent (TRM) and for every child. Uses: training-time halting (row turnover), detached input to the
join (prefer children predicted correct), vote tie-break at inference. Children's q heads are the "scoring head" of
resilient MCL. Nothing rewards stopping early; evaluation always runs 16 rounds.

### 3.4 Gradient-flow table

| parameter | term 1 | term 2 | term 3 | term 4 (gate) |
|---|---|---|---|---|
| shared blocks f | ✓ (parent post cycle + children via join) | ✓ (children) | ✓ | — |
| fork codes e_k, A_k, B, noise map W_ξ | ✓ via join | ✓ | ✓ | — |
| join W_q, W_k, β | ✓ | — | — (q enters detached) | — |
| lm_head | ✓ | ✓ | — | — |
| q_head | — | — | ✓ | — |
| puzzle embedding | ✓ | ✓ | ✓ | — |
| gate head (`fork_gate`) | — | — | — | ✓ (only) |
| pre-fork parent cycle | no grad by design (TRM) — it has no parameters of its own | | | |

On declined rows terms 2 and 3's child half are absent, so their loss trains exactly what the K = 0 loss trains.
`tests/test_model.py::test_fork_gate` asserts the first column from term 1 alone; `test_every_param_gets_grad` asserts that
no parameter is left out; `tests/test_gate.py` asserts the last column and the declined rows' isolation.

### 3.5 Normalisation

Token-mean CE per row makes rows of different sizes comparable; the per-step mean over rows makes step gradients
comparable across buckets whose row counts differ by 10×. TRM sums per-row means and divides by the global batch size,
which is the same thing at a fixed batch. In c_bπ the intermediates are averaged, so a depth-4 composite row weighs the
same as a depth-2 one.

### 3.6 Term 4 — fork gate (only with `model.fork_gate`)

The gate decides per row and round whether the round forks (`DESIGN.md` §4.8). It is trained as a classifier on
measured outcomes, never through the loss it influences:

- **Label.** On a label row, benefit b = CE(no-fork continuation, y*) − CE(parent after the fork, y*), both token-mean
  stablemax CE from the same pre-fork state, detached; label = b > price. The continuation is exactly what the row
  would have computed had it declined (TRM's remaining cycles, forward only), so the label answers "did this fork pay
  at least the price" for this state.
- **Label rows.** Rows that forked with the model's own join: not under a counterfactual, and (training) parent slot
  not dropped — a forced-children join measures a handicapped fork, not the one evaluation would run. Declined rows
  give no label; the exploration floor (`train.fork_explore`) makes the gate's "no" states fork now and then so they do.
- **Loss.** λ_gate · mean over label rows of BCE(g, label), λ_gate = 0.1; a plain mean (not scaled by the label-row
  fraction), 0 when there is none. The gate's inputs ([y₀ ‖ mean over cells of y] of the no-grad pre-fork state) are
  detached, so the term trains only the gate head and terms 1–3 never reach it.
- **Price.** `objective.fork_price` (default 0: fork whenever it helps). With `train.fork_rate_target > 0` the trainer
  adjusts it after each gated step: price ← max(0, price + `fork_price_lr` · (fork rate − target)); the objective takes
  the current value as `price`.
- **Why not a learned soft or REINFORCE gate.** frmv1's soft gate learned the cycle index (98.9% of its variance) and
  discrete decisions need high-variance estimators (§5). Supervised labels from measured counterfactual benefit, a
  forced-fork familiarisation phase (`train.fork_force_frac` = 0.3: every row forks, every state is labelled), the
  exploration floor and no step-index input are the mitigations; `fork.rate_by_round` and `gate_acc` vs `gate_pos`
  detect a relapse (§7).

### 3.7 Error-focused children (`objective.child_focus`, gated models)

Warm-started from a finished TRM, a child that reproduces TRM's own round earns most of its set-matching loss on cells
TRM already gets right, and a fork that only repeats TRM's round buys nothing for its price (`DESIGN.md` §4.10). With
`child_focus` = φ the children's final-target CE becomes a weighted token mean, cell weight 1 − φ where the no-fork
continuation (the label rows' `cf_logits`, already computed for the gate) predicts the cell right and 1 where it is
wrong:

$$
\mathrm{CE}_\phi(\hat y_k, y) = \frac{\sum_p w_p\,\mathrm{CE}_p}{\sum_p w_p},\qquad
w_p = 1 - \varphi\,[\arg\max \mathrm{cf}_p = y_p] .
$$

This is boosting's reweighting (later learners concentrate on the earlier learner's mistakes) applied inside one
round: the children's capacity goes to the cells where TRM is wrong, which is exactly where the per-cell join can
profit from them. Sub-goal targets are not reweighted (TRM's correctness on the final answer says nothing about an
intermediate grid); rows without a label (no fork, dropped, counterfactual) keep uniform weights; the metric
`child_ce_min` stays unweighted. φ = 0 (default) is the plain objective (`test_child_focus_sends_the_childrens_loss_to_cells_trm_gets_wrong`).

## 4. How the objective incentivises depth — the argument in one place

1. **New information every round, even at a fixed point.** K children with fresh noise produce K new proposals each
   round; set matching keeps them distinct where it matters. Later rounds can improve the answer without the input
   changing (DEPTH.md M2).
2. **Credit for setting up the answer.** Term 1's gradient reaches the children's computation, so a round that spends a
   child on an intermediate state the parent can finish is rewarded (M1). Term 2 teaches that move explicitly where the
   ground truth exists.
3. **Gradient where single trajectories stall.** Term 2 adds gradient only on rows whose children differ in quality —
   the stuck, later-round rows (M2, M4).
4. **Demand.** Composites with k ≤ 4 and ≥50% genuine detours make later rounds necessary on part of the data (M4).
5. **No early-exit bias.** No discounting, no geometric priors, fixed 16-round evaluation (M5).

None of these terms can be satisfied by making an earlier round worse (R5): no term compares two model-controlled
losses with gradient; assignment weights are computed without gradient, and the gate's label (a difference of two
detached CEs) trains only the gate, which cannot move either CE.

## 5. Considered and rejected

| idea | why not |
|---|---|
| diversity/disagreement bonus between children | R5: gameable by producing garbage; noise + set matching create diversity only where a real grid or a lower best loss rewards it |
| load balancing across children (MoE-style) | forces equal use when one child is legitimately better; ε already prevents dead children |
| per-target independent softmin for intermediates (this repo's first version) | one child could claim the final answer *and* every intermediate, leaving the others identical — replaced by one-to-one matching |
| noise only at test time (PTRM) | trained-in noise lets the codes learn *how* to use it; test-time-only noise perturbs a model that never saw it |
| learned "whether to fork" gate trained end to end (soft) or by REINFORCE/Gumbel (discrete) | frmv1's soft gate learned the cycle index (98.9% of its variance) [M]; discrete decisions need REINFORCE-style variance on top of a weak signal. The adopted gate (the user's decision, §3.6) is instead a classifier on measured counterfactual benefit, with a forced-fork phase and an exploration floor |
| TD value potential with gradient into the state | unstable critic exploitation at this scale [I]; revisit only if the fork path's credit proves too short |
| canvas-improvement reward or hinge | telescopes (no signal) or reweights stalled rows (already done by set matching without a round-dependent factor) |
| final-round-only grading | frmv1: detours appear but are never rewarded; loses TRM's training-time benefit of dense grading |
| per-step targets from a corruption schedule (DIS) | enforces monotone improvement, forbids detours |
| CTM min-over-ticks loss | needs all rounds in one graph; incompatible with TRM's online carry |
| PonderNet / geometric halting prior | early-exit bias (R7) |
| step-index embeddings | break extrapolation (Huginn; tiny B3) |

## 6. Defaults and what deserves a sweep

Only K is both expensive to get wrong and cheap to compare (each child adds ≈20 pass-units per row-round,
`DESIGN.md` §9); `EXPERIMENTS.md` compares K ∈ {2, 4} at matched compute. λ_wta, λ_sub, ε, τ and the noise scale keep
the values above unless an instrument in §7 fires; then the listed response is applied, not a sweep. No seed sweeps:
single-seed arms, paired per-task statistics, seed choice at the end.

## 7. Failure signatures of the objective

| signature | meaning | response |
|---|---|---|
| `win{k}` ≈ 0 for some k after the τ anneal | dead child | raise ε; slow the τ anneal |
| `distinct` → 1 while dev exact is low | collapse under uncertainty | check the learned noise scale (‖W_ξ‖); raise `noise_init` |
| `parent_mass` → 1, `cf.sham` ≈ 0 | parent ignores children | raise λ_wta; join-dropout of the parent slot (p = 0.25; `model.join_drop_parent`, applied after P1's first run hit parent mass 0.95–0.98) |
| `sub_coverage` ≈ 0 after τ has annealed | children never produce intermediates | raise λ_sub; check composite detour rate |
| q accuracy high but q never > 0 | halting never fires (all rows run 16 rounds) | expected early; if persistent, compare throughput with the control |
| `child_ce_min` ≪ `ce` | join failing to adopt good children | inspect β and `fork.adopt`; more join capacity |
| gated: `fork_rate` → 0 or 1 after the forced phase, or `fork.rate_by_round` a step function of r | the gate collapsed or became a round schedule | raise `train.fork_explore`; lengthen `train.fork_force_frac`; set `train.fork_rate_target` |
| gated: `gate_acc` ≈ max(`gate_pos`, 1 − `gate_pos`) | the gate predicts only the base rate | more label rows (exploration); check `benefit`'s spread — forks may not differ in value by state |
| gated: `cf.allfork` > 0 at eval | the gate declines forks that pay | lower `objective.fork_price` or the rate target |
| gated: `fork.skill` (eval at `fork_top` 1) AUC ≈ 0.5, or `top0.4.share` ≤ 0.4 | the gate cannot rank held-out forks by what they are worth (measured on `F` at 36k: AUC 0.56, the top 40% keep 8% of the CE benefit; `EXPERIMENTS.md` §4c G) | evaluate at a fixed rate (`eval.fork_top`) rather than logit > 0; a targeted rate then selects forks at random at best |
