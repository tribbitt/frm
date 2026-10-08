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

Per round, per row, then averaged over the rows of the step (a step = one bucket's carry for one round):

$$
\mathcal{L} \;=\; \underbrace{\mathrm{CE}(\hat y_{\text{parent}}, y^\*)}_{\text{deep supervision}}
\;+\; \lambda_{\text{wta}} \sum_{k=1}^{K} w_k\, \mathrm{CE}(\hat y_k, y^\*)
\;+\; \lambda_{\text{sub}}\, \frac{1}{|G|}\sum_{g\in G} \sum_{k=1}^{K} v_{gk}\, \mathrm{CE}(\hat y_k, g)
\;+\; \lambda_q \Big[\mathrm{BCE}(q, e) + \tfrac{1}{K}\textstyle\sum_k \mathrm{BCE}(q_k, e_k)\Big]
$$

$$
w = (1-\varepsilon)\,\mathrm{softmax}\!\big(-\mathrm{CE}(\hat y_{1..K}, y^\*)/\tau\big) + \varepsilon/K,\qquad
v_{g\cdot} = \mathrm{softmax}\!\big(-\mathrm{CE}(\hat y_{1..K}, g)/\tau\big),\qquad w, v\ \text{without gradient}
$$

- CE = TRM stablemax cross-entropy, token-mean per row over non-PAD labels (EOS supervised, so size is predicted).
- ŷ_parent = readout after the join and the post-join cycle; ŷ_k = child k's readout after its cycle; e, e_k = exact match.
- G = intermediate grids of the row (composites only; empty otherwise).
- τ anneals log-linearly from 1.0 to 0.02 nats/token over the first 30% of the compute budget, then stays.

Defaults: λ_wta = 0.5, λ_sub = 0.5, λ_q = 0.5 (TRM's q weight), ε = 0.05.

### 3.1 Term 1 — parent deep supervision (R1, R2, R3)

Unchanged from TRM in form, changed in what it trains. Because the graded readout comes *after* the join and a full
post-join cycle, its gradient runs `post cycle → join → child cycles → fork codes / shared weights`. A child that turns
the canvas into something the parent can finish — a detour — is paid for by this term even though nothing grades the
detour itself. The credit horizon is two cycles along the fork path (TRM: one), an intermediate horizon (§2.1).

Uniform weight across rounds: reweighting changed nothing in three lineage attempts and decaying weights push toward early
exit (DEPTH.md §4). No term multiplies the CE by a round-dependent factor.

### 3.2 Term 2 — annealed relaxed WTA over children (R4, R3, R7)

- *Why WTA.* Children are the same tied network with different codes. Grading all of them against y* makes them
  identical (collapse); grading only the best makes them cover different answers — exactly the "multiple guesses" that
  escape spurious fixed points (DEPTH.md M2) and the two attempts ARC allows.
- *Why annealed.* Early in training no child is meaningfully better; a hard WTA would hand one child every win and starve
  the rest. τ = 1 nat/token makes w ≈ uniform (every child learns the task); τ → 0.02 makes it a hard WTA (specialise).
  Annealed MCL shows this schedule avoids WTA's bad minima [L].
- *Why ε.* A floor of ε/K per child keeps every code on a gradient (Rupprecht) — the instrument `fork/win_rate[k]`
  checks it.
- *Why it drives depth.* When the parent is already right, the children agree with it and the term equals the parent's CE
  — no extra pressure, no wasted gradient. When the parent is stuck at a wrong fixed point, the child that moved closest
  to y* *from the current state* takes the gradient, and the parent's own term trains the join to adopt it. The training
  signal therefore concentrates exactly on the rows and rounds where a single trajectory stalls (the hard tail at later
  rounds), without any round-dependent weight.
- *Detours and WTA do not conflict much.* A child that serves as a detour usually loses the WTA; it then feels only the
  ε share pulling it toward y*, while the parent's term pays it for being useful.

### 3.3 Term 3 — sub-goal set matching (R3, R4; composites only)

Every intermediate grid g of a composite row must be produced by *some* child: each g is matched to children by a
softmin over their CE against g, so any child may claim it, in any order, and a child may claim several. This is DIS's
"per-step targets" idea with two differences that answer the lineage's failures: targets come from the actual program
(detours included, not a monotone corruption schedule) and they are a *set* (frmv1's ordered trace supervision lost
because orderings are not unique). The parent is never graded on intermediates — it keeps y* — so "produce the
intermediate, let the parent finish" is the behaviour that minimises terms 1 + 3 together.

### 3.4 Term 4 — stop/score heads (R7, R1)

BCE(q, exact) for the parent (TRM) and for every child. Uses: training-time halting (row turnover), detached input to the
join (prefer children predicted correct), vote tie-break at inference. Children's q heads are the "scoring head" of
resilient MCL. Nothing rewards stopping early; evaluation always runs 16 rounds.

### 3.5 Gradient-flow table

| parameter | term 1 | term 2 | term 3 | term 4 |
|---|---|---|---|---|
| shared blocks f | ✓ (parent post cycle + children via join) | ✓ (children) | ✓ | ✓ |
| fork codes e_k, A_k, B | ✓ via join | ✓ | ✓ | ✓ |
| join W_q, W_k, β | ✓ | — | — | — (q enters detached) |
| lm_head | ✓ | ✓ | ✓ | — |
| q_head | — | — | — | ✓ |
| puzzle embedding | ✓ | ✓ | ✓ | ✓ |
| pre-fork parent cycle | no grad by design (TRM) — it has no parameters of its own | | | |

`tests/test_model.py::test_fork_gate` asserts the first column from term 1 alone; `test_every_param_gets_grad` asserts that
no parameter is left out.

### 3.6 Normalisation

Token-mean CE per row makes rows of different sizes comparable; the per-step mean over rows makes step gradients
comparable across buckets whose row counts differ by 10×. TRM sums per-row means and divides by the global batch size,
which is the same thing at a fixed batch.

## 4. How the objective incentivises depth — the argument in one place

1. **New information every round.** K children with distinct codes produce K proposals from the current state; WTA keeps
   them distinct while the answer is uncertain. Later rounds can therefore improve the answer without the input changing
   (DEPTH.md M2).
2. **Credit for setting up the answer.** Term 1's gradient reaches the children's computation, so a round that spends a
   child on an intermediate state the parent can finish is rewarded (M1). Term 3 teaches that move explicitly where the
   ground truth exists.
3. **Gradient where single trajectories stall.** Term 2 adds gradient only on rows whose children disagree in quality —
   the stuck, later-round rows (M2, M4).
4. **Demand.** Composites with k ≤ 4 and ≥50% genuine detours make later rounds necessary on a quarter of the data (M4).
5. **No early-exit bias.** No discounting, no geometric priors, fixed 16-round evaluation (M5).

None of these terms can be satisfied by making an earlier round worse (R5): no term compares two model-controlled
losses; WTA weights and sub-goal assignments are computed without gradient.

## 5. Considered and rejected

| idea | why not |
|---|---|
| diversity/disagreement bonus between children | R5: gameable by producing garbage; MCL creates diversity only where it lowers the best loss |
| load balancing across children (MoE-style) | forces equal use when one child is legitimately better; WTA + ε already prevents dead children |
| learned "whether to fork" gate (soft or discrete) | frmv1's soft gate learned the cycle index (98.9% of its variance) [M]; discrete decisions need REINFORCE-style variance on top of a weak signal |
| TD value potential with gradient into the state | unstable critic exploitation at this scale [I]; revisit only if the fork path's credit proves too short |
| canvas-improvement reward or hinge | telescopes (no signal) or reweights stalled rows (already done by WTA without a round-dependent factor) |
| final-round-only grading | frmv1: detours appear but are never rewarded; loses TRM's training-time benefit of dense grading |
| per-step targets from a corruption schedule (DIS) | enforces monotone improvement, forbids detours |
| CTM min-over-ticks loss | needs all rounds in one graph; incompatible with TRM's online carry |
| PonderNet / geometric halting prior | early-exit bias (R7) |
| step-index embeddings | break extrapolation (Huginn; tiny B3) |

## 6. Defaults and what deserves a sweep

Only two choices are expensive to get wrong and cheap to compare: **K** (cost grows 15 units per child, `DESIGN.md` §9)
and **λ_wta** (how hard children are pushed toward y* versus left free to serve the parent). Everything else keeps TRM's
value or a literature value with no reason to deviate. `EXPERIMENTS.md` compares K ∈ {2, 4} at matched compute and
leaves λ_wta at 0.5 unless the fork instruments show collapse (raise ε, slow τ) or irrelevance (raise λ_wta). No seed
sweeps: single-seed arms, paired per-task statistics, seed choice at the end.

## 7. Failure signatures of the objective

| signature | meaning | response |
|---|---|---|
| `fork/win_rate[k]` ≈ 0 for some k | dead child | raise ε; slow the τ anneal |
| `fork/distinct` → 1 while dev exact is low | collapse under uncertainty | per-child code noise; check `codes` norms |
| `fork/parent_mass` → 1, `cf/sham` ≈ 0 | parent ignores children | raise λ_wta; join-dropout of the parent slot (p = 0.25) |
| `sub/coverage` ≈ 0 after τ has annealed | children never produce intermediates | raise λ_sub; check composite detour rate |
| q accuracy high but q never > 0 | halting never fires (all rows run 16 rounds) | expected early; if persistent, compare throughput with the control |
| child CE ≪ parent CE | join failing to adopt good children | inspect β and `fork/adopt`; more join capacity |
