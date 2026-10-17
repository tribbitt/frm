# The frm objective

Deliverable 3: research on loss design for recursive refiners with branches, and the objective frm trains with
(`frm/objective.py`). Read `DEPTH.md` first — every term below answers a mechanism listed there.

Tags: **[M]** measured, **[L]** literature, **[I]** inference. All arXiv IDs below were checked against the arXiv API.

---

## 1. Requirements

An objective for frm has to:

| # | requirement | why (evidence) |
|---|---|---|
| R1 | match the metric: whole-grid exact match, two attempts | ARC scores exact pass@2; cell metrics are dominated by copy cells [M: tinyabstractor] |
| R2 | stay dense: grade every round | deep supervision is what makes recursion pay *in training* (+13pp 1→2 loops [L: ARC Prize HRM analysis]); frmv1's final-only grading lost [M] |
| R3 | pay for detours | the greedy optimum of per-round CE is an idempotent solver (DEPTH.md M1) |
| R4 | make forks useful *and* different | forks collapsed (tiny, w₀=0.993), became irrelevant (tinyabstractor), or learned a schedule (frmv1) [M] |
| R5 | contain no term that rewards a difference between two model-controlled quantities | such terms are gamed by making the subtrahend worse (a discounted-early-segment "gain" was segment 1 getting worse [M: frmv2]) |
| R6 | add no forward passes | compute is the binding constraint (`DESIGN.md` §9) |
| R7 | not reward early exit | geometric/decaying priors starve deep steps of credit [L: Ouro 2510.25741; CGAR 2511.08653] |
| R8 | leave no parameter reachable only through a no-grad path | tinyv2.2's init states and warm-up step embeddings, ARM's latent tokens and H-level never trained [M] |

## 2. Research

### 2.1 Deep supervision in recursive models

- **TRM** ([2510.04871](https://arxiv.org/abs/2510.04871)): stablemax CE on every supervision step against y*, one optimizer
  step per supervision step, detached carry, gradient through the last full cycle; halting = BCE(q, current answer
  exactly correct), weight 0.5; the Q-learning "continue" loss was dropped (86.1 → 87.4%). Ablations: 1-step gradient
  56.5% vs 87.4%; no EMA 79.9%. → The backbone of the objective; change only what depth and forks need.
- **Progressive loss** (Bansal et al., [2202.05826](https://arxiv.org/abs/2202.05826)): n ~ U{0..m−1} iterations
  without gradient, then k with gradient, graded at the end; with input re-injection ("recall") this stops
  "overthinking" and extrapolates to far more iterations. → TRM's detached carry is already a progressive loss with
  k = one cycle: it buys path independence, not multi-step strategies [I].
- **Random depth with truncated backprop** (Huginn, [2502.05171](https://arxiv.org/abs/2502.05171)): sample the recurrence
  count per batch, backprop through the last 8; step-index conditioning breaks extrapolation. → No step inputs.
- **Intermediate gradient horizon** (Shakhvalieva et al., [2609.39967](https://arxiv.org/abs/2609.39967)): neither 1-step nor
  full BPTT, plus large physical batches and controlled state updates; 59.5% pass@2 ARC-AGI-1. → Extend the horizon
  modestly; keep physical batches large.
- **Deep Improvement Supervision** ([2511.16886](https://arxiv.org/abs/2511.16886)): each step gets its own target
  contracting toward y* (a corruption schedule of y*): 24% ARC-AGI-1 at 0.8M params, 18× fewer forward passes. →
  Per-step targets make every step work, but corruption targets *force monotone* progress — the opposite of detours.
  Use only targets from real decompositions (§3.2).
- **Continuous Thought Machines** ([2505.05522](https://arxiv.org/abs/2505.05522)): loss = ½(L at the min-loss tick + L at
  the max-certainty tick); intermediate ticks are free. → WTA over time is incompatible with one optimizer step per
  round, but the principle (grade the best, not every) transfers to WTA over children.

### 2.2 Multi-hypothesis objectives

- **Multiple Choice Learning** (Guzman-Rivera et al., NeurIPS 2012): loss = min_k L(ŷ_k, y); only the best hypothesis
  is trained and the set learns to cover the modes. **sMCL** (Lee et al., [1606.07839](https://arxiv.org/abs/1606.07839)):
  the same with SGD and shared parameters — hypotheses may share everything except a head or a code.
- **Relaxed WTA** (Rupprecht et al., [1612.00197](https://arxiv.org/abs/1612.00197)): winner weight 1−ε, others share ε,
  so hypotheses that never win still train. **Evolving WTA** (Makansi et al.,
  [1906.03631](https://arxiv.org/abs/1906.03631)): top-k winners with k shrinking. **Annealed MCL** (Perera et al.,
  [2407.15580](https://arxiv.org/abs/2407.15580)): softmin assignment with a temperature schedule — from "all hypotheses
  equal" to WTA, escaping WTA's bad local minima.
- **Resilient MCL** (Letzelter et al., [2311.01052](https://arxiv.org/abs/2311.01052)): a score head per hypothesis
  predicts whether it wins, giving a probabilistic reading and a ranking. → frm's per-child q head.
- **Pass@k training** (Chen et al., [2508.10751](https://arxiv.org/abs/2508.10751)): optimising pass@k instead of pass@1
  improves exploration. ARC scores pass@2, so a set objective is metric-aligned.
- **Collapse** is the default without these: two children with zero-init codes, zero-init arbitration and the same
  target collapsed onto child 0 [M: tiny]. A diversity bonus would be R5-gameable (produce garbage to differ);
  MCL-family objectives create diversity only where it lowers the best loss.
- **WTA's blind spot.** With one deterministic target per row, min-over-children pulls the winner to y* and leaves the
  losers where they are; nothing moves them *apart*. Two answers apply. **IMLE** (Li & Malik,
  [1809.09087](https://arxiv.org/abs/1809.09087)): fresh noise for every sample at every step, pull only the sample
  nearest the target — maximum likelihood under conditions, and it cannot drop modes, because every data point keeps
  pulling its nearest sample. **Set prediction with bipartite matching** (DETR, Carion et al.,
  [2005.12872](https://arxiv.org/abs/2005.12872)): match several distinct targets one-to-one to the predictions, which
  "forces unique predictions". frm uses both (§3.2).
- **Noise as escape from fixed points.** PTRM (Sghaier, Parviz & Jolicoeur-Martineau — TRM's author —
  [2605.19943](https://arxiv.org/abs/2605.19943)): TRM's "deterministic recursion can lead to convergence at suboptimal
  solutions, without escape mechanism"; Gaussian noise at each recursion step, parallel trajectories and selection by
  the Q head lift Sudoku-Extreme 87.4% → 98.75% *without retraining*. frm trains the noise in instead.

### 2.3 Credit for detours

- **Potential-based shaping** (Ng, Harada, Russell 1999): F = γΦ(s′) − Φ(s) leaves the optimum unchanged; with Φ = −CE
  of the canvas the per-round terms telescope to final − initial loss [I]. Canvas improvement rewards add no signal
  (and, without a stop-gradient, invite R5 gaming).
- **Learned values** (TD-trained V(state) ≈ P(eventual solve)) can credit detours, but backpropagating a critic into
  the state it scores is DDPG-like and unstable at this scale; "outcome-trained heads read detached inputs" is right
  for halting and selection [I].
- **Gradient paths through ungraded intermediates**: the cleanest credit for a detour is a gradient that reaches the
  detour's computation from a later graded readout *without* grading the detour. Fork → join → post cycle provides
  this inside one round (§3.1).
- **Intermediate targets in a fixed order** failed (−2.9σ: 66.6% of tasks admit other orderings) [M: frmv1]; an
  order-free *set* of sub-goals does not have that problem (§3.2).

### 2.4 Halting

ACT ponder cost ([1603.08983](https://arxiv.org/abs/1603.08983)) and PonderNet's KL to a geometric prior
([2107.05407](https://arxiv.org/abs/2107.05407)) both reward stopping early; Ouro found a uniform prior better and an
improvement-gated exit (trained on detached I_t = max(0, L_{t−1} − L_t)) +2–3pp [L: 2510.25741]. TRM's correctness BCE
is simple and, with evaluation always running 16 rounds, only controls *training-time row turnover*; it never
truncates inference. → Keep TRM's halting; feed q into the join detached.

### 2.5 Stability and ARC practice

Stablemax CE (TRM) avoids softmax collapse on long training; EMA 0.999 (−7.5pp on Sudoku without it [L: TRM]); large
physical batches (a TRM reproduction at batch 80 reached 20.75% vs the paper's 44.6% [M: ARM]); augmentation voting
adds ~11pp to TRM [L: 2512.11847]; test-time training was the only neural lever with measured gains in predecessors
[M: frmv1, decomposition-solver].

## 3. The objective

### 3.0 Written out

One optimizer step runs one round on the $B$ rows of one canvas bucket. Every row is at its own round
$r_b \in \lbrace 1, \dots, 16 \rbrace$, and its carry $(y, z)$ comes from the previous round with the gradient cut (TRM's
online deep supervision). The step's loss is

$$
\mathcal{L} \;=\; \frac{1}{B}\sum_{b=1}^{B} \mathcal{L}_b \;+\; \lambda_{\mathrm{gate}}\,\mathcal{L}_{\mathrm{gate}}
$$

with, for row $b$,

$$
\mathcal{L}_b \;=\;
\underbrace{\mathrm{CE}(\hat y_b, y_b)}_{\text{(1) parent}}
\;+\; \lambda_q\,\underbrace{\mathrm{BCE}(q_b, e_b)}_{\text{(3) halting}}
\;+\; f_b \Big[\,
\underbrace{\lambda_{\mathrm{wta}}\Big((1-\varepsilon)\sum_{\pi\in\Pi_b} w_{b\pi}\,c_{b\pi}
\;+\; \frac{\varepsilon}{K}\sum_{k=1}^{K}\mathrm{CE}(\hat y_{bk}, y_b)\Big)}_{\text{(2) children: annealed set matching}}
\;+\; \underbrace{\frac{\lambda_q}{K}\sum_{k=1}^{K}\mathrm{BCE}(q_{bk}, e_{bk})}_{\text{(3) children's score heads}}
\,\Big],
$$

the cost and weight of an assignment $\pi$ (a one-to-one map of row $b$'s targets onto distinct children)

$$
c_{b\pi} \;=\; \mathrm{CE}\big(\hat y_{b,\pi(\star)},\, y_b\big)
\;+\; \frac{\lambda_{\mathrm{sub}}}{\lambda_{\mathrm{wta}}}\,\frac{1}{\lvert T_\pi\rvert}\sum_{t\in T_\pi}\mathrm{CE}\big(\hat y_{b,\pi(t)},\, g_{bt}\big),
\qquad
w_{b\pi} \;=\; \frac{\exp\big(-\operatorname{sg}(c_{b\pi})/\tau\big)}{\sum_{\pi'\in\Pi_b}\exp\big(-\operatorname{sg}(c_{b\pi'})/\tau\big)}
$$

(the intermediate part is 0 when $T_\pi = \varnothing$, i.e. on every plain row), and the gate's term (4)

$$
\mathcal{L}_{\mathrm{gate}} \;=\; \frac{1}{\lvert G\rvert}\sum_{b\in G}
\mathrm{BCE}\Big(\gamma_b,\; \mathbb{1}\big[\,\mathrm{CE}(\tilde y_b, y_b) - \mathrm{CE}(\hat y_b, y_b) > p\,\big]\Big)
\qquad (0 \text{ if } G = \varnothing).
$$

CE is TRM's stablemax cross-entropy, a token mean over the row's supervised positions $V$ (colour and EOS cells of the
target; PAD cells are ignored), with 12 classes:

$$
\mathrm{CE}(\hat y, y) \;=\; -\frac{1}{\lvert V\rvert}\sum_{i\in V}\log\frac{s(\hat y_{i,y_i})}{\sum_{c=1}^{12} s(\hat y_{i,c})},
\qquad
s(x) \;=\; \begin{cases} x + 1 & x \ge 0 \\[2pt] \dfrac{1}{1 - x} & x < 0. \end{cases}
$$

The temperature falls log-linearly with the run's progress $u$ (the largest of the spent fractions of the compute,
step and hour budgets) and then stays at its end value:

$$
\tau(u) \;=\; \tau_0\,\Big(\frac{\tau_1}{\tau_0}\Big)^{\min(1,\; u/u_\tau)} .
$$

Symbols:

| symbol | meaning |
|---|---|
| $y_b$ | the row's target canvas: the output's colour tokens and its EOS border |
| $\hat y_b$ | the parent's logits: pre-fork cycle → fork → join → post-join cycle on a forked row, TRM's round on a declined one |
| $\hat y_{bk}$ | child $k$'s logits after its one cycle |
| $g_{bt}$ | intermediate grid $t$ of a composite row ($t \le M = 3$); none on ARC, BARC and RE-ARC rows |
| $\Pi_b$ | assignments of the final target $\star$ to one child and of $\min(M_b, K-1)$ of the row's $M_b$ intermediates, $T_\pi$, to other, distinct children. Plain rows: the $K$ maps $\star \mapsto k$ |
| $e_b,\ e_{bk}$ | 1 if the parent's / child $k$'s canvas is exactly right on every supervised token, else 0 |
| $q_b,\ q_{bk}$ | halting / score logits, `q_head` on prefix position 0 of $y$ |
| $f_b$ | 1 if row $b$ forked this round (always 1 without `model.fork_gate`) |
| $\gamma_b$ | the gate's logit for row $b$ |
| $\tilde y_b$ | the no-fork continuation: TRM's remaining cycles from the same pre-fork state, no gradient |
| $G$ | label rows: forked, with the model's own join (no counterfactual) and the parent slot not dropped |
| $p$ | fork price, nats per supervised token |
| $\operatorname{sg}$ | stop-gradient |

Settings:

| setting | default | ARC-AGI-2 final (`configs/final_frm_a2tn.json`) |
|---|---|---|
| $K$ | 4 | 4 |
| $\lambda_{\mathrm{wta}},\ \lambda_{\mathrm{sub}},\ \lambda_q$ | 0.5, 0.5, 0.5 ($\lambda_q$ is TRM's value) | the same |
| $\varepsilon$ | 0.05 | 0.05 |
| $\tau_0 \to \tau_1$ over $u_\tau$ | $1 \to 0.02$ over $u_\tau = 0.3$ | $1 \to 0.02$ over $u_\tau = 0.048$ of 100,000 PF, i.e. the first 4,800 PF |
| $\lambda_{\mathrm{gate}}$ | 0.1 | 0.1 |
| $p$ | 0, fixed | a controller holding the fork rate at 0.4 (§3.6): 0 at the start, ≈ 0.002 at 45k PF |
| `child_focus` $\varphi$ (§3.7) | 0 | 0 |

With no intermediates, term (2) reduces exactly to annealed relaxed winner-takes-all (§3.2). A declined row has
$f_b = 0$, so its loss is exactly the $K = 0$ (TRM) loss.

Where each term lives:

| term | code | what it computes |
|---|---|---|
| (1) | `Objective.__call__`: `row_ce(out.logits, labels)` | $\hat y_b$ comes from `Refiner.round` (§3.1) |
| (2) | `set_match(ce_final, ce_sub, mask, tau, λ_sub/λ_wta)`, then `set_term` masked by `out.fork` | assignments from the cached `_assignment_table(K, M)`: 4 on plain rows, 136 at $K = 4$, $M = 3$, evaluated as one gather per step |
| (3) | `F.binary_cross_entropy_with_logits` on `out.q` and `out.child_q` | the children's mean is masked by `out.fork` |
| (4) | `benefit = (row_ce(out.cf_logits) − ce_parent).detach()`, BCE over `out.gate_rows` | $\tilde y_b$ = `cf_logits`, from `Refiner._trm_tail` under `torch.no_grad` |
| $\tau$, $p$ | `tau_at(cfg, progress)`; `Trainer.price` | the price controller runs in `Trainer.train_step` |

### 3.1 Term 1 — parent deep supervision (R1, R2, R3)

**What.** TRM's stablemax cross-entropy of the parent's readout against the final answer, at every round.

**Why.** Unchanged from TRM in form, changed in what it trains. Because the graded readout comes *after* the join and a
full post-join cycle, its gradient runs post cycle → join → child cycles → fork codes and shared weights. A child that
turns the canvas into something the parent can finish — a detour — is paid for by this term even though nothing grades
the detour itself. The credit horizon is two cycles along the fork path (TRM: one), an intermediate horizon (§2.1).
It is also the only term that trains the join's $W_q$, $W_k$ and $\beta$ (§3.4).

Uniform weight across rounds: reweighting changed nothing in three predecessor attempts and decaying weights push toward
early exit (DEPTH.md §4). No term multiplies the CE by a round-dependent factor.

**Implementation.** `row_ce(out.logits, labels)`. `token_ce` computes log-probabilities in float64; `log_stablemax`
clamps only the $x < 0$ branch, so the gradient at $x = 0$ is not zero. The forward behind $\hat y_b$ is
`Refiner.round`: every row first runs one pre-fork cycle without gradient. A forked row then runs `_node` — the $K$
child cycles batched as $[B \cdot K, L, D]$, `_join` and one post-join cycle, all with gradient — then `lm_head`. A
declined row runs `_trm_tail` (one cycle without gradient, one with), so its term (1) is exactly TRM's. A cycle is
$n = 4$ updates $z \leftarrow f(z,\, y + x + c)$ then one $y \leftarrow f(y,\, z)$, with the same two blocks $f$
everywhere ($c = 0$ for the parent, the fork code $c_k$ for child $k$).

### 3.2 Term 2 — children: annealed set matching (R3, R4, R7)

**What.** Each child is scored against the row's targets. The assignment of targets to children that fits best takes
almost all the weight $w_{b\pi}$ (a soft-min computed without gradient), and $\varepsilon$ spreads a small share evenly
over every child.

**Why.** Five mechanisms make children *non-identical*, each with a different job:

1. **Distinct static codes** $e_k \sim \mathcal N(0,\, 0.5^2 I)$, never zero: breaks the exact symmetry that collapsed
   tiny's children.
2. **Parent-conditioned per-cell codes** $A_k B\,\mathrm{rms}(z[p])$ ($B$: $512 \to 64$, shared; $A_k$: $64 \to 512$
   per child, zero-initialised): the parent can give each child a different job per region.
3. **Fresh noise every round** (model, not loss): $c_k \mathrel{+}= W_\xi\,\xi_k$, $\xi_k \sim \mathcal N(0, I_{16})$
   drawn per row, child and round (`ModelCfg.noise_dim`; $W_\xi$ learned, initial output RMS 0.25; seeded at
   evaluation). Children become samples of a learned proposal distribution instead of fixed functions of the parent
   state. So min-over-children training is IMLE, which cannot collapse modes (§2.2), and a parent stuck at a fixed
   point still receives *new* proposals each round — without noise the whole round, fork included, is a deterministic
   map with the same fixed point (DEPTH.md M2; PTRM).
4. **One-to-one matching on composite rows**: the final answer and the intermediates go to *different* children, so
   two near-identical children are pushed apart, one toward each grid
   (`test_set_match_sends_distinct_targets_to_distinct_children`). This is diversity pressure with ground truth: it
   cannot be gamed by producing garbage, because every matched child is graded against a real grid. Matching is over a
   set, so step order does not matter (ordered trace supervision lost −2.9σ because orderings are not unique
   [M: frmv1]); the parent is never graded on intermediates, so "produce the intermediate, let the parent finish" is
   what minimises terms 1 + 2 together.
5. **Annealed relaxed WTA on plain rows**: only the best child (in the soft sense) is pulled to $y_b$; where the model
   is uncertain, different children win on different rows. $\tau = 1$ starts with every child learning the task (no
   early monopoly); $\tau \to 0.02$ specialises (annealed MCL); $\varepsilon/K$ per child keeps every code on a
   gradient (no dead children).

Why it drives depth: when the parent is already right, children agree and the term costs nothing extra. When the
parent is stuck, the child whose noise sample moved it closest to $y_b$ takes the gradient and term 1 trains the join
to adopt it, so gradient concentrates on stuck rows at later rounds without any round-dependent weight.

**Implementation.** `set_match(ce_final, ce_sub, mask, tau, sub_ratio)`:

- `_assignment_table(K, M)` enumerates, once per $(K, M)$ and device, every one-to-one map of the final target plus up
  to $K - 1$ intermediates onto distinct children (child index per target, $-1$ = unmatched).
- The cost of every assignment is `ce_final[:, A[:, 0]]` plus $\lambda_{\mathrm{sub}}/\lambda_{\mathrm{wta}}$ ($= 1$)
  times the mean CE of its matched intermediates, one gather from `ce_sub` $[B, K, M]$.
- An assignment is valid if it matches exactly $\min(M_b, K - 1)$ intermediates, all of which the row has
  (`subgoal_mask`).
- $w$ = `softmax(-cost.detach() / tau)` over the valid assignments. `set_match` returns
  $\sum_\pi w_{b\pi} c_{b\pi}$ and the arg-max assignment, which feeds the win and coverage logs.
- `set_term = lambda_wta * ((1 - eps) * match + eps * ce_final.mean(-1))`, added only where `out.fork` is true; this
  is term (2) of §3.0 line for line.

Because $w$ carries no gradient, on a plain row child $k$ receives its own CE gradient scaled by

$$
\frac{\partial \mathcal{L}_b}{\partial\,\mathrm{CE}(\hat y_{bk}, y_b)} \;=\; \lambda_{\mathrm{wta}}\Big[(1-\varepsilon)\,w_{bk} + \frac{\varepsilon}{K}\Big]
$$

(`tests/test_objective.py::test_children_gradient_is_weighted_ce_gradient`). Once $\tau = 0.02$ the weights are nearly
one-hot: about $0.481$ on the winning child and $0.00625$ on each other child, against $1$ on the parent's term (1).

### 3.3 Term 3 — stop/score heads (R7, R1)

**What.** $\mathrm{BCE}(q_b, e_b)$ for the parent (TRM) and $\frac{1}{K}\sum_k \mathrm{BCE}(q_{bk}, e_{bk})$ for the
children of a forked row, both weighted by $\lambda_q$.

**Why.** Uses: training-time halting (row turnover), detached input to the join (prefer children predicted correct),
vote tie-break at inference. Children's q heads are the "scoring head" of resilient MCL. Nothing rewards stopping
early; evaluation always runs 16 rounds.

**Implementation.** `F.binary_cross_entropy_with_logits(out.q, exact_parent)`, and the children's mean masked by
`out.fork`. `q_head` is a $512 \to 1$ linear on prefix position 0 of $y$ (weight 0, bias $-5$ at initialisation, as in
TRM). After its round $r$ the trainer halts a row when

$$
\big(r \ge 16 \;\lor\; q_b > 0\big) \;\land\; r \ge m_b,
\qquad
m_b \;=\; \begin{cases} \text{uniform on } \lbrace 2, \dots, 16 \rbrace & \text{with probability } 0.1, \\[2pt] 0 & \text{otherwise,} \end{cases}
$$

and refills it with a fresh row (`Trainer._refill`). The children's $q_{bk}$ enter the join only as
$\beta\,\operatorname{sg}(q_{bk})$ (§3.8), so the join cannot bend the score heads.

### 3.4 Gradient-flow table

| parameter | term 1 | term 2 | term 3 | term 4 (gate) |
|---|---|---|---|---|
| shared blocks f | ✓ (parent post cycle + children via join) | ✓ (children) | ✓ | — |
| fork codes $e_k$, $A_k$, $B$, noise map $W_\xi$ | ✓ via join | ✓ | ✓ | — |
| join $W_q$, $W_k$, $\beta$ | ✓ | — | — ($q$ enters detached) | — |
| lm_head | ✓ | ✓ | — | — |
| q_head | — | — | ✓ | — |
| puzzle embedding | ✓ | ✓ | ✓ | — |
| gate head (`fork_gate`) | — | — | — | ✓ (only) |
| pre-fork parent cycle | no grad by design (TRM) — it has no parameters of its own | | | |

On declined rows terms 2 and 3's child half are absent, so their loss trains exactly what the K = 0 loss trains.
`tests/test_model.py::test_fork_gate` asserts the first column from term 1 alone; `test_every_param_gets_grad` that
no parameter is left out; `tests/test_gate.py` the last column and the declined rows' isolation.

### 3.5 Normalisation

Token-mean CE per row makes rows of different sizes comparable; the per-step mean over rows makes step gradients
comparable across buckets whose row counts differ by 10×. TRM sums per-row means and divides by the global batch
size, the same thing at a fixed batch. In $c_{b\pi}$ the intermediates are averaged, so a depth-4 composite row
weighs the same as a depth-2 one.

### 3.6 Term 4 — fork gate (only with `model.fork_gate`)

The gate decides per row and round whether the round forks (`DESIGN.md` §4.8). It is trained as a classifier on
measured outcomes, never through the loss it influences:

- **Label.** On a label row the benefit $\Delta_b = \mathrm{CE}(\tilde y_b, y_b) - \mathrm{CE}(\hat y_b, y_b)$
  compares the no-fork continuation with the parent after the fork, both token-mean stablemax CE from the same pre-fork
  state, detached; the label is $\mathbb{1}[\Delta_b > p]$. The continuation is exactly what the row would have
  computed had it declined (TRM's remaining cycles, forward only), so the label answers "did this fork pay at least the
  price" for this state.
- **Label rows.** Rows that forked with the model's own join: not under a counterfactual, and (training) parent slot
  not dropped — a forced-children join measures a handicapped fork, not the one evaluation runs. Declined rows give no
  label; the exploration floor (`train.fork_explore`) makes the gate's "no" states fork now and then so they do.
- **Loss.** $\lambda_{\mathrm{gate}}\,\mathcal{L}_{\mathrm{gate}}$ with $\lambda_{\mathrm{gate}} = 0.1$: a plain mean
  over label rows (not scaled by the label-row fraction), 0 when there is none. The gate computes
  $\gamma_b = \mathrm{MLP}\big([\,y_b[0] \,\Vert\, \bar y_b\,]\big)$, $1024 \to 128 \to \mathrm{SiLU} \to 1$, where
  $\bar y_b$ is the mean of $y_b$ over grid cells. Both inputs come from the pre-fork state, computed without gradient,
  so the term trains only the gate head and terms (1)–(3) never reach it.
- **Decision.** During the forced phase (progress $u <$ `train.fork_force_frac`: 0.3 by default, the first 4,800 PF
  in the final) every row forks. After it, row $b$ forks if $\gamma_b > 0$, or anyway with probability `fork_explore`
  (0.1). Evaluation forks the top `eval.fork_top` of each batch and round by $\gamma_b$ (0.4 in the final; padding
  rows excluded).
- **Price.** `objective.fork_price` (default 0: fork whenever it helps). With `train.fork_rate_target` $\rho > 0$ the
  trainer moves it after each gated step: $p \leftarrow \max\big(0,\; p + \eta_p(\bar f - \rho)\big)$, where $\bar f$
  is the step's fork rate, exploration included, and $\eta_p$ = `fork_price_lr`. The final uses $\rho = 0.4$ and
  $\eta_p = 10^{-4}$. The price is checkpointed; adaptation starts from the price the checkpoint ended at.
- **Implementation.** In `Refiner.round` the label rows get $\tilde y_b$ from `_trm_tail` under `torch.no_grad`
  (`cf_logits`: two extra forward cycles per label row, charged in `round_flops`). In `Objective.__call__`,
  `benefit = (row_ce(cf_logits) − ce_parent).detach()`, `target = benefit > price`, and the BCE is averaged over
  `gate_rows`.
- **Why not a learned soft or REINFORCE gate.** frmv1's soft gate learned the cycle index (98.9% of its variance) and
  discrete decisions need high-variance estimators (§5). Supervised labels from measured counterfactual benefit, a
  forced-fork phase (every row forks, every state is labelled), the exploration floor and no step-index input are the
  mitigations; `fork.rate_by_round` and `gate_acc` vs `gate_pos` detect a relapse (§7).

### 3.7 Error-focused children (`objective.child_focus`, gated models)

Off ($\varphi = 0$) by default and in the final configuration. With `child_focus` $= \varphi > 0$, on label rows the
children's final-target CE becomes a weighted token mean: cells the no-fork continuation (`cf_logits`, already computed
for the gate) predicts right weigh $1 - \varphi$, the rest 1:

$$
\mathrm{CE}_\varphi(\hat y_{bk}, y_b) \;=\; \frac{\sum_{i\in V}\omega_{bi}\,\mathrm{CE}_i(\hat y_{bk}, y_b)}{\sum_{i\in V}\omega_{bi}},
\qquad
\omega_{bi} \;=\; 1 - \varphi\;\mathbb{1}[b \in G]\;\mathbb{1}\big[\arg\max_c \tilde y_{b,i,c} = y_{b,i}\big],
$$

with $\mathrm{CE}_i$ the stablemax CE of token $i$. It replaces $\mathrm{CE}(\hat y_{bk}, y_b)$ inside $c_{b\pi}$ and
the $\varepsilon$ share of term (2). The motive is boosting inside one round: after a warm start from a finished TRM,
a child that repeats TRM's round earns most of its loss on cells TRM already gets right (`DESIGN.md` §4.10). Sub-goal
targets and `child_ce_min` stay unweighted (`test_child_focus_sends_the_childrens_loss_to_cells_trm_gets_wrong`).

### 3.8 Around the loss: what else shapes training

None of these are loss terms, but every term above depends on them (`DESIGN.md` §4 and §6 have the full design).

- **Online deep supervision (TRM).** One optimizer step is one round of one bucket's carry. The carry is detached
  between rounds, so a round's loss never reaches earlier rounds. A forked row's gradient window is its $K$ child cycles
  plus the post-join cycle: 25 passes of $f$ with gradient, against TRM's 5.
- **Fork codes** (`Refiner._child_codes`). Child $k$ adds
  $c_k[p] = e_k + A_k B\,\mathrm{rms}(z[p]) + W_\xi\,\xi_k$ to the input of every one of its $z$-updates (§3.2).
- **Join** (`Refiner._join`). For every cell $p$ and every candidate $j \in \lbrace 0, 1, \dots, K\rbrace$
  (0 = the pre-fork parent), with $W_q, W_k : 512 \to 64$:

$$
s_j[p] \;=\; \frac{\big\langle W_q\,\mathrm{rms}(z[p]),\; W_k\,\mathrm{rms}(z_j[p])\big\rangle}{\sqrt{64}} + \beta\,\operatorname{sg}(q_j),
\qquad
\alpha[p] \;=\; \operatorname{softmax}_j\, s[p],
\qquad
(y, z)[p] \;\leftarrow\; \sum_{j=0}^{K} \alpha_j[p]\,(y_j, z_j)[p].
$$

- **Parent-slot dropout.** In training, the parent slot $j = 0$ is masked on each forked row with probability 0.25
  (`model.join_drop_parent`), so the post-join cycle must build on the children. Without it, an early run's parent
  slot kept 0.95 of the join mass ("parent ignores children"). These rows give no gate label.
- **Fork schedule.** Forced forking, then the gate with exploration and the price controller (§3.6).
- **Optimizer.** AdamW (betas 0.9 / 0.95, weight decay 0.1; lr $5 \cdot 10^{-5}$ in the final, the fork modules at
  $4\times$), gradient clip 1.0, SignSGD at lr $10^{-2}$ on the puzzle-embedding rows a step touched, EMA 0.999
  (evaluation and the shipped model use the EMA weights). Warm-up 1,000 steps, constant, then a $(1 - \sqrt{u'})$
  cooldown to 0 over the last 15% of the budget, $u'$ being the fraction of the cooldown elapsed.
- **Data.** Rows are 0.50 ARC, 0.15 composites (the only rows with intermediates $g_{bt}$), 0.25 BARC, 0.10 RE-ARC.
- **Left out on purpose.** Improvement rewards across rounds, round-dependent weights, diversity bonuses and geometric
  halting priors (§5).

### 3.9 What each term does in the ARC-AGI-2 final

Measured during training of the final ARC-AGI-2 run (`configs/final_frm_a2tn.json`): training-log readings compare the
forced phase (0–4,800 PF) with 40,000–47,000 PF. "TRM" is the K = 0 control (`configs/final_trm_a2tn.json`, same warm
start, data and budget). Evaluations use 32 voters; composites are held-out pairs of synthetic tasks.

| term | reading |
|---|---|
| (1) parent | training CE 0.433 → 0.220. Held-out composites pass@2 FRM − TRM +0.118 (90% CI +0.094…+0.143), pooled over 32,500–52,500 PF. On ARC-AGI-2 evaluation tasks at matched PF (≈ 53,500 PF, 4 voters) −0.006 (CI −0.031…+0.019): the composite lead did not transfer to real tasks |
| (2) children | set term 0.292 → 0.189; best child's CE 0.428 → 0.297, above the parent's 0.220. Win shares 0.16 / 0.21 / 0.48 / 0.16 (no dead child); 3.45 distinct answers of 4 (composites, 45k PF); intermediates reproduced exactly by their matched child 4% → 29%. The children are proposals, not answers: children-only votes score 0.17 on composites against the parent's 0.54, yet replacing them with another task's children (`cf.sham`) costs 0.067 composite exact. The parent slot's share of the join fell from 0.45 to 0.15 |
| (3) halting | accuracy 0.97 → 0.94, BCE 0.078 → 0.143 |
| (4) gate | label base rate 0.59 → 0.55, gate accuracy 0.61 → 0.63, mean benefit of labelled forks +0.009 → +0.013 nats/token. Of the labelled forks the gate itself would have declined, 26% paid (45% in the forced phase, where every row forks). Price 0 → 0.0020, training fork rate 0.42 against the 0.40 target. Held-out ranking is near chance (AUC 0.56 in a gated, warm-started ARC-AGI-1 run at 36k PF), which is why evaluation forks a fixed top 40% |

Depth (evaluations pooled over 32,500–45,000 PF): on held-out composites and ARC-AGI-2 evaluation tasks both models
gain from round 1 to round 4 and nothing after (composites row exact FRM 0.280 → 0.345 → 0.348 at rounds 1 / 4 / 16,
TRM 0.200 → 0.275 → 0.276). At 45k PF, resetting FRM's carry at round 4 changes composite exact by −0.0005, and
switching forks off at evaluation costs 0.014.

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
| learned "whether to fork" gate trained end to end (soft) or by REINFORCE/Gumbel (discrete) | frmv1's soft gate learned the cycle index (98.9% of its variance) [M]; discrete decisions need REINFORCE-style variance on top of a weak signal. The adopted gate (§3.6) is instead a classifier on measured counterfactual benefit, with a forced-fork phase and an exploration floor |
| TD value potential with gradient into the state | unstable critic exploitation at this scale [I]; revisit only if the fork path's credit proves too short |
| canvas-improvement reward or hinge | telescopes (no signal) or reweights stalled rows (already done by set matching without a round-dependent factor) |
| final-round-only grading | frmv1: detours appear but are never rewarded; loses TRM's training-time benefit of dense grading |
| per-step targets from a corruption schedule (DIS) | enforces monotone improvement, forbids detours |
| CTM min-over-ticks loss | needs all rounds in one graph; incompatible with TRM's online carry |
| PonderNet / geometric halting prior | early-exit bias (R7) |
| step-index embeddings | break extrapolation (Huginn; tiny) |

## 6. Defaults and what deserves a sweep

Only K is both expensive to get wrong and cheap to compare (each child adds ≈14 pass-units per row-round, ≈19 with
child checkpointing, `DESIGN.md` §9); K = 4 has not been compared against K = 2 at matched compute. λ_wta, λ_sub, ε, τ
and the noise scale keep the values above unless an instrument in §7 fires; then the listed response is applied, not a
sweep. No seed sweeps: single-seed runs, paired per-task statistics, seed choice at the end.

## 7. Failure signatures of the objective

| signature | meaning | response |
|---|---|---|
| `win{k}` ≈ 0 for some k after the τ anneal | dead child | raise ε; slow the τ anneal |
| `distinct` → 1 while dev exact is low | collapse under uncertainty | check the learned noise scale (‖W_ξ‖); raise `noise_init` |
| `parent_mass` → 1, `cf.sham` ≈ 0 | parent ignores children | raise λ_wta; join-dropout of the parent slot (p = 0.25; `model.join_drop_parent`, adopted after an early run hit parent mass 0.95–0.98) |
| `sub_coverage` ≈ 0 after τ has annealed | children never produce intermediates | raise λ_sub; check composite detour rate |
| q accuracy high but q never > 0 | halting never fires (all rows run 16 rounds) | expected early; if persistent, compare throughput with the control |
| `child_ce_min` ≪ `ce` | join failing to adopt good children | inspect β and `fork.adopt`; more join capacity |
| gated: `fork_rate` → 0 or 1 after the forced phase, or `fork.rate_by_round` a step function of r | the gate collapsed or became a round schedule | raise `train.fork_explore`; lengthen `train.fork_force_frac`; set `train.fork_rate_target` |
| gated: `gate_acc` ≈ max(`gate_pos`, 1 − `gate_pos`) | the gate predicts only the base rate | more label rows (exploration); check `benefit`'s spread — forks may not differ in value by state |
| gated: `cf.allfork` > 0 at eval | the gate declines forks that pay | lower `objective.fork_price` or the rate target |
| gated: `fork.skill` (eval at `fork_top` 1) AUC ≈ 0.5, or `top0.4.share` ≤ 0.4 | the gate cannot rank held-out forks by what they are worth (in a gated, warm-started ARC-AGI-1 run at 36k PF: AUC 0.56, the top 40% keep 8% of the CE benefit) | evaluate at a fixed rate (`eval.fork_top`) rather than logit > 0; a targeted rate then selects forks at random at best |
