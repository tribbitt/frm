# Why recursive refiners stop using depth

Deliverable 1. The question: why do TRM and every predecessor make most of their progress in the first revision, little
after, and roughly none by the fourth? What follows is the evidence, the mechanisms ranked by how well the evidence
supports them, what has already been tried and failed, and how frm answers each mechanism with something measurable.

Tags: **[M]** measured (file or paper table), **[L]** literature claim, **[I]** inference.

---

## 1. The observation, everywhere

| system | evidence of unused depth | source |
|---|---|---|
| TRM (released ARC checkpoint) | ARC-AGI-1 pass@1 by supervision step: 38.25 → 40.38 → 40.13 → 40.50%, flat to step 6. Step 1 = 94% of the final score | [M] Roye-Azar et al., [arXiv 2512.11847](https://arxiv.org/abs/2512.11847) |
| TRM (same checkpoint, tiny's harness) | halting histogram {1 step: 35, 16 steps: 29} of 80 pairs: pairs are solved at once or never | [M] `tiny/results/cmp/trm_flops.json` |
| HRM (ARC Prize analysis) | training with 16 refinement loops lifts **1-loop inference** by >15pp; extra inference loops add far less | [M] [arcprize.org/blog/hrm-analysis](https://arcprize.org/blog/hrm-analysis), Fig. 5 |
| HRM (Sudoku, per sample) | loss plateaus for many segments, then drops to 0 in one; trajectories trapped at spurious fixed points | [M] Ren & Liu, [arXiv 2601.10679](https://arxiv.org/abs/2601.10679) §4 |
| frmv1 | inference depth worth +0.0003 on a trained checkpoint; canvas accuracy flat to 0.001 over 12 steps | [M] `frmv1/MILESTONES.md` 225-253, `frmv1/FRM_HANDOFF.md` 2.4 |
| tiny (TinyRM) | `ce_step0` 0.6193 vs `ce_final` 0.6183 at step 10k; eval exact@t [2,3,3,3]/126 | [M] `tiny/runs/fork_main_log.jsonl:206`, `trans_ramp_log.jsonl:313` |
| tinyabstractor | dev exact/NLL within ±0.02 from segment 1 to 3 in all 22 evals; cells changed per segment 3.3%, 0.8%, 0.13%, 0.09%, 0.03% | [M] `tinyabstractor/insights/data/runs/main/dev.jsonl`, `probes/spark/depth_dev_41079.log` |
| frmv2 | dev s1→s3 NLL gain +0.07…0.12 at step 2k, ≈0 (−0.006…+0.009) by step 8k in every arm without new information between segments | [M] `frmv2/runs/*/dev.jsonl` |
| decomposition-solver transducer | per-step exact 0.1276 → 0.128 | [M] `decomposition-solver-arc/docs/TRANSDUCE.md` §9.4 |
| Ouro/LoopLM 1.4B | MMLU by loop step 41.2 → 60.4 → 66.7 → 67.5, then degrades beyond trained depth | [L] [arXiv 2510.25741](https://arxiv.org/abs/2510.25741) |

Two facts stand out. First, the pattern does not depend on architecture (HRM, TRM, in-context encoders, branching
heaps, stack machines), scale (0.8M–1.4B) or data (synthetic, ARC, Sudoku). Second, recursion clearly *helps training*
(ARC Prize: 1→2 training loops +13pp; frmv1: training depth 1→12 +0.010) while *inference* recursion beyond the first
few rounds does almost nothing. Depth is acting as a training-time regulariser and data multiplier, not as computation
the trained model performs.

## 2. What "using depth" should mean

A model uses depth when later rounds turn wrong answers into right ones on inputs it was not trained on. Measured as
(definitions in `DESIGN.md` §7.3):

- **late gain** = acc@16 − acc@4 (exact / voted pass@2), with a paired bootstrap CI over tasks;
- **first-solve histogram**: the round at which a row first becomes exact; mass beyond round 4 is depth being used;
- **regressions**: rows correct at round r−1 and wrong at r (a model that "uses" depth by oscillating is not using it);
- **carry reliance** `cf/reset@r`: reset the carry at round r; if accuracy does not drop, round r+1 recomputes from
  scratch and the recursion is a loop of one-shot solvers;
- **depth by composition length** on composites with known k.

Cell accuracy and NLL are not depth metrics: copy cells dominate them (~77% of cells in tinyabstractor's data [M]), and
"loop gain" read without the step-1 value was misread once already (frmv1 retracted +0.0299 because step 1 had fallen
by 0.0319 [M]).

## 3. Mechanisms, ranked by evidence

### M1 — Greedy credit: same target every round, detached carry, one-cycle gradient window **(strongest)**

TRM's round r is trained, by its own optimizer step, to make y_r as close to y* as possible from whatever carry it
receives; the carry is detached (`trm.py:219`), so round r receives no gradient from round r+1's loss, and only the last
cycle of the round is on the gradient path (`trm.py:208-216`). The optimum of this objective is an **idempotent solver
map**: f(x, y) ≈ y* for every reachable y. Iterating an idempotent map adds nothing. A *detour* — a round that makes
the canvas worse so that the answer becomes reachable (shuffle quadrants, then gravity) — costs loss at round r and is
credited by nothing.

Evidence. Structural [M: `trm.py`]. frmv1 freed the intermediate steps (final-only supervision): the model genuinely
detoured (step-1 cell 0.5157, below the copy floor) and repaired back to parity — detours appear when not penalised,
but they were never *rewarded*, so they bought nothing (peak −0.0023, final −0.0142) [M: `frmv1/FRM_HANDOFF.md` 4.0.3].
Ren & Liu name the same mechanism: with one-step gradients "segments are disentangled and all contribute to the same
goal — solving the puzzle in one step" [L: 2601.10679 §3.2].

### M2 — Fixed points and no new information between rounds **(strong)**

The round function is deterministic and re-injects the same x. Once the state converges, later rounds are no-ops; if it
converges to the wrong attractor, they are no-ops forever. HRM trajectories plateau at spurious fixed points and escape,
if at all, by a sudden jump [L: 2601.10679 §4–5]; "scaling guesses" (input relabelling, checkpoint bootstrapping) lifted
Sudoku-Extreme 55% → 96.9% — the gain came from *diversity plus selection*, not from more steps along one trajectory.
In the lineage, depth paid off only when something new arrived between segments: frmv2's checker arm, which fed the
model's own check on a held-out demonstration back into later segments, kept a late gain (+0.023) while every other arm
went to ≈0 [M: `frmv2/runs/p2_c/dev.jsonl`].

### M3 — The first round can already do everything the model can do **(strong)**

(a) **Lookup.** TRM's task conditioning is a per-augmented-puzzle embedding learned from the task's demonstrations;
replacing it with a blank or random id gives 0% [M: 2512.11847 Table 2]. The transformation is retrieved, not
computed, so round 1 does it. (b) **Capacity per round.** One TRM round is 3 cycles × 5 module calls × 2 layers = 30
layer applications (42 with n=6). Whatever one round of that depth can solve, it solves; what it cannot, a second round
of the same map rarely solves either (M2).

### M4 — The training distribution never demands depth **(strong, by construction)**

Nearly every training pair is solvable at round 1 or not at all; under ACT, solved rows halt and the rest are mostly
unsolvable, so later rounds see a gradient-free head and a hopeless tail. There are no "solvable only in k steps"
examples to learn from. The lineage removed them on purpose: tinyabstractor's grammar refused "chains whose second step
hides the first (tile then gravity …)" [M: `tinyabstractor/synth/compose.py:29-32`], and frmv1's corpus had branching
factor 1 in 100% of tasks [M]. A null on such data cannot distinguish "the model cannot use depth" from "there was
nothing to use depth on".

### M5 — Halting and step weighting reward early exits **(moderate)**

TRM's halting head is BCE on "the current answer is correct", so the trained policy is "stop at the first correct
canvas" — fine for efficiency, but it means training compute at later rounds is spent on the hopeless tail (M4). Priors
that favour early exits make it worse: PonderNet's geometric prior and Ouro's analysis show exit priors "concentrate mass
on shallow steps, reducing credit assignment to deeper computation"; a uniform prior did better [L: 2510.25741].
Decaying per-step weights push the same way (CGAR [L: 2511.08653]).

### M6 — Gradient horizon **(moderate)**

A one-step gradient is catastrophic (TRM Sudoku 56.5% vs 87.4% with the full last cycle [L: 2510.04871 Table 1]);
full BPTT through everything is also not best — an *intermediate* horizon is optimal [L: Shakhvalieva et al.,
[arXiv 2609.39967](https://arxiv.org/abs/2609.39967)]. TRM's horizon is one cycle; anything that needs two cycles of
coordinated computation is not learnable by credit, only by accident.

### M7 — One canvas holds one hypothesis **(moderate)**

A single deterministic y cannot represent alternatives; TRM's authors note it "can only provide a single deterministic
answer" [L: 2510.04871]. When the rule is ambiguous, the model commits at round 1 and spends later rounds defending the
commitment (M2).

### M8 — Measurement artefacts **(real, recurrent)**

Metrics that cannot fail hid problems for weeks in the lineage: a fork arbitration metric that is 1/K by construction
(tiny), buffer occupancy logged after a reseed (frmv1), hidden-norm measured right after an RMSNorm (tiny), cell
accuracy with PAD (frmv1), background-inflated recall (tinyabstractor). Depth claims also failed when decided on
early-training NLL (frmv2 pilots at 8k steps, ~1/4000 of TRM's samples, dev exact ≈0 everywhere).

### M9 — Depth beyond the trained range **(minor here)**

Performance degrades past the trained depth (Ouro), and step-index inputs break extrapolation (Huginn
[arXiv 2502.05171](https://arxiv.org/abs/2502.05171); tiny's untrained step embeddings at T≥5, bug B3 [M]). Not a cause
of the plateau, but a trap for any fix that adds round-specific parameters.

## 4. What has already been tried and did not work

| attempt | result | source |
|---|---|---|
| discount early segments (0.1/0.3/1) | s1 got worse, s3 unchanged, exact unchanged; the "gain" was s1 falling | [M] frmv2 P1, `runs/p1_rule.json` |
| ramp step weights | +0.021 cell, still depth-invariant | [M] tiny RESULTS §8 |
| final-step-only supervision | detours appear, are repaired, nothing gained; real ARC worse | [M] frmv1 7a0 |
| intermediate targets in a fixed order ("trace supervision") | −2.9σ; 66.6% of tasks admit other orderings, 35% contain no-op steps | [M] frmv1 MILESTONES |
| extra "DEEP" segments + learned depth head | training z −0.029; head ≡ its nuisance head | [M] tinyabstractor I-10 |
| cross-segment credit through a global latent h | h carried no episode information (whole-state swap cost −0.0014±0.0027 NLL) | [M] `tinyabstractor/insights/data/probes/laptop/s_synth_val_85340.log` |
| canvas-improvement rewards Φ(s′)−Φ(s) with Φ = −CE | telescope to final − initial loss; no new signal [I, Ng et al. 1999] | — |
| exponentially decaying step weights | faster, fewer steps, slightly worse: pushes toward early exit | [L] CGAR 2511.08653 |

The common thread: reweighting *when* the same target is graded never changes *what* the model can gain from a later
round. Either later rounds receive something new (information, alternatives) or a credit path pays an earlier round for
enabling a later one. Everything that did neither failed.

## 5. FRM's answer, mechanism by mechanism

| mechanism | frm response | where | instrument that would show it failing |
|---|---|---|---|
| M1 greedy credit | children are never graded on the final answer except through the set term (only the best-matched child is pushed toward y*); their computation is paid for by the parent's post-join readout through a **two-cycle credit path** (child cycle → join → post cycle). On composites, children are matched one-to-one to the final answer and the intermediate grids, so "produce the intermediate" is a rewarded move | `refiner.py` `_node`/`_join`; `objective.py` `set_match` | `cf.sham ≈ 0` (parent ignores child content); `sub_coverage ≈ 0` |
| M2 fixed points | every round forks K children whose codes include **fresh noise drawn each round**; the join can adopt a child that escaped the parent's attractor. Without the noise this would not hold: children are then a deterministic function of the parent state, so at a parent fixed point the whole round (fork included) repeats itself — the first version of this design had exactly that hole. With it, each round contains K new proposals even when the parent has stopped moving (PTRM, [2605.19943](https://arxiv.org/abs/2605.19943): noise + Q-head selection, 87.4% → 98.75% on Sudoku-Extreme at test time) | `Refiner._child_codes` | `distinct → 1`, `fork.adopt` low, first-solve mass stays at round 1 |
| M3 one round does all | the round's **sequential** depth is unchanged (3 cycles); the extra compute is parallel width that one trajectory cannot use. Lookup is not removed (it is the task conditioning) — depth is therefore measured on held-out test inputs and on composites, where the rule must be *applied* | `DESIGN.md` §4.2, §7.3 | `late_gain ≈ 0` on composites with k ≥ 3 |
| M4 no demand | composite tasks f_k∘…∘f_1, k ≤ 4, ≥50% genuine detours, 15% of training rows in the default mix, with intermediate grids | `frm/data/composite.py` | `depth/by_k`: accuracy at k=3,4 flat across rounds |
| M5 early-exit incentives | keep TRM's correctness halting only for row turnover in training; no geometric or decaying priors; evaluation always runs 16 rounds and reports every round | `train.py`, `evaluate.py` | first-solve histogram |
| M6 horizon | horizon grows from one cycle to two along the fork path (intermediate, not full BPTT); a `trm_h2` control isolates horizon from forking | `ModelCfg.grad_cycles` | `trm_h2` ≥ FRM at matched FLOPs → the gain was horizon, not forking |
| M7 one hypothesis | K + 1 states per round, K hypotheses at the final round available to voting | `RoundOut.child_logits` | `fork/children_only` pass@2 ≈ parent's |
| M8 artefacts | exact/pass@2 per round on held-out pairs; every fork/depth metric chosen so it moves in the direction of its failure; decisions only after pass@2 is non-trivial | `DESIGN.md` §7, `EXPERIMENTS.md` | — |
| M9 extrapolation | no step-index inputs; training max rounds = evaluation rounds = 16 | `ModelCfg` | — |

## 6. Predictions (pre-registered; tested in `EXPERIMENTS.md`)

- **P-D1.** At matched training FLOPs, FRM's late gain on dev and on composites exceeds the `trm` control's by more than
  its paired 95% CI. Failure → M2/M1 responses insufficient; inspect `cf/sham`, `fork/adopt`.
- **P-D2.** FRM's first-solve histogram has more mass after round 4 than the control's, with no rise in regressions.
- **P-D3.** On composites, FRM's accuracy rises with rounds more for k = 3–4 than for k = 1; the control's does not.
- **P-D4.** `cf/reset@4` costs FRM more than the control: FRM's later rounds build on the carry instead of recomputing.
- **P-D5.** If `trm_h2` matches FRM on P-D1–P-D3, the depth gain is the gradient horizon (M6), not forking; that is a
  finding to report, and the fork redesign (DESIGN.md §12) is the response, not removal.

**Status (2026-10-15, `HANDOFF.md` §6):** P-D1–P-D4 are not supported so far. Late gain ≈ 0 in every arm (W, R1, R2:
≤ 0.007 row exact r4 → r16 at every composition depth; both ARC-AGI-2 finals to 32.5k PF: ≤ 0.010, FRM's k = 3 mean
+0.005 against TRM's +0.002), 1–4% of first solves after round 4 (ARC-AGI-1 dev and composites), `cf.reset@4` ≈ 0.
Children add breadth to round 1, not depth across rounds; on composites forking shows as a steeper slope per PF
instead. P-D5 is untested (`trm_h2_a1` never ran).

## 7. What this analysis does not claim

It does not claim that more inference rounds must help on every task: tasks whose rule is retrieved and applied in one
round should halt at round 1. The claim is narrower: on tasks that need composition, a refiner that receives new
proposals every round and is paid for setting up later rounds should keep improving after round 4, where a greedy,
single-trajectory refiner stops.
