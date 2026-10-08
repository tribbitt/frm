"""Objective properties (DESIGN.md §5, OBJECTIVE.md)."""

import math

import torch

from frm.config import ObjectiveCfg
from frm.objective import (
    IGNORE, Objective, _assignment_table, _distinct, log_stablemax, row_ce, row_exact, set_match, tau_at,
)
from frm.model.refiner import Carry, RoundOut


def test_stablemax_normalised():
    x = torch.randn(5, 12) * 4
    torch.testing.assert_close(log_stablemax(x).exp().sum(-1), torch.ones(5))


def test_stablemax_gradient_finite_for_large_logits():
    """Regression: the unused 1/(1-x) branch at x >= 1 used to put NaN into the gradient through torch.where."""
    x = torch.tensor([[1.0, 2.5, -3.0, 0.0, 7.0]], requires_grad=True)
    row_ce(x[:, None, :], torch.tensor([[1]])).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_stablemax_gradient_nonzero_at_zero_logits():
    """Regression: clamping the x+1 branch made the gradient vanish at exactly x = 0."""
    x = torch.zeros(1, 1, 5, requires_grad=True)
    row_ce(x, torch.tensor([[2]])).sum().backward()
    assert x.grad[0, 0, 2] < 0 and (x.grad[0, 0, [0, 1, 3, 4]] > 0).all()



def test_row_ce_ignores_padding():
    logits = torch.randn(2, 10, 12)
    labels = torch.randint(1, 12, (2, 10))
    labels[:, 6:] = IGNORE
    a = row_ce(logits, labels)
    logits[:, 6:] = torch.randn(2, 4, 12) * 10
    torch.testing.assert_close(a, row_ce(logits, labels))


def test_row_exact():
    labels = torch.tensor([[3, 4, IGNORE], [IGNORE, IGNORE, IGNORE]])
    logits = torch.nn.functional.one_hot(torch.tensor([[3, 4, 0], [0, 0, 0]]), 12).float()
    assert row_exact(logits, labels).tolist() == [True, False]


def test_assignment_table_is_one_to_one():
    A = _assignment_table(4, 3)
    assert (A[:, 0] >= 0).all(), "the final target is always matched"
    for row in A.tolist():
        kids = [k for k in row if k >= 0]
        assert len(kids) == len(set(kids)) <= 4
    assert len(A) == 4 + 3 * 12 + 3 * 24 + 24
    assert ((_assignment_table(2, 3)[:, 1:] >= 0).sum(-1) <= 1).all(), "K=2 leaves one child for intermediates"


def test_set_match_without_intermediates_is_annealed_wta():
    ce = torch.tensor([[0.3, 0.1, 0.9, 0.5]])
    hard, best = set_match(ce, None, None, tau=1e-4, sub_ratio=1.0)
    soft, _ = set_match(ce, None, None, tau=1e4, sub_ratio=1.0)
    torch.testing.assert_close(hard, torch.tensor([0.1]))
    torch.testing.assert_close(soft, ce.mean(-1), atol=1e-4, rtol=0)
    assert best[0, 0].item() == 1


def test_set_match_sends_distinct_targets_to_distinct_children():
    """Two near-identical children (both ~ y*) and two different targets: the matching hands them one target each,
    so the gradient pulls them apart (the diversity pressure plain WTA lacks on composite rows)."""
    L = 5
    y = torch.randint(2, 12, (1, L))
    g = (y + 3).clamp(max=11)
    g[0, 0] = 2 if y[0, 0] != 2 else 3
    oh_g = torch.nn.functional.one_hot(g[0], 12).float()
    near_y = torch.nn.functional.one_hot(y, 12).float()[:, None].repeat(1, 2, 1, 1) * 4  # both children ~ y
    near_y[0, 0] += 0.1 * oh_g  # child 0 a hair closer to g: breaks the tie the way noise does in training
    child = near_y.clone().requires_grad_(True)
    ce_final = row_ce(child, y[:, None])
    ce_sub = row_ce(child[:, :, None], g[:, None, None])  # [1, 2, 1]
    loss, best = set_match(ce_final, ce_sub, torch.tensor([[True]]), tau=1e-4, sub_ratio=1.0)
    assert best[0].tolist() == [1, 0], "y* -> child 1, g -> child 0: different children"
    loss.sum().backward()
    toward_g = [-(child.grad[0, k] * oh_g).sum().item() for k in range(2)]
    assert toward_g[0] > 0 and toward_g[1] < 0, "child 0 is pulled toward g, child 1 away from it (toward y*)"


def test_set_match_respects_masks_and_spare_children():
    B, K, M = 2, 2, 3
    ce_final = torch.rand(B, K)
    ce_sub = torch.rand(B, K, M)
    mask = torch.tensor([[True, True, True], [False, False, False]])
    _, best = set_match(ce_final, ce_sub, mask, tau=1e-4, sub_ratio=1.0)
    assert (best[0, 1:] >= 0).sum() == 1, "K=2: one spare child, so exactly one intermediate is matched"
    assert (best[1, 1:] < 0).all(), "no valid intermediates: none matched"


def test_tau_schedule_endpoints():
    cfg = ObjectiveCfg(tau_start=1.0, tau_end=0.02, tau_anneal_frac=0.3)
    assert math.isclose(tau_at(cfg, 0.0), 1.0) and math.isclose(tau_at(cfg, 0.3), 0.02)
    assert math.isclose(tau_at(cfg, 1.0), 0.02) and 0.02 < tau_at(cfg, 0.15) < 1.0


def test_children_gradient_is_weighted_ce_gradient():
    """Without intermediates each child's gradient is w_k times its own CE gradient, w = (1-eps) softmin + eps/K."""
    K, L, eps = 3, 4, 0.05
    labels = torch.randint(2, 12, (1, L))
    target = torch.nn.functional.one_hot(labels, 12).float() * 6
    base = torch.randn(1, K, L, 12)
    base[0, 2] = target[0]  # child 2 is (nearly) right
    child = base.clone().requires_grad_(True)
    out = RoundOut(
        logits=target.clone(),
        q=torch.zeros(1),
        child_logits=child,
        child_q=torch.zeros(1, K),
        join_alpha=torch.full((1, K + 1), 1 / (K + 1)),
        join_entropy=torch.zeros(1),
        carry=Carry(torch.zeros(1), torch.zeros(1)),
    )
    cfg = ObjectiveCfg(lambda_q=0.0, lambda_wta=1.0, wta_eps=eps, tau_end=1e-4)
    loss, m = Objective(cfg)(out, {"labels": labels}, progress=1.0)
    loss.backward()
    assert m["winner"].item() == 2
    expected_w = torch.tensor([eps / K, eps / K, 1 - eps + eps / K])
    for k in range(K):
        own = base.clone().requires_grad_(True)
        row_ce(own[:, k], labels).sum().backward()
        torch.testing.assert_close(child.grad[0, k], expected_w[k] * own.grad[0, k], atol=1e-6, rtol=1e-4)


def test_objective_mixes_composite_and_plain_rows():
    """Rows with and without intermediates in one batch; coverage is NaN on plain rows and 1 where matched exactly."""
    K, M, L = 4, 3, 5
    labels = torch.randint(2, 12, (2, L))
    sub = torch.full((2, M, L), IGNORE)
    sub[0, 0] = (labels[0] + 1).clamp(max=11)
    mask = torch.zeros(2, M, dtype=torch.bool)
    mask[0, 0] = True
    child = torch.randn(2, K, L, 12)
    child[0, 1] = torch.nn.functional.one_hot(sub[0, 0], 12).float() * 8
    child[0, 2] = torch.nn.functional.one_hot(labels[0], 12).float() * 8
    out = RoundOut(
        logits=torch.randn(2, L, 12), q=torch.zeros(2), child_logits=child, child_q=torch.zeros(2, K),
        join_alpha=torch.full((2, K + 1), 0.2), join_entropy=torch.zeros(2), carry=None,
    )
    _, m = Objective(ObjectiveCfg(tau_end=1e-3))(out, {"labels": labels, "subgoals": sub, "subgoal_mask": mask}, 1.0)
    assert m["sub_coverage"][0].item() == 1.0 and torch.isnan(m["sub_coverage"][1])
    assert m["winner"][0].item() == 2


def test_distinct_counts_answers_on_supervised_cells():
    labels = torch.tensor([[2, 3, IGNORE]])
    preds = torch.tensor([[[2, 3, 5], [2, 3, 7], [4, 3, 5]]])  # children 0 and 1 agree on supervised cells
    assert _distinct(preds, labels).item() == 2
