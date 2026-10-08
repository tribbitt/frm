"""Objective properties (DESIGN.md §5, OBJECTIVE.md)."""

import math

import torch

from frm.config import ObjectiveCfg
from frm.objective import IGNORE, Objective, _distinct, log_stablemax, row_ce, row_exact, tau_at, wta_weights
from frm.model.refiner import Carry, RoundOut


def test_stablemax_normalised():
    x = torch.randn(5, 12) * 4
    torch.testing.assert_close(log_stablemax(x).exp().sum(-1), torch.ones(5))


def test_stablemax_gradient_finite_for_large_logits():
    """Regression: the unused 1/(1-x) branch at x >= 1 used to put NaN into the gradient through torch.where."""
    x = torch.tensor([[1.0, 2.5, -3.0, 0.0, 7.0]], requires_grad=True)
    row_ce(x[:, None, :], torch.tensor([[1]])).sum().backward()
    assert torch.isfinite(x.grad).all()


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


def test_wta_weights_limits():
    ce = torch.tensor([[0.3, 0.1, 0.9, 0.5]])
    eps, K = 0.05, 4
    hard = wta_weights(ce, tau=1e-4, eps=eps)
    soft = wta_weights(ce, tau=1e4, eps=eps)
    torch.testing.assert_close(hard.sum(-1), torch.ones(1))
    assert hard[0, 1] >= 1 - eps + eps / K - 1e-6 and torch.allclose(hard[0, [0, 2, 3]], torch.full((3,), eps / K))
    torch.testing.assert_close(soft, torch.full((1, 4), 0.25), atol=1e-3, rtol=0)


def test_tau_schedule_endpoints():
    cfg = ObjectiveCfg(tau_start=1.0, tau_end=0.02, tau_anneal_frac=0.3)
    assert math.isclose(tau_at(cfg, 0.0), 1.0) and math.isclose(tau_at(cfg, 0.3), 0.02)
    assert math.isclose(tau_at(cfg, 1.0), 0.02) and 0.02 < tau_at(cfg, 0.15) < 1.0


def test_wta_gradient_goes_to_the_winner():
    K, L = 3, 4
    labels = torch.randint(2, 12, (1, L))
    target = torch.nn.functional.one_hot(labels, 12).float() * 6
    child = torch.randn(1, K, L, 12)
    child[0, 2] = target[0]  # child 2 is (nearly) right
    child.requires_grad_(True)
    out = RoundOut(
        logits=target.clone(),
        q=torch.zeros(1),
        child_logits=child,
        child_q=torch.zeros(1, K),
        join_alpha=torch.full((1, K + 1), 1 / (K + 1)),
        join_entropy=torch.zeros(1),
        carry=Carry(torch.zeros(1), torch.zeros(1)),
    )
    loss, m = Objective(ObjectiveCfg(lambda_q=0.0, tau_end=1e-3))(out, {"labels": labels}, progress=1.0)
    loss.backward()
    g = child.grad.abs().sum((-1, -2))[0]
    assert m["winner"].item() == 2
    assert g[2] > 0, "the winner must receive gradient"
    assert g[0] / g.sum() < 0.1 and g[1] / g.sum() < 0.1, "losers get only the eps share"


def test_subgoal_term_claims_closest_child():
    K, M, L = 3, 3, 5
    labels = torch.randint(2, 12, (1, L))
    sub = torch.full((1, M, L), IGNORE)
    sub[0, 0] = torch.randint(2, 12, (L,))
    mask = torch.tensor([[True, False, False]])
    child = torch.randn(1, K, L, 12)
    child[0, 1] = torch.nn.functional.one_hot(sub[0, 0], 12).float() * 8
    loss, cov = Objective._subgoal_term(child, sub, mask, tau=1e-3)
    assert cov.item() == 1.0
    assert loss.item() < row_ce(child[:, 0], sub[:, 0]).item()
    no_mask_loss, no_cov = Objective._subgoal_term(child, sub, torch.zeros_like(mask), tau=1e-3)
    assert no_mask_loss.item() == 0.0 and torch.isnan(no_cov).all()


def test_distinct_counts_answers_on_supervised_cells():
    labels = torch.tensor([[2, 3, IGNORE]])
    preds = torch.tensor([[[2, 3, 5], [2, 3, 7], [4, 3, 5]]])  # children 0 and 1 agree on supervised cells
    assert _distinct(preds, labels).item() == 2
