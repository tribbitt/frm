"""Adaptive forking (DESIGN.md §4.8): declined rows are TRM rounds, forked rows are FRM rounds, rows are independent,
the gate learns from measured counterfactual benefit only, and the accounting follows the actual decisions."""

import dataclasses
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from frm.config import Cfg, ModelCfg, ObjectiveCfg
from frm.data import build as build_mod
from frm.data.composite import generate
from frm.data.loader import Dataset
from frm.evaluate import evaluate_split
from frm.model.refiner import Carry, Refiner, RoundOut
from frm.objective import Objective, row_ce
from frm.train import Trainer

B, S = 4, 6
CPU = torch.device("cpu")
MIXED = torch.tensor([True, False, True, False])


def make(K=2, **kw):
    torch.manual_seed(0)
    m = Refiner(ModelCfg(hidden=32, heads=2, K=K, code_rank=8, join_dim=8, gate_hidden=16, **kw), num_ids=8)
    m.train()
    return m


def share(dst: Refiner, src: Refiner) -> Refiner:
    """Copy every tensor the two models have in common (embeddings, blocks, heads, inits, puzzle table)."""
    own = dst.state_dict()
    dst.load_state_dict({k: v for k, v in src.state_dict().items() if k in own}, strict=False)
    return dst


def inputs():
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 12, (B, S * S), generator=g)
    labels = torch.randint(1, 12, (B, S * S), generator=g)
    labels[:, S * S // 2 :] = -100
    return x, torch.tensor([1, 2, 3, 4]), labels


def warm_carry(m, x, ids):
    """A carry after one round, so rows differ in state and not only in input."""
    with torch.no_grad():
        return m.round(m.init_carry(B, S, CPU, torch.float32), x, ids, fork_mode="all",
                       gen=torch.Generator().manual_seed(9)).carry


def force_decision(m, mask):
    """Make fork_mode="gate" take the given decision (the gate's own logits are tested elsewhere)."""
    inner = m._decide
    m._decide = lambda logit, mode, explore, n, dev: mask.clone() if mode == "gate" else inner(logit, mode, explore, n, dev)


def gen(seed=7):
    return torch.Generator().manual_seed(seed)


def sub(carry, idx):
    return Carry(carry.y[idx], carry.z[idx])


# ---------------------------------------------------------------------------------------------------- model
def test_gate_needs_depth_one():
    with pytest.raises(ValueError):
        make(fork_gate=True, fork_depth=2)


def test_fork_mode_none_equals_trm_control():
    g, t = make(fork_gate=True), make(K=0)
    share(t, g)
    x, ids, _ = inputs()
    cg = ct = g.init_carry(B, S, CPU, torch.float32)
    for _ in range(2):
        og, ot = g.round(cg, x, ids, fork_mode="none"), t.round(ct, x, ids)
        torch.testing.assert_close(og.logits, ot.logits)
        torch.testing.assert_close(og.q, ot.q)
        torch.testing.assert_close(og.carry.y, ot.carry.y)
        cg, ct = og.carry, ot.carry
    assert not og.fork.any() and not og.gate_rows.any() and torch.isnan(og.join_alpha).all()
    assert og.fork_logit.shape == (B,) and not og.cf_logits.any()


def test_fork_mode_all_equals_ungated_round():
    u = make(join_drop_parent=0.5)
    g = share(make(fork_gate=True, join_drop_parent=0.5), u)
    x, ids, _ = inputs()
    carry = warm_carry(u, x, ids)
    torch.manual_seed(3)  # the parent-slot dropout draws from the global RNG
    ou = u.round(carry, x, ids, gen=gen())
    torch.manual_seed(3)
    og = g.round(carry, x, ids, gen=gen(), fork_mode="all")
    for name in ("logits", "q", "child_logits", "child_q", "join_alpha", "join_entropy"):
        torch.testing.assert_close(getattr(og, name), getattr(ou, name), equal_nan=True)
    assert og.fork.all() and ou.fork.all() and ou.fork_logit is None and ou.cf_logits is None


def test_mixed_batch_rows_follow_their_own_path():
    """Gathering, running and scattering the subsets leaves each row exactly as if it ran its own path alone."""
    g = make(fork_gate=True)
    g.eval()
    x, ids, _ = inputs()
    carry = warm_carry(g, x, ids)
    force_decision(g, MIXED)
    f, n = MIXED.nonzero().squeeze(1), (~MIXED).nonzero().squeeze(1)
    with torch.no_grad():
        mixed = g.round(carry, x, ids, gen=gen())
        alone_f = g.round(sub(carry, f), x[f], ids[f], gen=gen(), fork_mode="all")
        alone_n = g.round(sub(carry, n), x[n], ids[n], fork_mode="none")
    tol = dict(atol=1e-5, rtol=1e-4)
    for name in ("logits", "q", "child_logits", "join_alpha", "cf_logits"):
        torch.testing.assert_close(getattr(mixed, name)[f], getattr(alone_f, name), **tol)
    torch.testing.assert_close(mixed.carry.y[f], alone_f.carry.y, **tol)
    torch.testing.assert_close(mixed.logits[n], alone_n.logits, **tol)
    torch.testing.assert_close(mixed.carry.z[n], alone_n.carry.z, **tol)
    assert torch.isnan(mixed.join_alpha[n]).all() and torch.isnan(mixed.join_entropy[n]).all()
    assert mixed.fork.tolist() == MIXED.tolist() and mixed.gate_rows.tolist() == MIXED.tolist()
    assert not mixed.cf_logits[n].any()


def test_cf_logits_are_the_trm_continuation():
    """A gate row's label readout is TRM's round from the same carry (H-1 no-grad cycles, one more), restated here."""
    g = make(fork_gate=True)
    x, ids, _ = inputs()
    carry = warm_carry(g, x, ids)
    out = g.round(carry, x, ids, gen=gen(), fork_mode="all")
    with torch.no_grad():
        y, z = carry.y, carry.z
        xe, cs = g.embed(x, ids, y.dtype), g.rope(S, y.device)
        for _ in range(g.cfg.H_cycles):
            for _ in range(g.cfg.L_cycles):
                z = g._f(z, y + xe, cs)
            y = g._f(y, z, cs)
    assert out.gate_rows.all() and out.cf_logits.grad_fn is None
    torch.testing.assert_close(out.cf_logits, g.lm_head(y)[:, g.cfg.prefix_len :])


def test_dropped_parent_slot_gives_no_gate_label():
    g = make(fork_gate=True, join_drop_parent=1.0)
    x, ids, labels = inputs()
    out = g.round(warm_carry(g, x, ids), x, ids, fork_mode="all")
    assert out.fork.all() and not out.gate_rows.any() and not out.cf_logits.any()
    _, m = Objective(ObjectiveCfg())(out, {"labels": labels}, 0.0)
    assert torch.isnan(m["gate_acc"]).all()


def test_no_fork_batch_sends_no_gradient_into_the_fork():
    g = make(fork_gate=True)
    with torch.no_grad():
        g.code_up.normal_(0, 0.1)
    x, ids, labels = inputs()
    carry = warm_carry(g, x, ids)
    loss, _ = Objective(ObjectiveCfg())(g.round(carry, x, ids, fork_mode="none"), {"labels": labels}, 0.0)
    loss.backward()
    for p in (g.codes, g.code_up, g.code_down.weight, g.join_q.weight, g.join_k.weight, g.join_beta):
        assert p.grad is None or not p.grad.any()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in g.blocks.parameters())  # the TRM round trains
    # In a mixed batch, the declined rows' loss alone reaches no fork parameter either.
    g.zero_grad(set_to_none=True)
    force_decision(g, MIXED)
    out = g.round(carry, x, ids)
    n = (~MIXED).nonzero().squeeze(1)
    row_ce(out.logits[n], labels[n]).sum().backward()
    assert g.codes.grad is None or not g.codes.grad.any()


def test_decide_gate_and_exploration():
    g, u = make(fork_gate=True), make()
    neg, pos = torch.full((B,), -1.0), torch.full((B,), 1.0)
    assert not g._decide(neg, "gate", 0.0, B, CPU).any() and g._decide(pos, "gate", 0.0, B, CPU).all()
    assert g._decide(neg, "gate", 1.0, B, CPU).all()  # exploration forks declined rows
    assert g._decide(neg, "all", 0.0, B, CPU).all() and not g._decide(pos, "none", 0.0, B, CPU).any()
    assert u._decide(None, "gate", 0.0, B, CPU).all()  # ungated: "gate" means every row forks


# ---------------------------------------------------------------------------------------------------- objective
def synthetic_out(fork, K=3, L=5, gated=False):
    g = torch.Generator().manual_seed(4)
    n = fork.shape[0]
    return RoundOut(
        logits=torch.randn(n, L, 12, generator=g), q=torch.randn(n, generator=g),
        child_logits=torch.randn(n, K, L, 12, generator=g).requires_grad_(True), child_q=torch.randn(n, K, generator=g),
        join_alpha=torch.full((n, K + 1), 0.25), join_entropy=torch.zeros(n), carry=None, fork=fork,
        fork_logit=torch.randn(n, generator=g) if gated else None,
        cf_logits=torch.randn(n, L, 12, generator=g) if gated else None,
        gate_rows=fork.clone() if gated else None,
    )


def test_declined_rows_have_exactly_the_trm_loss():
    labels = torch.randint(2, 12, (B, 5))
    obj = Objective(ObjectiveCfg())
    none = synthetic_out(torch.zeros(B, dtype=torch.bool))
    loss_none, m = obj(none, {"labels": labels}, 0.5)
    trm = dataclasses.replace(none, child_logits=None, child_q=None, join_alpha=None, join_entropy=None, fork=None)
    assert torch.equal(loss_none, obj(trm, {"labels": labels}, 0.5)[0])
    loss_none.backward()
    assert not none.child_logits.grad.any()
    assert all(torch.isnan(m[k]).all() for k in ("set", "child_ce_min", "child_exact_any", "winner", "distinct"))
    # Each forked row adds its own child terms and nothing else: the per-row increments add up to the all-fork loss.
    deltas = []
    for b in range(B):
        one = torch.zeros(B, dtype=torch.bool)
        one[b] = True
        deltas.append(obj(synthetic_out(one), {"labels": labels}, 0.5)[0] - loss_none)
    loss_all, m_all = obj(synthetic_out(torch.ones(B, dtype=torch.bool)), {"labels": labels}, 0.5)
    torch.testing.assert_close(loss_none + sum(deltas), loss_all)
    assert all(d > 0 for d in deltas) and not torch.isnan(m_all["set"]).any() and bool((m_all["fork_rate"] == 1).all())


def test_gate_label_is_benefit_above_price():
    labels = torch.randint(2, 12, (B, 5))
    out = synthetic_out(MIXED.clone(), gated=True)
    out.cf_logits[0] = F.one_hot(labels[0], 12).float() * 8  # row 0: the no-fork continuation is right -> no benefit
    out.logits[2] = F.one_hot(labels[2], 12).float() * 8  # row 2: the fork is right -> benefit
    benefit = row_ce(out.cf_logits, labels) - row_ce(out.logits, labels)
    base, _ = Objective(ObjectiveCfg(lambda_gate=0.0))(out, {"labels": labels}, 0.5)
    for price in (0.0, float(benefit[2]) + 1e-3, 100.0):
        loss, m = Objective(ObjectiveCfg(lambda_gate=1.0))(out, {"labels": labels}, 0.5, price=price)
        target = (benefit > price).float()
        g = MIXED
        assert m["gate_pos"][g].tolist() == target[g].tolist() and torch.isnan(m["gate_pos"][~g]).all()
        torch.testing.assert_close(m["benefit"][g], benefit[g])
        bce = F.binary_cross_entropy_with_logits(out.fork_logit[g], target[g])
        torch.testing.assert_close(loss - base, bce)  # the plain mean over gate rows
    assert m["gate_pos"][g].tolist() == [0.0, 0.0]
    assert Objective(ObjectiveCfg(fork_price=100.0))(out, {"labels": labels}, 0.5)[1]["gate_pos"][2] == 0.0
    no_rows = dataclasses.replace(out, gate_rows=torch.zeros(B, dtype=torch.bool))
    assert torch.equal(Objective(ObjectiveCfg(lambda_gate=1.0))(no_rows, {"labels": labels}, 0.5)[0], base)


def test_child_focus_sends_the_childrens_loss_to_cells_trm_gets_wrong():
    """With child_focus = 1 the children's final-target CE ignores every cell TRM's own round predicts right: making a
    child perfect there changes nothing, improving it on a cell TRM gets wrong lowers the loss."""
    labels = torch.randint(2, 12, (B, 5))
    out = synthetic_out(torch.ones(B, dtype=torch.bool), gated=True)
    out.cf_logits[:, :3] = F.one_hot(labels[:, :3], 12).float() * 8  # TRM's round right on cells 0-2
    out.cf_logits[:, 3:] = -F.one_hot(labels[:, 3:], 12).float() * 8  # and wrong on cells 3-4
    obj = Objective(ObjectiveCfg(child_focus=1.0, lambda_gate=0.0, wta_eps=1.0))  # eps = 1: plain mean child CE
    base = obj(out, {"labels": labels}, 1.0)[0]
    easy, hard = (dataclasses.replace(out, child_logits=out.child_logits.detach().clone()) for _ in range(2))
    easy.child_logits[:, :, :3] = F.one_hot(labels[:, None, :3], 12).float() * 8
    hard.child_logits[:, :, 3:] = F.one_hot(labels[:, None, 3:], 12).float() * 8
    torch.testing.assert_close(obj(easy, {"labels": labels}, 1.0)[0], base)
    assert obj(hard, {"labels": labels}, 1.0)[0] < base - 0.1
    plain = Objective(ObjectiveCfg(lambda_gate=0.0, wta_eps=1.0))
    assert plain(easy, {"labels": labels}, 1.0)[0] < plain(out, {"labels": labels}, 1.0)[0]  # off: every cell counts


# ---------------------------------------------------------------------------------------------------- accounting
@pytest.mark.parametrize("ckpt", [False, True])
def test_round_flops_boundaries(ckpt):
    g, t = make(fork_gate=True, grad_ckpt_children=ckpt), make(K=0)
    assert g.round_flops(S, B, B, 0) == pytest.approx(B * g.train_flops_per_row(S))
    assert g.round_flops(S, B, 0, 0) == pytest.approx(B * t.train_flops_per_row(S))
    assert g.round_flops(S, B, B, 0, train=False) == pytest.approx(B * g.eval_flops_per_row(S))
    assert g.round_flops(S, B, 0, 0, train=False) == pytest.approx(B * t.eval_flops_per_row(S))


def test_round_flops_match_executed_passes():
    g = make(fork_gate=True)
    counts = {True: 0, False: 0}
    f = g._f

    def counted(h, u, cs):
        counts[torch.is_grad_enabled()] += h.shape[0]
        return f(h, u, cs)

    g._f = counted
    force_decision(g, MIXED)
    x, ids, _ = inputs()
    out = g.round(g.init_carry(B, S, CPU, torch.float32), x, ids)
    nf, ncf = int(out.fork.sum()), int(out.gate_rows.sum())
    (fn, fg, _), (tn, tg, _) = g.passes_per_round(), g.passes_per_round(fork=False)
    cont = 2 * (g.cfg.L_cycles + 1)
    assert counts[True] == nf * fg + (B - nf) * tg
    assert counts[False] == nf * fn + (B - nf) * tn + ncf * cont
    expected = (nf * (fn + 3 * fg) + (B - nf) * (tn + 3 * tg) + ncf * cont) * g.flops_per_row_pass(S)
    assert g.round_flops(S, B, nf, ncf) == pytest.approx(expected)


# ---------------------------------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("gate")
    (d / "comp.json").write_text(json.dumps(generate(10, depth_range=(1, 2), seed=3, max_side=5)))
    build_mod.main(["--composite", str(d / "comp.json"), "--out", str(d / "data"), "--n_aug_composite", "2"])
    return d / "data"


def tiny_cfg(data_dir) -> Cfg:
    return Cfg.from_dict({
        "name": "gate",
        "model": {"hidden": 32, "heads": 2, "L_cycles": 1, "K": 2, "code_rank": 8, "join_dim": 8, "gate_hidden": 8,
                  "fork_gate": True, "join_drop_parent": 0.25},
        "data": {"root": str(data_dir), "tokens_per_step": 2000, "max_rows": 8, "source_weights": None},
        "train": {"lr": 1e-3, "warmup_steps": 2, "max_rounds": 3, "max_steps": 6, "budget_pflops": 1e9,
                  "eval_every_pflops": 1e9, "log_every": 1, "ckpt_minutes": 100, "device": "cpu",
                  "fork_force_frac": 0.5, "fork_explore": 1.0, "fork_rate_target": 0.25, "fork_price_lr": 0.1},
        "eval": {"splits": ["dev_composite"], "max_versions": 2, "rounds": 3, "tokens_per_batch": 4000,
                 "max_rows": 16, "cf_rows": 8},
    })


def test_gated_training_steps_price_and_resume(data_dir, tmp_path):
    cfg = tiny_cfg(data_dir)
    tr = Trainer(cfg, tmp_path / "run", "none")
    infos = [tr.train_step() for _ in range(6)]
    # Steps 0-2 are forced (progress < 0.5): the price stays; then every row forks (explore 1.0) at rate 1 > 0.25.
    assert [i["price"] for i in infos[:4]] == [0.0] * 4
    assert tr.price == pytest.approx(3 * 0.1 * 0.75)
    summary = tr.stats.summary()
    assert summary["fork_rate"] == 1.0 and 0.0 <= summary["gate_acc"] <= 1.0 and "benefit" in summary
    tr.save()
    again = Trainer(cfg, tmp_path / "run", "auto")
    assert again.price == pytest.approx(tr.price)


def test_fork_lr_mult_scales_only_the_fork_modules(data_dir, tmp_path):
    cfg = tiny_cfg(data_dir)
    cfg.train.fork_lr_mult = 4.0
    tr = Trainer(cfg, tmp_path / "run", "none")
    tr.train_step()
    base, forks = tr.opt.param_groups
    names = {id(p): n for n, p in tr.model.named_parameters()}
    assert {names[id(p)] for p in forks["params"]} >= {"codes", "join_q.weight", "join_beta"}
    assert not any(names[id(p)].startswith(("blocks.", "lm_head", "q_head")) for p in forks["params"])
    assert forks["lr"] == pytest.approx(4.0 * base["lr"]) and base["lr"] > 0
    tr.save()
    assert Trainer(cfg, tmp_path / "run", "auto").opt.param_groups[1]["lr_mult"] == 4.0  # resumes with both groups


def test_gated_evaluate_split_reports_rates_and_counterfactuals(data_dir):
    cfg = tiny_cfg(data_dir)
    torch.manual_seed(0)
    ds = Dataset(str(data_dir))
    model = Refiner(cfg.model, ds.num_ids)
    with torch.no_grad():
        model.gate[-1].bias.fill_(-10.0)  # a gate that declines everything: rate 0, children instruments undefined
    ec = cfg.eval
    res = evaluate_split(model, ds, "dev_composite", ec, CPU, torch.float32, counterfactuals=True)
    assert res["fork"]["rate"] == 0.0 and res["fork"]["rate_by_round"] == [0.0] * 3
    assert res["fork"]["forked_rows"] == 0 and res["fork"]["distinct"] is None
    assert {"nofork", "sham", "reset@4", "allfork", "gateoff"} <= set(res["cf"])
    assert res["cf"]["gateoff"] == 0.0  # the gate declined everything: gateoff is the base run
    with torch.no_grad():
        model.gate[-1].bias.fill_(10.0)
    res = evaluate_split(model, ds, "dev_composite", ec, CPU, torch.float32, counterfactuals=True)
    assert res["fork"]["rate"] == 1.0 and res["fork"]["forked_rows"] == res["rows"]
    assert res["cf"]["allfork"] == 0.0 and np.isfinite(res["fork"]["distinct"])
