"""Model gates (DESIGN.md §4.7, §10). The fork gate fails if the training forward does not fork."""

import pytest
import torch

from frm.config import ModelCfg, ObjectiveCfg
from frm.model.refiner import Refiner
from frm.objective import Objective, row_ce
from frm.train import load_pretrained

B, S = 3, 6


def make(K=4, **kw):
    torch.manual_seed(0)
    cfg = ModelCfg(hidden=32, heads=2, K=K, code_rank=8, join_dim=8, **kw)
    m = Refiner(cfg, num_ids=8)
    m.train()
    return m


def inputs():
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 12, (B, S * S), generator=g)
    labels = torch.randint(1, 12, (B, S * S), generator=g)
    labels[:, S * S // 2 :] = -100
    return x, torch.tensor([1, 2, 3]), labels


def record_cycles(model):
    calls = []
    inner = model._cycle_fn

    def wrapped(y, z, x, c, cos, sin, n, ckpt=False):
        calls.append((y.shape[0], torch.is_grad_enabled(), c is not None))
        return inner(y, z, x, c, cos, sin, n, ckpt)

    model._cycle_fn = wrapped
    return calls


def nonzero_grad(p):
    return p.grad is not None and bool(p.grad.abs().sum() > 0)


def test_child_checkpointing_is_exact():
    """Per-pass activation checkpointing of child cycles changes memory, not values or gradients."""
    grads = []
    for ckpt in (False, True):
        m = make(K=3, grad_ckpt_children=ckpt)
        x, ids, labels = inputs()
        out = m.round(m.init_carry(B, S, torch.device("cpu"), torch.float32), x, ids)
        row_ce(out.logits, labels).mean().backward()
        grads.append([None if p.grad is None else p.grad.clone() for p in m.parameters()])
        assert m.passes_per_round()[2] == (3 * (m.cfg.L_cycles + 1) if ckpt else 0)
    for a, b in zip(*grads):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b)


def test_fork_gate():
    """Children run with grad inside the round; the parent loss alone trains codes, join and children."""
    m = make(K=4)
    with torch.no_grad():  # at init q_head.weight = 0 makes every q equal, so beta's gradient is exactly 0 by symmetry
        m.q_head.weight.normal_(0, 0.1)
    calls = record_cycles(m)
    x, ids, labels = inputs()
    out = m.round(m.init_carry(B, S, torch.device("cpu"), torch.float32), x, ids)
    assert (B * 4, True, True) in calls, f"no child cycle with grad: {calls}"
    assert calls[0] == (B, False, False), "pre-fork parent cycle must run without grad"
    row_ce(out.logits, labels).mean().backward()  # parent loss only
    scale = m.join_q.weight.grad.abs().mean()
    assert scale > 0
    for name in ("codes", "code_up", "join_beta"):
        g = getattr(m, name).grad
        assert g is not None and g.abs().mean() > 1e-3 * scale, f"{name} receives no gradient from the parent loss"
    assert m.join_k.weight.grad.abs().mean() > 1e-3 * scale
    assert all(nonzero_grad(p) for p in m.blocks.parameters())


def test_code_down_trains_once_code_up_is_nonzero():
    m = make(K=4)
    with torch.no_grad():
        m.code_up.normal_(0, 0.1)
    x, ids, labels = inputs()
    out = m.round(m.init_carry(B, S, torch.device("cpu"), torch.float32), x, ids)
    row_ce(out.logits, labels).mean().backward()
    assert nonzero_grad(m.code_down.weight)


def reference_trm_round(m, carry, x_tokens, ids):
    """Independent restatement of TRM's round (TRM `models/recursive_reasoning/trm.py`)."""
    cfg = m.cfg
    y, z = carry.y, carry.z
    x = m.embed(x_tokens, ids, y.dtype)
    cs = m.rope(S, y.device)
    with torch.no_grad():
        for _ in range(cfg.H_cycles - 1):
            for _ in range(cfg.L_cycles):
                z = m._f(z, y + x, cs)
            y = m._f(y, z, cs)
    for _ in range(cfg.L_cycles):
        z = m._f(z, y + x, cs)
    y = m._f(y, z, cs)
    return m.lm_head(y)[:, cfg.prefix_len :], m.q_head(y[:, 0]).squeeze(-1)


def test_trm_mode_matches_reference_structure():
    m = make(K=0)
    x, ids, _ = inputs()
    carry = m.init_carry(B, S, torch.device("cpu"), torch.float32)
    out = m.round(carry, x, ids)
    ref_logits, ref_q = reference_trm_round(m, carry, x, ids)
    torch.testing.assert_close(out.logits, ref_logits)
    torch.testing.assert_close(out.q, ref_q)
    assert out.child_logits is None


def test_carry_detached_and_reset():
    m = make(K=4)
    x, ids, _ = inputs()
    out = m.round(m.init_carry(B, S, torch.device("cpu"), torch.float32), x, ids)
    assert out.carry.y.grad_fn is None and out.carry.z.grad_fn is None
    reset = torch.tensor([True, False, True])
    c = m.reset_carry(out.carry, reset)
    assert torch.equal(c.y[0], m.y_init.expand_as(c.y[0])) and torch.equal(c.y[1], out.carry.y[1])


def test_every_param_gets_grad():
    m = make(K=4)
    with torch.no_grad():
        m.code_up.normal_(0, 0.1)
    x, ids, labels = inputs()
    out = m.round(m.init_carry(B, S, torch.device("cpu"), torch.float32), x, ids)
    loss, _ = Objective(ObjectiveCfg())(out, {"labels": labels}, progress=0.0)
    loss.backward()
    dead = [n for n, p in m.named_parameters() if not nonzero_grad(p)]
    assert dead == [], f"parameters without gradient: {dead}"
    assert m.puzzle_emb._rows.grad is not None and m.puzzle_emb._rows.grad.abs().sum() > 0


def test_inference_equals_training_forward():
    m = make(K=4)
    x, ids, _ = inputs()
    carry = m.init_carry(B, S, torch.device("cpu"), torch.float32)
    train_out = m.round(carry, x, ids, gen=torch.Generator().manual_seed(7))
    m.eval()
    with torch.no_grad():
        eval_out = m.round(carry, x, ids, gen=torch.Generator().manual_seed(7))
    torch.testing.assert_close(train_out.logits.detach(), eval_out.logits)
    torch.testing.assert_close(train_out.child_logits.detach(), eval_out.child_logits)


def test_children_get_fresh_noise_each_round():
    """Children get fresh noise each round; a seeded generator keeps evaluation reproducible."""
    m = make(K=4)
    x, ids, _ = inputs()
    carry = m.init_carry(B, S, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        gen = torch.Generator().manual_seed(3)
        a, b = m.round(carry, x, ids, gen=gen).child_logits, m.round(carry, x, ids, gen=gen).child_logits
        again = m.round(carry, x, ids, gen=torch.Generator().manual_seed(3)).child_logits
    assert not torch.allclose(a, b), "same parent state, new round: children must differ"
    torch.testing.assert_close(a, again)
    quiet = make(K=4, noise_dim=0)
    with torch.no_grad():
        r1, r2 = (quiet.round(carry, x, ids).child_logits for _ in range(2))
    torch.testing.assert_close(r1, r2)  # without noise the fork is a deterministic map of the parent state


@pytest.mark.parametrize("K,depth", [(0, 1), (4, 1)])
def test_pass_accounting_matches_execution(K, depth):
    m = make(K=K, fork_depth=depth)
    counts = {True: 0, False: 0}
    f = m._f

    def counted(h, u, cs):
        counts[torch.is_grad_enabled()] += h.shape[0]
        return f(h, u, cs)

    m._f = counted
    x, ids, _ = inputs()
    m.round(m.init_carry(B, S, torch.device("cpu"), torch.float32), x, ids)
    nograd, grad, _ = m.passes_per_round()
    assert counts[False] == nograd * B and counts[True] == grad * B


def test_counterfactual_modes():
    m = make(K=4)
    x, ids, _ = inputs()
    carry = m.init_carry(B, S, torch.device("cpu"), torch.float32)

    def run(**kw):
        return m.round(carry, x, ids, gen=torch.Generator().manual_seed(0), **kw).logits

    with torch.no_grad():
        base, nofork, sham = run(), run(cf="nofork"), run(cf="sham")
        m.codes.add_(1.0)
        nofork_moved = run(cf="nofork")
    torch.testing.assert_close(nofork, nofork_moved)  # with the join on the parent slot, children cannot matter
    assert not torch.allclose(base, nofork) and not torch.allclose(base, sham)


def test_join_parent_dropout_forces_children_in_training_only():
    """A dropped row's joined state comes from the children alone; evaluation never drops."""
    m = make(K=4, join_drop_parent=1.0)
    L, D = m.cfg.prefix_len + S * S, m.cfg.hidden
    g = torch.Generator().manual_seed(3)
    y, z = torch.randn(B, L, D, generator=g), torch.randn(B, L, D, generator=g)
    cy, cz = torch.randn(B, 1, L, D, generator=g), torch.randn(B, 1, L, D, generator=g)
    ys, zs, q, qc = cy.expand(B, 4, L, D), cz.expand(B, 4, L, D), torch.zeros(B), torch.zeros(B, 4)
    y_new, z_new, mass, ent, drop = m._join(y, z, ys, zs, q, qc, None, None)
    torch.testing.assert_close(y_new, cy[:, 0])
    torch.testing.assert_close(z_new, cz[:, 0])
    assert torch.isnan(mass).all() and torch.isnan(ent).all() and bool(drop.all())
    m.eval()
    _, _, mass_eval, _, drop_eval = m._join(y, z, ys, zs, q, qc, None, None)
    assert drop_eval is None
    assert not torch.isnan(mass_eval).any() and bool((mass_eval[:, 0] > 0).all())


def test_puzzle_embedding_counts_updates_and_reads_old_checkpoints():
    m = make(K=0)
    x, ids, labels = inputs()
    out = m.round(m.init_carry(B, S, torch.device("cpu"), torch.float32), x, ids)
    row_ce(out.logits, labels).mean().backward()
    m.puzzle_emb.step(1e-2, 0.1)
    assert m.puzzle_emb.visits.tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    legacy = {k: v for k, v in m.state_dict().items() if k != "puzzle_emb.visits"}  # written before the counter
    m2 = make(K=0)
    m2.load_state_dict(legacy)
    assert m2.puzzle_emb.trained().tolist() == m.puzzle_emb.trained().tolist()


def test_load_pretrained_resets_new_rows_to_mean_of_trained_rows(tmp_path):
    """Never-trained (zero) rows do not dilute the reset-to-mean."""
    src = make(K=0)
    with torch.no_grad():
        src.puzzle_emb.table[1] = 1.0
        src.puzzle_emb.table[2] = 3.0
        src.puzzle_emb.visits[1:3] = 5
    torch.save({"model": src.state_dict()}, tmp_path / "ck.pt")
    dst = Refiner(src.cfg, num_ids=5)
    load_pretrained(dst, str(tmp_path / "ck.pt"), torch.device("cpu"))
    assert bool((dst.puzzle_emb.table[1:] == 2.0).all()) and bool((dst.puzzle_emb.table[0] == 0).all())
    assert int(dst.puzzle_emb.visits.sum()) == 0  # the new identifiers are untrained until adaptation updates them
