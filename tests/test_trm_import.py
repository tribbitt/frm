"""Importing a TRM checkpoint: 1-D RoPE geometry, identifier-aligned embedding transfer, warm starts."""

import json
import os
import sys

import pytest
import torch

from frm.config import Cfg, ModelCfg
from frm.data.augment import Version
from frm.data.build import trm_versions
from frm.model.layers import RopeTRM1D, apply_rope_1d
from frm.model.refiner import Refiner
from frm.train import load_pretrained

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from import_trm import DENSE, PREFIX, convert  # noqa: E402


def trm_rotary(dim: int, positions: int, base: float = 10000.0):
    """Independent restatement of TRM `models/layers.py` RotaryEmbedding."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = torch.outer(torch.arange(positions, dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def test_trm1d_positions_are_trms_canvas_indices():
    cos_t, sin_t = trm_rotary(64, 16 + 900)
    rope = RopeTRM1D(64, 16, 10000.0)
    cos, sin = rope(30, torch.device("cpu"))
    torch.testing.assert_close(cos, cos_t)  # S = 30: TRM's own table
    torch.testing.assert_close(sin, sin_t)
    cos12, _ = rope(12, torch.device("cpu"))  # a smaller bucket keeps every token's TRM position
    idx = list(range(16)) + [16 + 30 * r + c for r in range(12) for c in range(12)]
    torch.testing.assert_close(cos12, cos_t[idx])
    x = torch.randn(2, len(idx), 3, 64)
    half = 32
    ref = x * cos_t[idx][:, None] + torch.cat((-x[..., half:], x[..., :half]), -1) * sin_t[idx][:, None]
    torch.testing.assert_close(apply_rope_1d(x, *rope(12, torch.device("cpu"))), ref)


def _tiny_cfg() -> Cfg:
    cfg = Cfg()
    cfg.model = ModelCfg(hidden=32, heads=2, K=0, pos_encoding="trm1d")
    return cfg


def test_import_copies_weights_and_aligns_embedding_rows_by_name():
    cfg = _tiny_cfg()
    ref = Refiner(cfg.model, num_ids=1).state_dict()
    trm_ids = ["<blank>", "aaaa", "aaaa|||t3|||0213456789", "bbbb"]
    trm = {PREFIX + theirs: torch.randn_like(ref[ours]) for ours, theirs in DENSE.items()}
    trm.update({PREFIX + "H_init": torch.randn(32).bfloat16(), PREFIX + "L_init": torch.randn(32).bfloat16(),
                PREFIX + "q_head.weight": torch.randn(2, 32), PREFIX + "q_head.bias": torch.randn(2),
                PREFIX + "puzzle_emb.weights": torch.randn(4, 32)})
    ours_ids = ["<blank>", Version.identity().name("aaaa"), "aaaa|||t3|||0213456789", Version.identity().name("cccc")]
    state, stats = convert(trm, trm_ids, ours_ids, cfg)
    for ours, theirs in DENSE.items():
        torch.testing.assert_close(state[ours], trm[PREFIX + theirs])
    torch.testing.assert_close(state["q_head.weight"], trm[PREFIX + "q_head.weight"][:1])  # the halt logit
    table, w = state["puzzle_emb.table"], trm[PREFIX + "puzzle_emb.weights"]
    torch.testing.assert_close(table[1], w[1])  # TRM's bare id is frm's identity version
    torch.testing.assert_close(table[2], w[2])
    torch.testing.assert_close(table[3], w[1:].mean(0))  # not in TRM's list: the mean row
    assert state["puzzle_emb.visits"].tolist() == [0, 1, 1, 0] and stats["mapped"] == 2
    Refiner(cfg.model, num_ids=4).load_state_dict(state)  # loads strictly into the configured model


def test_versions_from_reads_trms_table_identity_first(tmp_path):
    p = tmp_path / "identifiers.json"
    p.write_text(json.dumps(["<blank>", "aaaa", "aaaa|||t7|||0987654321", "aaaa|||t1|||0123456798", "bbbb"]))
    table = trm_versions(str(p))
    assert table["aaaa"] == [Version.identity(), Version(7, (0, 9, 8, 7, 6, 5, 4, 3, 2, 1)),
                             Version(1, (0, 1, 2, 3, 4, 5, 6, 7, 9, 8))]
    assert table["bbbb"] == [Version.identity()]


@pytest.mark.parametrize("extra", [{}, {"fork_gate": True}])
def test_continue_mode_keeps_table_and_gives_forks_a_fresh_start(tmp_path, extra):
    torch.manual_seed(0)
    src = Refiner(ModelCfg(hidden=32, heads=2, K=0, pos_encoding="trm1d"), num_ids=5)
    with torch.no_grad():
        src.puzzle_emb.table.normal_()
        src.puzzle_emb.visits[1:3] = 7
    torch.save({"model": src.state_dict(), "num_ids": 5}, tmp_path / "ck.pt")
    dst = Refiner(ModelCfg(hidden=32, heads=2, K=4, code_rank=8, join_dim=8, pos_encoding="trm1d", **extra), num_ids=5)
    fresh = {k: v.clone() for k, v in dst.state_dict().items() if k not in src.state_dict()}
    load_pretrained(dst, str(tmp_path / "ck.pt"), torch.device("cpu"), mode="continue")
    for k, v in src.state_dict().items():
        torch.testing.assert_close(dst.state_dict()[k], v)  # shared weights, table and visits carried over
    for k, v in fresh.items():
        torch.testing.assert_close(dst.state_dict()[k], v)  # fork modules untouched
    other = Refiner(ModelCfg(hidden=32, heads=2, K=0, pos_encoding="trm1d"), num_ids=6)
    with pytest.raises(ValueError):  # continue mode refuses a checkpoint of another dataset
        load_pretrained(other, str(tmp_path / "ck.pt"), torch.device("cpu"), mode="continue")
