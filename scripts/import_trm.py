"""Import a TRM checkpoint (SamsungSAILMontreal/TinyRecursiveModels) into a frm checkpoint for a frm dataset.

    python scripts/import_trm.py --trm CKPT/arc_v2_public/step_723914 \
        --trm_ids TRM_DATA/arc2concept-aug-1000/identifiers.json --config configs/final_trm_a2tn.json \
        --data data/built/a2tn --out runs/trm_v2/import_a2tn.pt

Dense weights map one to one; the model must be K=0 with ``pos_encoding="trm1d"``. The dataset must be built with
``--versions_from`` the same identifiers.json. Identifiers found in TRM's list get TRM's puzzle-embedding row;
others start at the mean row, untrained. Load with ``frm.evaluate --ckpt PATH`` or train from it with
``train.init_from=PATH train.init_mode=continue`` (K>0 models: fork modules start fresh).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frm.config import Cfg  # noqa: E402
from frm.data.augment import SEP, Version  # noqa: E402
from frm.model.refiner import Refiner  # noqa: E402

PREFIX = "_orig_mod.model.inner."
DENSE = {  # frm name -> TRM name (inside the inner model)
    "tok_emb": "embed_tokens.embedding_weight",
    "lm_head.weight": "lm_head.weight",
    **{f"blocks.{i}.{a}": f"L_level.layers.{i}.{b}" for i in range(2) for a, b in (
        ("attn.qkv.weight", "self_attn.qkv_proj.weight"), ("attn.o.weight", "self_attn.o_proj.weight"),
        ("mlp.gate_up.weight", "mlp.gate_up_proj.weight"), ("mlp.down.weight", "mlp.down_proj.weight"))},
}


def frm_name(trm_name: str) -> str:
    """TRM names the identity version by the bare task id; frm always writes the transform."""
    return trm_name if SEP in trm_name else Version.identity().name(trm_name)


def convert(trm_state: dict, trm_ids: list[str], ids: list[str], cfg: Cfg) -> tuple[dict, dict]:
    sd = {k[len(PREFIX):] if k.startswith(PREFIX) else k: v for k, v in trm_state.items()}
    if cfg.model.pos_encoding != "trm1d" or cfg.model.K != 0:
        raise SystemExit("import into a K=0 model with model.pos_encoding=trm1d (forks start fresh at init_from)")
    model = Refiner(cfg.model, num_ids=len(ids))
    state = model.state_dict()
    for ours, theirs in DENSE.items():
        if state[ours].shape != sd[theirs].shape:
            raise SystemExit(f"{ours} {tuple(state[ours].shape)} != TRM {theirs} {tuple(sd[theirs].shape)}")
        state[ours] = sd[theirs].float().clone()
    state["y_init"], state["z_init"] = sd["H_init"].float().clone(), sd["L_init"].float().clone()
    state["q_head.weight"] = sd["q_head.weight"][:1].float().clone()  # TRM's halt logit (continue is unused)
    state["q_head.bias"] = sd["q_head.bias"][:1].float().clone()
    table = sd["puzzle_emb.weights"].float()
    if table.shape[0] != len(trm_ids):
        raise SystemExit(f"TRM table has {table.shape[0]} rows, its identifiers.json lists {len(trm_ids)}")
    row = {frm_name(n): i for i, n in enumerate(trm_ids) if i > 0}
    emb = state["puzzle_emb.table"]
    emb.copy_(table[1:].mean(0).expand_as(emb))
    emb[0].zero_()
    visits = state["puzzle_emb.visits"]
    visits.zero_()
    hit = [(i, row[n]) for i, n in enumerate(ids) if n in row]
    if not hit:
        raise SystemExit("no frm identifier is in TRM's list: build the dataset with --versions_from")
    dst, src = (torch.tensor(x) for x in zip(*hit))
    emb[dst] = table[src]
    visits[dst] = 1
    by_source: dict[str, list[int]] = {}
    for n in ids[1:]:
        base = n.split(SEP)[0]
        src_name = base.split(":")[0] if ":" in base else base.split("/")[0] if "/" in base else "arc"
        c = by_source.setdefault(src_name, [0, 0])
        c[0] += 1
        c[1] += int(n in row)
    stats = {"trm_rows": len(trm_ids) - 1, "frm_ids": len(ids) - 1, "mapped": len(hit),
             "trm_rows_unused": len(trm_ids) - 1 - len(hit),
             "by_source": {s: {"ids": c[0], "mapped": c[1]} for s, c in sorted(by_source.items())}}
    return state, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trm", required=True, help="TRM state dict (torch.save of the compiled ACT model)")
    ap.add_argument("--trm_ids", required=True, help="identifiers.json of the TRM dataset the checkpoint trained on")
    ap.add_argument("--config", required=True, help="frm config (K=0, model.pos_encoding=trm1d)")
    ap.add_argument("--data", required=True, help="frm dataset built with --versions_from TRM_IDS")
    ap.add_argument("--out", required=True)
    ap.add_argument("overrides", nargs="*", help="config key=value overrides")
    args = ap.parse_args(argv)
    cfg = Cfg.load(args.config, args.overrides)
    cfg.data.root = args.data
    with open(args.trm_ids) as f:
        trm_ids = json.load(f)
    with open(os.path.join(args.data, "identifiers.json")) as f:
        ids = json.load(f)
    trm_state = torch.load(args.trm, map_location="cpu", weights_only=False)
    state, stats = convert(trm_state, trm_ids, ids, cfg)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"model": state, "cfg": cfg.to_dict(), "num_ids": len(ids), "step": 0, "flops": 0.0,
                "row_rounds": 0, "import": {"trm": args.trm, **stats}}, args.out)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
