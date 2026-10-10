"""Data-layer gates: augmentation, encoding, size rule, leakage, loader semantics (DESIGN.md §3, §10)."""

from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest

from frm.data.arc import load_subset
from frm.data.augment import DIHEDRAL_INVERSE, Version, _puzzle_hash, dihedral, inverse_dihedral, make_versions
from frm.data.build import build, dev_ids, parse_args
from frm.data.encode import EOS, IGNORE, PAD, crop, encode_grid, labels_from
from frm.data.loader import Dataset, TrainStream, eval_batches
from frm.data.sizerule import BUCKETS, choose_bucket, fits, predict_output_size

REAL = os.path.join(os.path.dirname(__file__), "..", "..", "trm-samsung", "kaggle", "combined")
SPLITS = ("dev", "dev_composite", "test")


def _rand_grid(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    return rng.integers(0, 10, (h, w)).astype(np.uint8)


def _rand_version(rng: np.random.Generator) -> Version:
    return Version(int(rng.integers(0, 8)), (0, *(int(c) for c in rng.permutation(np.arange(1, 10)))))


# ---------------------------------------------------------------- fixture


def _pick(tasks: dict, n: int, want_big: bool) -> list[str]:
    ids = sorted(tasks)[:n]
    if want_big:
        big = next(k for k in sorted(tasks) if max(len(ex["input"]) for ex in tasks[k]["train"]) >= 28)
        ids.append(big)
    return ids


def _write_subset(raw: str, subset: str, tasks: dict, ids: list[str]) -> None:
    ch = {k: {"train": tasks[k]["train"], "test": [{"input": ex["input"]} for ex in tasks[k]["test"]]} for k in ids}
    so = {k: [ex["output"] for ex in tasks[k]["test"]] for k in ids}
    with open(os.path.join(raw, f"arc-agi_{subset}_challenges.json"), "w") as f:
        json.dump(ch, f)
    with open(os.path.join(raw, f"arc-agi_{subset}_solutions.json"), "w") as f:
        json.dump(so, f)


def _composites(seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    steps = [np.fliplr, lambda g: np.where(g == 1, 2, g), lambda g: g.T]
    tasks = []
    for t, depth in enumerate((1, 2, 3, 3)):
        pairs = []
        for _ in range(8):
            x = _rand_grid(rng, int(rng.integers(2, 7)), int(rng.integers(2, 9)))
            gs = [x]
            for f in steps[:depth]:
                gs.append(np.ascontiguousarray(f(gs[-1])))
            pairs.append({"input": x.tolist(), "output": gs[-1].tolist(), "subgoals": [g.tolist() for g in gs[1:-1]]})
        tasks.append({"id": f"c{t}", "program": "test", "depth": depth, "train": pairs[:6], "test": pairs[6:]})
    return tasks


def _make_raw(root: str, poison: bool) -> tuple[str, str]:
    """Small arc1-shaped raw dir from real tasks; ``poison`` replaces evaluation test outputs by random grids."""
    raw = os.path.join(root, "raw")
    os.makedirs(raw, exist_ok=True)
    for subset, n, big in (("training", 10, True), ("concept", 3, False), ("evaluation", 6, True)):
        tasks = load_subset(REAL, subset)
        ids = _pick(tasks, n, big)
        if subset == "evaluation":
            ids.append("ad7e01d0")  # test output 25x25 overflows its predicted bucket 24
            if poison:
                rng = np.random.default_rng(123)
                tasks = copy.deepcopy(tasks)
                for k in ids:
                    for ex in tasks[k]["test"]:
                        h, w = np.shape(ex["output"])
                        ex["output"] = _rand_grid(rng, h, w).tolist()
        _write_subset(raw, subset, tasks, ids)
    comp = os.path.join(root, "comp.json")
    with open(comp, "w") as f:
        json.dump(_composites(), f)
    return raw, comp


def _build(root: str, poison: bool = False) -> str:
    raw, comp = _make_raw(root, poison)
    out = os.path.join(root, "built")
    args = ["--raw", raw, "--sources", "arc1", "--composite", comp, "--out", out, "--n_aug", "6",
            "--n_aug_composite", "3", "--dev_n", "2", "--seed", "0"]
    build(parse_args(args))
    return out


@pytest.fixture(scope="session")
def built(tmp_path_factory) -> str:
    if not os.path.isdir(REAL):
        pytest.skip("needs ../trm-samsung/kaggle/combined")
    return _build(str(tmp_path_factory.mktemp("clean")))


@pytest.fixture(scope="session")
def poisoned(tmp_path_factory) -> str:
    if not os.path.isdir(REAL):
        pytest.skip("needs ../trm-samsung/kaggle/combined")
    return _build(str(tmp_path_factory.mktemp("poison")), poison=True)


@pytest.fixture(scope="session")
def ds(built) -> Dataset:
    return Dataset(built)


def _version(ds: Dataset, pid: int) -> tuple[str, Version]:
    return Version.parse(ds.identifiers[pid])


def _train_task_grids(ds: Dataset) -> dict[str, list[dict]]:
    """task id -> base training pairs (numpy) reconstructed from the stored arrays."""
    out: dict[str, list[dict]] = {}
    for S in ds.buckets:
        if not ds.has_split(S, "train"):
            continue
        d = ds.split_data(S, "train")
        for t, task_id in enumerate(d.tasks):
            pairs = []
            for e in range(d.ex_start[t], d.ex_start[t + 1]):
                (h, w), (oh, ow) = d.inp_hw[e], d.out_hw[e]
                subs = []
                if d.sub is not None:
                    for m in range(3):
                        sh, sw = d.sub_hw[e, m]
                        if sh:
                            subs.append(d.sub[0, e * 3 + m, :sh, :sw])
                pairs.append({"input": d.inp[0, e, :h, :w], "output": d.out[0, e, :oh, :ow], "subgoals": subs})
            out[task_id] = pairs
    return out


# ---------------------------------------------------------------- augmentation / encoding


def test_dihedral_inverse_all_tids():
    rng = np.random.default_rng(0)
    g = _rand_grid(rng, 5, 7)
    for tid in range(8):
        assert np.array_equal(inverse_dihedral(dihedral(g, tid), tid), g)
        assert np.array_equal(dihedral(dihedral(g, tid), DIHEDRAL_INVERSE[tid]), g)
    assert len({dihedral(g, t).tobytes() + bytes(dihedral(g, t).shape) for t in range(8)}) == 8


def test_version_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(200):
        v = _rand_version(rng)
        g = _rand_grid(rng, int(rng.integers(1, 31)), int(rng.integers(1, 31)))
        a = v.apply(g)
        assert np.array_equal(v.invert(a), g)
        assert np.array_equal(a == 0, dihedral(g == 0, v.tid))  # colour 0 fixed
        assert Version.parse(v.name("abc"))[1] == v and Version.parse(v.name("abc"))[0] == "abc"
    assert Version.identity().name("x") == "x|||t0|||0123456789"


def test_make_versions_dedup():
    rng = np.random.default_rng(2)
    task = {"train": [{"input": _rand_grid(rng, 3, 4).tolist(), "output": _rand_grid(rng, 2, 2).tolist()}],
            "test": [{"input": _rand_grid(rng, 3, 4).tolist(), "output": None}]}
    vs = make_versions(task, 30, np.random.default_rng(0))
    assert vs[0] == Version.identity() and len(vs) == 31
    pairs = [(np.asarray(task["train"][0]["input"], np.uint8), np.asarray(task["train"][0]["output"], np.uint8)),
             (np.asarray(task["test"][0]["input"], np.uint8), None)]
    hashes = {_puzzle_hash([(v.apply(i), None if o is None else v.apply(o)) for i, o in pairs]) for v in vs}
    assert len(hashes) == len(vs)
    # all-black square task: every version is the same puzzle -> identity only
    black = {"train": [{"input": [[0, 0], [0, 0]], "output": [[0]]}], "test": [{"input": [[0, 0], [0, 0]]}]}
    assert make_versions(black, 10, np.random.default_rng(0)) == [Version.identity()]


def test_encode_crop_round_trip():
    rng = np.random.default_rng(3)
    for S in BUCKETS:
        for _ in range(50):
            side = S if S == 30 else S - 1
            h, w = int(rng.integers(1, side + 1)), int(rng.integers(1, side + 1))
            g = _rand_grid(rng, h, w)
            seq = encode_grid(g, S)
            assert seq.dtype == np.uint8 and seq.shape == (S * S,)
            assert np.array_equal(crop(seq, S), g)
            lab = labels_from(seq)
            assert lab.dtype == np.int16 and np.all((lab == IGNORE) == (seq == PAD))
            if S < 30:
                assert (seq == EOS).sum() == h + w
    full = _rand_grid(rng, 30, 30)
    assert np.array_equal(crop(encode_grid(full, 30), 30), full)
    assert not np.any(encode_grid(full, 30) == EOS)


def test_size_rule_ignores_test_outputs():
    rng = np.random.default_rng(4)
    for subset in ("training", "evaluation"):
        tasks = load_subset(REAL, subset) if os.path.isdir(REAL) else {}
        for k in sorted(tasks)[:150]:
            t = tasks[k]
            pois = copy.deepcopy(t)
            for ex in pois["test"]:
                ex["output"] = _rand_grid(rng, 30, 30).tolist()
            gone = copy.deepcopy(t)
            for ex in gone["test"]:
                del ex["output"]
            assert choose_bucket(t) == choose_bucket(pois) == choose_bucket(gone)
            for a, b in zip(t["test"], pois["test"]):
                assert predict_output_size(t, a["input"]) == predict_output_size(pois, b["input"])


def test_size_rule_cases():
    def task(pairs, test_shape):
        return ({"train": [{"input": np.zeros(i, int).tolist(), "output": np.zeros(o, int).tolist()} for i, o in pairs],
                 "test": [{"input": np.zeros(test_shape, int).tolist()}]})

    t = task([((2, 3), (4, 6)), ((3, 4), (6, 8))], (5, 4))  # ratio 2 on both axes
    assert predict_output_size(t, t["test"][0]["input"]) == (10, 8)
    t = task([((2, 3), (3, 3)), ((4, 5), (3, 3))], (9, 9))  # constant
    assert predict_output_size(t, t["test"][0]["input"]) == (3, 3)
    t = task([((2, 3), (4, 4)), ((5, 6), (7, 7))], (10, 1))  # difference +2/+1
    assert predict_output_size(t, t["test"][0]["input"]) == (12, 2)
    t = task([((2, 2), (3, 3)), ((3, 3), (7, 2))], (4, 4))  # no rule
    assert predict_output_size(t, t["test"][0]["input"]) is None
    assert choose_bucket(t) == 12  # max side 7 + 4 = 11 -> 12
    assert fits(29, 30) and fits(30, 30) and fits(7, 8) and not fits(8, 8)


# ---------------------------------------------------------------- build / leakage


def _arrays(root: str) -> dict[str, bytes]:
    out = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            with open(os.path.join(dirpath, fn), "rb") as f:
                out[os.path.relpath(os.path.join(dirpath, fn), root)] = f.read()
    return out


def test_poisoning_leaves_everything_but_test_outputs_unchanged(built, poisoned):
    """Poisoning perturbs every ARC-AGI-1 evaluation test output; dev tasks are evaluation tasks, so dev outputs
    change too. Nothing else may."""
    a, b = _arrays(built), _arrays(poisoned)
    assert a.keys() == b.keys()
    differ = {k for k in a if a[k] != b[k]}
    allowed = {"test_tasks.json", "meta.json"} | {k for k in a if "/test/out" in k or "/dev/out" in k}
    assert differ and differ <= allowed, differ - allowed
    ma, mb = (json.loads(x["meta.json"]) for x in (a, b))
    ma.pop("overflow"), mb.pop("overflow")
    ma.pop("composite"), mb.pop("composite")
    assert ma == mb


def test_dev_rule_md5_order(tmp_path):
    """dev = the n evaluation tasks with >= 3 demos with the smallest md5(task id) hex digests."""
    import hashlib

    demo = {"input": [[0]], "output": [[1]]}
    ids = [f"t{i:02d}" for i in range(12)]
    few = set(ids[::4])  # 2 demos: never dev
    ch = {k: {"train": [demo] * (2 if k in few else 3), "test": [{"input": [[0]]}]} for k in ids}
    (tmp_path / "arc-agi_evaluation_challenges.json").write_text(json.dumps(ch))
    order = sorted((k for k in ids if k not in few), key=lambda k: hashlib.md5(k.encode()).hexdigest())
    assert dev_ids(str(tmp_path), 4) == sorted(order[:4])
    assert not few & set(dev_ids(str(tmp_path), 100))
    assert len(dev_ids(str(tmp_path), 100)) == len(ids) - len(few)


def test_dev_and_test_splits(ds, built):
    raw = os.path.join(os.path.dirname(built), "raw")
    ev = load_subset(raw, "evaluation")
    want = dev_ids(raw, 2)
    assert ds.meta["dev_ids"] == want and len(want) == 2
    assert all(len(ev[k]["train"]) >= 3 for k in want)
    assert set(ds.tasks("dev")) == set(want)
    assert set(ds.tasks("test")) == set(ev) - set(want)
    grids = _train_task_grids(ds)
    for sub in ("training", "concept"):  # every training/concept task's test pairs are training data
        for k, t in load_subset(raw, sub).items():
            assert len(grids[k]) == len(t["train"]) + len(t["test"])
    for k, t in ev.items():  # demos of dev and test tasks are trained on (transductive), their test pairs never
        assert len(grids[k]) == len(t["train"])
    cf = ds.meta["content_filter"]
    assert cf["arc_tasks_dropped"] == [] and cf["composite_pairs_dropped"] == 0


def test_dev_pool_evaluation2_takes_dev_from_arc_agi2_evaluation(tmp_path):
    """For a model whose pretraining saw ARC-AGI-2 training test outputs, dev comes from evaluation2: its dev tasks'
    test pairs are dev, the other evaluation2 tasks' are test, and every training2 test pair stays training data."""
    if not os.path.isdir(REAL):
        pytest.skip("needs ../trm-samsung/kaggle/combined")
    raw = str(tmp_path / "raw")
    os.makedirs(raw)
    for subset, n in (("training2", 6), ("concept", 2), ("evaluation2", 5)):
        tasks = load_subset(REAL, subset)
        _write_subset(raw, subset, tasks, _pick(tasks, n, False))
    out = str(tmp_path / "built")
    build(parse_args(["--raw", raw, "--sources", "arc2", "--dev_subset", "evaluation2", "--dev_n", "2", "--out", out,
                      "--n_aug", "2", "--seed", "0"]))
    d = Dataset(out)
    ev2 = load_subset(raw, "evaluation2")
    assert set(d.tasks("dev")) == set(dev_ids(raw, 2, "evaluation2")) and set(d.tasks("dev")) <= set(ev2)
    assert set(d.tasks("test")) == set(ev2) - set(d.tasks("dev"))
    grids = _train_task_grids(d)
    for k, t in load_subset(raw, "training2").items():
        assert len(grids[k]) == len(t["train"]) + len(t["test"])


def test_composite_bucket_ignores_held_out_targets():
    """Held-out (dev_composite) outputs and sub-goals must not choose the canvas; demo sub-goals must fit it."""
    from frm.data.build import Record, _bucket

    rng = np.random.default_rng(0)
    demos = [{"input": _rand_grid(rng, 4, 4).tolist(), "output": _rand_grid(rng, 4, 4).tolist(),
              "subgoals": [_rand_grid(rng, 4, 4).tolist()]} for _ in range(4)]
    small = [{"input": _rand_grid(rng, 4, 4).tolist(), "output": _rand_grid(rng, 4, 4).tolist(),
              "subgoals": [_rand_grid(rng, 4, 4).tolist()]}]
    big = [dict(small[0], output=_rand_grid(rng, 20, 20).tolist(), subgoals=[_rand_grid(rng, 25, 25).tolist()])]
    rec = lambda held: Record("comp:x", "composite", {"train": demos, "test": held}, demos, "dev_composite", held)  # noqa: E731
    assert _bucket(rec(small)) == _bucket(rec(big)) == 8
    demos[0]["subgoals"] = [_rand_grid(rng, 11, 11).tolist()]
    assert _bucket(rec(small)) == 12


def test_no_test_output_in_train_rows(poisoned):
    ds = Dataset(poisoned)
    train_grids = {(g.shape, g.tobytes()) for pairs in _train_task_grids(ds).values()
                   for p in pairs for g in (p["input"], p["output"], *p["subgoals"])}
    test = {k: t for split in ("dev", "test") for k, t in ds.tasks(split).items()}
    assert ds.tasks("dev") and ds.tasks("test")
    for t in test.values():
        for ex in t["test"]:
            g = np.asarray(ex["output"], np.uint8)
            assert (g.shape, g.tobytes()) not in train_grids
    # and the stream never renders one, in any version
    for S in ds.buckets:
        ts = TrainStream(ds, S, np.random.default_rng(0), translate_p=0.0)
        b = ts.take(64)
        for k in range(len(b["puzzle_ids"])):
            task_id, v = _version(ds, int(b["puzzle_ids"][k]))
            lab = np.where(b["labels"][k] == IGNORE, 0, b["labels"][k])
            g = v.invert(crop(lab, S))
            for t in test.values():
                for ex in t["test"]:
                    assert not np.array_equal(g, np.asarray(ex["output"], np.uint8))


def test_meta_and_splits(ds):
    m = ds.meta
    assert m["P"] == 16 and m["vocab"] == 12 and m["M"] == 3
    assert ds.splits() == ["train", "dev", "dev_composite", "test"]
    assert len(ds.identifiers) == ds.num_ids and ds.identifiers[0] == "<blank>"
    assert {o[1] for o in m["overflow"]} == {"ad7e01d0"}
    assert all(ds.tasks("dev_composite")[k]["test"] for k in ds.tasks("dev_composite"))
    assert all(k.startswith("comp:") for k in ds.tasks("dev_composite"))
    assert sum(ds.num_train_examples(S) for S in ds.buckets) > 0
    assert 30 in ds.buckets


def test_eval_rows_untranslated_and_keys(ds):
    for split in SPLITS:
        tasks = ds.tasks(split)
        n_rows = 0
        for S in ds.buckets:
            batches = list(eval_batches(ds, split, S, batch_rows=7, max_versions=3))
            again = list(eval_batches(ds, split, S, batch_rows=7, max_versions=3))
            assert [b["keys"] for b in batches] == [b["keys"] for b in again]
            seen_first: set[tuple[str, int]] = set()
            for b in batches:
                assert b["inputs"].dtype == np.uint8 and b["labels"].dtype == np.int16
                # sub-goal arrays exist only where the stored split has sub-goals, which only composites have
                assert "subgoals" not in b or split == "dev_composite"
                for k, (task_id, idx, pid) in enumerate(b["keys"]):
                    n_rows += 1
                    tid, v = _version(ds, pid)
                    assert tid == task_id and pid == b["puzzle_ids"][k]
                    if (task_id, idx) not in seen_first:
                        assert v == Version.identity()
                        seen_first.add((task_id, idx))
                    ex = tasks[task_id]["test"][idx]
                    want = encode_grid(v.apply(np.asarray(ex["input"], np.uint8)), S)
                    assert np.array_equal(b["inputs"][k], want)  # offset (0, 0)
                    out = np.asarray(ex["output"], np.uint8)
                    if fits(max(out.shape), S):
                        assert np.array_equal(b["labels"][k], labels_from(encode_grid(v.apply(out), S)))
                    else:
                        assert not np.any(b["labels"][k] == IGNORE)
        n_want = 0
        for task_id, t in tasks.items():
            nv = min(3, sum(1 for name in ds.identifiers if name.rsplit("|||", 2)[0] == task_id))
            n_want += len(t["test"]) * nv
        assert n_rows == n_want


def test_kaggle_unknown_outputs(tmp_path):
    if not os.path.isdir(REAL):
        pytest.skip("needs ../trm-samsung/kaggle/combined")
    tasks = load_subset(REAL, "evaluation2")
    ids = sorted(tasks)[:3]
    ch = tmp_path / "ch.json"
    ch.write_text(json.dumps({k: {"train": tasks[k]["train"], "test": [{"input": e["input"]} for e in tasks[k]["test"]]}
                              for k in ids}))
    out = str(tmp_path / "k")
    build(parse_args(["--sources", "kaggle", "--challenges", str(ch), "--out", out, "--n_aug", "2"]))
    ds = Dataset(out)
    assert ds.splits() == ["train", "test"]
    for S in ds.buckets:
        for b in eval_batches(ds, "test", S, 16):
            assert np.all(b["labels"] == IGNORE)
    assert all(ex["output"] is None for t in ds.tasks("test").values() for ex in t["test"])


# ---------------------------------------------------------------- training stream


def test_take_shapes_and_content(ds):
    grids = _train_task_grids(ds)
    n_comp = 0
    for S in ds.buckets:
        ts = TrainStream(ds, S, np.random.default_rng(S), translate_p=1.0)
        b = ts.take(200)
        assert b["inputs"].shape == (200, S * S) and b["inputs"].dtype == np.uint8
        assert b["labels"].shape == (200, S * S) and b["labels"].dtype == np.int16
        assert b["puzzle_ids"].shape == (200,) and b["puzzle_ids"].dtype == np.int32
        assert b["subgoals"].shape == (200, 3, S * S) and b["subgoals"].dtype == np.int16
        assert b["subgoal_mask"].shape == (200, 3) and b["subgoal_mask"].dtype == bool
        assert np.all(b["labels"] != PAD) and np.all(np.isin(b["labels"], [IGNORE, *range(1, 12)]))
        moved = 0
        for k in range(200):
            task_id, v = _version(ds, int(b["puzzle_ids"][k]))
            inp = b["inputs"][k].reshape(S, S)
            rr, cc = np.nonzero(inp)
            off = (int(rr.min()), int(cc.min()))
            moved += off != (0, 0)
            match = None
            for p in grids[task_id]:
                ai, ao = v.apply(p["input"]), v.apply(p["output"])
                try:
                    if np.array_equal(b["inputs"][k], encode_grid(ai, S, off)) and np.array_equal(
                        b["labels"][k], labels_from(encode_grid(ao, S, off))
                    ):
                        match = p
                        break
                except ValueError:
                    continue
            assert match is not None, (task_id, k)
            if S < 30:
                assert (b["inputs"][k] == EOS).sum() == sum(v.apply(match["input"]).shape)
            nsub = len(match["subgoals"])
            assert b["subgoal_mask"][k].tolist() == [m < nsub for m in range(3)]
            for m in range(3):
                if m < nsub:
                    want = labels_from(encode_grid(v.apply(match["subgoals"][m]), S, off))
                    assert np.array_equal(b["subgoals"][k, m], want)
                else:
                    assert np.all(b["subgoals"][k, m] == IGNORE)
            n_comp += task_id.startswith("comp:")
        assert moved > 0
        b0 = TrainStream(ds, S, np.random.default_rng(0), translate_p=0.0).take(50)
        for k in range(50):
            rr, cc = np.nonzero(b0["inputs"][k].reshape(S, S))
            assert rr.min() == 0 and cc.min() == 0
    assert n_comp > 0


def test_group_balanced_draws_and_carry(ds):
    S = max(ds.buckets, key=ds.num_train_examples)
    d = ds.split_data(S, "train")
    size = dict(zip(d.tasks, np.diff(d.ex_start)))
    ts = TrainStream(ds, S, np.random.default_rng(5))
    n_tasks = len(d.tasks)
    b = ts.take(int(np.diff(d.ex_start).sum()))  # one epoch worth of rows, minus carry-over effects
    pids = b["puzzle_ids"]
    runs = np.split(pids, np.flatnonzero(np.diff(pids)) + 1)
    names = [_version(ds, int(r[0]))[0] for r in runs]
    for r, name in zip(runs[:-1], names[:-1]):
        assert len(r) % min(size[name], 16) == 0  # whole draws (adjacent equal versions merge)
    assert len(set(names)) >= 0.5 * n_tasks
    pending = ts.pending_vid.copy()
    nxt = ts.take(len(pending) + 1)
    assert np.array_equal(nxt["puzzle_ids"][: len(pending)], pending)


def test_source_weights(ds):
    S = next(S for S in ds.buckets if ds.meta["counts"]["train"][str(S)].get("composite"))
    only_comp = TrainStream(ds, S, np.random.default_rng(0), source_weights={"arc": 0.0, "composite": 1.0}).take(100)
    assert all(ds.identifiers[p].startswith("comp:") for p in only_comp["puzzle_ids"])
    only_arc = TrainStream(ds, S, np.random.default_rng(0), source_weights={"arc": 1.0}).take(100)
    assert not any(ds.identifiers[p].startswith("comp:") for p in only_arc["puzzle_ids"])


def test_state_dict_resume(ds):
    S = max(ds.buckets, key=ds.num_train_examples)
    a = TrainStream(ds, S, np.random.default_rng(7))
    a.take(13)
    state = json.loads(json.dumps(a.state_dict()))
    want = [a.take(n) for n in (29, 64, 5)]
    b = TrainStream(ds, S, np.random.default_rng(999))
    b.load_state_dict(state)
    got = [b.take(n) for n in (29, 64, 5)]
    for w, g in zip(want, got):
        for k in w:
            assert np.array_equal(w[k], g[k]), k


def test_per_draw_cap_and_resume(ds):
    """A draw yields a random subset of min(task size, cap) distinct pairs; streams resume exactly with the cap."""
    S = max(ds.buckets, key=ds.num_train_examples)
    d = ds.split_data(S, "train")
    cap = 3
    big = np.flatnonzero(np.diff(d.ex_start) > cap)
    assert len(big)
    ts = TrainStream(ds, S, np.random.default_rng(3), max_pairs_per_draw=cap)
    assert ts.rows_per_draw <= cap
    seen: dict[int, set[int]] = {}
    for _ in range(40):
        ts._draw(25)
        ex, vid = ts.pending_ex, ts.pending_vid
        ts.pending_ex, ts.pending_vid = ex[:0], vid[:0]
        task = np.searchsorted(d.ex_start, ex, side="right") - 1
        a = 0
        while a < len(ex):  # draws are blocks of min(size, cap) rows of one task and one version
            t = int(task[a])
            n = min(int(d.ex_start[t + 1] - d.ex_start[t]), cap)
            block = ex[a : a + n]
            assert np.all(task[a : a + n] == t) and np.all(vid[a : a + n] == vid[a]) and len(set(block)) == n
            assert d.vstart[t] <= vid[a] < d.vstart[t] + d.vcount[t]
            seen.setdefault(t, set()).update(block.tolist())
            a += n
    t = int(big[0])
    assert seen[t] == set(range(d.ex_start[t], d.ex_start[t + 1]))  # a random subset each time, not a prefix
    a = TrainStream(ds, S, np.random.default_rng(7), max_pairs_per_draw=cap)
    a.take(13)
    state = json.loads(json.dumps(a.state_dict()))
    want = [a.take(n) for n in (29, 64, 5)]
    b = TrainStream(ds, S, np.random.default_rng(999), max_pairs_per_draw=cap)
    b.load_state_dict(state)
    for w, g in zip(want, [b.take(n) for n in (29, 64, 5)]):
        for k in w:
            assert np.array_equal(w[k], g[k]), k


def test_take_speed(ds):
    """Not a gate on CI hardware; guards against accidental de-vectorisation (target < 50 ms at 4096 x S=16)."""
    import time

    S = 16 if 16 in ds.buckets else ds.buckets[0]
    ts = TrainStream(ds, S, np.random.default_rng(0))
    ts.take(4096)
    t = time.perf_counter()
    ts.take(4096)
    assert time.perf_counter() - t < 0.5
