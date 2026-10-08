"""External pools and contamination: canonical digests, ingest filter, --external builds, the build's content filter."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
from test_data import REAL, _make_raw, _rand_grid, _write_subset

from frm.data.arc import load_subset
from frm.data.augment import Version, dihedral
from frm.data.build import _bucket, build, dev_ids, external_records, parse_args
from frm.data.heldout import HeldOut, canon, hits, informative
from frm.data.loader import Dataset, TrainStream

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ingest = _script("ingest_external")
make_adapt_dev = _script("make_adapt_dev")


def _disguise(g: np.ndarray, k: int, perm: np.ndarray) -> np.ndarray:
    """A held-out grid as a leak would look: recoloured (any of the 10 colours, 0 included) and transformed."""
    return np.ascontiguousarray(dihedral(perm[np.asarray(g, np.uint8)], k))


def _write_pool(path: str, tasks: list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]], **extra: np.ndarray) -> None:
    """A tinyabstractor FlatPool npz (``extra``: more arrays, e.g. the ingested ``heldout_ids`` and ``source``)."""
    data, off, hw, pair, pool_off, ids = [], [0], [], [], [0], []
    for tid, prs in tasks:
        ids.append(tid)
        for a, b in prs:
            for g in (a, b):
                g = np.asarray(g, np.int8)
                data.append(g.reshape(-1))
                off.append(off[-1] + g.size)
                hw.append(g.shape)
            pair.append((len(hw) - 2, len(hw) - 1))
        pool_off.append(len(pair))
    np.savez_compressed(path, data=np.concatenate(data), off=np.asarray(off, np.int64), hw=np.asarray(hw, np.int16),
                        pair=np.asarray(pair, np.int32), pool_off=np.asarray(pool_off, np.int32), ids=np.array(ids),
                        adm=np.zeros((len(ids), 4), bool), fresh=np.array(False), **extra)


def _pairs(rng: np.random.Generator, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(_rand_grid(rng, int(rng.integers(3, 9)), int(rng.integers(3, 9))),
             _rand_grid(rng, int(rng.integers(3, 9)), int(rng.integers(3, 9)))) for _ in range(n)]


def test_generator_tasks_split_by_bucket(tmp_path):
    """rearc/synth tasks become per-bucket sub-tasks (small groups merge upward); barc and arc tasks stay whole."""
    rng = np.random.default_rng(5)

    def sized(n: int, side: int) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(_rand_grid(rng, side, int(rng.integers(1, side + 1))), _rand_grid(rng, 3, 3)) for _ in range(n)]

    # g1: 5 pairs fit S=8, 2 fit S=12 (merge into the S=16 group of 4), 3 need S=30; mixed order
    g1 = sized(3, 7) + sized(2, 10) + sized(3, 25) + sized(4, 14) + sized(2, 5)
    # g2: 4 pairs at S=8 and one at S=30: the largest group is too small and merges down -> one task at S=30
    g2 = sized(4, 6) + sized(1, 30)
    tasks = [("g1", g1), ("g2", g2)]
    out = {}
    for source in ("rearc", "synth", "barc", "arc"):
        path = str(tmp_path / f"{source}.npz")
        _write_pool(path, tasks, heldout_ids=np.array(["x"]), source=np.array(source))
        out[source] = {r.task_id: r for r in external_records(source, path, set())}
    for source in ("rearc", "synth"):
        recs = out[source]
        assert sorted(recs) == [f"{source}:g1@16", f"{source}:g1@30", f"{source}:g1@8", f"{source}:g2@30"]
        sides = {k: [max(*np.shape(ex["input"]), *np.shape(ex["output"])) for ex in r.train] for k, r in recs.items()}
        assert sides[f"{source}:g1@8"] == [7, 7, 7, 5, 5]  # original order kept inside a group
        assert sides[f"{source}:g1@16"] == [10, 10, 14, 14, 14, 14]
        assert sides[f"{source}:g1@30"] == [25, 25, 25] and sides[f"{source}:g2@30"] == [6, 6, 6, 6, 30]
        for k, r in recs.items():
            assert _bucket(r) == int(k.rsplit("@", 1)[1]) and r.original["train"] is r.train and r.external
    assert sorted(out["barc"]) == ["barc:g1", "barc:g2"] and len(out["barc"]["barc:g1"].train) == len(g1)
    assert sorted(out["arc"]) == ["g1", "g2"] and len(out["arc"]["g2"].train) == len(g2)
    # through a build: sub-tasks land in their buckets with their own versions
    paths = [f"rearc={tmp_path}/rearc.npz", f"barc={tmp_path}/barc.npz"]
    build(parse_args(["--external", *paths, "--out", str(tmp_path / "b"), "--n_aug_external", "2"]))
    ds = Dataset(str(tmp_path / "b"))
    where = {t: S for S in ds.buckets for t in ds.split_data(S, "train").tasks}
    assert where == {"rearc:g1@8": 8, "rearc:g1@16": 16, "rearc:g1@30": 30, "rearc:g2@30": 30,
                     "barc:g1": 30, "barc:g2": 30}
    assert {Version.parse(n)[0] for n in ds.identifiers[1:]} == set(where)


# ---------------------------------------------------------------- canonical form


def test_canon_invariant_and_discriminating():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = _rand_grid(rng, 5, 7), _rand_grid(rng, 4, 3)
        ca, cab = canon(a), canon(a, b)
        for k in range(8):
            for _ in range(4):
                p = rng.permutation(10).astype(np.uint8)
                assert canon(_disguise(a, k, p)) == ca
                assert canon(_disguise(a, k, p), _disguise(b, k, p)) == cab
        a2 = a.copy()
        a2[0, 0] = (a2[0, 0] + 1) % 10
        assert canon(a2) != ca and canon(a2, b) != cab
        assert canon(a, b) != canon(b, a)
        assert canon(a, b) != canon(a, np.rot90(b).copy())  # one transform for the whole pair
        assert canon(a) != canon(a[:, :-1].copy())


def test_informative_threshold():
    g = np.zeros((6, 6), np.uint8)
    g.flat[:8] = 3
    assert informative(g)
    g.flat[7] = 0
    assert not informative(g)  # 7 cells off the background
    assert not informative(np.arange(35).reshape(5, 7) % 10)  # 35 cells
    h = np.full((6, 6), 4, np.uint8)
    h.flat[:8] = 0  # background = most frequent colour, not colour 0
    assert informative(h)


def test_heldout_hits():
    rng = np.random.default_rng(1)
    x, y, d_in, d_out = (_rand_grid(rng, 7, 7) for _ in range(4))
    tasks = {"t": {"train": [{"input": d_in, "output": d_out}], "test": [{"input": x, "output": y}]}}
    p = rng.permutation(10).astype(np.uint8)
    full, tests_out = HeldOut(tasks), HeldOut(tasks, demos=False, inputs=False)
    z = _rand_grid(rng, 7, 7)
    assert full.hit(_disguise(x, 2, p), _disguise(y, 2, p)) == "pair"
    assert full.hit(z, _disguise(y, 5, p)) == "grid" and tests_out.hit(_disguise(y, 5, p), z) == "grid"
    assert full.hit(_disguise(d_in, 1, p), z) == "grid" and tests_out.hit(_disguise(d_in, 1, p), z) is None
    assert full.hit(_disguise(d_in, 1, p), _disguise(d_out, 1, p)) == "pair"
    assert tests_out.hit(_disguise(x, 0, p), z) is None  # test inputs are visible at inference
    assert full.hit(z, _rand_grid(rng, 7, 7)) is None
    groups = [[(z, _disguise(y, k, p)), (z, z)] for k in range(8)] + [[(_disguise(x, 3, p), _disguise(y, 3, p))]]
    assert hits(groups, full, workers=2) == hits(groups, full, workers=1) == [["grid", None]] * 8 + [["pair"]]


# ---------------------------------------------------------------- ingest


@pytest.fixture(scope="module")
def env(tmp_path_factory) -> dict:
    """Fixture raw dir (arc1-shaped plus a small evaluation2), and ingested pools with planted contamination."""
    if not os.path.isdir(REAL):
        pytest.skip("needs ../trm-samsung/kaggle/combined")
    root = str(tmp_path_factory.mktemp("ext"))
    raw, comp = _make_raw(root, poison=False)
    ev2 = load_subset(REAL, "evaluation2")
    _write_subset(raw, "evaluation2", ev2, sorted(ev2)[:3])
    ev = load_subset(raw, "evaluation")
    dev = dev_ids(raw, 2)
    target = next(k for k in sorted(ev) if k not in dev and informative(np.asarray(ev[k]["test"][0]["output"])))
    x, y = (np.asarray(ev[target]["test"][0][f], np.uint8) for f in ("input", "output"))
    demo = ev[dev[0]]["train"][0]
    rng = np.random.default_rng(2)
    perm = rng.permutation(10).astype(np.uint8)
    clean = _pairs(rng, 5)
    barc = [
        ("barc/clean0", clean),
        ("barc/grid1", _pairs(rng, 4) + [(_rand_grid(rng, 5, 5), _disguise(y, 1, perm))]),
        ("barc/pair2", _pairs(rng, 4) + [(_disguise(x, 3, perm), _disguise(y, 3, perm))]),
        ("barc/dup3", _pairs(rng, 4) + [clean[1]] * 3),
        ("barc/small4", _pairs(rng, 2) + [(_disguise(y, 5, perm), _rand_grid(rng, 4, 4))]),
        ("barc/demo5", _pairs(rng, 3) + [(_disguise(demo["input"], 4, perm), _disguise(demo["output"], 4, perm))]),
    ]
    a2_new = [("a2clean1", _pairs(rng, 4)),
              ("a2dirty2", _pairs(rng, 3) + [(_rand_grid(rng, 4, 4), _disguise(y, 6, perm))])]
    src, out = os.path.join(root, "src"), os.path.join(root, "external")
    os.makedirs(src)
    _write_pool(os.path.join(src, "barc.npz"), barc)
    _write_pool(os.path.join(src, "a2_new.npz"), a2_new)
    reports = ingest.main(["--src", src, "--raw", raw, "--out", out, "--pools", "barc", "a2_new", "--workers", "1"])
    return {"root": root, "raw": raw, "comp": comp, "src": src, "out": out, "dev": dev, "target": target,
            "y": y, "perm": perm, "reports": {r["pool"]: r for r in reports}, "barc": barc}


def test_ingest_drops_planted(env):
    rb, ra = env["reports"]["barc"], env["reports"]["a2_new"]
    assert rb["by_task"]["barc/grid1"] == {"pair": 0, "grid": 1, "duplicate": 0}
    assert rb["by_task"]["barc/pair2"]["pair"] == 1
    assert rb["by_task"]["barc/dup3"] == {"pair": 0, "grid": 0, "duplicate": 2}
    assert rb["by_task"]["barc/small4"] == {"pair": 0, "grid": 1, "duplicate": 0, "dropped": "too_few_pairs"}
    assert rb["by_task"]["barc/demo5"]["pair"] == 1  # held-out demos are filter targets at ingest
    assert "barc/clean0" not in rb["by_task"]
    assert rb["contaminated_pairs"] == {"pair": 2, "grid": 2} and rb["duplicate_pairs"] == 2
    assert rb["tasks_out"] == 5 and rb["pairs_out"] == 5 + 4 + 4 + 5 + 3
    assert ra["dropped_task_ids"]["contaminated"] == ["a2dirty2"] and ra["tasks_out"] == 1 and ra["pairs_out"] == 4
    z = np.load(os.path.join(env["out"], "barc.npz"), allow_pickle=False)
    assert str(z["source"]) == "barc"
    assert [str(i) for i in z["ids"]] == ["barc/clean0", "barc/grid1", "barc/pair2", "barc/dup3", "barc/demo5"]
    want_held = set(load_subset(env["raw"], "evaluation")) | set(load_subset(env["raw"], "evaluation2"))
    assert [str(i) for i in z["heldout_ids"]] == sorted(want_held)
    cells = z["data"].view(np.uint8)

    def grid(g):
        return cells[z["off"][g] : z["off"][g + 1]].reshape(z["hw"][g])

    got = [(grid(i), grid(o)) for i, o in z["pair"][: z["pool_off"][1]]]
    assert all(np.array_equal(a, c) and np.array_equal(b, d) for (a, b), (c, d) in zip(got, env["barc"][0][1]))
    held = ingest.load_heldout(env["raw"])
    assert all(held.hit(grid(i), grid(o)) is None for i, o in z["pair"])
    a2 = np.load(os.path.join(env["out"], "a2_new.npz"), allow_pickle=False)
    assert [str(i) for i in a2["ids"]] == ["a2clean1"] and str(a2["source"]) == "arc"


def test_ingest_rejects_heldout_ids(env, tmp_path):
    rng = np.random.default_rng(3)
    _write_pool(str(tmp_path / "rearc.npz"), [(env["target"], _pairs(rng, 4))])
    with pytest.raises(SystemExit, match="held-out ids"):
        ingest.main(["--src", str(tmp_path), "--raw", env["raw"], "--out", str(tmp_path / "o"), "--pools", "rearc",
                     "--workers", "1"])


# ---------------------------------------------------------------- builds


def _build_args(env: dict, out: str, *extra: str) -> list[str]:
    return ["--raw", env["raw"], "--sources", "arc1", "--composite", env["comp"], "--out", out, "--n_aug", "2",
            "--n_aug_composite", "1", "--n_aug_external", "2", "--dev_n", "2", *extra]


def test_build_with_external(env, tmp_path):
    out = str(tmp_path / "b")
    meta = build(parse_args(_build_args(env, out, "--external", f"barc={env['out']}/barc.npz",
                                        f"arc={env['out']}/a2_new.npz")))
    ds = Dataset(out)
    assert meta["dev_ids"] == env["dev"] and set(ds.tasks("dev")) == set(env["dev"])
    assert meta["n_aug_external"] == 2
    train = {S: ds.split_data(S, "train") for S in ds.buckets if ds.has_split(S, "train")}
    where = {t: (S, int(d.source[i])) for S, d in train.items() for i, t in enumerate(d.tasks)}
    assert where["a2clean1"][1] == 0 and "a2dirty2" not in where  # a2_new keeps its ARC id and source "arc"
    barc_ids = {f"barc:{t}" for t in ("barc/clean0", "barc/grid1", "barc/pair2", "barc/dup3", "barc/demo5")}
    assert barc_ids == {t for t in where if t.startswith("barc:")} and all(where[t][1] == 2 for t in barc_ids)
    assert sum(c.get("barc", {}).get("examples", 0) for c in meta["counts"]["train"].values()) == 21
    for split in ("dev", "dev_composite", "test"):
        assert not any(k.startswith("barc:") for k in ds.tasks(split))
    S = next(int(S) for S, c in meta["counts"]["train"].items() if "barc" in c and "arc" in c)
    b = TrainStream(ds, S, np.random.default_rng(0), source_weights={"barc": 1.0}).take(64)
    assert all(ds.identifiers[p].startswith("barc:") for p in b["puzzle_ids"])
    mixed = TrainStream(ds, S, np.random.default_rng(0), source_weights={"arc": 0.5, "barc": 0.5}).take(2000)
    frac = np.mean([ds.identifiers[p].startswith("barc:") for p in mixed["puzzle_ids"]])
    assert 0.4 < frac < 0.6
    # the grids stored for a barc task are the ingested ones
    S, _ = where["barc:barc/clean0"]
    d = train[S]
    t = d.tasks.index("barc:barc/clean0")
    for e, (i, o) in zip(range(d.ex_start[t], d.ex_start[t + 1]), env["barc"][0][1], strict=True):
        (h, w), (oh, ow) = d.inp_hw[e], d.out_hw[e]
        assert np.array_equal(d.inp[0, e, :h, :w], i) and np.array_equal(d.out[0, e, :oh, :ow], o)


def test_build_external_checks(env, tmp_path):
    z = dict(np.load(f"{env['out']}/barc.npz", allow_pickle=False))
    held = [str(i) for i in z["heldout_ids"]]
    missing = next(i for i in held if i in set(load_subset(env["raw"], "evaluation")) and i not in env["dev"])
    z["heldout_ids"] = np.array([i for i in held if i != missing])
    np.savez(tmp_path / "stale.npz", **z)
    with pytest.raises(ValueError, match="not filtered against 1 dev/test"):
        build(parse_args(_build_args(env, str(tmp_path / "x"), "--external", f"barc={tmp_path}/stale.npz")))
    with pytest.raises(ValueError, match="ingested as source 'barc'"):
        build(parse_args(_build_args(env, str(tmp_path / "y"), "--external", f"rearc={env['out']}/barc.npz")))
    with pytest.raises(SystemExit):
        parse_args(_build_args(env, str(tmp_path / "z"), "--external", f"gen={env['out']}/barc.npz"))


def test_content_filter_drops_planted(env, tmp_path):
    """ARC training task / composite pair holding a recoloured, rotated dev or test output is dropped."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for f in os.listdir(env["raw"]):
        (raw / f).write_bytes((Path(env["raw"]) / f).read_bytes())
    rng = np.random.default_rng(4)
    y, perm = env["y"], env["perm"]
    ch = json.loads((raw / "arc-agi_training_challenges.json").read_text())
    so = json.loads((raw / "arc-agi_training_solutions.json").read_text())
    leak = _disguise(y, 2, perm).tolist()
    ch["planted0"] = {"train": [{"input": _rand_grid(rng, 5, 5).tolist(), "output": leak},
                                {"input": _rand_grid(rng, 5, 5).tolist(), "output": _rand_grid(rng, 5, 5).tolist()}],
                      "test": [{"input": _rand_grid(rng, 5, 5).tolist()}]}
    so["planted0"] = [_rand_grid(rng, 5, 5).tolist()]
    dev_out = np.asarray(load_subset(env["raw"], "evaluation")[env["dev"][0]]["test"][0]["output"], np.uint8)
    if informative(dev_out):  # a dev output in a training task's test pair
        ch["planted1"] = {"train": ch["planted0"]["train"][1:] * 2, "test": [{"input": _rand_grid(rng, 5, 5).tolist()}]}
        so["planted1"] = [_disguise(dev_out, 7, perm).tolist()]
    (raw / "arc-agi_training_challenges.json").write_text(json.dumps(ch))
    (raw / "arc-agi_training_solutions.json").write_text(json.dumps(so))
    comps = json.loads(Path(env["comp"]).read_text())
    comps[0]["train"][2]["output"] = leak
    comps[1]["train"][0]["subgoals"] = [_disguise(y, 4, perm).tolist()]
    comps[2]["test"][0]["output"] = leak
    comp = tmp_path / "comp.json"
    comp.write_text(json.dumps(comps))
    # RE-ARC-style generators: one for the dropped (rule-copying) task, one unrelated
    held_ids = np.array(sorted(set(load_subset(env["raw"], "evaluation")) | set(load_subset(env["raw"], "evaluation2"))))
    _write_pool(str(tmp_path / "rearc.npz"), [("planted0", _pairs(rng, 4)), ("cleangen", _pairs(rng, 4))],
                source=np.array("rearc"), heldout_ids=held_ids)
    args = _build_args(env, str(tmp_path / "b"), "--external", f"rearc={tmp_path}/rearc.npz")
    args[args.index("--raw") + 1], args[args.index("--composite") + 1] = str(raw), str(comp)
    meta = build(parse_args(args))
    cf = meta["content_filter"]
    assert cf["arc_tasks_dropped"] == ["planted0"] + (["planted1"] if "planted1" in ch else [])
    assert cf["composite_pairs_dropped"] == 3 and cf["composite_tasks_dropped"] == []
    assert cf["generator_tasks_dropped"] and all(t.startswith("rearc:planted0@") for t in cf["generator_tasks_dropped"])
    ds = Dataset(str(tmp_path / "b"))
    train_ids = {t for S in ds.buckets if ds.has_split(S, "train") for t in ds.split_data(S, "train").tasks}
    assert "planted0" not in train_ids and "comp:c0" in train_ids
    assert not any(t.startswith("rearc:planted0@") for t in train_ids), "a rule copy's generator must go too"
    assert any(t.startswith("rearc:cleangen@") for t in train_ids)
    assert len(ds.tasks("dev_composite")["comp:c2"]["test"]) == 1


# ---------------------------------------------------------------- adaptation dev


def _toy_task(rng: np.random.Generator, n_demo: int = 3) -> dict:
    ex = lambda: {"input": _rand_grid(rng, 4, 5).tolist(), "output": _rand_grid(rng, 3, 3).tolist()}  # noqa: E731
    return {"train": [ex() for _ in range(n_demo)], "test": [ex()]}


def test_make_adapt_dev(tmp_path):
    import hashlib

    rng = np.random.default_rng(6)
    cand = [f"c{i:02d}" for i in range(10)]
    t2 = {k: _toy_task(rng) for k in [*cand, "shared1"]}
    raw = str(tmp_path / "raw")
    os.makedirs(raw)
    _write_subset(raw, "training2", t2, sorted(t2))
    for sub, ids in (("training", ["tr1"]), ("evaluation", ["shared1"]), ("evaluation2", ["e2"])):
        tasks = {k: t2.get(k) or _toy_task(rng) for k in ids}
        _write_subset(raw, sub, tasks, ids)
    pool = str(tmp_path / "a2_new.npz")
    _write_pool(pool, [(k, _pairs(rng, 3)) for k in cand])
    out = str(tmp_path / "out")
    ids = make_adapt_dev.main(["--raw", raw, "--pool", pool, "--n", "4", "--out", out])
    assert ids == sorted(sorted(cand, key=lambda k: hashlib.md5(k.encode()).hexdigest())[:4])
    ch = json.loads(Path(out, "adev_challenges.json").read_text())
    so = json.loads(Path(out, "adev_solutions.json").read_text())
    assert json.loads(Path(out, "adev_ids.json").read_text()) == ids and sorted(ch) == sorted(so) == ids
    for k in ids:
        assert ch[k]["train"] == t2[k]["train"] and ch[k]["test"] == [{"input": t2[k]["test"][0]["input"]}]
        assert so[k] == [t2[k]["test"][0]["output"]]
    _write_pool(pool, [(k, _pairs(rng, 3)) for k in [*cand, "shared1"]])  # an ARC-AGI-1 task is no candidate
    with pytest.raises(SystemExit, match="not ARC-AGI-2-training-only"):
        make_adapt_dev.main(["--raw", raw, "--pool", pool, "--n", "4", "--out", out])


def test_ingest_heldout_pools(env, tmp_path):
    """Every pair of a --heldout_pools task is a filter target; the pool's own ids are held-out ids."""
    rng = np.random.default_rng(7)
    p, q = _rand_grid(rng, 7, 7), _rand_grid(rng, 7, 7)
    held_pool = str(tmp_path / "held.npz")
    _write_pool(held_pool, [("a2x", [(p, q)] + _pairs(rng, 3))])
    src = tmp_path / "src"
    src.mkdir()
    perm = rng.permutation(10).astype(np.uint8)
    _write_pool(str(src / "barc.npz"), [("barc/q1", _pairs(rng, 3) + [(_disguise(p, 3, perm), _disguise(q, 3, perm))]),
                                        ("barc/q2", _pairs(rng, 4))])
    base = ["--src", str(src), "--raw", env["raw"], "--pools", "barc", "--workers", "1"]
    plain = ingest.main([*base, "--out", str(tmp_path / "o1")])[0]
    assert plain["contaminated_pairs"] == {"pair": 0, "grid": 0}
    r = ingest.main([*base, "--out", str(tmp_path / "o2"), "--heldout_pools", held_pool])[0]
    assert r["by_task"]["barc/q1"]["pair"] == 1 and r["pairs_out"] == 7 and r["heldout"]["pools"][0]["sha256"]
    assert "a2x" in {str(i) for i in np.load(tmp_path / "o2" / "barc.npz")["heldout_ids"]}
    _write_pool(str(src / "a2_new.npz"), [("a2x", _pairs(rng, 3))])
    with pytest.raises(SystemExit, match="held-out ids"):
        ingest.main(["--src", str(src), "--raw", env["raw"], "--pools", "a2_new", "--workers", "1",
                     "--out", str(tmp_path / "o3"), "--heldout_pools", held_pool])


def test_exclude_ids(env, tmp_path):
    def run(name: str, exclude: list[str], *extra: str) -> dict:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(exclude))
        return build(parse_args(_build_args(env, str(tmp_path / name), "--exclude_ids", str(path), *extra)))

    train_id = sorted(load_subset(env["raw"], "training"))[0]
    with pytest.raises(ValueError, match="1 excluded tasks would enter"):
        run("arc", [train_id])
    a2 = dict(np.load(f"{env['out']}/a2_new.npz", allow_pickle=False))
    with pytest.raises(ValueError, match="not filtered against 1 dev/test/excluded"):  # pools must be filtered
        run("a2", ["a2clean1"], "--external", f"arc={env['out']}/a2_new.npz")  # against the excluded tasks too
    a2["heldout_ids"] = np.append(a2["heldout_ids"], "a2clean1")
    np.savez(tmp_path / "a2_new.npz", **a2)
    with pytest.raises(ValueError, match="1 excluded tasks would enter"):  # an a2_new task keeps its ARC id
        run("a2b", ["a2clean1"], "--external", f"arc={tmp_path}/a2_new.npz")
    z = dict(np.load(f"{env['out']}/barc.npz", allow_pickle=False))
    z["source"] = np.array("rearc")
    z["heldout_ids"] = np.append(z["heldout_ids"], ["barc/clean0", "zzzzzzzz"])
    np.savez(tmp_path / "rearc.npz", **z)  # generator sub-tasks rearc:{id}@{S} are matched on {id}
    with pytest.raises(ValueError, match="rearc:barc/clean0@"):
        run("gen", ["barc/clean0"], "--external", f"rearc={tmp_path}/rearc.npz")
    meta = run("ok", ["zzzzzzzz"], "--external", f"rearc={tmp_path}/rearc.npz")
    assert meta["exclude_ids"] == str(tmp_path / "ok.json")
