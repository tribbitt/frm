import numpy as np
import pytest

from frm.data import composite as C

N_TASKS = 80


@pytest.fixture(scope="module")
def tasks() -> list[dict]:
    return C.generate(N_TASKS, seed=11)


def _grids(ex: dict) -> list[np.ndarray]:
    return [np.array(ex["input"]), np.array(ex["output"]), *(np.array(g) for g in ex["subgoals"])]


def test_deterministic(tasks):
    assert C.generate(N_TASKS, seed=11) == tasks
    assert C.generate(5, seed=12) != tasks[:5]


def test_schema_shapes_and_values(tasks):
    assert [t["id"] for t in tasks] == [f"comp11_{i}" for i in range(N_TASKS)]
    for t in tasks:
        assert set(t) == {"id", "program", "depth", "detour", "train", "test"}
        assert 1 <= t["depth"] <= 4 and len(t["program"].split(">")) == t["depth"]
        assert 6 <= len(t["train"]) <= 10 and len(t["test"]) == 2
        for ex in t["train"] + t["test"]:
            assert set(ex) == {"input", "output", "subgoals"}
            for rows in [ex["input"], ex["output"], *ex["subgoals"]]:
                assert rows and len({len(r) for r in rows}) == 1
            for g in _grids(ex):
                assert 1 <= min(g.shape) and max(g.shape) <= 30
                assert g.min() >= 0 and g.max() <= 9
            assert 3 <= min(np.array(ex["input"]).shape) and max(np.array(ex["input"]).shape) <= 20
    assert set(t["depth"] for t in tasks) == {1, 2, 3, 4}


def test_subgoals_are_program_prefixes(tasks):
    for t in tasks:
        steps = C.parse_program(t["program"])
        assert C.format_program(steps) == t["program"]
        for ex in t["train"] + t["test"]:
            assert len(ex["subgoals"]) == t["depth"] - 1
            chain = C.run_program(steps, np.array(ex["input"], dtype=np.uint8))
            assert chain is not None
            assert [g.tolist() for g in chain[1:-1]] == ex["subgoals"]
            assert chain[-1].tolist() == ex["output"]


def test_filters_hold(tasks):
    for t in tasks:
        steps = C.parse_program(t["program"])
        exs = t["train"] + t["test"]
        for a, b in zip(steps, steps[1:]):
            assert not (a[0].family and a[0].family == b[0].family), t["program"]
        for ex in exs:
            stages = [ex["input"], *ex["subgoals"], ex["output"]]
            assert all(x != y for x, y in zip(stages, stages[1:])), t["program"]  # every step changes the grid
            assert len(np.unique(ex["output"])) >= 2
        if t["depth"] >= 2:
            pairs = [(np.array(e["input"], dtype=np.uint8), np.array(e["output"], dtype=np.uint8)) for e in t["train"]]
            assert not C._single_reproduces(pairs), t["program"]


def test_detour_flag_and_rate(tasks):
    multi = [t for t in tasks if t["depth"] >= 2]
    for t in multi:
        exs = t["train"] + t["test"]
        stages = [
            [np.array(e["input"]) for e in exs],
            *([np.array(e["subgoals"][j]) for e in exs] for j in range(t["depth"] - 1)),
            [np.array(e["output"]) for e in exs],
        ]
        assert C.is_detour(stages) == t["detour"]
    assert sum(t["detour"] for t in multi) / len(multi) >= 0.5
    forced = C.generate(30, depth_range=(2, 3), seed=5, detour_frac=0.9)
    assert sum(t["detour"] for t in forced) / len(forced) >= 0.9


def test_depth_weights():
    ts = C.generate(12, depth_range=(1, 4), seed=2, depth_weights=(0, 0, 1, 0))
    assert {t["depth"] for t in ts} == {3}


def test_shortcut_filter_catches_shorter_programs():
    rng = np.random.default_rng(0)
    xs = [rng.integers(0, 10, size=(5, 7)).astype(np.uint8) for _ in range(4)]

    def stages_of(program: str) -> tuple[list, list]:
        steps = C.parse_program(program)
        chains = [C.run_program(steps, x) for x in xs]
        return steps, [list(s) for s in zip(*chains)]

    for program in ("flip_h>flip_v", "rot90>colour_swap(1,2)>rot270", "upscale_x2>transpose>downscale_x2"):
        steps, stages = stages_of(program)
        assert C._has_shortcut(rng, steps, stages, len(xs)), program
    steps, stages = stages_of("gravity('down',None)>tile_2x2")
    assert not C._has_shortcut(rng, steps, stages, len(xs))


def _example_input(prim: C.Primitive, rng: np.random.Generator) -> tuple[np.ndarray, tuple]:
    for _ in range(2000):
        grids = C._inputs(rng, 1, 3, 12)
        if grids is None:
            continue
        params = prim.sample(rng, grids)
        if params is not None and (out := prim(grids[0], params)) is not None and not C._eq(out, grids[0]):
            return grids[0], params
    raise AssertionError(f"no input found for {prim.name}")


@pytest.mark.parametrize("prim", C.PRIMITIVES, ids=lambda p: p.name)
def test_every_primitive_is_pure_and_valid(prim):
    rng = np.random.default_rng(sum(map(ord, prim.name)))
    g, params = _example_input(prim, rng)
    before = g.copy()
    out = prim(g, params)
    assert np.array_equal(g, before)
    assert out.dtype == np.uint8 and out.ndim == 2 and max(out.shape) <= 30
    assert out.min() >= 0 and out.max() <= 9
    assert np.array_equal(prim(g, params), out)
    assert C.parse_program(C.format_program([(prim, params)])) == [(prim, params)]
    if prim.space is not None:
        colours = C._colours([g, out])
        assert params in prim.space(colours)
        assert C._single_reproduces([(g, out)])
    else:
        assert prim.fit([(g, out)])


def test_primitive_semantics():
    P = C.BY_NAME
    g = np.array([[0, 3, 0], [0, 0, 0], [2, 0, 3]], dtype=np.uint8)
    assert P["gravity"](g, ("down", None)).tolist() == [[0, 0, 0], [0, 0, 0], [2, 3, 3]]
    assert np.array_equal(P["gravity"](g, ("down", 2)), g)  # colour 2 already rests on the floor
    assert P["gravity"](g, ("left", 3)).tolist() == [[3, 0, 0], [0, 0, 0], [2, 3, 0]]
    ring = np.zeros((5, 5), dtype=np.uint8)
    ring[1:4, 1:4] = 4
    ring[2, 2] = 0
    assert P["fill_enclosed"](ring, (7,))[2, 2] == 7
    solid = np.zeros((5, 5), dtype=np.uint8)
    solid[1:4, 1:4] = 4
    assert P["hollow"](solid, ()).tolist() == np.where(ring == 4, 4, 0).tolist()
    pair = np.array([[5, 0, 0, 5, 0]], dtype=np.uint8).repeat(3, axis=0)
    pair[1] = 0
    assert P["connect"](pair, ("h",))[0].tolist() == [5, 5, 5, 5, 0]
    assert P["extend_rays"](np.array([[0, 6, 0, 0]] * 2, dtype=np.uint8), ("right",))[0].tolist() == [0, 6, 6, 6]
    q = np.arange(16, dtype=np.uint8).reshape(4, 4) % 10
    rot = P["quad_rotate"](q, ("cw",))
    assert np.array_equal(rot[:2, 2:], q[:2, :2]) and np.array_equal(rot[2:, :2], q[2:, 2:])
    assert np.array_equal(P["quad_rotate"](rot, ("ccw",)), q)
    big = P["upscale_x2"](q, ())
    assert big.shape == (8, 8) and np.array_equal(P["downscale_x2"](big, ()), q)
    assert P["tile_2x2"](np.zeros((16, 16), dtype=np.uint8), ()) is None  # 32 > 30
    dot = np.zeros((4, 4), dtype=np.uint8)
    dot[0, 0], dot[3, 3] = 1, 2
    shifted = P["shift"](dot, (1, 1))
    assert shifted[1, 1] == 1 and (shifted == 2).sum() == 0  # leaving cells are dropped
    labels, sizes = C._components(np.array([[1, 1, 0, 2], [0, 1, 0, 2], [3, 0, 0, 0]], dtype=np.uint8), 0)
    assert sorted(sizes[1:].tolist()) == [1, 2, 3] and labels[0, 0] == labels[1, 1]
