"""Synthetic composite tasks f_k∘…∘f_1 with recorded intermediate grids (DESIGN §3.5).

k parameterised primitives applied to 8–12 random inputs; each example stores intermediates g_j = f_j(g_{j-1}).
Filters (rejected attempts are resampled): every step changes every example; no step returns to an earlier grid;
no consecutive steps share a family; outputs non-trivial and ≤ 30×30; no shorter program (single primitive on any
window of ≥ 2 steps, any one step dropped, sampled length-2 compositions for k ≥ 3) reproduces the demos.
Detour: some g_j is, on every example, not strictly Hamming-closer to the output than the input (shape mismatch =
not closer). Deterministic in `seed`.

Usage: python -m frm.data.composite --n 2000 --seed 0
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

Grid = np.ndarray  # 2-D uint8, values 0..9
Params = tuple
MAX_SIDE = 30
DIRECTIONS = ("up", "down", "left", "right")
VECTORS = tuple((dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if (dr, dc) != (0, 0))
SIZE_THRESHOLDS = (2, 3, 4, 5, 6)
STEP_TRIES = 60
MAX_ATTEMPTS = 20000


# ---------------------------------------------------------------------------------------------------------------
# grid helpers


def background(g: Grid) -> int:
    """Most frequent colour (lowest on ties)."""
    return int(np.bincount(g.ravel(), minlength=10).argmax())


def _colours(grids: Sequence[Grid]) -> list[int]:
    counts = np.zeros(10, dtype=np.int64)
    for g in grids:
        counts += np.bincount(g.ravel(), minlength=10)
    return [int(c) for c in np.flatnonzero(counts)]


def _eq(a: Grid, b: Grid) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _components(g: Grid, bg: int) -> tuple[np.ndarray, np.ndarray]:
    """4-connected single-colour components: labels (0 = background) and sizes; min-label pointer jumping."""
    h, w = g.shape
    fg = g != bg
    big = h * w
    lab = np.where(fg, np.arange(big).reshape(h, w), big)
    join_r = fg[:, 1:] & fg[:, :-1] & (g[:, 1:] == g[:, :-1])
    join_d = fg[1:] & fg[:-1] & (g[1:] == g[:-1])
    while True:
        new = lab.copy()
        new[:, 1:] = np.minimum(new[:, 1:], np.where(join_r, lab[:, :-1], big))
        new[:, :-1] = np.minimum(new[:, :-1], np.where(join_r, lab[:, 1:], big))
        new[1:] = np.minimum(new[1:], np.where(join_d, lab[:-1], big))
        new[:-1] = np.minimum(new[:-1], np.where(join_d, lab[1:], big))
        flat = new.ravel()
        inside = flat < big
        flat[inside] = flat[flat[inside]]
        if np.array_equal(new, lab):
            break
        lab = new
    roots = np.unique(lab[fg])
    labels = np.zeros((h, w), dtype=np.int64)
    labels[fg] = np.searchsorted(roots, lab[fg]) + 1
    return labels, np.bincount(labels.ravel(), minlength=len(roots) + 1)


def _oriented(g: Grid, direction: str, fn_left: Callable[[Grid], Grid | None]) -> Grid | None:
    """Apply `fn_left` (written for the leftward direction) in `direction` by rotating there and back."""
    k = {"left": 0, "up": 1, "right": 2, "down": 3}[direction]
    out = fn_left(np.rot90(g, k))
    return None if out is None else np.rot90(out, -k)


# ---------------------------------------------------------------------------------------------------------------
# primitives: pure grid -> grid | None (None = not applicable to this input)


def _swap_halves_lr(g: Grid) -> Grid | None:
    h, w = g.shape
    if w < 2:
        return None
    w2 = w // 2
    out = g.copy()
    out[:, :w2], out[:, w - w2:] = g[:, w - w2:], g[:, :w2]
    return out


def _swap_halves_ud(g: Grid) -> Grid | None:
    out = _swap_halves_lr(g.T)
    return None if out is None else out.T


def _quadrants(g: Grid) -> tuple[tuple[slice, slice], ...] | None:
    """TL, TR, BR, BL quadrant slices (a middle row/column of an odd side stays in place)."""
    h, w = g.shape
    if h < 2 or w < 2:
        return None
    h2, w2 = h // 2, w // 2
    top, bot = slice(0, h2), slice(h - h2, h)
    left, right = slice(0, w2), slice(w - w2, w)
    return (top, left), (top, right), (bot, right), (bot, left)


def _quad_swap_diag(g: Grid) -> Grid | None:
    q = _quadrants(g)
    if q is None:
        return None
    out = g.copy()
    out[q[0]], out[q[2]] = g[q[2]], g[q[0]]
    out[q[1]], out[q[3]] = g[q[3]], g[q[1]]
    return out


def _quad_rotate(g: Grid, sense: str) -> Grid | None:
    """Move whole quadrants one place clockwise (TL→TR→BR→BL) or counter-clockwise; contents keep orientation."""
    q = _quadrants(g)
    if q is None:
        return None
    step = 1 if sense == "cw" else -1
    out = g.copy()
    for i in range(4):
        out[q[(i + step) % 4]] = g[q[i]]
    return out


def _gravity_left(g: Grid, colour: int | None) -> Grid | None:
    bg = background(g)
    if colour == bg:
        return None
    movable = (g != bg) if colour is None else (g == colour)
    obstacle = (g != bg) & ~movable
    seg = np.cumsum(obstacle, axis=1)
    key = 3 * seg + np.where(obstacle, 0, np.where(movable, 1, 2))
    return np.take_along_axis(g, np.argsort(key, axis=1, kind="stable"), axis=1)


def _gravity(g: Grid, direction: str, colour: int | None) -> Grid | None:
    """Slide movable cells (all foreground, or one colour) towards `direction`; other colours are obstacles."""
    return _oriented(g, direction, lambda a: _gravity_left(a, colour))


def _colour_swap(g: Grid, a: int, b: int) -> Grid:
    out = g.copy()
    out[g == a] = b
    out[g == b] = a
    return out


def _colour_map(g: Grid, table: str) -> Grid:
    return np.frombuffer(table.encode(), dtype=np.uint8)[g] - ord("0")


def _fits_colour_map(pairs: Sequence[tuple[Grid, Grid]]) -> bool:
    if any(x.shape != y.shape for x, y in pairs):
        return False
    codes = np.unique(np.concatenate([x.ravel().astype(np.int64) * 10 + y.ravel() for x, y in pairs]))
    return len(np.unique(codes // 10)) == len(codes)


def _invert_2colour(g: Grid) -> Grid | None:
    present = np.flatnonzero(np.bincount(g.ravel(), minlength=10))
    if len(present) != 2:
        return None
    return _colour_swap(g, int(present[0]), int(present[1]))


def _unique_largest(sizes: np.ndarray) -> int | None:
    if len(sizes) < 2:
        return None
    order = np.argsort(sizes[1:])[::-1] + 1
    if len(order) > 1 and sizes[order[0]] == sizes[order[1]]:
        return None
    return int(order[0])


def _recolour_largest(g: Grid, c: int) -> Grid | None:
    labels, sizes = _components(g, background(g))
    top = _unique_largest(sizes)
    if top is None:
        return None
    out = g.copy()
    out[labels == top] = c
    return out


def _recolour_by_size(g: Grid, t: int, c: int) -> Grid | None:
    """Objects with at least `t` cells take colour `c`."""
    labels, sizes = _components(g, background(g))
    big = (sizes >= t)[labels] & (labels > 0)
    if not big.any():
        return None
    out = g.copy()
    out[big] = c
    return out


def _keep_largest(g: Grid) -> Grid | None:
    bg = background(g)
    labels, sizes = _components(g, bg)
    top = _unique_largest(sizes)
    if top is None:
        return None
    return np.where((labels == top) | (labels == 0), g, bg).astype(np.uint8)


def _remove_smallest(g: Grid) -> Grid | None:
    bg = background(g)
    labels, sizes = _components(g, bg)
    if len(sizes) < 2:
        return None
    small = (sizes == sizes[1:].min())[labels] & (labels > 0)
    if small.sum() == (labels > 0).sum():
        return None
    return np.where(small, bg, g).astype(np.uint8)


def _hollow(g: Grid) -> Grid | None:
    """Clear object interiors: foreground cells whose four neighbours all have their colour."""
    bg = background(g)
    inner = np.zeros(g.shape, dtype=bool)
    m = g[1:-1, 1:-1]
    inner[1:-1, 1:-1] = (
        (m != bg) & (m == g[:-2, 1:-1]) & (m == g[2:, 1:-1]) & (m == g[1:-1, :-2]) & (m == g[1:-1, 2:])
    )
    if not inner.any():
        return None
    return np.where(inner, bg, g).astype(np.uint8)


def _outline(g: Grid, c: int) -> Grid | None:
    """Background cells 8-adjacent to foreground take colour `c`."""
    bg = background(g)
    fg = np.pad(g != bg, 1)
    h, w = g.shape
    near = np.zeros((h, w), dtype=bool)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            near |= fg[dr:dr + h, dc:dc + w]
    ring = near & (g == bg)
    if not ring.any():
        return None
    out = g.copy()
    out[ring] = c
    return out


def _fill_enclosed(g: Grid, c: int) -> Grid | None:
    """Background cells not 4-connected to the border through background take colour `c`."""
    free = g == background(g)
    reach = np.zeros_like(free)
    reach[[0, -1], :] = free[[0, -1], :]
    reach[:, [0, -1]] |= free[:, [0, -1]]
    while True:
        new = reach.copy()
        new[1:] |= reach[:-1]
        new[:-1] |= reach[1:]
        new[:, 1:] |= reach[:, :-1]
        new[:, :-1] |= reach[:, 1:]
        new &= free
        if np.array_equal(new, reach):
            break
        reach = new
    hole = free & ~reach
    if not hole.any():
        return None
    out = g.copy()
    out[hole] = c
    return out


def _crop_content(g: Grid) -> Grid | None:
    rows, cols = np.nonzero(g != background(g))
    if len(rows) == 0:
        return None
    return g[rows.min():rows.max() + 1, cols.min():cols.max() + 1].copy()


def _downscale_x2(g: Grid) -> Grid | None:
    h, w = g.shape
    if h % 2 or w % 2:
        return None
    small = g[::2, ::2]
    return small.copy() if np.array_equal(np.kron(small, np.ones((2, 2), dtype=np.uint8)), g) else None


def _border(g: Grid, c: int) -> Grid:
    out = g.copy()
    out[[0, -1], :] = c
    out[:, [0, -1]] = c
    return out


def _shift(g: Grid, dr: int, dc: int) -> Grid | None:
    """Translate the foreground by (dr, dc); cells leaving the grid are dropped, rejected when none remain."""
    bg = background(g)
    rows, cols = np.nonzero(g != bg)
    nr, nc = rows + dr, cols + dc
    h, w = g.shape
    keep = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
    if not keep.any():
        return None
    out = np.full_like(g, bg)
    out[nr[keep], nc[keep]] = g[rows[keep], cols[keep]]
    return out


def _rays_left(g: Grid) -> Grid:
    """Each foreground cell paints the background to its left until the next foreground cell or the edge."""
    bg = background(g)
    w = g.shape[1]
    idx = np.where(g != bg, np.arange(w), w)
    nearest = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
    src = np.take_along_axis(g, np.minimum(nearest, w - 1), axis=1)
    return np.where(nearest < w, src, g).astype(np.uint8)


def _connect_rows(g: Grid) -> Grid | None:
    """Background runs between two equal-coloured cells of one row take that colour."""
    bg = background(g)
    w = g.shape[1]
    fg = g != bg
    cols = np.arange(w)
    left = np.maximum.accumulate(np.where(fg, cols, -1), axis=1)
    right = np.minimum.accumulate(np.where(fg, cols, w)[:, ::-1], axis=1)[:, ::-1]
    lc = np.take_along_axis(g, np.maximum(left, 0), axis=1)
    rc = np.take_along_axis(g, np.minimum(right, w - 1), axis=1)
    fill = ~fg & (left >= 0) & (right < w) & (lc == rc)
    return np.where(fill, lc, g).astype(np.uint8)


def _connect(g: Grid, axis: str) -> Grid | None:
    out = _connect_rows(g if axis == "h" else g.T)
    return out if axis == "h" else out.T


def _symmetrize(g: Grid, axis: str) -> Grid:
    """Overlay the mirror image onto the background (axis 'h' mirrors left-right, 'v' top-bottom)."""
    return np.where(g == background(g), np.fliplr(g) if axis == "h" else np.flipud(g), g).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------------
# parameter samplers (per task, from the grids a step sees) and enumerable spaces (from the colours of a pair)


def _no_params(rng: np.random.Generator, grids: Sequence[Grid]) -> Params:
    return ()


def _one_setting(colours: Sequence[int]) -> list[Params]:
    return [()]


def _pick_colour(rng: np.random.Generator, grids: Sequence[Grid]) -> int:
    bg = background(grids[0])
    return int(rng.choice([c for c in range(10) if c != bg]))


def _colour_param(rng: np.random.Generator, grids: Sequence[Grid]) -> Params:
    return (_pick_colour(rng, grids),)


def _colour_space(colours: Sequence[int]) -> list[Params]:
    return [(c,) for c in colours]


def _choice_param(options: Sequence) -> Callable[[np.random.Generator, Sequence[Grid]], Params]:
    return lambda rng, grids: (options[int(rng.integers(len(options)))],)


def _choice_space(options: Sequence) -> Callable[[Sequence[int]], list[Params]]:
    return lambda colours: [(o,) for o in options]


def _gravity_param(rng: np.random.Generator, grids: Sequence[Grid]) -> Params:
    direction = DIRECTIONS[int(rng.integers(4))]
    if rng.random() < 0.5:
        return direction, None
    bg = background(grids[0])
    fg = [c for c in _colours(grids) if c != bg]
    return direction, (int(rng.choice(fg)) if fg else None)


def _gravity_space(colours: Sequence[int]) -> list[Params]:
    return [(d, c) for d in DIRECTIONS for c in (None, *colours)]


def _swap_param(rng: np.random.Generator, grids: Sequence[Grid]) -> Params | None:
    present = _colours(grids)
    if len(present) < 2:
        return None
    a, b = sorted(int(c) for c in rng.choice(present, size=2, replace=False))
    return a, b


def _swap_space(colours: Sequence[int]) -> list[Params]:
    return list(combinations(colours, 2))


def _map_param(rng: np.random.Generator, grids: Sequence[Grid]) -> Params:
    bg = background(grids[0])
    others = np.array([c for c in range(10) if c != bg])
    table = np.arange(10)
    table[others] = rng.permutation(others)
    return ("".join(str(int(c)) for c in table),)


def _size_param(rng: np.random.Generator, grids: Sequence[Grid]) -> Params:
    return int(rng.choice(SIZE_THRESHOLDS)), _pick_colour(rng, grids)


def _size_space(colours: Sequence[int]) -> list[Params]:
    return [(t, c) for t in SIZE_THRESHOLDS for c in colours]


def _vector_param(rng: np.random.Generator, grids: Sequence[Grid]) -> Params:
    return VECTORS[int(rng.integers(len(VECTORS)))]


def _vector_space(colours: Sequence[int]) -> list[Params]:
    return list(VECTORS)


# ---------------------------------------------------------------------------------------------------------------
# library


@dataclass(frozen=True)
class Primitive:
    """Parameterised grid transform. `space` enumerates settings for the shortcut search (`fit` when too large);
    consecutive steps never share a non-empty `family`."""

    name: str
    fn: Callable[..., Grid | None]
    sample: Callable[[np.random.Generator, Sequence[Grid]], Params | None]
    space: Callable[[Sequence[int]], list[Params]] | None
    family: str = ""
    fit: Callable[[Sequence[tuple[Grid, Grid]]], bool] | None = None

    def __call__(self, g: Grid, params: Params) -> Grid | None:
        """Apply with `params`; None when inapplicable, empty, or larger than 30×30."""
        out = self.fn(g, *params)
        if out is None or out.size == 0 or max(out.shape) > MAX_SIDE:
            return None
        return np.ascontiguousarray(out, dtype=np.uint8)


def _plain(name: str, fn: Callable[[Grid], Grid | None], family: str = "") -> Primitive:
    return Primitive(name, fn, _no_params, _one_setting, family)


PRIMITIVES: tuple[Primitive, ...] = (
    _plain("rot90", lambda g: np.rot90(g, 1), "dihedral"),
    _plain("rot180", lambda g: np.rot90(g, 2), "dihedral"),
    _plain("rot270", lambda g: np.rot90(g, 3), "dihedral"),
    _plain("flip_h", np.fliplr, "dihedral"),
    _plain("flip_v", np.flipud, "dihedral"),
    _plain("transpose", np.transpose, "dihedral"),
    _plain("swap_halves_lr", _swap_halves_lr, "blocks"),
    _plain("swap_halves_ud", _swap_halves_ud, "blocks"),
    _plain("quad_swap_diag", _quad_swap_diag, "blocks"),
    Primitive("quad_rotate", _quad_rotate, _choice_param(("cw", "ccw")), _choice_space(("cw", "ccw")), "blocks"),
    Primitive("gravity", _gravity, _gravity_param, _gravity_space),
    Primitive("colour_swap", _colour_swap, _swap_param, _swap_space, "colour"),
    Primitive("colour_map", _colour_map, _map_param, None, "colour", _fits_colour_map),
    _plain("invert_2colour", _invert_2colour, "colour"),
    Primitive("recolour_largest", _recolour_largest, _colour_param, _colour_space),
    Primitive("recolour_by_size", _recolour_by_size, _size_param, _size_space),
    _plain("keep_largest", _keep_largest),
    _plain("remove_smallest", _remove_smallest),
    _plain("hollow", _hollow),
    Primitive("outline", _outline, _colour_param, _colour_space),
    Primitive("fill_enclosed", _fill_enclosed, _colour_param, _colour_space),
    _plain("crop_content", _crop_content),
    _plain("tile_2x2", lambda g: np.tile(g, (2, 2))),
    _plain("mirror_concat_h", lambda g: np.hstack([g, np.fliplr(g)])),
    _plain("mirror_concat_v", lambda g: np.vstack([g, np.flipud(g)])),
    _plain("upscale_x2", lambda g: np.kron(g, np.ones((2, 2), dtype=np.uint8))),
    _plain("downscale_x2", _downscale_x2),
    Primitive("border", _border, _colour_param, _colour_space),
    Primitive("roll", lambda g, dr, dc: np.roll(g, (dr, dc), axis=(0, 1)), _vector_param, _vector_space, "translate"),
    Primitive("shift", _shift, _vector_param, _vector_space, "translate"),
    Primitive("extend_rays", lambda g, d: _oriented(g, d, _rays_left), _choice_param(DIRECTIONS),
              _choice_space(DIRECTIONS)),
    Primitive("connect", _connect, _choice_param(("h", "v")), _choice_space(("h", "v"))),
    Primitive("symmetrize", _symmetrize, _choice_param(("h", "v")), _choice_space(("h", "v"))),
)
BY_NAME: dict[str, Primitive] = {p.name: p for p in PRIMITIVES}
# Structure-dependent primitives are often infeasible, so they are proposed more often.
SAMPLING_WEIGHT = {
    "hollow": 12.0, "fill_enclosed": 12.0, "downscale_x2": 6.0, "connect": 4.0, "keep_largest": 4.0,
    "crop_content": 3.0, "recolour_largest": 3.0, "remove_smallest": 3.0, "recolour_by_size": 2.0, "gravity": 2.0,
}
_DRAW_W = np.array([SAMPLING_WEIGHT.get(p.name, 1.0) for p in PRIMITIVES])
_DRAW_P = _DRAW_W / _DRAW_W.sum()

Step = tuple[Primitive, Params]


def format_program(steps: Sequence[Step]) -> str:
    return ">".join(p.name + (f"({','.join(map(repr, a))})" if a else "") for p, a in steps)


def parse_program(s: str) -> list[Step]:
    steps = []
    for part in s.split(">"):
        m = re.fullmatch(r"(\w+)(?:\((.*)\))?", part)
        if m is None:
            raise ValueError(f"bad program step {part!r}")
        steps.append((BY_NAME[m[1]], tuple(ast.literal_eval(f"({m[2]},)")) if m[2] else ()))
    return steps


def run_program(steps: Sequence[Step], g: Grid) -> list[Grid] | None:
    """All grids of the chain [x, f_1(x), …, f_k(…)], or None if some step is inapplicable."""
    chain = [g]
    for prim, params in steps:
        nxt = prim(chain[-1], params)
        if nxt is None:
            return None
        chain.append(nxt)
    return chain


# ---------------------------------------------------------------------------------------------------------------
# input generator


KINDS = ("rect", "line", "blob", "sprite", "pair")
PROFILES = (  # task-level object mixes: probabilities over KINDS
    (0.3, 0.15, 0.25, 0.2, 0.1),
    (1.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.4, 0.0, 0.0, 0.6),
    (0.2, 0.0, 0.8, 0.0, 0.0),
    (0.0, 0.0, 0.2, 0.8, 0.0),
)
PROFILE_P = (0.35, 0.3, 0.1, 0.1, 0.15)


@dataclass(frozen=True)
class _Look:
    """Task-level object statistics, shared by every input so that object-wise rules apply to all of them."""

    kind_p: tuple[float, ...]
    rect_min: int
    hollow_p: float
    n_obj: tuple[int, int]


def _sample_look(rng: np.random.Generator) -> _Look:
    lo = 1 if rng.random() < 0.3 else 2
    return _Look(
        kind_p=PROFILES[int(rng.choice(len(PROFILES), p=PROFILE_P))],
        rect_min=3 if rng.random() < 0.5 else 1,
        hollow_p=float(rng.choice((0.0, 1.0))),
        n_obj=(lo, int(rng.integers(lo, 7))),
    )


def _object_patch(rng: np.random.Generator, look: _Look, palette: Sequence[int], h: int, w: int) -> np.ndarray:
    """A colour patch with -1 for transparent cells: rectangle, line, blob, sprite, or two cells with a gap."""
    kind = KINDS[int(rng.choice(len(KINDS), p=look.kind_p))]
    colour = int(rng.choice(palette))
    if kind == "rect":
        lo = min(look.rect_min, h, w)
        ph, pw = int(rng.integers(lo, min(6, h) + 1)), int(rng.integers(lo, min(6, w) + 1))
        patch = np.full((ph, pw), colour, dtype=np.int16)
        if ph >= 3 and pw >= 3 and rng.random() < look.hollow_p:
            patch[1:-1, 1:-1] = -1
        return patch
    if kind in ("line", "pair"):
        vertical = rng.random() < 0.5
        side = h if vertical else w
        lo = 3 if kind == "pair" else 2
        n = int(rng.integers(lo, min(8, side) + 1)) if side >= lo else 1
        patch = np.full(n, colour, dtype=np.int16)
        if kind == "pair" and n >= 3:
            patch[1:-1] = -1
        return patch.reshape((n, 1) if vertical else (1, n))
    ph, pw = int(rng.integers(2, min(5, h) + 1)) if h >= 2 else 1, int(rng.integers(2, min(5, w) + 1)) if w >= 2 else 1
    if kind == "blob":
        mask = np.zeros((ph, pw), dtype=bool)
        r, c = ph // 2, pw // 2
        mask[r, c] = True
        for _ in range(int(rng.integers(2, ph * pw + 1))):
            dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            r, c = min(ph - 1, max(0, r + dr)), min(pw - 1, max(0, c + dc))
            mask[r, c] = True
        return np.where(mask, colour, -1).astype(np.int16)
    colours = rng.choice(palette, size=(ph, pw))
    mask = rng.random((ph, pw)) < 0.65
    mask[ph // 2, pw // 2] = True
    return np.where(mask, colours, -1).astype(np.int16)


def _scene(rng: np.random.Generator, h: int, w: int, bg: int, palette: Sequence[int], look: _Look, n_obj: int) -> Grid:
    """Background plus up to `n_obj` objects kept one cell apart, so each is its own component."""
    g = np.full((h, w), bg, dtype=np.uint8)
    taken = np.zeros((h, w), dtype=bool)
    for _ in range(n_obj):
        patch = _object_patch(rng, look, palette, h, w)
        ph, pw = patch.shape
        if ph > h or pw > w:
            continue
        for _ in range(10):
            r, c = int(rng.integers(0, h - ph + 1)), int(rng.integers(0, w - pw + 1))
            if taken[max(0, r - 1):r + ph + 1, max(0, c - 1):c + pw + 1].any():
                continue
            solid = patch >= 0
            g[r:r + ph, c:c + pw][solid] = patch[solid]
            taken[r:r + ph, c:c + pw] = True
            break
    return g


def _panels(rng: np.random.Generator, h: int, w: int, bg: int, palette: Sequence[int], look: _Look) -> Grid:
    """Even-sided grid whose four quadrants are filled independently."""
    h2, w2 = h // 2, w // 2
    g = np.full((2 * h2, 2 * w2), bg, dtype=np.uint8)
    for r in (0, h2):
        for c in (0, w2):
            g[r:r + h2, c:c + w2] = _scene(rng, h2, w2, bg, palette, look, int(rng.integers(1, 3)))
    return g


def _sparse(rng: np.random.Generator, h: int, w: int, bg: int, palette: Sequence[int]) -> Grid:
    g = np.full((h, w), bg, dtype=np.uint8)
    hit = rng.random((h, w)) < rng.uniform(0.05, 0.2)
    g[hit] = rng.choice(palette, size=int(hit.sum()))
    return g


STYLES = ("objects", "panels", "blocky", "sparse", "two_colour")
STYLE_P = (0.4, 0.2, 0.1, 0.1, 0.2)


def _inputs(rng: np.random.Generator, n: int, min_side: int, max_side: int) -> list[Grid] | None:
    """`n` distinct inputs of one task-level style, look, background and palette; None when duplicates arise."""
    style = str(rng.choice(STYLES, p=STYLE_P))
    look = _sample_look(rng)
    bg = 0 if rng.random() < 0.8 else int(rng.integers(1, 10))
    others = [c for c in range(10) if c != bg]
    n_pal = 1 if style == "two_colour" else int(rng.integers(1, 5))
    palette = [int(c) for c in rng.choice(others, size=n_pal, replace=False)]
    lo = max(min_side, 4) if style in ("panels", "blocky") else min_side
    if look.rect_min >= 3:  # room for rectangles with an interior
        lo = max(lo, min(5, max_side))
    hi = int(rng.integers(max(lo, 6), max_side + 1)) if max_side > max(lo, 6) else max_side
    fixed = (int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1))) if rng.random() < 0.5 else None
    grids: list[Grid] = []
    for _ in range(n):
        h, w = fixed or (int(rng.integers(lo, hi + 1)), int(rng.integers(lo, hi + 1)))
        n_obj = int(rng.integers(look.n_obj[0], look.n_obj[1] + 1))
        if style == "panels":
            g = _panels(rng, h, w, bg, palette, look)
        elif style == "blocky":
            g = np.kron(_scene(rng, h // 2, w // 2, bg, palette, look, n_obj), np.ones((2, 2), dtype=np.uint8))
        elif style == "sparse":
            g = _sparse(rng, h, w, bg, palette)
        else:
            g = _scene(rng, h, w, bg, palette, look, n_obj)
        if not (g != bg).any():
            g[int(rng.integers(g.shape[0])), int(rng.integers(g.shape[1]))] = palette[0]
        grids.append(g)
    keys = {(g.shape, g.tobytes()) for g in grids}
    return grids if len(keys) == n else None


# ---------------------------------------------------------------------------------------------------------------
# program sampling and filters


def _sample_program(rng: np.random.Generator, inputs: list[Grid], k: int) -> tuple[list[Step], list[list[Grid]]] | None:
    """Greedy chain of k steps, each changing every example; returns steps and the stages [inputs, g_1, …, outputs]."""
    stages = [inputs]
    steps: list[Step] = []
    for _ in range(k):
        for _ in range(STEP_TRIES):
            prim = PRIMITIVES[int(rng.choice(len(PRIMITIVES), p=_DRAW_P))]
            if steps and prim.family and prim.family == steps[-1][0].family:
                continue
            params = prim.sample(rng, stages[-1])
            if params is None:
                continue
            outs = []
            for g in stages[-1]:
                o = prim(g, params)
                if o is None or _eq(o, g):
                    break
                outs.append(o)
            if len(outs) < len(inputs):
                continue
            if any(all(_eq(o, s) for o, s in zip(outs, stage)) for stage in stages[:-1]):
                continue
            steps.append((prim, params))
            stages.append(outs)
            break
        else:
            return None
    return steps, stages


def _settings(colours: Sequence[int]) -> list[Step]:
    return [(p, a) for p in PRIMITIVES if p.space is not None for a in p.space(colours)]


def _reproduces(step: Step, pairs: Sequence[tuple[Grid, Grid]]) -> bool:
    prim, params = step
    for x, y in pairs:
        out = prim(x, params)
        if out is None or not _eq(out, y):
            return False
    return True


def _single_reproduces(pairs: Sequence[tuple[Grid, Grid]]) -> bool:
    """Whether any primitive, under any of its settings, maps every x to its y."""
    colours = _colours([g for pair in pairs for g in pair])
    if any(_reproduces(s, pairs) for s in _settings(colours)):
        return True
    return any(p.fit(pairs) for p in PRIMITIVES if p.fit is not None)


def _pair_reproduces(rng: np.random.Generator, pairs: Sequence[tuple[Grid, Grid]], n_first: int, n_second: int) -> bool:
    """Sampled length-2 compositions: n_first first steps, each followed by n_second sampled steps and every fit."""
    settings = _settings(_colours([g for pair in pairs for g in pair]))
    fits = [p for p in PRIMITIVES if p.fit is not None]
    x0, y0 = pairs[0]
    for i in rng.choice(len(settings), size=min(n_first, len(settings)), replace=False):
        first = settings[int(i)]
        z0 = first[0](x0, first[1])
        if z0 is None:
            continue
        picks = rng.choice(len(settings), size=min(n_second, len(settings)), replace=False)
        seconds = [settings[int(j)] for j in picks]
        hits = [s for s in seconds if (o := s[0](z0, s[1])) is not None and _eq(o, y0)]
        fit_hits = [p for p in fits if p.fit([(z0, y0)])]
        if not hits and not fit_hits:
            continue
        mids = [first[0](x, first[1]) for x, _ in pairs]
        if any(m is None for m in mids):
            continue
        mid_pairs = [(m, y) for m, (_, y) in zip(mids, pairs)]
        if any(_reproduces(s, mid_pairs) for s in hits) or any(p.fit(mid_pairs) for p in fit_hits):
            return True
    return False


def _has_shortcut(rng: np.random.Generator, steps: Sequence[Step], stages: Sequence[list[Grid]], n_train: int) -> bool:
    """Whether a program shorter than `steps` reproduces the demos (see module docstring)."""
    k = len(steps)
    xs, ys = stages[0][:n_train], stages[-1][:n_train]
    for length in range(k, 1, -1):
        for i in range(k - length + 1):
            if _single_reproduces(list(zip(stages[i][:n_train], stages[i + length][:n_train]))):
                return True
    if k >= 3:
        for i in range(k):
            rest = [*steps[:i], *steps[i + 1:]]
            if all((c := run_program(rest, x)) is not None and _eq(c[-1], y) for x, y in zip(xs, ys)):
                return True
        if _pair_reproduces(rng, list(zip(xs, ys)), n_first=24, n_second=8):
            return True
    return False


def _hamming(a: Grid, b: Grid) -> float:
    return float((a != b).sum()) if a.shape == b.shape else float("inf")


def is_detour(stages: Sequence[Sequence[Grid]]) -> bool:
    """Some intermediate is, on every example, not strictly closer to the output than the input is."""
    xs, ys = stages[0], stages[-1]
    return any(
        all(_hamming(g, y) >= _hamming(x, y) for g, x, y in zip(stage, xs, ys)) for stage in stages[1:-1]
    )


def _attempt(
    rng: np.random.Generator, k: int, n_train: int, n_test: int, min_side: int, max_side: int, need_detour: bool
) -> tuple[list[Step], list[list[Grid]], bool] | None:
    inputs = _inputs(rng, n_train + n_test, min_side, max_side)
    if inputs is None:
        return None
    built = _sample_program(rng, inputs, k)
    if built is None:
        return None
    steps, stages = built
    outs = stages[-1]
    if any(_eq(x, y) for x, y in zip(inputs, outs)):
        return None
    if all(_eq(outs[0], y) for y in outs[1:]) or any(len(_colours([y])) < 2 for y in outs):
        return None
    detour = k >= 2 and is_detour(stages)
    if need_detour and not detour:
        return None
    if k >= 2 and _has_shortcut(rng, steps, stages, n_train):
        return None
    return steps, stages, detour


def _make_task(
    rng: np.random.Generator,
    k: int,
    n_train: tuple[int, int],
    n_test: int,
    min_side: int,
    max_side: int,
    need_detour: bool,
) -> tuple[list[Step], list[list[Grid]], bool, int]:
    n_tr = int(rng.integers(n_train[0], n_train[1] + 1))
    for _ in range(MAX_ATTEMPTS):
        got = _attempt(rng, k, n_tr, n_test, min_side, max_side, need_detour)
        if got is not None:
            return (*got, n_tr)
    raise RuntimeError(f"no depth-{k} task accepted in {MAX_ATTEMPTS} attempts")


def _example(stages: Sequence[list[Grid]], e: int) -> dict:
    return {
        "input": stages[0][e].tolist(),
        "output": stages[-1][e].tolist(),
        "subgoals": [stage[e].tolist() for stage in stages[1:-1]],
    }


def generate(
    n_tasks: int,
    depth_range: tuple[int, int] = (1, 4),
    seed: int = 0,
    n_train: tuple[int, int] = (6, 10),
    n_test: int = 2,
    max_side: int = 20,
    detour_frac: float = 0.5,
    depth_weights: Sequence[float] | None = None,
    min_side: int = 3,
) -> list[dict]:
    """`n_tasks` composite tasks; deterministic in `seed`. At least `detour_frac` of k ≥ 2 tasks are detours."""
    depths = np.arange(depth_range[0], depth_range[1] + 1)
    p = None if depth_weights is None else np.asarray(depth_weights, dtype=float) / np.sum(depth_weights)
    rng = np.random.default_rng(seed)
    tasks = []
    n_multi = n_detour = 0
    for i in range(n_tasks):
        k = int(rng.choice(depths, p=p))
        need = k >= 2 and n_detour < detour_frac * (n_multi + 1)
        steps, stages, detour, n_tr = _make_task(rng, k, n_train, n_test, min_side, max_side, need)
        if k >= 2:
            n_multi += 1
            n_detour += detour
        n_ex = len(stages[0])
        tasks.append({
            "id": f"comp{seed}_{i}",
            "program": format_program(steps),
            "depth": k,
            "detour": bool(detour),
            "train": [_example(stages, e) for e in range(n_tr)],
            "test": [_example(stages, e) for e in range(n_tr, n_ex)],
        })
    return tasks


def summarise(tasks: Sequence[dict]) -> dict:
    multi = [t for t in tasks if t["depth"] >= 2]
    usage = Counter(step.split("(")[0] for t in tasks for step in t["program"].split(">"))
    return {
        "per_depth": dict(sorted(Counter(t["depth"] for t in tasks).items())),
        "detour_rate": sum(t["detour"] for t in multi) / len(multi) if multi else 0.0,
        "usage": {p.name: usage.get(p.name, 0) for p in PRIMITIVES},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None, help="default data/raw/composite_s{seed}.json")
    ap.add_argument("--depth_min", type=int, default=1)
    ap.add_argument("--depth_max", type=int, default=4)
    ap.add_argument("--max_side", type=int, default=20)
    ap.add_argument("--detour_frac", type=float, default=0.5)
    args = ap.parse_args()
    t0 = time.perf_counter()
    tasks = generate(args.n, (args.depth_min, args.depth_max), args.seed, max_side=args.max_side,
                     detour_frac=args.detour_frac)
    elapsed = time.perf_counter() - t0
    out = Path(args.out or f"data/raw/composite_s{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tasks, separators=(",", ":")))
    s = summarise(tasks)
    print(f"wrote {len(tasks)} tasks to {out} in {elapsed:.1f}s")
    print("per depth: " + ", ".join(f"k={k}: {n}" for k, n in s["per_depth"].items()))
    print(f"detour rate (k>=2): {s['detour_rate']:.3f}")
    print("primitive usage:")
    for name, n in sorted(s["usage"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:18s} {n}")


if __name__ == "__main__":
    main()
