"""Token layout (TRM-compatible): PAD=0, EOS=1, colour c -> c+2 on an S x S canvas."""

from __future__ import annotations

import numpy as np

PAD = 0
EOS = 1
IGNORE = -100
VOCAB = 12

# dihedral id -> (transpose, flip rows, flip cols), so that dihedral(g, tid) == flips(transpose?(g))
_DIHEDRAL_PARTS = np.array(
    [[0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool
)


def encode_grid(grid: np.ndarray, S: int, offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Write ``grid`` at ``offset`` with its EOS row/column (dropped where they leave the canvas); -> uint8[S*S]."""
    grid = np.asarray(grid, dtype=np.uint8)
    h, w = grid.shape
    r0, c0 = offset
    if r0 < 0 or c0 < 0 or r0 + h > S or c0 + w > S:
        raise ValueError(f"grid {h}x{w} at {offset} does not fit canvas {S}")
    canvas = np.zeros((S, S), dtype=np.uint8)
    canvas[r0 : r0 + h, c0 : c0 + w] = grid + 2
    if r0 + h < S:
        canvas[r0 + h, c0 : c0 + w] = EOS
    if c0 + w < S:
        canvas[r0 : r0 + h, c0 + w] = EOS
    return canvas.reshape(-1)


def labels_from(seq: np.ndarray) -> np.ndarray:
    """Token sequence -> int16 labels with PAD ignored."""
    seq = np.asarray(seq)
    return np.where(seq == PAD, np.int16(IGNORE), seq.astype(np.int16))


def crop(seq: np.ndarray, S: int) -> np.ndarray:
    """TRM ``_crop`` on an S x S canvas: largest top-left rectangle of colour tokens, minus 2."""
    canvas = np.asarray(seq).reshape(S, S)
    colour = (canvas >= 2) & (canvas <= 11)
    # run[r] = length of the leading colour run of row r, capped by the runs of the rows above it
    run = np.where(colour.all(axis=1), S, np.argmin(colour, axis=1))
    run = np.minimum.accumulate(run)
    area = run * np.arange(1, S + 1)
    best = int(np.argmax(area))
    if area[best] == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    return (canvas[: best + 1, : run[best]] - 2).astype(np.uint8)


def canvas_stack(grids: list[np.ndarray], S: int) -> tuple[np.ndarray, np.ndarray]:
    """Base grids -> (canvases uint8[2, E, S, S] = raw colours top-left, plain and transposed; dims uint8[E, 2])."""
    base = np.zeros((len(grids), S, S), dtype=np.uint8)
    hw = np.zeros((len(grids), 2), dtype=np.uint8)
    for e, g in enumerate(grids):
        h, w = g.shape
        base[e, :h, :w] = g
        hw[e] = (h, w)
    return np.stack([base, base.transpose(0, 2, 1)]), hw


def augmented_dims(hw: np.ndarray, tid: np.ndarray) -> np.ndarray:
    """Dims [n, 2] of the grids after dihedral ``tid`` [n]."""
    swap = _DIHEDRAL_PARTS[tid, 0]
    return np.where(swap[:, None], hw[:, ::-1], hw)


def render(
    canvases: np.ndarray, ex: np.ndarray, hw: np.ndarray, tid: np.ndarray, perm: np.ndarray, offset: np.ndarray
) -> np.ndarray:
    """Vectorised ``encode_grid(Version(tid, perm).apply(grid[ex]), S, offset)`` for n rows at once.

    ``canvases``/``hw`` from :func:`canvas_stack`; ``ex`` [n] grid indices; ``hw`` [n, 2] base dims of those grids;
    ``tid`` [n]; ``perm`` uint8 [n, 10]; ``offset`` [n, 2]. Returns uint8 [n, S*S].
    """
    n, S = len(ex), canvases.shape[-1]
    parts = _DIHEDRAL_PARTS[tid]
    dims = augmented_dims(hw, tid).astype(np.int16)  # (h', w')
    ar = np.arange(S, dtype=np.int16)
    i = ar[None, :] - offset[:, :1].astype(np.int16)  # [n, S] augmented row of each canvas row
    j = ar[None, :] - offset[:, 1:].astype(np.int16)
    hp, wp = dims[:, :1], dims[:, 1:]
    valid_r = (i >= 0) & (i < hp)
    valid_c = (j >= 0) & (j < wp)
    ri = np.where(parts[:, 1:2], hp - 1 - i, i).clip(0, S - 1).astype(np.int32)
    ci = np.where(parts[:, 2:3], wp - 1 - j, j).clip(0, S - 1).astype(np.int32)
    E = canvases.shape[1]
    base = ((parts[:, 0] * E + ex) * (S * S)).astype(np.int32)
    colours = canvases.reshape(-1).take(base[:, None, None] + (ri * S)[:, :, None] + ci[:, None, :])
    lut = (perm + np.uint8(2)).reshape(-1)
    tokens = lut.take(np.arange(0, 10 * n, 10, dtype=np.int32)[:, None, None] + colours)
    inside = valid_r[:, :, None] & valid_c[:, None, :]
    eos = ((i == hp)[:, :, None] & valid_c[:, None, :]) | (valid_r[:, :, None] & (j == wp)[:, None, :])
    return np.where(inside, tokens, eos.view(np.uint8)).reshape(n, S * S)
