"""Size buckets: predicted test-output size from a demo-verified rule, and the task's canvas size.

Never reads test outputs."""

from __future__ import annotations

from fractions import Fraction

from frm.data.arc import Grid, Task

BUCKETS = (8, 12, 16, 20, 24, 30)
MAX_SIDE = 30
ABSTAIN_MARGIN = 4


def fits(side: int, S: int) -> bool:
    """A grid of max side ``side`` fits canvas S with its EOS margin (S=30 holds 30 without it, as TRM)."""
    return side + 1 <= S or (S == MAX_SIDE and side <= MAX_SIDE)


def smallest_bucket(side: int) -> int:
    for S in BUCKETS:
        if fits(side, S):
            return S
    raise ValueError(f"grid side {side} exceeds {MAX_SIDE}")


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _axis_rule(ins: list[int], outs: list[int], x: int) -> int | None:
    """Constant output, else constant ratio, else constant difference; must hold on every demo."""
    if len(set(outs)) == 1:
        return outs[0]
    ratios = {Fraction(o, i) for i, o in zip(ins, outs)}
    if len(ratios) == 1:
        y = ratios.pop() * x
        return int(y) if y.denominator == 1 else None
    diffs = {o - i for i, o in zip(ins, outs)}
    if len(diffs) == 1:
        return x + diffs.pop()
    return None


def predict_output_size(task: Task, test_input: Grid) -> tuple[int, int] | None:
    """Per-axis demo-verified size rule; None when either axis abstains or predicts outside 1..30."""
    shapes = [(_shape(ex["input"]), _shape(ex["output"])) for ex in task["train"]]
    if not shapes:
        return None
    x = _shape(test_input)
    pred = []
    for a in (0, 1):
        y = _axis_rule([i[a] for i, _ in shapes], [o[a] for _, o in shapes], x[a])
        if y is None or not 1 <= y <= MAX_SIDE:
            return None
        pred.append(y)
    return pred[0], pred[1]


def choose_bucket(task: Task) -> int:
    """Smallest bucket fitting every demo grid, every test input and each test input's predicted output size
    (max side + 4 when the rule abstains, capped at 30)."""
    demo_side = max((max(_shape(ex[k])) for ex in task["train"] for k in ("input", "output")), default=0)
    side = demo_side
    for ex in task["test"]:
        in_side = max(_shape(ex["input"]))
        pred = predict_output_size(task, ex["input"])
        out_side = max(pred) if pred is not None else min(max(demo_side, in_side) + ABSTAIN_MARGIN, MAX_SIDE)
        side = max(side, in_side, out_side)
    return smallest_bucket(side)
