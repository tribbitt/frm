"""ARC task JSON loading: challenges (+ solutions when present) -> {task_id: {"train": [...], "test": [...]}}."""

from __future__ import annotations

import json
import os

Grid = list[list[int]]
Task = dict[str, list[dict]]  # {"train": [{"input", "output"}], "test": [{"input", "output" | None}]}


def load_tasks(challenges: str, solutions: str | None = None) -> dict[str, Task]:
    """Load a challenges file and merge test outputs from a solutions file (``None`` outputs when absent)."""
    with open(challenges) as f:
        raw = json.load(f)
    sols = None
    if solutions is not None and os.path.isfile(solutions):
        with open(solutions) as f:
            sols = json.load(f)
    tasks: dict[str, Task] = {}
    for tid, t in raw.items():
        train = [{"input": ex["input"], "output": ex["output"]} for ex in t["train"]]
        test = []
        for i, ex in enumerate(t["test"]):
            out = ex.get("output")
            if sols is not None:
                out = sols[tid][i]
            test.append({"input": ex["input"], "output": out})
        tasks[tid] = {"train": train, "test": test}
    return tasks


def load_subset(raw_dir: str, subset: str) -> dict[str, Task]:
    """Load ``{raw_dir}/arc-agi_{subset}_{challenges,solutions}.json``."""
    prefix = os.path.join(raw_dir, f"arc-agi_{subset}")
    return load_tasks(f"{prefix}_challenges.json", f"{prefix}_solutions.json")
