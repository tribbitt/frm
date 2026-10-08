"""Evaluation helpers: official scoring, submission shape, sham children from other tasks, which versions vote."""

from types import SimpleNamespace

import numpy as np
import torch

from frm.data.loader import eval_versions
from frm.evaluate import Votes, attempts, score, sham_perm


def test_only_trained_versions_vote_identity_first():
    d = SimpleNamespace(vstart=np.array([0, 5]), vcount=np.array([5, 3]))
    trained = np.array([0, 1, 0, 1, 1, 0, 0, 0], dtype=bool)
    assert eval_versions(d, 0, None, trained).tolist() == [1, 3, 4]
    assert eval_versions(d, 0, 2, trained).tolist() == [1, 3]  # the cap keeps version order
    assert eval_versions(d, 1, 32, trained).tolist() == [5]  # nothing trained: the identity alone, a blind attempt


def test_score_official_metric_and_per_task():
    a, b = np.array([[1, 2]], np.uint8), np.array([[3]], np.uint8)
    tasks = {
        "t1": {"test": [{"output": a}, {"output": b}]},  # two test inputs, one solved -> 0.5
        "t2": {"test": [{"output": b}]},  # solved by the second attempt -> pass@2 only
        "t3": {"test": [{"output": None}]},  # unknown output: not scored
    }
    v = Votes()
    v.add(("t1", 0), a, 0.9)
    v.add(("t1", 1), a, 0.9)
    for _ in range(2):
        v.add(("t2", 0), a, 0.5)
    v.add(("t2", 0), b, 0.9)
    s = score(v, tasks)
    assert s["n_tasks"] == 2 and s["per_task"] == {"t1": 0.5, "t2": 1.0}
    assert s["pass@2"] == 0.75 and s["pass@1"] == 0.25


def test_votes_rank_by_count_then_q():
    a, b, c = (np.full((1, 1), i, np.uint8) for i in range(3))
    v = Votes()
    v.add(("t", 0), a, 0.1)
    v.add(("t", 0), b, 0.9)
    v.add(("t", 0), c, 0.2)
    v.add(("t", 0), c, 0.2)
    ranked = v.ranked(("t", 0))
    assert [int(g[0, 0]) for g in ranked] == [2, 1, 0]


def test_attempts_one_entry_per_test_input_padded():
    g = np.array([[4]], np.uint8)
    v = Votes()
    v.add(("t", 0), g, 1.0)
    sub = attempts(v, {"t": {"test": [{"input": [[0]]}, {"input": [[0]]}]}})
    assert sub == {"t": [{"attempt_1": [[4]], "attempt_2": [[4]]}, {"attempt_1": [[0]], "attempt_2": [[0]]}]}


def test_sham_perm_crosses_tasks():
    tasks = ["a"] * 6 + ["b"] * 3 + ["c"]
    perm = sham_perm(tasks, torch.device("cpu")).tolist()
    assert all(tasks[j] != tasks[i] for i, j in enumerate(perm))
    assert sham_perm(["a", "a"], torch.device("cpu")).tolist() == [1, 0]  # single-task batch: fallback
