"""End-to-end CPU smoke (DESIGN.md §10): build, train, SIGTERM, resume, DONE, evaluate, adapt, submission."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from frm.data import build as build_mod
from frm.data.composite import generate

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(args, **kw):
    return subprocess.run([PY, "-m", *args], cwd=ROOT, capture_output=True, text=True, timeout=600, **kw)


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("smoke")
    tasks = generate(10, depth_range=(1, 2), seed=3, max_side=5)
    (d / "comp.json").write_text(json.dumps(tasks))
    build_mod.main(["--composite", str(d / "comp.json"), "--out", str(d / "data"), "--n_aug_composite", "2"])
    cfg = {
        "name": "smoke",
        "model": {"hidden": 32, "heads": 2, "L_cycles": 1, "K": 2, "code_rank": 8, "join_dim": 8},
        "data": {"root": str(d / "data"), "tokens_per_step": 2000, "max_rows": 8, "source_weights": None},
        "train": {"lr": 1e-3, "warmup_steps": 2, "max_rounds": 3, "max_steps": 30, "budget_pflops": 1e9,
                  "eval_every_pflops": 1e9, "log_every": 1, "ckpt_minutes": 100, "device": "cpu"},
        "eval": {"splits": ["dev_composite"], "max_versions": 2, "rounds": 3, "tokens_per_batch": 4000,
                 "max_rows": 16, "cf_rows": 8},
    }
    (d / "cfg.json").write_text(json.dumps(cfg))
    # A Kaggle-format challenge file (and its solutions) from the same composite tasks.
    chall = {t["id"]: {"train": [{"input": p["input"], "output": p["output"]} for p in t["train"]],
                       "test": [{"input": p["input"]} for p in t["test"]]} for t in tasks[:4]}
    sols = {t["id"]: [p["output"] for p in t["test"]] for t in tasks[:4]}
    (d / "chall.json").write_text(json.dumps(chall))
    (d / "sols.json").write_text(json.dumps(sols))
    return d


def test_train_sigterm_resume_eval_adapt(workdir):
    run_dir = workdir / "run"
    proc = subprocess.Popen(
        [PY, "-m", "frm.train", "--config", str(workdir / "cfg.json"), "--run_dir", str(run_dir)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    log = run_dir / "train.jsonl"
    deadline = time.time() + 300
    while not (log.exists() and log.read_text().count("\n") >= 2):
        assert proc.poll() is None, proc.stdout.read()
        assert time.time() < deadline, "trainer produced no log lines"
        time.sleep(0.2)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 0, out
    assert (run_dir / "last.pt").exists() and not (run_dir / "DONE").exists()
    steps_before = [json.loads(line)["step"] for line in log.read_text().splitlines()]

    res = run(["frm.train", "--config", str(workdir / "cfg.json"), "--run_dir", str(run_dir)])
    assert res.returncode == 0, res.stdout + res.stderr
    assert "resumed" in res.stdout and (run_dir / "DONE").exists()
    steps = [json.loads(line)["step"] for line in log.read_text().splitlines()]
    assert steps[len(steps_before)] == steps_before[-1] + 1, "resume must continue the step count"
    assert steps[-1] == 30

    ev = run(["frm.evaluate", "--run_dir", str(run_dir), "--split", "dev_composite", "--ckpt", "last"])
    assert ev.returncode == 0, ev.stdout + ev.stderr
    report = json.loads(next((run_dir / "eval").glob("dev_composite_*.json")).read_text())
    assert len(report["by_round"]["pass@2"]) == 3 and "fork" in report and "cf" in report
    assert run(["frm.evaluate", "--run_dir", str(run_dir), "--split", "test"]).returncode == 2

    sub = workdir / "submission.json"
    ad = run([
        "frm.adapt", "--ckpt", str(run_dir / "last.pt"), "--challenges", str(workdir / "chall.json"),
        "--solutions", str(workdir / "sols.json"), "--work", str(workdir / "adapt"), "--n_aug", "2",
        "--hours", "1", "--max_steps", "4", "--submission", str(sub), "train.device=cpu",
    ])
    assert ad.returncode == 0, ad.stdout + ad.stderr
    submission = json.loads(sub.read_text())
    chall = json.loads((workdir / "chall.json").read_text())
    assert set(submission) == set(chall)
    for tid, entries in submission.items():
        assert len(entries) == len(chall[tid]["test"])
        for e in entries:
            assert set(e) == {"attempt_1", "attempt_2"}
            assert all(isinstance(row, list) for row in e["attempt_1"])
    summary = json.loads((workdir / "adapt" / "adapt_summary.json").read_text())
    assert summary["steps"] == 4 and summary["n_tasks"] == 4


if os.environ.get("FRM_SKIP_SMOKE"):
    pytest.skip("FRM_SKIP_SMOKE set", allow_module_level=True)
