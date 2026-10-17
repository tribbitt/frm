"""Run BARC programs, which are untrusted LLM-written code, contained.  Never in this process: every program runs in
a worker process (``_worker.py``) under the sandbox venv's interpreter (``BARC_PYTHON`` overrides it).

    python -m frm.external.barc.sandbox --selftest [--dir data/external/barc] [--workers 2]

The self-test runs HOSTILE (escapes, rebinding, loops, memory, nondeterminism) and exits 1 unless each is refused or
contained as expected.

The layers, outermost first:
    OS sandbox     macOS: sandbox-exec, no network, no file writes, no fork (PROFILE).  Elsewhere
                   RLIMIT_NPROC 0 stops forks; there is no network or write barrier below the
                   audit hook, so `Sandbox` refuses to start off macOS unless BARC_UNSANDBOXED=1 (an explicit
                   opt-out: on Linux run it inside a container or VM without network)
    rlimits        set by the worker on itself before its imports: CPU seconds (its life), file
                   size 0, no core files, NOFILE descriptors; on Linux also address space and
                   RLIMIT_NPROC 0 (macOS refuses RLIMIT_AS, see memory)
    process        its own session, an empty working directory, an empty environment, `python -I
                   -B` (no user site-packages, no PYTHON* variables, no bytecode written)
    memory         the worker's peak RSS checked after every call, and a watchdog thread here that
                   kills any worker whose RSS passes MEM_MB (macOS does not enforce RLIMIT_AS)
    time           a CALL_S timer per call inside the worker; a deadline per request here, after
                   which the worker is killed and replaced (code that swallows the timer)
    audit hook     in the worker, after warm-up: no new imports, no file opens, no os.* events, no
                   subprocess, socket, ctypes, compile or exec but the worker's own
    namespace      restricted builtins, an importer of `policy.ALLOWED_IMPORTS` only, a fresh copy
                   of BARC's common.py per program
    static policy  `policy.check` before a program is sent, and again inside the worker
Answers are JSON (never pickle), read with a size cap and a deadline.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from frm.external.barc import fetch, policy

WORKER = Path(__file__).resolve().parent / "_worker.py"
BARC_DIR = WORKER.parent
CALL_S = 2.0                      # per call
MEM_MB = 1536                     # per worker, peak RSS
CPU_S = 4 * 3600                  # a worker's whole life (RLIMIT_CPU); workers are recycled long before
NOFILE = 256
START_S = 60.0
MAX_ANSWER = 64 << 20
RECYCLE = 5000                    # programs per worker before a fresh one
PROFILE = "(version 1)(allow default)(deny network*)(deny file-write*)(deny process-fork)"
SEEDS = (12345, 987654)           # two numpy / random states: a program must not depend on them


def python(root: Path) -> str:
    env = os.environ.get("BARC_PYTHON")
    return os.path.abspath(env) if env else str(fetch.venv(root) / "bin" / "python")


def available(root: Path) -> str | None:
    """None if programs can run here, else what is missing."""
    if not Path(python(root)).exists():
        return f"no sandbox interpreter at {python(root)} (python -m frm.external.barc.fetch)"
    if not (fetch.lib(root) / "common.py").exists():
        return f"no {fetch.lib(root) / 'common.py'} (python -m frm.external.barc.fetch)"
    if sys.platform == "darwin" and not shutil.which("sandbox-exec"):
        return "sandbox-exec not found"
    if sys.platform != "darwin" and os.environ.get("BARC_UNSANDBOXED") != "1":
        return "no OS sandbox on this platform (macOS sandbox-exec); BARC_UNSANDBOXED=1 to accept rlimits + hook"
    return None


class WorkerDied(Exception):
    pass


class Worker:
    """One sandboxed worker process.  `request` returns the parsed answer, or raises
    WorkerDied (deadline, crash, malformed answer) after killing the process."""

    def __init__(self, home: str, root: Path):
        r, w = os.pipe()
        prefix = ["sandbox-exec", "-p", PROFILE] if sys.platform == "darwin" else []
        cmd = [*prefix, python(root), "-I", "-B", str(WORKER), str(fetch.lib(root)), str(BARC_DIR), str(w),
               str(MEM_MB), str(CALL_S), str(CPU_S), str(NOFILE)]
        # the worker sets its own rlimits before it imports anything (a pre-exec hook is unsafe from threads)
        # one BLAS / OpenMP thread: n workers already fill the cores, and on Linux RLIMIT_NPROC 0
        # refuses the thread pool OpenBLAS would start on import
        env = {"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  pass_fds=(w,), cwd=home, env=env, start_new_session=True)
        os.close(w)
        self.fd = r
        self.buf = b""
        self.programs = 0
        self.spent = False           # a memory or tainted answer: the worker exits, take a fresh one
        hello = self._read(START_S)
        if not hello.get("ready"):
            self.kill()
            raise WorkerDied(f"worker did not start: {hello}")
        self.info = hello

    def _read(self, timeout: float) -> dict:
        end = time.monotonic() + timeout
        while b"\n" not in self.buf:
            left = end - time.monotonic()
            if left <= 0 or len(self.buf) > MAX_ANSWER:
                self.kill()
                raise WorkerDied("deadline" if left <= 0 else "answer too large")
            ready, _, _ = select.select([self.fd], [], [], left)
            if not ready:
                continue
            chunk = os.read(self.fd, 1 << 20)
            if not chunk:
                self.kill()
                raise WorkerDied("worker exited")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        try:
            out = json.loads(line)
        except ValueError:
            self.kill()
            raise WorkerDied("malformed answer") from None
        if not isinstance(out, dict):
            self.kill()
            raise WorkerDied("malformed answer")
        if out.get("memory") or out.get("tainted"):
            self.spent = True
        return out

    def request(self, msg: dict | bytes, timeout: float) -> dict:
        """One request (a dict, or a JSON line already encoded) and its answer."""
        line = msg if isinstance(msg, bytes) else (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        try:
            self.p.stdin.write(line)
            self.p.stdin.flush()
        except (BrokenPipeError, OSError):
            self.kill()
            raise WorkerDied("worker exited") from None
        return self._read(timeout)

    def alive(self) -> bool:
        return self.p.poll() is None

    def kill(self) -> None:
        if self.p.poll() is None:
            self.p.kill()
            self.p.wait()
        try:
            os.close(self.fd)
        except OSError:
            pass
        self.fd = -1


def _rss_kb(pids: list[int]) -> dict[int, int]:
    if not pids:
        return {}
    out = subprocess.run(["ps", "-o", "pid=,rss=", "-p", ",".join(map(str, pids))], capture_output=True, text=True,
                         check=False)
    got = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            got[int(parts[0])] = int(parts[1])
    return got


class Sandbox:
    """n workers driven by n threads.  `map(job, items)` yields (item, result) in completion
    order; `job(worker, item)` talks to its worker and returns a JSON-able result.  A job
    whose worker dies gets {"died": reason}, and the worker is replaced.  `init` messages
    (e.g. the held-out demonstrations) are sent to every fresh worker."""

    def __init__(self, n: int, root: Path, init: list[dict] | None = None):
        root = Path(root).resolve()  # workers run in an empty temporary directory: no relative paths
        why = available(root)
        if why:
            raise SystemExit(f"BARC sandbox unavailable: {why}")
        self.n, self.root, self.init = n, root, init or []
        self.home = tempfile.mkdtemp(prefix="barc_sandbox_")
        self.workers: list[Worker | None] = [None] * n
        self.killed_rss = 0
        self.stop = threading.Event()
        self.watch = threading.Thread(target=self._watchdog, daemon=True)
        self.watch.start()

    def _watchdog(self) -> None:
        while not self.stop.wait(0.25):
            live = {w.p.pid: w for w in self.workers if w is not None and w.alive()}
            for pid, kb in _rss_kb(list(live)).items():
                if kb > MEM_MB * 1024:
                    self.killed_rss += 1
                    live[pid].p.kill()

    def _fresh(self, i: int) -> Worker:
        for attempt in range(3):
            try:
                w = Worker(self.home, self.root)
                for msg in self.init:
                    ans = w.request(msg, START_S)
                    if not ans.get("ok"):
                        w.kill()
                        raise WorkerDied(f"init failed: {ans}")
                self.workers[i] = w
                return w
            except WorkerDied:
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    def _loop(self, i: int, job, it, lock, out: queue.Queue) -> None:
        w = None
        try:
            while True:
                with lock:
                    item = next(it, StopIteration)
                if item is StopIteration:
                    break
                if w is None or w.spent or not w.alive() or w.programs >= RECYCLE:
                    if w is not None:
                        w.kill()
                    w = self._fresh(i)
                try:
                    res = job(w, item)
                    w.programs += 1
                except WorkerDied as e:
                    res = {"died": str(e)}
                    w = None
                out.put((item, res))
        except BaseException as e:  # noqa: BLE001  (re-raised in the caller's thread)
            out.put(e)
        finally:
            if w is not None:
                w.kill()
            out.put(None)

    def map(self, job, items):
        """(item, result) pairs as they finish (completion order; the caller sorts)."""
        it, lock, out = iter(items), threading.Lock(), queue.Queue()
        threads = [threading.Thread(target=self._loop, args=(i, job, it, lock, out), daemon=True)
                   for i in range(self.n)]
        for t in threads:
            t.start()
        live = len(threads)
        while live:
            got = out.get()
            if got is None:
                live -= 1
            elif isinstance(got, BaseException):
                raise got
            else:
                yield got
        for t in threads:
            t.join()

    def close(self) -> None:
        self.stop.set()
        for w in self.workers:
            if w is not None:
                w.kill()
        shutil.rmtree(self.home, ignore_errors=True)


# --------------------------------------------------------------------------
# requests
# --------------------------------------------------------------------------


def load(w: Worker, source: str) -> dict:
    return w.request({"op": "load", "source": source}, CALL_S + 10.0)


def run(w: Worker, pairs: list, *, out: bool = False) -> dict:
    """Each pair's input under both SEEDS: 'ok' (same output twice, equal to the stored one
    when given), 'mismatch', 'nondet', 'error', 'timeout', 'invalid'."""
    msg = {"op": "run", "pairs": pairs, "seeds": list(SEEDS), "out": out}
    return w.request(msg, len(pairs) * len(SEEDS) * CALL_S + 10.0)


def check_and_run(w: Worker, source: str, pairs: list) -> dict:
    """load + run: the program's verdict on its own stored pairs."""
    ld = load(w, source)
    if not ld.get("ok"):
        return {"load": ld.get("why", ld.get("error", "load"))}
    r = run(w, pairs)
    if "status" not in r or r.get("memory"):
        return {"load": "ok", "run": "memory" if r.get("memory") else r.get("error", "run")}
    return {"load": "ok", "status": r["status"]}


# --------------------------------------------------------------------------
# self-test: hostile programs are refused or contained
# --------------------------------------------------------------------------

_GRID = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
HOSTILE = {   # name: (source, the verdicts that contain it, "|"-separated)
    "import os": ("import os\ndef main(g):\n    os.system('touch /tmp/barc_pwn')\n    return g\n", "refused"),
    "typing.sys": ("import typing\ndef main(g):\n    typing.sys.modules['os'].system('x')\n    return g\n", "refused"),
    "open": ("def main(g):\n    open('/tmp/barc_pwn', 'w')\n    return g\n", "refused"),
    "class walk": ("def main(g):\n    return ().__class__.__bases__[0].__subclasses__()\n", "refused"),
    "format walk": ("def main(g):\n    return '{0.__class__}'.format(g)\n", "refused"),
    "patch numpy": ("import numpy as np\ndef main(g):\n    np.array_equal = None\n    return g\n", "refused"),
    "patch numpy.random": ("from numpy import random as R\ndef main(g):\n    R.seed = None\n    return g\n",
                           "refused"),
    "rebind numpy": ("import numpy as np\ndef main(g):\n    m = np\n    m.flip = None\n    return np.asarray(g)\n",
                     "tainted"),
    "ctypes": ("from numpy import ctypeslib\ndef main(g):\n    return g\n", "refused"),
    "loop": ("def main(g):\n    while True:\n        pass\n", "timeout"),
    "swallow": (("def main(g):\n    while True:\n        try:\n            while True:\n                pass\n"
                 "        except BaseException:\n            pass\n"), "timeout|died"),
    # the RSS watchdog or the call timer may win the race; either way the allocation is stopped
    "memory": ("import numpy as np\ndef main(g):\n    return np.ones((1 << 31,), dtype=np.int64)\n",
               "died|memory|error|timeout"),
    "random": ("import numpy as np\ndef main(g):\n    return (np.asarray(g) + np.random.randint(1, 9)) % 10\n",
               "nondet"),
    "benign": ("import numpy as np\ndef main(g):\n    return np.asarray(g)[::-1]\n", "ok"),
}


def selftest(root: Path, n: int = 2) -> dict[str, tuple[str, str]]:
    """name -> (verdict, the verdicts that contain it) for every HOSTILE program."""

    def job(w: Worker, item):
        _, (src, _want) = item
        if policy.check(src):
            return "refused"
        r = check_and_run(w, src, [[_GRID, None]])
        if "status" in r:
            return r["status"][0]
        return str(r.get("run") if r.get("load") == "ok" else r.get("load"))

    sb = Sandbox(n, root)
    try:
        got = {}
        for (name, (_, want)), r in sb.map(job, list(HOSTILE.items())):
            got[name] = ("died" if isinstance(r, dict) and "died" in r else r, want)
    finally:
        sb.close()
    return got


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="run the HOSTILE programs (the only action)")
    ap.add_argument("--dir", type=Path, default=fetch.DIR, help="BARC work directory (lib/, venv/)")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    if not a.selftest:
        ap.error("nothing to do: pass --selftest")
    got = selftest(a.dir, a.workers)
    bad = 0
    for name in HOSTILE:
        verdict, want = got[name]
        ok = verdict in want.split("|")
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {name:20s} {verdict} (want {want})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
