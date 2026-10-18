"""The sandbox worker: runs BARC programs for ``sandbox.py``.

Never imported by this package.  `sandbox.Worker` starts it by path under the sandbox venv's
interpreter (`python -I -B`), inside the OS sandbox (macOS: sandbox-exec, no network, no file
writes, no fork), in its own session, in an empty directory, with an empty environment:

    python -I -B _worker.py LIB_DIR BARC_DIR OUT_FD MEM_MB CALL_S CPU_S NOFILE

It sets its own rlimits first (CPU_S seconds of CPU for its life, no file growth, no core,
NOFILE descriptors; on Linux also address space and no new processes): the parent starts
workers from threads, where a pre-exec hook is unsafe.  Requests are JSON lines on stdin; each
answer is one JSON line on OUT_FD (a pipe only this code holds: the programs' prints go to a
sink).  After the warm-up (numpy, scipy.ndimage, BARC's common.py) an audit hook refuses every
new import, every file open, every os, process, socket, ctypes and code-execution event, so
whatever a program reaches it cannot open, spawn, connect or import.  Each program runs in its
own namespace with fresh restricted builtins, an importer that admits only
`policy.ALLOWED_IMPORTS`, and a fresh copy of `common` (a program that mutates Color or a helper
cannot reach the next one); the entries of the modules in _SHARED (numpy, numpy.random,
numpy.linalg, random, math, itertools, collections, typing, scipy.ndimage, builtins) are compared
with their warm-up state after every request, and a program that rebound one gets a 'tainted'
answer and ends the process (numpy's other submodules and class attributes are not compared).
So does a memory answer (ru_maxrss never falls).  Each call has
a CALL_S timer; the parent's deadline and RSS watchdog are the backstop for code that swallows
it.

    {"op": "load", "source": str}                              -> {"ok": bool, "why"?: str}
    {"op": "run", "pairs": [[x, y|null], ...], "seeds": [a, b]} -> {"status": [code, ...], "out": [...]?}
    {"op": "heldout", "tasks": [[[x, y], ...], ...]}           -> {"ok": true}
    {"op": "symmetry", "pairs": [[x, y], ...]}                 -> {"sym": [d, ...]}
    {"op": "audit", "cases": [[t, [d, ...], [lut, ...]], ...], "budget_s": s}
                                                               -> {"hit": [t, d, lut] | null, "calls": n, ...}
"""

from __future__ import annotations

# every allowed module (policy.ALLOWED_IMPORTS) is loaded before the audit hook closes imports
import builtins
import collections
import collections.abc
import itertools
import json
import math
import os
import random
import resource
import signal
import sys
import time
import types
import typing

LIB, BARC_DIR, OUT_FD, MEM_MB, CALL_S, CPU_S, NOFILE = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]),
    int(sys.argv[7]))
for _lim, _val in ((resource.RLIMIT_CPU, CPU_S), (resource.RLIMIT_FSIZE, 0), (resource.RLIMIT_CORE, 0),
                   (resource.RLIMIT_NOFILE, NOFILE)):
    resource.setrlimit(_lim, (_val, _val))
if sys.platform != "darwin":         # macOS refuses RLIMIT_AS; there the parent's RSS watchdog holds
    resource.setrlimit(resource.RLIMIT_AS, (4 * MEM_MB << 20, 4 * MEM_MB << 20))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
sys.path[:0] = [LIB, BARC_DIR]

import numpy as np  # noqa: E402  (after the rlimits, which must bind before any library loads)
import policy  # noqa: E402
import scipy.ndimage  # noqa: E402
from scipy import ndimage  # noqa: E402, F401

_OUT = os.fdopen(OUT_FD, "w", buffering=1)
_IN = sys.stdin
_asarray, _ascontig, _rot90 = np.asarray, np.ascontiguousarray, np.rot90
_maxrss_scale = 1 if sys.platform == "darwin" else 1024        # ru_maxrss: bytes on macOS, KiB on Linux
with open(os.path.join(LIB, "common.py")) as f:
    _COMMON = compile(f.read(), "common.py", "exec")
_warm = types.ModuleType("common")
exec(_COMMON, _warm.__dict__)  # noqa: S102  (BARC's common.py, pinned: imports what it imports at top level)


class _Sink:
    def write(self, s):
        return len(s)

    def flush(self):
        pass


class _Timeout(BaseException):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)
sys.stdout = sys.stderr = _Sink()

_real_import = builtins.__import__
_RUNTIME_BANNED = {"open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr", "globals",
                   "locals", "vars", "dir", "breakpoint", "input", "help", "exit", "quit", "memoryview",
                   "__loader__", "__spec__"}
_current_common = None


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in policy.ALLOWED_IMPORTS:
        raise ImportError(name)
    if name == "common":
        return _current_common
    return _real_import(name, globals, locals, fromlist, level)


_SAFE = {k: v for k, v in vars(builtins).items() if k not in _RUNTIME_BANNED}
_SAFE["__import__"] = _import

# modules every program shares (each gets its own common and builtins): their entries at warm-up,
# compared by identity after every request.  A program that rebinds one (`m = np; m.sum = f`, which
# the static policy cannot follow) would change every later program's behaviour in this worker,
# so its own answer becomes an error and the worker exits (the parent starts a fresh one)
_SHARED = (np, np.random, np.linalg, random, math, itertools, collections, collections.abc, typing,
           scipy.ndimage, builtins)
_SNAP = [(m, dict(vars(m))) for m in _SHARED]
_NP_ERR = np.geterr()
_MISSING = object()


def _intact() -> bool:
    return all(vars(m).get(k, _MISSING) is v for m, snap in _SNAP for k, v in snap.items())

# compile / exec events: only the ones this worker issues itself, counted (a program cannot reach
# the counter; it is spent by the events it pays for)
_EXEC_BUDGET = [0]
_DENY = ("open", "os.", "subprocess", "socket", "ctypes", "shutil", "sys.settrace", "sys.setprofile",
         "sys.addaudithook", "code.__new__", "function.__new__", "pty", "webbrowser", "urllib", "http", "ftplib",
         "smtplib", "poplib", "imaplib", "nntplib", "telnetlib", "syslog", "fcntl", "mmap", "resource.setrlimit",
         "signal.", "cpython.", "winreg", "msvcrt", "_winapi")


def _hook(event, args):
    if event in ("compile", "exec"):
        if _EXEC_BUDGET[0] <= 0:
            raise RuntimeError(f"sandbox: {event}")
        _EXEC_BUDGET[0] -= 1
        return
    if event == "import":
        name = str(args[0])
        if name.split(".")[0] not in policy.ALLOWED_IMPORTS or name not in sys.modules:
            raise RuntimeError(f"sandbox: import {name}")
        return
    if event.startswith(_DENY):
        raise RuntimeError(f"sandbox: {event}")


def _send(obj) -> None:
    _OUT.write(json.dumps(obj, separators=(",", ":")) + "\n")


def _grid(x):
    return _ascontig(_asarray(x, dtype=np.int64))


def _valid(o):
    """The program's output as an int64 grid, or None (not a 2-D grid of colours 0..9, sides 1..30)."""
    try:
        a = _asarray(o)
        if a.ndim != 2 or not (1 <= a.shape[0] <= 30 and 1 <= a.shape[1] <= 30) or a.dtype.kind not in "biuf":
            return None
        if a.dtype.kind == "f" and not bool((a == a.round()).all()):
            return None
        a = _ascontig(a.astype(np.int64))
        if int(a.min()) < 0 or int(a.max()) > 9:
            return None
        return a
    except Exception:  # noqa: BLE001  (whatever a program returns)
        return None


def _call(fn, g, seed):
    """('ok', grid) | ('timeout' | 'error' | 'invalid', None) for one call, CALL_S on the timer."""
    signal.setitimer(signal.ITIMER_REAL, CALL_S)
    try:
        np.random.seed(seed)
        random.seed(seed)
        out = fn(g.copy())
    except _Timeout:
        return "timeout", None
    except BaseException:  # noqa: BLE001  (a program may raise anything, RecursionError and MemoryError included)
        return "error", None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    o = _valid(out)
    return ("ok", o) if o is not None else ("invalid", None)


def _same(a, b) -> bool:
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def _memory_ok() -> bool:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _maxrss_scale < MEM_MB * (1 << 20)


_main = None
_held: list = []                      # [task][dihedral] -> [(x, y), ...]


def _load(source):
    global _main, _current_common, _sym
    _main, _sym = None, [0]
    _EXEC_BUDGET[0] = 1                      # ast.parse in the policy check
    try:
        why = policy.check(source)
    finally:
        _EXEC_BUDGET[0] = 0
    if why:
        return {"ok": False, "why": why}
    common = types.ModuleType("common")
    ns = {"__builtins__": dict(_SAFE), "__name__": "barc_program"}
    np.seterr(**_NP_ERR)                     # a previous program's seterr ends with it
    try:
        _EXEC_BUDGET[0] = 3                  # exec common, compile the program, exec the program
        exec(_COMMON, common.__dict__)  # noqa: S102  (a fresh common per program)
        code = compile(source, "<barc>", "exec")
        _current_common = common
        signal.setitimer(signal.ITIMER_REAL, CALL_S)
        exec(code, ns)  # noqa: S102  (the program, policy-checked, restricted builtins, hook armed)
    except _Timeout:
        return {"ok": False, "why": "load_timeout"}
    except BaseException as e:  # noqa: BLE001  (a program's module body may raise anything)
        return {"ok": False, "why": f"load_error:{type(e).__name__}"}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        _EXEC_BUDGET[0] = 0
    fn = ns.get("main")
    if not callable(fn):
        return {"ok": False, "why": "no_main"}
    _main = fn
    return {"ok": True}


def _run(pairs, seeds, want_out):
    status, outs = [], []
    for x, y in pairs:
        g = _grid(x)
        got = [_call(_main, g, s) for s in seeds]
        bad = next((st for st, _ in got if st != "ok"), None)
        o = got[0][1]
        if bad:
            status.append(bad)
        elif any(not _same(o, oo) for _, oo in got[1:]):
            status.append("nondet")
        elif y is not None and not _same(o, _grid(y)):
            status.append("mismatch")
        else:
            status.append("ok")
        outs.append(o.tolist() if want_out and o is not None else None)
        if not _memory_ok():
            return {"status": status, "memory": True}
    return {"status": status, **({"out": outs} if want_out else {})}


def _dihedral(g, k):
    if k >= 4:
        g = g[:, ::-1]
    return _ascontig(_rot90(g, k % 4))


# the dihedral group: _COMPOSE[a][b] = c where _dihedral(_dihedral(z, b), a) == _dihedral(z, c)
def _compose_table():
    z = np.arange(6).reshape(2, 3)
    forms = [_dihedral(z, k) for k in range(8)]

    def index(g):
        return next(k for k, f in enumerate(forms) if f.shape == g.shape and bool((f == g).all()))
    return [[index(_dihedral(forms[b], a)) for b in range(8)] for a in range(8)]


_COMPOSE = _compose_table()
_sym = [0]                            # the loaded program's dihedral symmetries (`_symmetry`)


def _heldout(tasks):
    _held.clear()
    for demos in tasks:
        arr = [(_grid(x), _grid(y)) for x, y in demos]
        _held.append([[(_dihedral(x, k), _dihedral(y, k)) for x, y in arr] for k in range(8)])
    return {"ok": True}


def _symmetry(pairs):
    """The dihedral transforms g the loaded program commutes with on its own stored pairs,
    main(g(x)) == g(y), closed into the group they generate."""
    global _sym
    own = [(_grid(x), _grid(y)) for x, y in pairs]
    sym = {0}
    for g in range(1, 8):
        if all((r := _call(_main, _dihedral(x, g), 0))[0] == "ok" and _same(r[1], _dihedral(y, g)) for x, y in own):
            sym.add(g)
    grown = True
    while grown:
        more = {_COMPOSE[a][b] for a in sym for b in sym} - sym
        sym |= more
        grown = bool(more)
    _sym = sorted(sym)
    return {"sym": _sym}


def _classes(ds):
    """One transform per class {g . d : g a symmetry}: if main commutes with g then
    main(tau(g(d(x)))) == g(main(tau(d(x)))), so d and g . d stand or fall together."""
    kept = []
    for d in ds:
        if not any(_COMPOSE[g][k] == d for k in kept for g in _sym):
            kept.append(d)
    return kept


def _order(n: int) -> list[int]:
    """The LUT order: three probes spread over the bindings (first, middle, last: they bind the
    rule's first named colour differently), then the rest."""
    probes = sorted({0, n // 2, n - 1}) if n else []
    return probes + [i for i in range(n) if i not in probes]


def _audit(cases, budget_s, max_timeouts):
    """Every case (held-out task t, dihedral transforms, colour LUTs): is there a (d, lut) with
    main(lut[d(x)]) == lut[d(y)] on every demonstration?  One transform per class of the
    program's symmetries; demonstration 0 first.  A case ends early only when three spread
    probes all run and agree on one output shape that is not the demonstration's: the shape
    does not depend on the colours there (an error, a timeout or two shapes keep every binding
    in play).  Any timeout leaves its binding untested: the parent reads `timeouts` > 0 as
    inconclusive."""
    t0 = time.process_time()
    n = {"calls": 0, "timeouts": 0, "errors": 0, "sym": len(_sym)}

    def run(g):
        st, o = _call(_main, g, 0)
        n["calls"] += 1
        n["timeouts"] += st == "timeout"
        n["errors"] += st == "error"
        return st, o

    for t, ds, luts in cases:
        luts = [_asarray(lut, dtype=np.int64) for lut in luts]
        order = _order(len(luts))
        for d in _classes(ds):
            demos = _held[t][d]
            x0, y0 = demos[0]
            probe = []
            for j, i in enumerate(order):
                if time.process_time() - t0 > budget_s or n["timeouts"] > max_timeouts:
                    return {"hit": None, **n, "budget": True}
                lut = luts[i]
                st, o = run(lut[x0])
                if not _memory_ok():
                    return {"hit": None, **n, "memory": True}
                if j < 3:
                    probe.append(o.shape if st == "ok" else None)
                    if j == 2 and None not in probe and len(set(probe)) == 1 and probe[0] != y0.shape:
                        break
                if st != "ok" or not _same(o, lut[y0]):
                    continue
                rest = (run(lut[x]) + (lut[y],) for x, y in demos[1:])
                if all(st2 == "ok" and _same(o2, want) for st2, o2, want in rest):
                    return {"hit": [t, d, lut.tolist()], **n}
    return {"hit": None, **n}


def _answer(req: dict) -> dict:
    op = req["op"]
    if op == "load":
        return _load(req["source"])
    if op == "heldout":
        return _heldout(req["tasks"])
    if _main is None:
        return {"error": "no_program"}
    if op == "run":
        return _run(req["pairs"], req["seeds"], bool(req.get("out")))
    if op == "symmetry":
        return _symmetry(req["pairs"])
    if op == "audit":
        return _audit(req["cases"], float(req["budget_s"]), int(req.get("max_timeouts", 3)))
    return {"error": f"op {op}"}


def main() -> None:
    sys.addaudithook(_hook)
    _EXEC_BUDGET[0] = 0
    _send({"ready": True, "pid": os.getpid(), "numpy": np.__version__, "python": sys.version.split()[0]})
    for line in _IN:
        try:
            req = json.loads(line)
            if req.get("op") == "exit":
                return
            ans = _answer(req)
        except _Timeout:
            ans = {"error": "timeout"}
        except Exception as e:  # noqa: BLE001  (a malformed request is answered, not fatal)
            ans = {"error": f"{type(e).__name__}"}
        if not _intact():
            # a program rebound a shared module: its own answer is void, and no later program
            # may run in this process
            _send({"error": "tainted", "tainted": True})
            return
        _send(ans)
        if ans.get("memory"):
            return          # ru_maxrss is a lifetime peak: this process would refuse every later program


if __name__ == "__main__":
    main()
