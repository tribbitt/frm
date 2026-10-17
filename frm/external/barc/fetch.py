"""Fetch BARC ARC-Heavy and what its code needs into the BARC work directory (default data/external/barc).

    python -m frm.external.barc.fetch [--dir data/external/barc] [--threads 16]

Writes, each pinned and checked: ``data_100k.jsonl`` of the Hugging Face dataset REPO at REVISION (LFS sha256);
``lib/common.py``, BARC's ``seeds/common.py`` at BARC_COMMIT (sha256; the BARC repository shows no licence, so the
file is fetched at build time and never committed); ``venv/``, the sandbox interpreter (uv, Python VENV_PYTHON with
VENV_PACKAGES: common.py imports scipy.ndimage); ``MANIFEST.json``. Needs network access and ``uv``.

The card's "~200k" is two files of ~100k rows at this revision: the same ~110k GPT-4o descriptions coded twice by
GPT-4o-mini, plainly (data_100k.jsonl) and told to use common.py's functions (SECOND). The pool uses the first file
only: the second adds second implementations of the first's rules, not new rules, and its descriptions are
rewritten, so no description hash could keep an implementation pair on one side of barc_val.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DIR = Path("data/external/barc")

REPO = "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems"
REVISION = "4264798428e3c48cd67ec7a3402e0847f645da52"
FILES = {  # name: (bytes, LFS sha256)
    "data_100k.jsonl": (2798072950, "982aa76c55a22520b54217fba0e435f90d72b25068572b4dd4cbb83e09b2d8a2"),
}
SECOND = {"data_suggestfunction_100k.jsonl": (3389250697,  # recorded, not fetched (the docstring)
                                             "197a184678e86722fbc4891629cc0c163cec637998bc77426ce7176c333015bf")}
BARC_COMMIT = "a7b51a6b1ff969da3a78a71c533b6d79a93966e7"  # xu3kev/BARC
COMMON_SHA256 = "7b3922317470706ad8f61ae196d84539838c4587cb516efba7c8d61aa7a3f33b"
VENV_PYTHON = "3.12"
VENV_PACKAGES = ("numpy==1.26.4", "scipy==1.13.1")  # the releases current when BARC ran its code
PART = 16 << 20  # parts are recorded as they finish: a rerun resumes


def lib(root: Path) -> Path:
    """The sandbox's import path: common.py alone."""
    return root / "lib"


def venv(root: Path) -> Path:
    return root / "venv"


def _url(name: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def _part(url: str, path: Path, lo: int, hi: int, done: Path) -> None:
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "r+b") as f:
                f.seek(lo)
                got = 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    got += len(chunk)
            if got == hi - lo + 1:
                with open(done, "a") as f:  # one line per finished part
                    f.write(f"{lo}\n")
                return
        except OSError:
            pass
        time.sleep(2 * (attempt + 1))
    raise SystemExit(f"{url}: bytes {lo}-{hi} failed after 6 attempts")


def download(root: Path, name: str, size: int, digest: str, threads: int) -> None:
    path = root / name
    if path.exists() and path.stat().st_size == size and sha256(path) == digest:
        print(f"{name}: present, sha256 ok")
        return
    tmp, done = path.with_suffix(path.suffix + ".partial"), path.with_suffix(path.suffix + ".parts")
    if not (tmp.exists() and tmp.stat().st_size == size and done.exists()):
        with open(tmp, "wb") as f:
            f.truncate(size)
        done.write_text("")
    have = {int(x) for x in done.read_text().split()}
    t0 = time.time()
    ranges = [(lo, min(lo + PART, size) - 1) for lo in range(0, size, PART) if lo not in have]
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(lambda r: _part(_url(name), tmp, *r, done), ranges))
    got = sha256(tmp)
    if got != digest:
        done.unlink()
        raise SystemExit(f"{name}: sha256 {got} != {digest} (parts record removed: rerun downloads afresh)")
    tmp.replace(path)
    done.unlink()
    print(f"{name}: {size / 1e9:.2f} GB, {len(ranges)} parts in {time.time() - t0:.0f}s, sha256 ok")


def fetch_common(root: Path) -> None:
    path = lib(root) / "common.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path.exists() and sha256(path) == COMMON_SHA256):
        url = f"https://raw.githubusercontent.com/xu3kev/BARC/{BARC_COMMIT}/seeds/common.py"
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    if sha256(path) != COMMON_SHA256:
        raise SystemExit(f"common.py: sha256 {sha256(path)} != {COMMON_SHA256}")
    print("common.py: sha256 ok")


def make_venv(root: Path) -> dict:
    py = venv(root) / "bin" / "python"
    if not py.exists():
        uv = shutil.which("uv")
        if uv is None:
            raise SystemExit(f"uv is needed to build the sandbox venv ({venv(root)})")
        subprocess.run([uv, "venv", str(venv(root)), "--python", VENV_PYTHON, "-q"], check=True)
        subprocess.run([uv, "pip", "install", "-q", "--python", str(py), *VENV_PACKAGES], check=True)
    probe = "import sys, numpy, scipy; print(sys.version.split()[0], numpy.__version__, scipy.__version__)"
    out = subprocess.run([str(py), "-I", "-c", probe], check=True, capture_output=True, text=True).stdout.split()
    want = {p.split("==")[0]: p.split("==")[1] for p in VENV_PACKAGES}
    if out[1:] != [want["numpy"], want["scipy"]]:
        raise SystemExit(f"sandbox venv has numpy {out[1]}, scipy {out[2]}: remove {venv(root)} and rerun")
    print(f"sandbox venv: python {out[0]}, numpy {out[1]}, scipy {out[2]}")
    return {"python": out[0], "numpy": out[1], "scipy": out[2]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DIR, help="BARC work directory")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(argv)
    a.dir.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in FILES.items():
        download(a.dir, name, size, digest, a.threads)
    fetch_common(a.dir)
    versions = make_venv(a.dir)
    manifest = {"dataset": REPO, "revision": REVISION, "licence": "MIT (dataset card)",
                "files": {n: {"bytes": s, "sha256": d} for n, (s, d) in FILES.items()},
                "common_py": {"repo": "github.com/xu3kev/BARC", "commit": BARC_COMMIT, "path": "seeds/common.py",
                              "sha256": COMMON_SHA256, "licence": "none detected on GitHub"},
                "sandbox_venv": versions}
    (a.dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
