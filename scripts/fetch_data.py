"""Populate data/raw with arc-agi_{subset}_{challenges,solutions}.json for ARC-AGI-1, ARC-AGI-2 and ConceptARC.

Copies from ../trm-samsung/kaggle/combined when present (offline); otherwise downloads:
ARC-AGI-1 from github fchollet/ARC-AGI, ARC-AGI-2 from github arcprize/ARC-AGI-2 (per-task files, combined here),
ConceptARC from the TRM repo's kaggle/combined files.

    python scripts/fetch_data.py [--dest data/raw] [--local ../trm-samsung/kaggle/combined]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request

SUBSETS = ("training", "evaluation", "concept", "training2", "evaluation2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = os.path.join(ROOT, "..", "trm-samsung", "kaggle", "combined")

TRM_RAW = "https://raw.githubusercontent.com/SamsungSAILMontreal/TinyRecursiveModels/main/kaggle/combined"
# subset -> (GitHub repo, branch, directory of per-task JSON files)
GITHUB = {
    "training": ("fchollet/ARC-AGI", "master", "data/training"),
    "evaluation": ("fchollet/ARC-AGI", "master", "data/evaluation"),
    "training2": ("arcprize/ARC-AGI-2", "main", "data/training"),
    "evaluation2": ("arcprize/ARC-AGI-2", "main", "data/evaluation"),
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "frm-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _download_github(subset: str) -> tuple[dict, dict]:
    """Per-task {train, test} files -> (challenges without test outputs, solutions)."""
    repo, branch, path = GITHUB[subset]
    listing = json.loads(_get(f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"))
    challenges, solutions = {}, {}
    for item in sorted(listing, key=lambda x: x["name"]):
        if not item["name"].endswith(".json"):
            continue
        task = json.loads(_get(item["download_url"]))
        tid = item["name"][: -len(".json")]
        challenges[tid] = {"train": task["train"], "test": [{"input": ex["input"]} for ex in task["test"]]}
        solutions[tid] = [ex["output"] for ex in task["test"]]
    return challenges, solutions


def fetch(dest: str, local: str) -> None:
    os.makedirs(dest, exist_ok=True)
    for subset in SUBSETS:
        names = [f"arc-agi_{subset}_{kind}.json" for kind in ("challenges", "solutions")]
        if all(os.path.isfile(os.path.join(dest, n)) for n in names):
            print(f"{subset}: present")
            continue
        if all(os.path.isfile(os.path.join(local, n)) for n in names):
            for n in names:
                shutil.copyfile(os.path.join(local, n), os.path.join(dest, n))
            print(f"{subset}: copied from {local}")
            continue
        if subset in GITHUB:
            challenges, solutions = _download_github(subset)
            for n, obj in zip(names, (challenges, solutions)):
                with open(os.path.join(dest, n), "w") as f:
                    json.dump(obj, f)
        else:
            for n in names:
                with open(os.path.join(dest, n), "wb") as f:
                    f.write(_get(f"{TRM_RAW}/{n}"))
        print(f"{subset}: downloaded")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--local", default=LOCAL)
    args = ap.parse_args()
    fetch(args.dest, args.local)


if __name__ == "__main__":
    main()
