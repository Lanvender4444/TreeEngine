"""TE-038: a cheap, deterministic quality gate - no LLM, no downloaded model.

Runs the controlled suite (325 questions, bundled Markdown) with FTS, the policy controller
and a hashing-embedder hybrid, and compares recall@5 / MRR / recall within 2k context tokens
with ``tools/bench_gate.json``. A drop larger than ``tolerance`` fails (``--check``);
``--update`` rewrites the reference after an intended change.

    python tools/bench_gate.py --check
    python tools/bench_gate.py --update
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = Path(__file__).with_suffix(".json")
METRICS = ("recall@5", "mrr", "recall@2k_tok")
CMD = [
    "-m", "benchmarks.run_retrieval", "--suite", "controlled",
    "--strategies", "fts,policy_fts,policy_hybrid", "--vector", "--embedder", "hashing",
    "--quiet",
]  # fmt: skip


def measure() -> dict[str, dict[str, float]]:
    with tempfile.TemporaryDirectory() as out:
        subprocess.run(
            [sys.executable, *CMD, "--out", out], cwd=ROOT, check=True, stdout=subprocess.DEVNULL
        )
        data = json.loads(next(Path(out).glob("*.json")).read_text("utf-8"))
    return {s: {m: round(float(v[m]), 4) for m in METRICS} for s, v in data["summary"].items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--update", action="store_true")
    ap.add_argument("--tolerance", type=float, default=0.01)
    args = ap.parse_args()
    now = measure()
    if args.update:
        REF.write_text(json.dumps({"tolerance": args.tolerance, "metrics": now}, indent=1) + "\n")
        print(f"reference updated: {REF}")
        return 0
    ref = json.loads(REF.read_text("utf-8"))
    tol = float(ref.get("tolerance", args.tolerance))
    bad = []
    print("| strategy | metric | reference | now |\n|---|---|---|---|")
    for s, ms in ref["metrics"].items():
        for m, v in ms.items():
            cur = now.get(s, {}).get(m)
            print(f"| {s} | {m} | {v:.4f} | {cur if cur is None else f'{cur:.4f}'} |")
            if cur is None or cur < v - tol:
                bad.append(f"{s} {m}: {v:.4f} -> {cur}")
    if bad:
        print("quality gate FAILED:\n  " + "\n  ".join(bad))
        return 1
    print("quality gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
