"""TE-001: record a reproducible baseline - code version, environment, extras, SQLite / FTS5,
and the results of the test suite, lint and type checks (failures and skips kept separately).

    python tools/baseline.py [--out docs/operations/baseline.md] [--json baseline.json]

Compare two baselines before a release: same commands, same extras, different commits.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
import re
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRAS = {
    "pdf": ["pypdf", "pypdfium2"],
    "vector": ["fastembed", "sqlite-vec"],
    "mcp": ["mcp"],
    "server": ["fastapi", "uvicorn", "httpx"],
    "langchain": ["langchain-core"],
    "llamaindex": ["llama-index-core"],
    "dev": ["pytest", "ruff", "mypy", "reportlab", "build", "twine"],
}


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except OSError:
        return None


def _run(cmd: list[str]) -> tuple[int, str]:
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return out.returncode, (out.stdout + out.stderr)


def environment() -> dict[str, object]:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        fts5 = True
    except sqlite3.OperationalError:
        fts5 = False
    sys.path.insert(0, str(ROOT))
    import treeengine

    return {
        "treeengine": treeengine.__version__,
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "os": f"{platform.system()} {platform.release()} {platform.machine()}",
        "sqlite": sqlite3.sqlite_version,
        "fts5": fts5,
        "extras": {e: {d: _version(d) for d in ds} for e, ds in EXTRAS.items()},
        "recorded": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def checks() -> dict[str, object]:
    code, out = _run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider"])
    tail = out.strip().splitlines()[-1] if out.strip() else ""
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|error|errors)", tail)}
    skips = re.findall(r"^SKIPPED \[\d+\] (.*)$", out, re.M)
    failures = re.findall(r"^FAILED (.*)$", out, re.M)
    rc_ruff, ruff = _run([sys.executable, "-m", "ruff", "check", "."])
    rc_mypy, mypy = _run([sys.executable, "-m", "mypy", "treeengine"])
    return {
        "pytest": {"exit": code, "summary": tail, "counts": counts},
        "failures": failures,
        "skips": skips,
        "ruff": {"exit": rc_ruff, "summary": ruff.strip().splitlines()[-1:]},
        "mypy": {"exit": rc_mypy, "summary": mypy.strip().splitlines()[-1:]},
    }


def render(env: dict[str, object], res: dict[str, object]) -> str:
    ex = env["extras"]
    assert isinstance(ex, dict)
    rows = "\n".join(
        f"| `{e}` | " + ", ".join(f"{d} {v or '—'}" for d, v in ds.items()) + " |"
        for e, ds in ex.items()
    )
    py = res["pytest"]
    assert isinstance(py, dict)
    skips = res["skips"]
    fails = res["failures"]
    assert isinstance(skips, list) and isinstance(fails, list)
    return f"""# Baseline — treeengine {env["treeengine"]}

Recorded {env["recorded"]} with `python tools/baseline.py`. Reproduce: install the same extras,
run the same command, compare the two files.

| | |
|---|---|
| commit | `{env["commit"] or "n/a (not a git checkout)"}`{" (dirty)" if env["dirty"] else ""} |
| Python | {env["python"]} ({env["implementation"]}) |
| OS | {env["os"]} |
| SQLite | {env["sqlite"]}, FTS5 {"available" if env["fts5"] else "MISSING"} |

## Installed extras

| extra | distributions |
|---|---|
{rows}

## Checks

| check | result |
|---|---|
| `pytest -q -rs` | {py["summary"]} (exit {py["exit"]}) |
| `ruff check .` | {" ".join(res["ruff"]["summary"])} |
| `mypy treeengine` | {" ".join(res["mypy"]["summary"])} |

Failures ({len(fails)}): {", ".join(fails) or "none"}

Skips ({len(skips)}): {"; ".join(skips) or "none"}

## Benchmarks

Benchmark numbers are NOT re-measured here. Each report under `benchmarks/results/` records its
own commit, dataset split, models, budgets and seeds (see `benchmarks/README.md`); the cheap
deterministic gate is `python tools/bench_gate.py --check`.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "operations" / "baseline.md"))
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--commit", default=None, help="record this commit (e.g. 'abc123 + uncommitted changes')"
    )
    args = ap.parse_args()
    env, res = environment(), checks()
    if args.commit:
        env["commit"], env["dirty"] = args.commit, False
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(render(env, res), encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps({"env": env, "checks": res}, indent=1), "utf-8")
    print(f"baseline written to {args.out}")
    return 0 if res["pytest"]["exit"] == 0 else 1  # type: ignore[index]


if __name__ == "__main__":
    raise SystemExit(main())
