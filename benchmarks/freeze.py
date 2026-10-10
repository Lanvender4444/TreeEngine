"""Freeze the retrieval system before reading held-out results.

    python -m benchmarks.freeze --note "V0.4 phase 1"     # record the current fingerprint
    python -m benchmarks.freeze --check                    # what changed since the freeze?

Three fingerprints, so that a change in one place does not make another look invalid:

  retrieval   every source file that can change a retrieval result: TreeEngine's core / ingest /
              structure / pdf / storage / retrieval / embeddings packages and the benchmark
              strategies, plus the frozen chunk configuration and candidate pool. "Frozen" in a
              report means this one.
  dataset     manifests, queries and reference structures of every suite
  evaluation  metrics, judges, the answerer, context reconstruction (treeengine/context and
              its benchmark adapter) and the corpus loader

Reports, charts and run scripts are in none of them: changing how a number is shown does not
change the number. Every report states whether it was produced by the frozen retrieval system
and lists dataset / evaluation changes separately; held-out numbers cannot silently come from a
system tuned on held-out failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
ROOT = HERE.parent
FROZEN = HERE / "FROZEN.json"
PACKAGES = ("core", "ingest", "structure", "pdf", "storage", "retrieval", "embeddings")
DEFAULT_CHUNK = {"size": 600, "overlap": 100, "selected_by": "default (design doc)"}
# candidates every retriever returns before the answer context is filled / reconstructed;
# RRF results depend on it (top 5 of a depth-5 fusion != top 5 of a depth-50 fusion)
DEFAULT_RETRIEVAL = {"candidate_pool": 50}


GROUPS = ("retrieval", "dataset", "evaluation")


def source_files(group: str = "retrieval") -> list[Path]:
    files: list[Path] = []
    if group == "retrieval":
        for sub in PACKAGES:
            files += sorted((ROOT / "treeengine" / sub).glob("*.py"))
            files += sorted((ROOT / "treeengine" / sub).glob("*.sql"))
        files += sorted((HERE / "strategies").glob("*.py"))
    elif group == "dataset":
        ds = HERE / "datasets"
        for pattern in ("*/manifest.json", "*/queries.jsonl", "*/structures/*.json"):
            files += sorted(ds.glob(pattern))
    elif group == "evaluation":
        for sub in ("metrics", "judges"):
            files += sorted((HERE / sub).glob("*.py"))
        # the reading layer: what the answer model sees, not what retrieval finds
        files += sorted((ROOT / "treeengine" / "context").glob("*.py"))
        files += [
            HERE / "answer.py",
            HERE / "context.py",
            HERE / "loader.py",
            HERE / "datasets" / "__init__.py",
        ]
    else:
        raise ValueError(group)
    return [f for f in files if f.exists()]


def file_hashes(group: str = "retrieval") -> dict[str, str]:
    return {
        str(f.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
            f.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()[:16]
        for f in source_files(group)
    }


def group_fingerprint(hashes: dict[str, str]) -> str:
    h = hashlib.sha256()
    for k, v in sorted(hashes.items()):
        h.update(f"{k}:{v}\n".encode())
    return h.hexdigest()[:16]


def fingerprint(
    hashes: dict[str, str] | None = None,
    chunk: dict[str, Any] | None = None,
    pool: int | None = None,
) -> str:
    h = hashlib.sha256()
    for k, v in sorted((hashes or file_hashes()).items()):
        h.update(f"{k}:{v}\n".encode())
    c = chunk or load().get("chunk", DEFAULT_CHUNK)
    h.update(f"chunk:{c.get('size')}/{c.get('overlap')}".encode())
    h.update(f"candidate_pool:{pool or candidate_pool()}".encode())
    return h.hexdigest()[:16]


def load() -> dict[str, Any]:
    if FROZEN.exists():
        data: dict[str, Any] = json.loads(FROZEN.read_text(encoding="utf-8"))
        return data
    return {}


def candidate_pool() -> int:
    return int(load().get("retrieval", DEFAULT_RETRIEVAL).get("candidate_pool", 50))


def chunk_config() -> tuple[int, int]:
    c = load().get("chunk", DEFAULT_CHUNK)
    return int(c["size"]), int(c["overlap"])


def _diff(now: dict[str, str], then: dict[str, str]) -> list[str]:
    return sorted({k for k in now if then.get(k) != now[k]} | {k for k in then if k not in now})


def status() -> dict[str, Any]:
    """{'frozen': bool (retrieval), 'since': date, 'changed': [retrieval files],
    'dataset_changed': [...] | None, 'evaluation_changed': [...] | None} for report headers.
    ``None`` = that group was not recorded by the freeze (older FROZEN.json)."""
    data = load()
    if not data.get("fingerprint"):
        return {
            "frozen": False,
            "since": None,
            "changed": [],
            "note": "never frozen",
            "dataset_changed": None,
            "evaluation_changed": None,
        }
    now = file_hashes()
    changed = _diff(now, data.get("files", {}))
    pool = int(data.get("retrieval", DEFAULT_RETRIEVAL).get("candidate_pool", 50))
    same = fingerprint(now, data.get("chunk"), pool) == data["fingerprint"]
    out: dict[str, Any] = {
        "frozen": same and not changed,
        "since": data.get("frozen_at"),
        "changed": changed,
        "note": data.get("note", ""),
    }
    for group in ("dataset", "evaluation"):
        rec = data.get(group)
        out[f"{group}_changed"] = None if rec is None else _diff(file_hashes(group), rec["files"])
    return out


def _files(xs: list[str]) -> str:
    return ", ".join(xs[:5]) + (" …" if len(xs) > 5 else "")


def describe(st: dict[str, Any]) -> str:
    if st["since"] is None:
        return "retrieval not frozen yet (python -m benchmarks.freeze)"
    if st["frozen"]:
        text = f"frozen retrieval system ({st['since']}{'; ' + st['note'] if st['note'] else ''})"
    else:
        text = (
            f"**retrieval CHANGED since the freeze of {st['since']}**: "
            f"{_files(st['changed']) or 'chunk config'}"
        )
    for group in ("dataset", "evaluation"):
        ch = st.get(f"{group}_changed")
        if ch:
            text += f"; {group} changed since the freeze: {_files(ch)}"
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.freeze")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--note", default="")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--chunk-overlap", type=int, default=None)
    ap.add_argument("--chunk-selected-by", default=None)
    ap.add_argument("--candidate-pool", type=int, default=None)
    args = ap.parse_args(argv)
    if args.check:
        st = status()
        print(describe(st))
        for f in st["changed"]:
            print("  retrieval changed:", f)
        for group in ("dataset", "evaluation"):
            for f in st.get(f"{group}_changed") or []:
                print(f"  {group} changed:", f)
        return 0 if st["frozen"] else 1
    data = load()
    chunk = dict(data.get("chunk", DEFAULT_CHUNK))
    if args.chunk_size:
        chunk = {
            "size": args.chunk_size,
            "overlap": args.chunk_overlap if args.chunk_overlap is not None else chunk["overlap"],
            "selected_by": args.chunk_selected_by or "manual",
        }
    retrieval = dict(data.get("retrieval", DEFAULT_RETRIEVAL))
    if args.candidate_pool:
        retrieval["candidate_pool"] = args.candidate_pool
    hashes = file_hashes()
    data = {
        "frozen_at": time.strftime("%Y-%m-%d %H:%M"),
        "note": args.note,
        "fingerprint": fingerprint(hashes, chunk, retrieval["candidate_pool"]),  # retrieval
        "chunk": chunk,
        "retrieval": retrieval,
        "files": hashes,
    }
    for group in ("dataset", "evaluation"):
        gh = file_hashes(group)
        data[group] = {"fingerprint": group_fingerprint(gh), "files": gh}
    FROZEN.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(
        f"frozen retrieval {data['fingerprint']} ({len(hashes)} files, "
        f"chunk {chunk['size']}/{chunk['overlap']}, candidate_pool "
        f"{retrieval['candidate_pool']}); "
        f"dataset {data['dataset']['fingerprint']}; "
        f"evaluation {data['evaluation']['fingerprint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
