"""Freeze the retrieval system before reading held-out results.

    python -m benchmarks.freeze --note "V0.4 phase 1"     # record the current fingerprint
    python -m benchmarks.freeze --check                    # what changed since the freeze?

The fingerprint covers every source file that can change a retrieval result: TreeEngine's
core / ingest / structure / storage / retrieval / embeddings packages and the benchmark
strategies, plus the frozen traditional-RAG chunk configuration. Every report states whether it
was produced by the frozen system; a run whose fingerprint differs is marked, so held-out
numbers cannot silently come from a system tuned on held-out failures.
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
PACKAGES = ("core", "ingest", "structure", "storage", "retrieval", "embeddings")
DEFAULT_CHUNK = {"size": 600, "overlap": 100, "selected_by": "default (design doc)"}


def source_files() -> list[Path]:
    files = []
    for sub in PACKAGES:
        files += sorted((ROOT / "treeengine" / sub).glob("*.py"))
        files += sorted((ROOT / "treeengine" / sub).glob("*.sql"))
    files += sorted((HERE / "strategies").glob("*.py"))
    return files


def file_hashes() -> dict[str, str]:
    return {
        str(f.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
            f.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()[:16]
        for f in source_files()
    }


def fingerprint(hashes: dict[str, str] | None = None, chunk: dict[str, Any] | None = None) -> str:
    h = hashlib.sha256()
    for k, v in sorted((hashes or file_hashes()).items()):
        h.update(f"{k}:{v}\n".encode())
    c = chunk or load().get("chunk", DEFAULT_CHUNK)
    h.update(f"chunk:{c.get('size')}/{c.get('overlap')}".encode())
    return h.hexdigest()[:16]


def load() -> dict[str, Any]:
    if FROZEN.exists():
        data: dict[str, Any] = json.loads(FROZEN.read_text(encoding="utf-8"))
        return data
    return {}


def chunk_config() -> tuple[int, int]:
    c = load().get("chunk", DEFAULT_CHUNK)
    return int(c["size"]), int(c["overlap"])


def status() -> dict[str, Any]:
    """{'frozen': bool, 'since': date, 'changed': [files]} for the report header."""
    data = load()
    if not data.get("fingerprint"):
        return {"frozen": False, "since": None, "changed": [], "note": "never frozen"}
    now = file_hashes()
    then = data.get("files", {})
    changed = sorted({k for k in now if then.get(k) != now[k]} | {k for k in then if k not in now})
    same = fingerprint(now, data.get("chunk")) == data["fingerprint"]
    return {
        "frozen": same and not changed,
        "since": data.get("frozen_at"),
        "changed": changed,
        "note": data.get("note", ""),
    }


def describe(st: dict[str, Any]) -> str:
    if st["frozen"]:
        return f"frozen system ({st['since']}{'; ' + st['note'] if st['note'] else ''})"
    if st["since"] is None:
        return "retrieval not frozen yet (python -m benchmarks.freeze)"
    files = ", ".join(st["changed"][:5]) + (" …" if len(st["changed"]) > 5 else "")
    return f"**CHANGED since the freeze of {st['since']}**: {files or 'chunk config'}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.freeze")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--note", default="")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--chunk-overlap", type=int, default=None)
    ap.add_argument("--chunk-selected-by", default=None)
    args = ap.parse_args(argv)
    if args.check:
        st = status()
        print(describe(st))
        for f in st["changed"]:
            print("  changed:", f)
        return 0 if st["frozen"] else 1
    data = load()
    chunk = dict(data.get("chunk", DEFAULT_CHUNK))
    if args.chunk_size:
        chunk = {
            "size": args.chunk_size,
            "overlap": args.chunk_overlap if args.chunk_overlap is not None else chunk["overlap"],
            "selected_by": args.chunk_selected_by or "manual",
        }
    hashes = file_hashes()
    data = {
        "frozen_at": time.strftime("%Y-%m-%d %H:%M"),
        "note": args.note,
        "fingerprint": fingerprint(hashes, chunk),
        "chunk": chunk,
        "files": hashes,
    }
    FROZEN.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(
        f"frozen {data['fingerprint']} ({len(hashes)} files, "
        f"chunk {chunk['size']}/{chunk['overlap']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
