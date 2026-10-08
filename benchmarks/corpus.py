"""Benchmark corpus: real documents pinned to upstream commits (downloaded, sha256-verified,
not committed) + the synthetic fixtures from ``tests/fixtures``.

    python -m benchmarks.corpus          # download / verify everything in manifest.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
MANIFEST = HERE / "corpus" / "manifest.json"
FILES = HERE / "corpus" / "files"


@dataclass
class CorpusDoc:
    name: str
    path: Path
    lang: str
    kind: str
    synthetic: bool
    url: str | None = None
    sha256: str | None = None


def load_manifest() -> list[CorpusDoc]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out = []
    for d in data["documents"]:
        synthetic = bool(d.get("synthetic"))
        path = ROOT / d["path"] if synthetic else FILES / d["name"]
        out.append(
            CorpusDoc(
                name=d["name"],
                path=path,
                lang=d.get("lang", ""),
                kind=d.get("kind", ""),
                synthetic=synthetic,
                url=d.get("url"),
                sha256=d.get("sha256"),
            )
        )
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(docs: list[CorpusDoc] | None = None, quiet: bool = False) -> list[str]:
    """Download missing real documents and verify checksums. Returns problems (empty = ok)."""
    problems = []
    FILES.mkdir(parents=True, exist_ok=True)
    for d in docs or load_manifest():
        if d.synthetic:
            if not d.path.exists():
                problems.append(f"missing fixture {d.path}")
            continue
        if not d.path.exists():
            assert d.url
            if not quiet:
                print(f"downloading {d.name} ...", file=sys.stderr)
            try:
                with urllib.request.urlopen(d.url, timeout=60) as r:
                    d.path.write_bytes(r.read())
            except Exception as e:  # network errors are reported, not raised
                problems.append(f"{d.name}: download failed ({e})")
                continue
        if d.sha256 and _sha256(d.path) != d.sha256:
            problems.append(f"{d.name}: sha256 mismatch (upstream file changed?)")
    return problems


if __name__ == "__main__":
    errs = fetch()
    for e in errs:
        print("ERROR", e, file=sys.stderr)
    print("corpus ok" if not errs else f"{len(errs)} problem(s)")
    raise SystemExit(1 if errs else 0)
