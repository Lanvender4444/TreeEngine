"""Benchmark suites. Each suite is a directory with

* ``manifest.json``  documents: real ones pinned by URL + sha256 (downloaded into ``files/``,
  never committed), synthetic ones pointing into ``tests/fixtures``;
* ``queries.jsonl``  ground truth (see ``metrics.retrieval.Query``);
* optionally ``build.py``, the script that produced both from an upstream dataset.

Suites:

  controlled    hand-written, type-balanced queries over 28 real + 5 synthetic documents
                (evidence = text needles, node paths); retrieval only
  longdoc       MMLongBench-Doc, pure-text answerable subset (evidence = pages, gold answers)
  financebench  FinanceBench open-source sample (evidence = pages + text, gold answers)

    python -m benchmarks.datasets controlled     # download / verify one suite's documents
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
SUITES = ("controlled", "longdoc", "financebench")


@dataclass
class CorpusDoc:
    name: str
    path: Path
    lang: str
    kind: str
    synthetic: bool
    url: str | None = None
    sha256: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Suite:
    name: str
    root: Path
    docs: list[CorpusDoc]
    description: str = ""
    has_answers: bool = False  # gold answers -> usable for end-to-end QA

    @property
    def queries_path(self) -> Path:
        return self.root / "queries.jsonl"

    @property
    def files(self) -> Path:
        return self.root / "files"


def load_suite(name: str) -> Suite:
    root = HERE / name
    manifest = root / "manifest.json"
    if not manifest.exists():
        hint = f" (build it with: python -m benchmarks.datasets.{name}.build)"
        raise SystemExit(f"suite {name!r} has no manifest.json{hint}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    docs = []
    for d in data["documents"]:
        synthetic = bool(d.get("synthetic"))
        path = ROOT / d["path"] if synthetic else root / "files" / d["name"]
        extra = {
            k: v
            for k, v in d.items()
            if k not in ("name", "path", "lang", "kind", "synthetic", "url", "sha256")
        }
        docs.append(
            CorpusDoc(
                name=d["name"],
                path=path,
                lang=d.get("lang", ""),
                kind=d.get("kind", ""),
                synthetic=synthetic,
                url=d.get("url"),
                sha256=d.get("sha256"),
                meta=extra,
            )
        )
    return Suite(
        name=name,
        root=root,
        docs=docs,
        description=data.get("description", ""),
        has_answers=bool(data.get("has_answers", False)),
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(suite: Suite, docs: list[CorpusDoc] | None = None, quiet: bool = False) -> list[str]:
    """Download missing real documents and verify checksums. Returns problems (empty = ok)."""
    problems = []
    suite.files.mkdir(parents=True, exist_ok=True)
    for d in docs if docs is not None else suite.docs:
        if d.synthetic:
            if not d.path.exists():
                problems.append(f"missing fixture {d.path}")
            continue
        if not d.path.exists():
            if not d.url:
                problems.append(f"{d.name}: no url")
                continue
            if not quiet:
                print(f"downloading {d.name} ...", file=sys.stderr)
            tmp = d.path.with_suffix(d.path.suffix + ".part")
            try:
                req = urllib.request.Request(d.url, headers={"User-Agent": "treeengine-bench"})
                with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
                    for chunk in iter(lambda: r.read(1 << 20), b""):
                        f.write(chunk)
                tmp.replace(d.path)
            except Exception as e:  # network errors are reported, not raised
                tmp.unlink(missing_ok=True)
                problems.append(f"{d.name}: download failed ({e})")
                continue
        if d.sha256 and sha256_file(d.path) != d.sha256:
            problems.append(f"{d.name}: sha256 mismatch (upstream file changed?)")
    return problems


def main(argv: list[str] | None = None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or ["controlled"]
    bad = 0
    for name in names:
        errs = fetch(load_suite(name))
        for e in errs:
            print("ERROR", e, file=sys.stderr)
        print(f"{name}: " + ("ok" if not errs else f"{len(errs)} problem(s)"))
        bad += len(errs)
    return 1 if bad else 0


__all__ = ["SUITES", "CorpusDoc", "Suite", "fetch", "load_suite", "sha256_file"]
