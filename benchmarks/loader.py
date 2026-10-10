"""Shared plumbing for run_retrieval / run_qa: suites -> ingested corpus (cached) -> queries."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from treeengine.core.config import EngineConfig
from treeengine.core.models import Document
from treeengine.factory import build_components
from treeengine.ingest.base import detect_source_type, get_adapter
from treeengine.storage.sqlite import SQLiteRepository

from .datasets import CorpusDoc, Suite, fetch, load_suite
from .metrics.cost import count_tokens
from .metrics.retrieval import NodePaths, Query, expected_documents, page_count, validate
from .strategies.base import IndexStats

HERE = Path(__file__).parent
CACHE = HERE / ".cache"


def corpus_key(docs: list[CorpusDoc]) -> str:
    """Cache key: document checksums + every source file that influences ingestion."""
    import treeengine

    h = hashlib.sha256()
    for d in docs:
        h.update(d.name.encode())
        h.update(hashlib.sha256(d.path.read_bytes()).digest())
    root = Path(treeengine.__file__).parent
    for sub in ("core", "ingest", "structure", "pdf", "storage"):
        for f in sorted((root / sub).glob("*")):
            if f.suffix in (".py", ".sql"):
                h.update(f.read_bytes())
    return h.hexdigest()[:16]


@dataclass
class Corpus:
    repo: SQLiteRepository
    doc_ids: dict[str, str]  # corpus name -> document id
    ingest: IndexStats
    cached: bool


def _meta(repo: SQLiteRepository, key: str, value: Any = None) -> Any:
    repo.conn.execute("CREATE TABLE IF NOT EXISTS bench_meta (key TEXT PRIMARY KEY, value TEXT)")
    if value is not None:
        repo.conn.execute(
            "INSERT OR REPLACE INTO bench_meta VALUES (?, ?)", (key, json.dumps(value))
        )
        repo.conn.commit()
        return value
    row = repo.conn.execute("SELECT value FROM bench_meta WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def load_corpus(
    docs: list[CorpusDoc],
    use_cache: bool = True,
    quiet: bool = True,
    *,
    config: EngineConfig | None = None,
    llm: Any = None,
    prepare: Callable[[Document, CorpusDoc], None] | None = None,
    variant: str = "",
) -> Corpus:
    """Ingest (or reuse) the documents with TreeEngine's pipeline.

    ``config`` / ``llm`` / ``prepare`` (a hook that may edit the parsed Document before the
    structure is built, e.g. inject a reference outline) define structure-quality variants;
    ``variant`` must name them so each variant gets its own cache file."""
    path = None
    if use_cache:
        CACHE.mkdir(exist_ok=True)
        key = corpus_key(docs)
        if variant:
            key = hashlib.sha256(f"{key}:{variant}".encode()).hexdigest()[:16]
        path = CACHE / f"corpus-{key}.db"
        if path.exists():
            repo = SQLiteRepository(path)
            by_uri = {Path(d.uri or "").name: d.id for d in repo.list_documents()}
            ids = {d.name: by_uri[d.name] for d in docs if d.name in by_uri}
            if len(ids) == len(docs):
                secs = _meta(repo, "ingest_s") or 0.0
                return Corpus(repo, ids, _ingest_stats(repo, ids, secs, path), True)
            repo.close()
        tmp = path.with_suffix(".tmp")
        for f in CACHE.glob(f"{tmp.name}*"):
            f.unlink()
        repo = SQLiteRepository(tmp)
    else:
        repo = SQLiteRepository(":memory:")
    comps = build_components(repo, llm, config or EngineConfig())
    t0 = time.perf_counter()
    ids = {}
    for d in docs:
        if not quiet:
            print(f"ingesting {d.name}{' [' + variant + ']' if variant else ''}", file=sys.stderr)
        if prepare is None:
            ids[d.name] = comps.pipeline.ingest(d.path).id
        else:
            parsed = get_adapter(detect_source_type(str(d.path))).load(str(d.path))
            prepare(parsed, d)
            ids[d.name] = comps.pipeline.store(parsed).id
    secs = time.perf_counter() - t0
    if path is not None:
        _meta(repo, "ingest_s", secs)
        repo.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        repo.close()
        tmp = path.with_suffix(".tmp")
        shutil.move(str(tmp), str(path))
        for f in CACHE.glob(f"{tmp.name}*"):
            f.unlink()
        repo = SQLiteRepository(path)
    return Corpus(repo, ids, _ingest_stats(repo, ids, secs, path), False)


def _ingest_stats(
    repo: SQLiteRepository, ids: dict[str, str], secs: float, path: Path | None
) -> IndexStats:
    blocks = sum(len(repo.get_document_blocks(i)) for i in ids.values())
    return IndexStats(
        "corpus",
        documents=len(ids),
        blocks=blocks,
        elapsed_seconds=secs,
        index_bytes=path.stat().st_size if path is not None and path.exists() else 0,
        parts=["corpus"],
    )


@dataclass
class DocStats:
    name: str
    pages: int | None
    nodes: int
    tokens: int
    structure: str | None = None  # native / heuristic / flat / llm / oracle ...


def doc_stats(corpus: Corpus) -> dict[str, DocStats]:
    cached = _meta(corpus.repo, "doc_tokens") or {}
    out = {}
    for name, did in corpus.doc_ids.items():
        if name not in cached:
            doc = corpus.repo.get_document(did)
            cached[name] = count_tokens(doc.text or "") if doc else 0
        doc = corpus.repo.get_document(did)
        out[name] = DocStats(
            name,
            page_count(corpus.repo, did),
            corpus.repo.count_nodes(did),
            cached[name],
            (doc.metadata or {}).get("structure_method") if doc else None,
        )
    try:
        _meta(corpus.repo, "doc_tokens", cached)
    except Exception:  # read-only cache file: just recompute next time
        pass
    return out


def load_queries(path: Path) -> list[Query]:
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            out.append(Query.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError) as e:
            raise SystemExit(f"{path}:{i}: bad query line ({e})") from e
    ids = [q.id for q in out]
    dupes = {x for x in ids if ids.count(x) > 1}
    if dupes:
        raise SystemExit(f"duplicate query ids: {sorted(dupes)}")
    return out


def select_queries(
    queries: list[Query],
    docs: list[CorpusDoc],
    complete: bool,
    *,
    qtype: str | None = None,
    doc: str | None = None,
    split: str | None = None,
    per_type: int | None = None,
    limit: int | None = None,
) -> list[Query]:
    names = {d.name for d in docs}
    # corpus-wide queries are only meaningful against the complete corpus
    qs = [q for q in queries if (q.document is None and complete) or q.document in names]
    if qtype:
        qs = [q for q in qs if q.type == qtype]
    if doc:
        qs = [q for q in qs if q.document == doc]
    if split:
        qs = [q for q in qs if q.split == split]
    if per_type:
        # deterministic sample for cost control (e.g. LLM runs): first N queries of each type
        seen: Counter[str] = Counter()
        sampled = []
        for q in qs:
            if seen[q.type] < per_type:
                sampled.append(q)
                seen[q.type] += 1
        qs = sampled
    if limit:
        qs = qs[:limit]
    return qs


def check_queries(queries: list[Query], corpus: Corpus) -> tuple[NodePaths, list[str]]:
    """Validate ground truth and fill in expected_documents for corpus-wide queries."""
    paths = NodePaths(corpus.repo)
    all_ids = list(corpus.doc_ids.values())
    problems = []
    for q in queries:
        ids = [corpus.doc_ids[q.document]] if q.document else all_ids
        problems += validate(q, corpus.repo, paths, ids)
        q.expected_documents = expected_documents(q, corpus.repo, paths, all_ids)
    return paths, problems


def breakdown(queries: list[Query]) -> str:
    split = Counter(q.split for q in queries)
    types = Counter(q.type for q in queries)
    return (
        "("
        + ", ".join(f"{k} {v}" for k, v in sorted(split.items()))
        + "; "
        + ", ".join(f"{k} {v}" for k, v in sorted(types.items()))
        + ")"
    )


def prepare_suite(
    name: str, *, synthetic_only: bool = False, quiet: bool = True
) -> tuple[Suite, list[CorpusDoc], bool]:
    suite = load_suite(name)
    docs = [d for d in suite.docs if d.synthetic] if synthetic_only else list(suite.docs)
    problems = fetch(suite, docs, quiet=quiet)
    if problems:
        for p in problems:
            print("corpus:", p, file=sys.stderr)
        docs = [d for d in docs if d.path.exists()]
        print(f"continuing with {len(docs)} available documents", file=sys.stderr)
    return suite, docs, len(docs) == len(suite.docs)


__all__ = [
    "CACHE",
    "Corpus",
    "DocStats",
    "breakdown",
    "check_queries",
    "corpus_key",
    "doc_stats",
    "load_corpus",
    "load_queries",
    "prepare_suite",
    "select_queries",
]
