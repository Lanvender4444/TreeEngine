"""Strategies under comparison. All read the same ingested documents (one Workspace).

TreeEngine (block index; structure-aware)
  fts                       B   FTS5 / BM25 over blocks
  vector                        block embeddings
  fts+vector                    FTS + block vectors, RRF
  tree_structure            E   tree navigation on structure only (titles, summaries, hierarchy)
  tree_lexical              F   tree navigation + lexical matching + subtree FTS signal
  tree_semantic                 tree navigation, subtree signal = max(FTS, vector)
  tree_structure+fts            tree_structure scope -> FTS inside the scope
  tree_lexical+fts          G   tree_lexical scope -> FTS inside the scope
  tree_lexical+vector       H   tree_lexical scope -> vector inside the scope
  tree_lexical+fts+vector   I   tree_lexical scope -> RRF(FTS, vector) inside the scope
  tree_semantic+fts+vector      tree_semantic scope -> RRF(FTS, vector) inside the scope
  managed                   J   rule planner (LOOKUP / DOCUMENT_REASONING / HYBRID)
  managed+vector                rule planner with FTS+vector fusion in its lexical steps
  tree_llm / tree_llm+fts       LLM navigation, one level at a time (needs an LLM)
  tree_llm+vector / +fts+vector LLM navigation scope -> vector / RRF(FTS, vector)
  vector_raw                    ablation: block embeddings of the text only (no section title)
  tree_lexical+vector_raw       ablation: tree scope -> raw block vectors
  managed_llm                   LLM planner (needs an LLM)

Traditional RAG (fixed-size chunks of the raw text; structure-blind)
  rag_bm25                      BM25 over 600-token chunks
  rag_vector                C   embeddings of 600-token chunks
  rag_hybrid                D   BM25 + vector over chunks, RRF  ("strong traditional baseline")

Baseline
  full_context              A   the whole document as context (QA only; skipped over budget)

Oracle (K) is computed from the outcomes in the report, never run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TypeVar

from treeengine.core.models import Evidence
from treeengine.retrieval.corpus import CorpusRetriever
from treeengine.retrieval.fts import FTSRetriever
from treeengine.retrieval.fusion import EvidenceMerger
from treeengine.retrieval.planner import RetrievalPlanner
from treeengine.retrieval.result import Trace
from treeengine.retrieval.scoped import tree_scoped_search
from treeengine.retrieval.tree import TreeRetriever

from .base import BenchmarkStrategy, IndexStats, RetrievalRun, Search, Strategy
from .chunk_rag import ChunkStore
from .full_context import FullContext
from .workspace import Workspace

LEXICAL = [
    "fts",
    "tree_structure",
    "tree_lexical",
    "tree_structure+fts",
    "tree_lexical+fts",
    "managed",
    "rag_bm25",
]
NEEDS_VECTOR = [
    "vector",
    "fts+vector",
    "tree_lexical+vector",
    "tree_lexical+fts+vector",
    "tree_semantic",
    "tree_semantic+fts+vector",
    "managed+vector",
    "rag_vector",
    "rag_hybrid",
    "vector_raw",
    "tree_lexical+vector_raw",
]
NEEDS_LLM = ["tree_llm", "tree_llm+fts", "managed_llm", "tree_llm+vector", "tree_llm+fts+vector"]
NEEDS_BOTH = ["tree_llm+vector", "tree_llm+fts+vector"]  # LLM navigation + embeddings
QA_ONLY = ["full_context"]
ALL = LEXICAL + NEEDS_VECTOR + NEEDS_LLM + QA_ONLY
RENAMED = {"tree": "tree_lexical", "tree+fts": "tree_lexical+fts"}  # V0.2 names
PRESETS = {
    # benchmark design doc, first round: is Tree useful, does Vector add semantics, is the
    # traditional hybrid already strong enough?
    "round1": [
        "fts",
        "rag_vector",
        "rag_hybrid",
        "tree_lexical",
        "tree_lexical+fts",
        "tree_lexical+vector",
        "tree_lexical+fts+vector",
    ],
    # design doc V0.4 Phase 1: the full lexical / semantic / hybrid / tree-aware matrix
    "v04": [
        "fts",
        "rag_bm25",
        "rag_vector",
        "rag_hybrid",
        "vector",
        "fts+vector",
        "tree_lexical",
        "tree_lexical+fts",
        "tree_lexical+vector",
        "tree_lexical+fts+vector",
        "managed",
        "managed+vector",
    ],
    # block vector ablation: granularity vs title metadata vs tree scope
    "vector_ablation": [
        "rag_vector",
        "vector_raw",
        "vector",
        "tree_lexical+vector_raw",
        "tree_lexical+vector",
    ],
    "llm_tree": [
        "tree_structure",
        "tree_lexical",
        "tree_llm",
        "tree_llm+fts",
        "tree_llm+vector",
    ],
}
T = TypeVar("T")
# (query, document_id, scope node ids, limit, trace) -> evidence
Inner = Callable[[str, str | None, Sequence[str] | None, int, Trace | None], list[Evidence]]
FAMILY = {"full_context": "baseline", **{s: "traditional" for s in ALL if s.startswith("rag_")}}


def build_strategies(ws: Workspace, names: Sequence[str]) -> dict[str, BenchmarkStrategy]:
    """Instantiate the requested strategies (indexes are built lazily on first use)."""
    repo, cfg = ws.repo, ws.config
    fts = FTSRetriever(repo, cfg)
    merger = EvidenceMerger()
    corpus = CorpusRetriever(repo)  # lexical document routing
    lazy: dict[str, object] = {}

    def vec():
        return ws.block_vectors()

    def vec_raw():
        return ws.block_vectors_raw()

    def corpus_raw() -> CorpusRetriever:
        if "corpus_raw" not in lazy:
            lazy["corpus_raw"] = CorpusRetriever(repo, vec_raw())
        return lazy["corpus_raw"]  # type: ignore[return-value]

    def corpus_v() -> CorpusRetriever:
        if "corpus_v" not in lazy:
            lazy["corpus_v"] = CorpusRetriever(repo, vec())
        return lazy["corpus_v"]  # type: ignore[return-value]

    tree_lex = TreeRetriever(repo, None, cfg, corpus=corpus)
    tree_struct = TreeRetriever(repo, None, replace(cfg, tree_use_fts_signal=False), corpus=corpus)

    def tree_sem() -> TreeRetriever:
        if "tree_sem" not in lazy:
            lazy["tree_sem"] = TreeRetriever(repo, None, cfg, corpus=corpus_v(), vector=vec())
        return lazy["tree_sem"]  # type: ignore[return-value]

    def tree_llm() -> TreeRetriever:
        if "tree_llm" not in lazy:
            lazy["tree_llm"] = TreeRetriever(repo, ws.llm, cfg, corpus=corpus)
        return lazy["tree_llm"]  # type: ignore[return-value]

    # ---- in-scope searches: (query, doc, scope, k, trace) -> evidence
    def fts_in(q: str, d: str | None, scope: Sequence[str] | None, k: int, t: Trace | None):
        return fts.fts_search(q, document_id=d, node_ids=scope, limit=k, trace=t)

    def vec_in(q: str, d: str | None, scope: Sequence[str] | None, k: int, t: Trace | None):
        return vec().vector_search(q, document_id=d, node_ids=scope, limit=k, trace=t)

    def vec_raw_in(q: str, d: str | None, scope: Sequence[str] | None, k: int, t: Trace | None):
        return vec_raw().vector_search(q, document_id=d, node_ids=scope, limit=k, trace=t)

    def fused_in(q: str, d: str | None, scope: Sequence[str] | None, k: int, t: Trace | None):
        a = fts_in(q, d, scope, k, t)
        b = vec_in(q, d, scope, k, t)
        return merger.rrf({"fts": a, "vector": b}, limit=k, trace=t)

    def whole(inner: Inner) -> Search:
        def run(q: str, d: str | None, k: int, t: Trace) -> list[Evidence]:
            return inner(q, d, None, k, t)

        return run

    def tree_only(tr: Callable[[], TreeRetriever]) -> Search:
        def run(q: str, d: str | None, k: int, t: Trace) -> list[Evidence]:
            return tr().search(q, document_id=d, limit=k, trace=t)

        return run

    def scoped(
        tr: Callable[[], TreeRetriever], inner: Inner, router: Callable[[], CorpusRetriever]
    ) -> Search:
        def run(q: str, d: str | None, k: int, t: Trace) -> list[Evidence]:
            ids = [d] if d else [c.document_id for c in router().route(q, trace=t)]
            return tree_scoped_search(tr(), q, ids, k, inner, t)

        return run

    def planner(make: Callable[[], RetrievalPlanner]) -> Search:
        def run(q: str, d: str | None, k: int, t: Trace) -> list[Evidence]:
            res = make().retrieve(q, document_id=d, limit=k)
            t.steps = res.trace
            return res.evidence

        return run

    def planners(kind: str) -> RetrievalPlanner:
        if kind not in lazy:
            if kind == "rule":
                lazy[kind] = RetrievalPlanner(tree_lex, fts, None, cfg, corpus=corpus)
            elif kind == "vector":
                lazy[kind] = RetrievalPlanner(
                    tree_lex, fts, None, cfg, corpus=corpus_v(), vector=vec(), use_vector=True
                )
            else:
                lazy[kind] = RetrievalPlanner(
                    tree_lex, fts, ws.llm, cfg, use_llm=True, corpus=corpus
                )
        return lazy[kind]  # type: ignore[return-value]

    def chunks() -> ChunkStore:
        store: ChunkStore = ws.chunks()
        return store

    def rag(kind: str) -> Search:
        def run(q: str, d: str | None, k: int, t: Trace) -> list[Evidence]:
            store = chunks()
            if kind == "bm25":
                return store.evidence(store.bm25(q, d, k), "rag_bm25")
            ws.chunk_vectors()
            if kind == "vector":
                return store.evidence(store.vector(q, d, k), "rag_vector")
            a = store.evidence(store.bm25(q, d, k), "bm25")
            b = store.evidence(store.vector(q, d, k), "vector")
            return merger.rrf({"bm25": a, "vector": b}, limit=k, trace=t)

        return run

    def r(f: T) -> Callable[[], T]:
        return lambda: f

    table: dict[str, tuple[Search, tuple[str, ...]]] = {
        "fts": (whole(fts_in), ()),
        "vector": (whole(vec_in), ("block_vectors",)),
        "fts+vector": (whole(fused_in), ("block_vectors",)),
        "tree_structure": (tree_only(r(tree_struct)), ()),
        "tree_lexical": (tree_only(r(tree_lex)), ()),
        "tree_semantic": (tree_only(tree_sem), ("block_vectors",)),
        "tree_structure+fts": (scoped(r(tree_struct), fts_in, r(corpus)), ()),
        "tree_lexical+fts": (scoped(r(tree_lex), fts_in, r(corpus)), ()),
        "tree_lexical+vector": (scoped(r(tree_lex), vec_in, corpus_v), ("block_vectors",)),
        "tree_lexical+fts+vector": (
            scoped(r(tree_lex), fused_in, corpus_v),
            ("block_vectors",),
        ),
        "tree_semantic+fts+vector": (scoped(tree_sem, fused_in, corpus_v), ("block_vectors",)),
        "managed": (planner(lambda: planners("rule")), ()),
        "managed+vector": (planner(lambda: planners("vector")), ("block_vectors",)),
        "tree_llm": (tree_only(tree_llm), ()),
        "tree_llm+fts": (scoped(tree_llm, fts_in, r(corpus)), ()),
        "tree_llm+vector": (scoped(tree_llm, vec_in, corpus_v), ("block_vectors",)),
        "tree_llm+fts+vector": (scoped(tree_llm, fused_in, corpus_v), ("block_vectors",)),
        "vector_raw": (whole(vec_raw_in), ("block_vectors_raw",)),
        "tree_lexical+vector_raw": (
            scoped(r(tree_lex), vec_raw_in, corpus_raw),
            ("block_vectors_raw",),
        ),
        "managed_llm": (planner(lambda: planners("llm")), ()),
        "rag_bm25": (rag("bm25"), ("chunks",)),
        "rag_vector": (rag("vector"), ("chunks", "chunk_vectors")),
        "rag_hybrid": (rag("hybrid"), ("chunks", "chunk_vectors")),
    }
    out: dict[str, BenchmarkStrategy] = {}
    for name in names:
        if name == "full_context":
            out[name] = FullContext(ws)
            continue
        if name not in table:
            raise KeyError(f"unknown strategy {name!r}; choose from {ALL}")
        fn, needs = table[name]
        requires = ("chunks",) + needs if name.startswith("rag_") else ("corpus",) + needs
        requires = tuple(dict.fromkeys(requires))
        out[name] = Strategy(
            name,
            FAMILY.get(name, "treeengine"),
            ws,
            fn,
            requires,
        )
    return out


__all__ = [
    "ALL",
    "LEXICAL",
    "NEEDS_BOTH",
    "NEEDS_LLM",
    "NEEDS_VECTOR",
    "PRESETS",
    "QA_ONLY",
    "RENAMED",
    "BenchmarkStrategy",
    "IndexStats",
    "RetrievalRun",
    "Strategy",
    "Workspace",
    "build_strategies",
]
