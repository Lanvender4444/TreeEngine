"""Retrieval strategies under comparison. All share one ingested Repository.

Names say what the algorithm actually uses:

  fts                     FTS5 / BM25 only
  tree_structure          Tree navigation on structure only (titles, summaries, hierarchy)
  tree_lexical            Tree navigation + lexical matching + subtree FTS signal  (V0.2 "tree")
  tree_llm                Tree navigation by an LLM, one level at a time            (needs --llm)
  tree_structure+fts      tree_structure scope -> FTS inside the scope
  tree_lexical+fts        tree_lexical scope   -> FTS inside the scope  (V0.2 "tree+fts")
  tree_llm+fts            tree_llm scope       -> FTS inside the scope  (needs --llm)
  vector                  block embeddings (needs --embedder)
  fts+vector              FTS and vector fused with RRF
  tree_lexical+vector     tree_lexical scope -> vector inside the scope
  tree_lexical+fts+vector tree_lexical scope -> RRF(FTS, vector) inside the scope
  tree_semantic           Tree navigation guided by FTS + vector subtree signals
  tree_semantic+fts+vector tree_semantic scope -> RRF(FTS, vector) inside the scope
  managed                 rule planner (LOOKUP / DOCUMENT_REASONING / HYBRID)
  managed_llm             LLM planner (needs --llm)

Planner evaluation also reports, from the outcomes above: always_fts (= fts),
always_tree (= tree_lexical), always_hybrid (= tree_lexical+fts), rule_planner (= managed),
llm_planner (= managed_llm) and two oracles (see report.py).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from treeengine.core.config import EngineConfig
from treeengine.core.models import Evidence
from treeengine.core.protocols import LLMProvider, Repository
from treeengine.llm.base import MeteredLLM
from treeengine.retrieval.corpus import CorpusRetriever
from treeengine.retrieval.fts import FTSRetriever
from treeengine.retrieval.fusion import EvidenceMerger
from treeengine.retrieval.planner import QueryType, RetrievalPlanner
from treeengine.retrieval.result import Trace
from treeengine.retrieval.scoped import tree_scoped_search
from treeengine.retrieval.tree import TreeRetriever
from treeengine.retrieval.vector import VectorRetriever


@dataclass
class RunOutput:
    evidence: list[Evidence]
    target_ids: list[str] | None
    visited_ratio: float | None
    latency_ms: float
    llm_calls: int = 0
    llm_tokens: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0


Runner = Callable[[str, str | None, int], RunOutput]

LEXICAL = [
    "fts",
    "tree_structure",
    "tree_lexical",
    "tree_structure+fts",
    "tree_lexical+fts",
    "managed",
]
NEEDS_LLM = ["tree_llm", "tree_llm+fts", "managed_llm"]
NEEDS_VECTOR = [
    "vector",
    "fts+vector",
    "tree_lexical+vector",
    "tree_lexical+fts+vector",
    "tree_semantic",
    "tree_semantic+fts+vector",
]
ALL = LEXICAL + NEEDS_VECTOR + NEEDS_LLM
RENAMED = {"tree": "tree_lexical", "tree+fts": "tree_lexical+fts"}  # V0.2 names


def _tree_stats(trace: Trace) -> tuple[list[str] | None, float | None]:
    steps = trace.of("tree")
    if not steps:
        return None, None
    targets = [t for s in steps for t in s.get("target_ids", [])]
    visited = sum(s["visited_nodes"] for s in steps)
    total = sum(s["total_nodes"] for s in steps)
    return targets, (visited / total if total else None)


def build_runners(
    repo: Repository,
    config: EngineConfig | None = None,
    llm: LLMProvider | None = None,
    vector: VectorRetriever | None = None,
) -> dict[str, Runner]:
    cfg = config or EngineConfig()
    fts = FTSRetriever(repo, cfg)
    corpus = CorpusRetriever(repo)  # lexical document routing for lexical strategies
    corpus_v = CorpusRetriever(repo, vector) if vector is not None else corpus
    merger = EvidenceMerger()
    tree_lex = TreeRetriever(repo, None, cfg, corpus=corpus)
    tree_struct = TreeRetriever(repo, None, replace(cfg, tree_use_fts_signal=False), corpus=corpus)
    meter = MeteredLLM(llm) if llm is not None and not isinstance(llm, MeteredLLM) else llm

    def timed(fn: Callable[[], tuple[list[Evidence], Trace]]) -> RunOutput:
        before = meter.usage.snapshot() if isinstance(meter, MeteredLLM) else None
        t0 = time.perf_counter()
        ev, trace = fn()
        ms = (time.perf_counter() - t0) * 1000
        targets, ratio = _tree_stats(trace)
        out = RunOutput(ev, targets, ratio, ms)
        if before is not None and isinstance(meter, MeteredLLM):
            used = meter.usage.since(before)
            out.llm_calls = used.calls
            out.llm_input_tokens = used.input_tokens
            out.llm_output_tokens = used.output_tokens
            out.llm_tokens = used.input_tokens + used.output_tokens
        return out

    def doc_ids(q: str, doc: str | None, router: CorpusRetriever, t: Trace) -> list[str]:
        return [doc] if doc else [c.document_id for c in router.route(q, trace=t)]

    # -- single retrievers
    def run_fts(q: str, doc: str | None, k: int) -> RunOutput:
        def go() -> tuple[list[Evidence], Trace]:
            t = Trace()
            return fts.fts_search(q, document_id=doc, limit=k, trace=t), t

        return timed(go)

    def tree_runner(tr: TreeRetriever) -> Runner:
        def run(q: str, doc: str | None, k: int) -> RunOutput:
            def go() -> tuple[list[Evidence], Trace]:
                t = Trace()
                return tr.search(q, document_id=doc, limit=k, trace=t), t

            return timed(go)

        return run

    def scoped_runner(
        tr: TreeRetriever,
        inner: Callable[[str, str, Sequence[str], int, Trace | None], list[Evidence]],
        router: CorpusRetriever,
    ) -> Runner:
        def run(q: str, doc: str | None, k: int) -> RunOutput:
            def go() -> tuple[list[Evidence], Trace]:
                t = Trace()
                ids = doc_ids(q, doc, router, t)
                return tree_scoped_search(tr, q, ids, k, inner, t), t

            return timed(go)

        return run

    def fts_inner(q: str, d: str, scope: Sequence[str], k: int, t: Trace | None) -> list[Evidence]:
        return fts.fts_search(q, document_id=d, node_ids=scope, limit=k, trace=t)

    def planner_runner(p: RetrievalPlanner) -> Runner:
        def run(q: str, doc: str | None, k: int) -> RunOutput:
            def go() -> tuple[list[Evidence], Trace]:
                res = p.retrieve(q, document_id=doc, limit=k)
                t = Trace()
                t.steps = res.trace
                return res.evidence, t

            return timed(go)

        return run

    runners: dict[str, Runner] = {
        "fts": run_fts,
        "tree_structure": tree_runner(tree_struct),
        "tree_lexical": tree_runner(tree_lex),
        "tree_structure+fts": scoped_runner(tree_struct, fts_inner, corpus),
        "tree_lexical+fts": scoped_runner(tree_lex, fts_inner, corpus),
        "managed": planner_runner(RetrievalPlanner(tree_lex, fts, None, cfg, corpus=corpus)),
    }

    if vector is not None:
        vec = vector

        def run_vector(q: str, doc: str | None, k: int) -> RunOutput:
            def go() -> tuple[list[Evidence], Trace]:
                t = Trace()
                return vec.vector_search(q, document_id=doc, limit=k, trace=t), t

            return timed(go)

        def fused(q: str, doc: str | None, scope: Sequence[str] | None, k: int, t: Trace | None):
            a = fts.fts_search(q, document_id=doc, node_ids=scope, limit=k, trace=t)
            b = vec.vector_search(q, document_id=doc, node_ids=scope, limit=k, trace=t)
            return merger.rrf({"fts": a, "vector": b}, limit=k, trace=t)

        def run_fused(q: str, doc: str | None, k: int) -> RunOutput:
            def go() -> tuple[list[Evidence], Trace]:
                t = Trace()
                if doc is not None:
                    return fused(q, doc, None, k, t), t
                return fused(q, None, None, k, t), t

            return timed(go)

        def vec_inner(
            q: str, d: str, scope: Sequence[str], k: int, t: Trace | None
        ) -> list[Evidence]:
            return vec.vector_search(q, document_id=d, node_ids=scope, limit=k, trace=t)

        def fused_inner(
            q: str, d: str, scope: Sequence[str], k: int, t: Trace | None
        ) -> list[Evidence]:
            return fused(q, d, scope, k, t)

        runners["vector"] = run_vector
        runners["fts+vector"] = run_fused
        runners["tree_lexical+vector"] = scoped_runner(tree_lex, vec_inner, corpus_v)
        runners["tree_lexical+fts+vector"] = scoped_runner(tree_lex, fused_inner, corpus_v)
        # navigation guided by FTS + vector subtree signals (semantic scope resolution)
        tree_sem = TreeRetriever(repo, None, cfg, corpus=corpus_v, vector=vec)
        runners["tree_semantic"] = tree_runner(tree_sem)
        runners["tree_semantic+fts+vector"] = scoped_runner(tree_sem, fused_inner, corpus_v)

    if meter is not None:
        tree_llm = TreeRetriever(repo, meter, cfg, corpus=corpus)
        runners["tree_llm"] = tree_runner(tree_llm)
        runners["tree_llm+fts"] = scoped_runner(tree_llm, fts_inner, corpus)
        runners["managed_llm"] = planner_runner(
            RetrievalPlanner(tree_lex, fts, meter, cfg, use_llm=True, corpus=corpus)
        )
    return runners


__all__ = ["ALL", "LEXICAL", "NEEDS_LLM", "NEEDS_VECTOR", "RENAMED", "QueryType", "build_runners"]
