"""Retrieval strategies under comparison. All share one ingested Repository.

A  fts             FTS only
B  tree            Tree, heuristic navigation (title + summary + subtree FTS signal)
B0 tree_structure  Tree, heuristic navigation on structure only (title + summary, no FTS signal)
C  tree+fts        Tree locates sections -> FTS inside them (planner HYBRID)
M  managed         Planner decides (LOOKUP / DOCUMENT_REASONING / HYBRID)
D  tree_llm        Tree, LLM navigation (needs an LLM)
D+ tree_llm+fts    Tree with LLM navigation -> scoped FTS (needs an LLM)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from treeengine.core.config import EngineConfig
from treeengine.core.models import Evidence
from treeengine.core.protocols import LLMProvider, Repository
from treeengine.llm.base import MeteredLLM
from treeengine.retrieval.fts import FTSRetriever
from treeengine.retrieval.planner import QueryType, RetrievalPlanner
from treeengine.retrieval.result import Trace
from treeengine.retrieval.tree import TreeRetriever


@dataclass
class RunOutput:
    evidence: list[Evidence]
    target_ids: list[str] | None
    visited_ratio: float | None
    latency_ms: float
    llm_calls: int = 0
    llm_tokens: int = 0


Runner = Callable[[str, str | None, int], RunOutput]

ALL = ["fts", "tree", "tree_structure", "tree+fts", "managed", "tree_llm", "tree_llm+fts"]
NEEDS_LLM = {"tree_llm", "tree_llm+fts"}


def _tree_stats(trace: Trace) -> tuple[list[str], float | None]:
    steps = trace.of("tree")
    targets = [t for s in steps for t in s.get("target_ids", [])]
    visited = sum(s["visited_nodes"] for s in steps)
    total = sum(s["total_nodes"] for s in steps)
    return targets, (visited / total if total else None)


def build_runners(
    repo: Repository, config: EngineConfig | None = None, llm: LLMProvider | None = None
) -> dict[str, Runner]:
    cfg = config or EngineConfig()
    fts = FTSRetriever(repo, cfg)
    tree = TreeRetriever(repo, None, cfg)
    tree_struct = TreeRetriever(repo, None, replace(cfg, tree_use_fts_signal=False))
    planner = RetrievalPlanner(tree, fts, None, cfg)
    meter = MeteredLLM(llm) if llm is not None and not isinstance(llm, MeteredLLM) else llm

    def timed(fn: Callable[[], tuple[list[Evidence], Trace | None]]) -> RunOutput:
        before = meter.usage.snapshot() if isinstance(meter, MeteredLLM) else None
        t0 = time.perf_counter()
        ev, trace = fn()
        ms = (time.perf_counter() - t0) * 1000
        targets, ratio = _tree_stats(trace) if trace is not None else (None, None)
        calls = tokens = 0
        if before is not None and isinstance(meter, MeteredLLM):
            used = meter.usage.since(before)
            calls, tokens = used.calls, used.input_tokens + used.output_tokens
        if trace is not None and not trace.of("tree"):
            targets = None
        return RunOutput(ev, targets, ratio, ms, calls, tokens)

    def run_fts(q: str, doc: str | None, k: int) -> RunOutput:
        return timed(lambda: (fts.fts_search(q, document_id=doc, limit=k), None))

    def tree_runner(tr: TreeRetriever) -> Runner:
        def run(q: str, doc: str | None, k: int) -> RunOutput:
            def go() -> tuple[list[Evidence], Trace]:
                t = Trace()
                return tr.search(q, document_id=doc, limit=k, trace=t), t

            return timed(go)

        return run

    def planner_runner(p: RetrievalPlanner, mode: QueryType | None) -> Runner:
        def run(q: str, doc: str | None, k: int) -> RunOutput:
            def go() -> tuple[list[Evidence], Trace]:
                res = p.retrieve(q, document_id=doc, limit=k, mode=mode)
                t = Trace()
                t.steps = res.trace
                return res.evidence, t

            return timed(go)

        return run

    runners: dict[str, Runner] = {
        "fts": run_fts,
        "tree": tree_runner(tree),
        "tree_structure": tree_runner(tree_struct),
        "tree+fts": planner_runner(planner, QueryType.HYBRID),
        "managed": planner_runner(planner, None),
    }
    if meter is not None:
        tree_llm = TreeRetriever(repo, meter, cfg)
        runners["tree_llm"] = tree_runner(tree_llm)
        runners["tree_llm+fts"] = planner_runner(
            RetrievalPlanner(tree_llm, fts, None, cfg), QueryType.HYBRID
        )
    return runners
