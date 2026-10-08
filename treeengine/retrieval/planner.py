"""Retrieval Planner V0: rules (+ optional small-model classification) decide whether to run
FTS first (LOOKUP), Tree first (DOCUMENT_REASONING) or Tree -> scoped FTS (HYBRID)."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum

from ..core.config import EngineConfig
from ..core.models import Evidence
from ..core.protocols import LLMProvider
from ..llm import prompts
from .base import dedupe
from .fts import FTSRetriever
from .result import SearchResult, SearchStats, Trace
from .tree import TreeRetriever

log = logging.getLogger(__name__)


class QueryType(StrEnum):
    LOOKUP = "LOOKUP"
    DOCUMENT_REASONING = "DOCUMENT_REASONING"
    HYBRID = "HYBRID"


_SECTION = re.compile(
    r"(章节|章|节|部分|一节|小节|段落|附录|\bsection\b|\bchapter\b|\bpart\b|\bappendix\b)", re.I
)
_MENTION = re.compile(
    r"(提到|提及|涉及|出现|谈到|讲到|哪些地方|哪里|哪儿|\bmention|\brefer|\bdiscuss|\bwhere\b)",
    re.I,
)
_LOOKUP_STRONG = re.compile(
    r"(多少|几个|几年|哪一年|哪年|何时|什么时候|是谁|叫什么|定义|数值|金额|比例|版本号|"
    r"\bhow (many|much)\b|\bwhat year\b|\bwhen\b|\bwho\b|\bwhat is the (value|number|name)\b)",
    re.I,
)
_REASONING = re.compile(
    r"(为什么|为何|如何|怎么|怎样|原因|解释|分析|影响|区别|比较|对比|总结|概括|论证|评价|意味着|"
    r"\bwhy\b|\bhow\b|\bexplain|\bsummar|\bcompare|\bdifference|\bimpact|\breason|\banaly)",
    re.I,
)
_IDENT = re.compile(r"(\d|\"|“|`|\w+\(\)|[a-z]+_[a-z_]+|[a-z]+[A-Z]\w+|[A-Z]{2,})")


@dataclass
class RetrievalPlan:
    query: str
    query_type: QueryType
    steps: list[str] = field(default_factory=list)
    by: str = "rules"


class RetrievalPlanner:
    def __init__(
        self,
        tree: TreeRetriever,
        fts: FTSRetriever,
        llm: LLMProvider | None = None,
        config: EngineConfig | None = None,
        use_llm: bool = False,
    ) -> None:
        self.tree = tree
        self.fts = fts
        self.llm = llm if use_llm else None
        self.config = config or EngineConfig()

    # ------------------------------------------------------------------ classification
    @staticmethod
    def classify_rules(query: str) -> QueryType:
        q = query.strip()
        if _SECTION.search(q) and _MENTION.search(q):
            return QueryType.HYBRID
        if _LOOKUP_STRONG.search(q):
            return QueryType.LOOKUP
        if _REASONING.search(q):
            return QueryType.DOCUMENT_REASONING
        if _IDENT.search(q) or len(q) <= 12 or len(q.split()) <= 3:
            return QueryType.LOOKUP
        return QueryType.HYBRID

    def classify(self, query: str) -> tuple[QueryType, str]:
        if self.llm is not None:
            try:
                reply = self.llm.complete(
                    prompts.PLANNER.format(question=query),
                    system=prompts.PLANNER_SYSTEM,
                    max_tokens=10,
                ).upper()
                for t in QueryType:
                    if t.value in reply:
                        return t, "llm"
            except Exception as e:
                log.warning("LLM planner failed, using rules: %s", e)
        return self.classify_rules(query), "rules"

    def plan(self, query: str, mode: QueryType | str | None = None) -> RetrievalPlan:
        if mode is not None:
            qt, by = QueryType(mode if isinstance(mode, str) else mode.value), "forced"
        else:
            qt, by = self.classify(query)
        steps = {
            QueryType.LOOKUP: ["fts", "tree (fallback if fts is empty)"],
            QueryType.DOCUMENT_REASONING: ["tree", "fts (fallback if tree is empty)"],
            QueryType.HYBRID: ["tree (locate sections)", "fts scoped to sections", "tree blocks"],
        }[qt]
        return RetrievalPlan(query, qt, steps, by)

    # ------------------------------------------------------------------ execution
    def search(
        self,
        query: str,
        document_id: str | None = None,
        limit: int = 10,
        mode: QueryType | str | None = None,
    ) -> tuple[list[Evidence], RetrievalPlan]:
        res = self.retrieve(query, document_id=document_id, limit=limit, mode=mode)
        return res.evidence, res.plan

    def retrieve(
        self,
        query: str,
        document_id: str | None = None,
        limit: int = 10,
        mode: QueryType | str | None = None,
    ) -> SearchResult:
        """Managed retrieval with trace and stats (LLM usage is filled in by the caller that
        owns the metered provider)."""
        t0 = time.perf_counter()
        trace = Trace()
        plan = self.plan(query, mode)
        trace.add("plan", query_type=plan.query_type.value, by=plan.by, steps=plan.steps)
        qt = plan.query_type
        ev: list[Evidence]
        if qt is QueryType.LOOKUP:
            ev = self.fts.fts_search(query, document_id=document_id, limit=limit, trace=trace)
            if not ev:
                trace.add("fallback", frm="fts", to="tree", reason="no fts hits")
                ev = self.tree.search(query, document_id=document_id, limit=limit, trace=trace)
        elif qt is QueryType.DOCUMENT_REASONING:
            ev = self.tree.search(query, document_id=document_id, limit=limit, trace=trace)
            if not ev:
                trace.add("fallback", frm="tree", to="fts", reason="no tree evidence")
                ev = self.fts.fts_search(query, document_id=document_id, limit=limit, trace=trace)
        else:
            ev = self._hybrid(query, document_id, limit, trace)
        for e in ev:
            e.metadata.setdefault("strategy", qt.value)
        ev = dedupe(ev)[:limit]

        stats = SearchStats(latency_ms=(time.perf_counter() - t0) * 1000)
        stats.fts_queries = len(trace.of("fts"))
        for step in trace.of("tree"):
            stats.tree_searches += 1
            stats.visited_nodes += step["visited_nodes"]
            stats.total_nodes += step["total_nodes"]
        trace.add("result", evidence=len(ev))
        return SearchResult(query, ev, plan, trace.steps, stats)

    def _hybrid(
        self, query: str, document_id: str | None, limit: int, trace: Trace
    ) -> list[Evidence]:
        doc_ids = (
            [document_id] if document_id else self.tree.candidate_documents(query, trace=trace)
        )
        scoped: list[Evidence] = []
        tree_ev: list[Evidence] = []
        for doc_id in doc_ids:
            loc = self.tree.locate(query, doc_id, limit=limit, trace=trace)
            tree_ev += loc.evidence
            if loc.scope_node_ids:
                scoped += self.fts.fts_search(
                    query,
                    document_id=doc_id,
                    node_ids=loc.scope_node_ids,
                    limit=limit,
                    trace=trace,
                )
        for e in scoped:
            e.metadata["scoped_by"] = "tree"
        scoped.sort(key=lambda e: -(e.score or 0))
        return dedupe(scoped + sorted(tree_ev, key=lambda e: -(e.score or 0)))
