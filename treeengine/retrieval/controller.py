"""Retrieval Controller: activates retrieval paths by policy, observes, decides whether to keep
going, enforces budgets, and hands the result to the ContextBuilder.

    Query
      -> fast paths (FTS / Vector, by weight)        observations
      -> evaluate: enough evidence?                   deterministic soft stop
      -> tree reasoning steps (LLM, bounded actions)  only when the policy allows / needs it
      -> weighted rank fusion of all paths            ranks only, never raw scores
      -> ContextBuilder                               ContextSpan[] within the context budget

The controller does no retrieval itself; it calls the existing retrievers and navigation
primitives (roots / children / read blocks / search in a subtree). Tree reasoning never filters
the global retrieval: its hits are one more ranked list.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..context import BlockContextBuilder
from ..core.config import EngineConfig
from ..core.models import Evidence, Node
from ..core.protocols import LLMProvider, Repository
from ..core.text import query_terms, truncate
from ..llm import prompts
from ..llm.base import estimate_tokens, extract_json
from .corpus import CorpusRetriever
from .fts import FTSRetriever
from .fusion import EvidenceMerger
from .policy import RetrievalPolicy
from .result import SearchResult, SearchStats
from .state import RetrievalAnchor, RetrievalState
from .tree import TreeRetriever
from .vector import VectorRetriever

ACTIONS = ("SELECT", "READ", "SEARCH", "PARENT", "STOP")
READ_BLOCKS = 5


@dataclass
class ControllerPlan:
    """What the controller decided (stands in for the planner's RetrievalPlan)."""

    policy: RetrievalPolicy
    mode: str
    paths: list[str] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def query_type(self) -> str:
        return f"POLICY:{self.mode.upper()}"


class RetrievalController:
    def __init__(
        self,
        repo: Repository,
        fts: FTSRetriever,
        *,
        vector: VectorRetriever | None = None,
        corpus: CorpusRetriever | None = None,
        tree: TreeRetriever | None = None,
        llm: LLMProvider | None = None,
        config: EngineConfig | None = None,
        token_counter: Callable[[str], int] = estimate_tokens,
    ) -> None:
        self.repo = repo
        self.fts = fts
        self.vector = vector
        self.corpus = corpus
        self.tree = tree
        self.llm = llm
        self.config = config or EngineConfig()
        self.count = token_counter

    # ------------------------------------------------------------------ main loop
    def retrieve(
        self,
        query: str,
        *,
        document_id: str | None = None,
        policy: RetrievalPolicy | None = None,
    ) -> SearchResult:
        policy = policy or RetrievalPolicy()
        t0 = time.perf_counter()
        st = RetrievalState(query=query, document_id=document_id)
        tr = st.trace
        tr.add(
            "policy",
            mode=policy.mode,
            weights={
                "fts": policy.fts_weight,
                "vector": policy.vector_weight,
                "tree": policy.tree_weight,
            },
            budgets={
                "max_steps": policy.max_steps,
                "max_llm_calls": policy.max_llm_calls,
                "max_tree_visits": policy.max_tree_visits,
                "candidate_pool": policy.candidate_pool,
                "context_budget": policy.context_budget,
            },
        )

        # 1) fast paths
        if policy.fts_weight > 0:
            hits = self.fts.fts_search(
                query, document_id=document_id, limit=policy.candidate_pool, trace=tr
            )
            st.fts_queries += 1
            st.add("fts", hits)
        if policy.vector_weight > 0:
            if self.vector is None:
                tr.add("vector", skipped="no embedder configured")
            else:
                hits = self.vector.vector_search(
                    query, document_id=document_id, limit=policy.candidate_pool, trace=tr
                )
                st.vector_queries += 1
                st.add("vector", hits)

        # 2) evaluate, 3) tree reasoning when the policy allows / needs it
        enough, why = self._enough(st, policy)
        tr.add("evaluate", at_step=st.step, enough=enough, why=why)
        run, steps, stop_on_enough = self._tree_plan(policy, enough)
        if run:
            self._tree_episode(st, policy, steps, stop_on_enough)
        elif not st.stop_reason:
            st.stop_reason = (
                "enough evidence after the first pass"
                if enough
                else ("tree reasoning disabled" if not policy.tree_enabled else "agentic off")
            )
        tr.add("stop", reason=st.stop_reason, steps=st.step, llm_calls=st.llm_calls)

        # 4) fuse paths by rank, 5) build the reading context
        evidence = self._fuse(st, policy)
        spans = BlockContextBuilder(
            self.repo, policy=policy.context_policy, token_counter=self.count
        ).build(evidence, token_budget=policy.context_budget)
        ctx_tokens = sum(s.token_count for s in spans)
        tr.add(
            "context",
            policy=policy.context_policy,
            spans=len(spans),
            tokens=ctx_tokens,
            budget=policy.context_budget,
        )
        stats = SearchStats(
            latency_ms=(time.perf_counter() - t0) * 1000,
            fts_queries=st.fts_queries,
            vector_queries=st.vector_queries,
            tree_searches=1 if st.tree_steps else 0,
            visited_nodes=len(st.visited_nodes),
            llm_calls=st.llm_calls,
            tree_steps=st.tree_steps,
            candidate_count=len(evidence),
            context_tokens=ctx_tokens,
        )
        plan = ControllerPlan(policy, policy.mode, list(st.used_retrievers), st.stop_reason)
        return SearchResult(query, evidence, plan, tr.steps, stats, context_spans=spans)

    # ------------------------------------------------------------------ decisions
    def _tree_plan(self, policy: RetrievalPolicy, enough: bool) -> tuple[bool, int, bool]:
        """-> (run tree reasoning?, step budget, stop as soon as evidence is enough?)"""
        if not policy.tree_enabled:
            return False, 0, False
        mode = policy.mode
        if not policy.fast_paths:  # tree reasoning is the retrieval itself
            steps = policy.max_steps if mode in ("guided", "full") else min(2, policy.max_steps)
            return True, steps, False
        if mode == "off":
            return False, 0, False
        if mode == "fallback":
            return (not enough), min(2, policy.max_steps), False
        if mode == "guided":
            return (not enough), policy.max_steps, True
        return True, policy.max_steps, False  # full: the LLM decides, budgets still apply

    def _enough(self, st: RetrievalState, policy: RetrievalPolicy) -> tuple[bool, str]:
        """Deterministic soft stop: enough distinct evidence that covers the query's terms."""
        top = self._fuse(st, policy)[:10]
        distinct = len({(e.document_id, e.block_id or e.node_id) for e in top})
        terms = query_terms(st.query)
        text = " ".join(e.content.lower() for e in top)
        coverage = sum(1 for t in terms if t.lower() in text) / len(terms) if terms else 1.0
        ok = distinct >= policy.min_evidence and coverage >= policy.min_term_coverage
        return ok, f"{distinct} distinct items, {coverage:.0%} of query terms covered"

    def _gate(self, st: RetrievalState, policy: RetrievalPolicy) -> str | None:
        """Hard budgets: never exceeded, whatever the LLM wants."""
        if st.step >= policy.max_steps:
            return "max_steps reached"
        if st.llm_calls >= policy.max_llm_calls:
            return "max_llm_calls reached"
        if len(st.visited_nodes) >= policy.max_tree_visits:
            return "max_tree_visits reached"
        return None

    # ------------------------------------------------------------------ tree reasoning
    def _document(self, st: RetrievalState) -> str | None:
        if st.document_id:
            return st.document_id
        counts = Counter(e.document_id for e in st.all_evidence()[:20])
        if counts:
            return counts.most_common(1)[0][0]
        if self.corpus is not None:
            routed = self.corpus.route(st.query, max_docs=1, trace=st.trace)
            if routed:
                return routed[0].document_id
        docs = self.repo.list_documents()
        return docs[0].id if docs else None

    def _tree_episode(
        self, st: RetrievalState, policy: RetrievalPolicy, steps: int, stop_on_enough: bool
    ) -> None:
        doc = self._document(st)
        if doc is None:
            st.stop_reason = "no document to navigate"
            return
        if self.llm is None:
            self._heuristic_tree(st, doc, policy)
            return
        document = self.repo.get_document(doc)
        position: Node | None = None
        path: list[str] = []
        for _ in range(steps):
            gate = self._gate(st, policy)
            if gate:
                st.stop_reason = gate
                return
            if stop_on_enough:
                enough, why = self._enough(st, policy)
                if enough:
                    st.stop_reason = f"enough evidence ({why})"
                    return
            options = self._options(doc, position)
            if not options and position is not None:
                self._read(st, doc, position, "leaf section reached")
                st.stop_reason = "leaf section read"
                return
            for n in options:
                st.visited_nodes.add(n.id)
            alias = {f"s{i + 1}": n for i, n in enumerate(options)}
            if position is not None:
                alias["here"] = position
            reply = self._ask(st, document.title if document else doc, path, alias, steps)
            st.step += 1
            st.tree_steps += 1
            action = str((reply or {}).get("action", "")).upper()
            node = alias.get(str((reply or {}).get("node", "")))
            reason = str((reply or {}).get("reason", ""))[:200]
            st.trace.add(
                "tree_step",
                at_step=st.step,
                action=action or "INVALID",
                node=node.title if node else None,
                options=len(options),
                reason=reason,
            )
            if action not in ACTIONS or (action in ("SELECT", "READ", "SEARCH") and node is None):
                st.stop_reason = "invalid tree-reasoning reply"
                return
            if action == "STOP":
                st.stop_reason = f"llm stop: {reason or 'enough'}"
                return
            if action == "PARENT":
                position = (
                    self.repo.get_node(position.parent_id)
                    if position and position.parent_id
                    else None
                )
                path = path[:-1]
            elif action == "SELECT":
                assert node is not None
                position = node
                path.append(node.title)
            elif action == "READ":
                assert node is not None
                self._read(st, doc, node, reason)
            elif action == "SEARCH":
                assert node is not None
                q = str((reply or {}).get("query") or st.query)
                hits = self.fts.fts_search(q, document_id=doc, node_id=node.id, limit=READ_BLOCKS)
                st.fts_queries += 1
                st.add("tree", [self._tag(e, node, reason) for e in hits])
                st.node_anchors.append(
                    RetrievalAnchor("node", doc, node_id=node.id, source="tree_search")
                )
        st.stop_reason = st.stop_reason or "tree step budget used"

    def _options(self, doc: str, position: Node | None) -> list[Node]:
        nodes = (
            self.repo.get_roots(doc) if position is None else self.repo.get_children(position.id)
        )
        return [n for n in nodes if n.node_type != "toc"]

    def _ask(
        self,
        st: RetrievalState,
        document: str,
        path: list[str],
        alias: dict[str, Node],
        steps: int,
    ) -> dict[str, Any] | None:
        assert self.llm is not None
        counts = self.repo.count_children([n.id for n in alias.values()])
        lines = []
        for a, n in alias.items():
            summary = truncate(" ".join((n.summary or n.text or "").split()), 160)
            label = "(current section)" if a == "here" else f"(subsections: {counts.get(n.id, 0)})"
            lines.append(f"- {a}: {n.title} {label} — {summary}")
        known = self._fuse_lists(st.lists, {})[:3]
        evidence = (
            "\n".join(f"- {truncate(' '.join(e.content.split()), 160)}" for e in known)
            or "(none yet)"
        )
        prompt = prompts.TREE_REASON.format(
            question=st.query,
            document=document,
            path=" > ".join(path) or "(top level)",
            options="\n".join(lines) or "(no subsections)",
            evidence=evidence,
            steps=max(0, steps - st.tree_steps),
        )
        st.llm_calls += 1
        try:
            reply = self.llm.complete(prompt, system=prompts.TREE_REASON_SYSTEM, max_tokens=200)
        except Exception as e:  # a failed call ends the episode, the fast paths still count
            st.trace.add("tree_step", error=repr(e))
            return None
        data = extract_json(reply)
        return data if isinstance(data, dict) else None

    def _read(self, st: RetrievalState, doc: str, node: Node, reason: str) -> None:
        """READ: resolve a node anchor to its most relevant blocks (search in the subtree),
        or its first blocks when the question's words do not occur there."""
        hits = self.fts.fts_search(st.query, document_id=doc, node_id=node.id, limit=READ_BLOCKS)
        st.fts_queries += 1
        if not hits:
            hits = [
                Evidence(doc, b.node_id, b.id, b.content, "tree", metadata={"page": b.page})
                for b in self.repo.get_blocks(node.id, limit=READ_BLOCKS)
            ]
        st.add("tree", [self._tag(e, node, reason) for e in hits])
        st.node_anchors.append(RetrievalAnchor("node", doc, node_id=node.id, source="tree_read"))

    @staticmethod
    def _tag(e: Evidence, node: Node, reason: str) -> Evidence:
        return Evidence(
            e.document_id,
            e.node_id,
            e.block_id,
            e.content,
            "tree",
            e.score,
            {**e.metadata, "tree_node": node.title, "tree_reason": reason},
        )

    def _heuristic_tree(self, st: RetrievalState, doc: str, policy: RetrievalPolicy) -> None:
        """No LLM configured: one heuristic navigation pass stands in for tree reasoning."""
        if self.tree is None:
            st.stop_reason = "tree reasoning needs an LLM"
            return
        loc = self.tree.locate(st.query, doc, limit=READ_BLOCKS, trace=st.trace)
        st.tree_steps += 1
        st.step += 1
        st.visited_nodes.update(n.id for n in loc.targets)
        hits = (
            self.fts.fts_search(
                st.query, document_id=doc, node_ids=loc.scope_node_ids, limit=READ_BLOCKS
            )
            if loc.scope_node_ids
            else []
        )
        st.fts_queries += 1 if loc.scope_node_ids else 0
        st.add(
            "tree",
            [
                Evidence(
                    e.document_id,
                    e.node_id,
                    e.block_id,
                    e.content,
                    "tree",
                    e.score,
                    {**e.metadata, "tree_node": "heuristic"},
                )
                for e in hits
            ],
        )
        st.stop_reason = "heuristic tree pass (no LLM)"

    # ------------------------------------------------------------------ fusion
    def _fuse(self, st: RetrievalState, policy: RetrievalPolicy) -> list[Evidence]:
        weights = {
            "fts": policy.fts_weight,
            "vector": policy.vector_weight,
            "tree": policy.tree_weight,
        }
        return self._fuse_lists(st.lists, weights)[: policy.candidate_pool]

    @staticmethod
    def _fuse_lists(lists: dict[str, list[Evidence]], weights: dict[str, float]) -> list[Evidence]:
        active = {k: v for k, v in lists.items() if v and weights.get(k, 1.0) > 0}
        if not active:
            return []
        fused = EvidenceMerger(weights={k: weights.get(k, 1.0) for k in active}).rrf(active)
        out = []
        for e in fused:
            paths = sorted((e.metadata.get("signals") or {}).keys()) or [e.source]
            out.append(
                Evidence(
                    e.document_id,
                    e.node_id,
                    e.block_id,
                    e.content,
                    e.source,
                    e.score,
                    {**e.metadata, "retrieval_paths": paths},
                )
            )
        return out


__all__ = ["ACTIONS", "ControllerPlan", "RetrievalController"]
