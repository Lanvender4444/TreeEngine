"""Tree-RAG: Query Tree -> Retrieval Search Tree -> Evidence Tree -> ContextSpans -> answer.

    engine.search(query, strategy="best_first" | "beam" | "mcts",
                  budget={"max_steps": 12, "max_llm_calls": 5, "max_context_tokens": 2000})

1. DECOMPOSE (optional, 1 LLM call): the question becomes up to ``max_subqueries``
   sub-questions (the Query Tree); a simple question stays one leaf.
2. Per leaf: SEARCH_TEXT + SEARCH_VECTOR (the fused fast paths, each candidate with a
   confidence prior; FTS / vector ranks recorded), then Evidence Search over reading units
   (READ_BLOCK / NEIGHBORS / PARENT / CHILD / PAGE / TABLE, FOLLOW_ENTITY, SEARCH_IN_SECTION),
   a local reranker blended with the prior as the node value, best-first / beam / MCTS as the
   policy, and VERIFY (LLM judge: answerable, coverage, missing) -> REWRITE_QUERY -> SEARCH
   rounds while LLM calls are left.
3. Context reconstruction: the units read for every leaf, packed into ContextSpans within
   ``max_context_tokens`` (filled from the leaves' candidates when room is left).
4. ANSWER (synthesis over all leaves' evidence).

Budgets: ``max_steps`` READ actions in total, ``max_llm_calls`` for DECOMPOSE + VERIFY (the
answer call is not counted), ``max_context_tokens`` for the answer context.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..context import BlockContextBuilder
from ..context.builder import as_anchors, resolve_policy
from ..context.span import ContextSpan
from ..core.models import AnswerResult
from ..esearch import EvidenceSearch, SearchSpec
from ..retrieval.policy import RetrievalPolicy
from .tree import UNIT_ACTIONS, RAGNode, SearchTree

DECOMPOSE_SYSTEM = (
    "You split questions about documents into the sub-questions that must be looked up. "
    "Reply with JSON only."
)
DECOMPOSE_PROMPT = """Question: {question}
{title}
If answering needs several separate facts that would be found in different places of the
document (different statements, years, notes or sections), list one short search question per
fact, in the document's own vocabulary (at most {n}). If one lookup is enough, return an empty
list.
Reply as JSON: {{"sub_questions": ["...", "..."]}}
"""

DEFAULT_BUDGET = {"max_steps": 12, "max_llm_calls": 4, "max_context_tokens": 2000}
ALL_ACTIONS = "child,table,entity,section"


@dataclass
class TreeRAGResult:
    """What ``engine.search(query, strategy=...)`` returns."""

    query: str
    strategy: str
    answer: str | None
    citations: list[Any]
    query_tree: list[str]  # the leaves searched (the question itself when not decomposed)
    search_tree: SearchTree
    evidence: list[Any]  # Evidence of the context spans (what the answer cites)
    context_spans: list[ContextSpan]
    trace: list[dict[str, Any]]
    stats: dict[str, Any] = field(default_factory=dict)

    def render(self, **kw: Any) -> str:
        return self.search_tree.render(**kw)

    def to_v1(self) -> Any:
        """A versioned ``ResearchResultV1`` record (``.to_dict()`` for JSON)."""
        from ..api.dto import research_v1

        return research_v1(None, self)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            "query": self.query,
            "strategy": self.strategy,
            "answer": self.answer,
            "citations": [asdict(c) for c in self.citations],
            "query_tree": self.query_tree,
            "search_tree": self.search_tree.to_dict(),
            "context_spans": [asdict(s) for s in self.context_spans],
            "trace": self.trace,
            "stats": self.stats,
        }


def parse_subquestions(reply: str | None, limit: int) -> list[str]:
    m = re.search(r"\{.*\}|\[.*\]", reply or "", re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return []
    raw = data.get("sub_questions") if isinstance(data, dict) else data
    out: list[str] = []
    for x in raw if isinstance(raw, list) else []:
        s = " ".join(str(x).split())[:200]
        if s and s not in out:
            out.append(s)
    return out[:limit]


class TreeRAG:
    """The Tree-RAG search. Works on any retriever: ``retrieve(query, document_id) -> ranked
    anchors / evidence`` (the engine's controller, or a benchmark strategy)."""

    def __init__(
        self,
        repo: Any,
        retrieve: Callable[[str, str | None], Sequence[Any]],
        llm: Any = None,
        *,
        answerer: Any = None,
        to_evidence: Callable[[list[ContextSpan]], list[Any]] | None = None,
        title_of: Callable[[str | None], str] | None = None,
    ) -> None:
        self.repo = repo
        self.retrieve = retrieve
        self.llm = llm
        self.answerer = answerer
        self.to_evidence = to_evidence or (lambda spans: [sp.to_evidence() for sp in spans])
        self.title_of = title_of or (lambda _d: "")
        self.llm_calls = 0
        self.llm_in = 0
        self.llm_out = 0

    @classmethod
    def from_engine(cls, engine: Any, policy: RetrievalPolicy | None = None) -> TreeRAG:
        pol = policy or RetrievalPolicy.recommended()
        seed_pol = replace(pol, context_policy="auto", context_budget=1)
        controller = engine.components.controller

        def retrieve(q: str, doc: str | None) -> list[Any]:
            return list(controller.retrieve(q, document_id=doc, policy=seed_pol).anchors)

        return cls(
            engine.repo,
            retrieve,
            engine.llm,
            answerer=engine.components.answerer,
            to_evidence=engine._span_evidence,
            title_of=controller._title,
        )

    def _complete(self, prompt: str, system: str, max_tokens: int) -> str:
        reply = self.llm.complete(prompt, system=system, max_tokens=max_tokens)
        self.llm_calls += 1
        self.llm_in += len(prompt) // 4
        self.llm_out += len(reply or "") // 4
        return str(reply or "")

    # ------------------------------------------------------------------ public
    def run(
        self,
        query: str,
        document_id: str | None = None,
        *,
        strategy: str = "best_first",
        budget: dict[str, int] | None = None,
        decompose: bool = True,
        max_subqueries: int = 3,
        actions: str = ALL_ACTIONS,
        answer: bool = True,
        reranker: str | None = None,
        raptor: bool | None = None,
        reader: str = "search",
        **params: Any,
    ) -> TreeRAGResult:
        """``reader``: how a leaf's evidence is read - "search" (Evidence Search: the reranker
        values reading units, ``strategy`` picks best-first / beam / MCTS) or "gate" (the V0.15
        confidence gate: confident candidates directly, one PageIndex-style LLM selection over
        the candidate catalog + page outline for the rest; VERIFY -> REWRITE_QUERY -> a second
        selection over what was not read yet)."""
        if reader not in ("search", "gate"):
            raise ValueError(f"unknown reader {reader!r} (search | gate)")
        if strategy not in ("best_first", "beam", "mcts"):
            raise ValueError(f"unknown strategy {strategy!r} (best_first | beam | mcts)")
        t0 = time.perf_counter()
        self.llm_calls = self.llm_in = self.llm_out = 0
        b = {**DEFAULT_BUDGET, **(budget or {})}
        llm_left = int(b["max_llm_calls"]) if self.llm is not None else 0
        steps_left = int(b["max_steps"])
        ctx_budget = int(b["max_context_tokens"])
        title = self.title_of(document_id)

        tree = SearchTree()
        trace: list[dict[str, Any]] = []
        root = tree.add(RAGNode("q", None, "QUERY", "", query))

        # 1) Query Tree
        leaves = [query]
        if decompose and llm_left > 1:
            reply = self._complete(
                DECOMPOSE_PROMPT.format(
                    question=query,
                    title=f"Document: {title}\n" if title else "",
                    n=max_subqueries,
                ),
                DECOMPOSE_SYSTEM,
                200,
            )
            llm_left -= 1
            subs = parse_subquestions(reply, max_subqueries)
            trace.append({"action": "DECOMPOSE", "sub_questions": subs})
            if len(subs) > 1:
                leaves = subs
        leaf_nodes = [root]
        if leaves != [query]:
            root.action = "DECOMPOSE"
            leaf_nodes = [
                tree.add(RAGNode(f"q{i}", root.id, "SUBQUERY", "DECOMPOSE", q))
                for i, q in enumerate(leaves, 1)
            ]

        # 2) one Retrieval Search Tree per leaf
        read_rows: list[tuple[Any, ...]] = []
        pools: list[list[Any]] = []
        retrievals = 0
        order = 0
        builder = BlockContextBuilder(self.repo, policy="auto")
        rx = None
        if raptor is not False and hasattr(self.repo, "conn"):
            from .raptor import RaptorIndex

            rx = RaptorIndex(self.repo)  # READ_SUMMARY where a summary tree was built
        summary_spans: list[ContextSpan] = []
        for li, (leaf, qn) in enumerate(zip(leaves, leaf_nodes, strict=True)):
            share = len(leaves) - li
            leaf_tokens = ctx_budget // len(leaves)
            leaf_steps = max(1, steps_left // share)
            leaf_llm = llm_left // share if llm_left > 0 else 0
            anchors = [_anchor(x) for x in self.retrieve(leaf, document_id)]
            retrievals += 1
            pools.append(anchors)
            search = tree.add(
                RAGNode(
                    f"{qn.id}.s",
                    qn.id,
                    "SEARCH",
                    "SEARCH_TEXT+VECTOR",
                    leaf,
                    observation=_sources(anchors),
                )
            )
            asked: list[str] = []

            def again(q: str, _asked: list[str] = asked) -> list[Any]:
                _asked.append(q)
                return [_anchor(x) for x in self.retrieve(q, document_id)]

            if reader == "gate":
                rows, used_llm, n_read = self._gate_leaf(
                    tree,
                    qn,
                    search,
                    builder,
                    leaf,
                    anchors,
                    again,
                    leaf_tokens,
                    leaf_llm,
                    order,
                    title,
                    trace,
                )
                read_rows += rows
                retrievals += len(asked)
                llm_left -= used_llm
                order += n_read
                continue

            kw: dict[str, Any] = {
                "algo": strategy,
                "prior": 0.5,
                "stop_rel": 0.2,
                "lam": 0.05,
                "actions": actions,
                "max_steps": leaf_steps,
                "loop": leaf_llm > 0,
                "loop_rounds": max(1, leaf_llm),
                **params,
            }
            if reranker:
                kw["reranker"] = reranker
            es = EvidenceSearch(
                builder,
                replace(SearchSpec(), **kw),
                retrieve=again,
                llm=self.llm if leaf_llm > 0 else None,
                raptor=rx,
            )
            try:
                _, info = es.run(leaf, anchors, leaf_tokens)
            except Exception as e:  # noqa: BLE001 - no reranker: read the candidates in order
                trace.append({"leaf": leaf, "fallback": str(e)[:200]})
                pol_auto = resolve_policy("auto", leaf_tokens)
                for a in as_anchors(anchors):
                    iv = builder.interval(a, pol_auto) if a.kind == "block" else None
                    if iv:
                        read_rows.append((*iv, str(a.block_id), "window"))
                continue
            retrievals += len(asked)
            llm_left -= es.llm_calls
            self.llm_calls += es.llm_calls
            self.llm_in += es.llm_in
            self.llm_out += es.llm_out
            steps_left -= int(info.get("read") or 0)
            trace.append({"leaf": leaf, **{k: v for k, v in info.items() if k != "trace"}})
            trace += [{"leaf": leaf, **t} for t in es.trace]
            order = self._grow(tree, qn, search, es, asked, order)
            for k in es._read:
                u = es.units[k]
                if u.kind == "summary":
                    summary_spans.append(es.summary_span(u))
                else:
                    read_rows.append((u.doc, u.lo, u.hi, f"es:{u.key}", _resolution(u.kind)))

        # 3) context reconstruction
        summary_spans = summary_spans[:4]
        rest = ctx_budget - sum(sp.token_count for sp in summary_spans)
        spans = summary_spans + self._pack(builder, read_rows, pools, max(0, rest))
        llm_retrieval = self.llm_calls

        # 4) answer
        result: AnswerResult | None = None
        evidence = self.to_evidence(spans)
        if answer and self.answerer is not None:
            q = query
            if len(leaves) > 1:
                q += "\n(Sub-questions looked up: " + "; ".join(leaves) + ")"
            result = self.answerer.answer(q, evidence)
            tree.add(RAGNode("a", root.id, "ANSWER", "SYNTHESIZE", query, result.answer))
        stats = {
            "steps": len(tree.read_nodes()),
            "nodes": len(tree),
            "pruned": sum(1 for n in tree.nodes.values() if n.pruned),
            "leaves": len(leaves),
            "retrieval_calls": retrievals,
            "llm_calls_retrieval": llm_retrieval,
            "llm_input_tokens_retrieval": self.llm_in,
            "llm_output_tokens_retrieval": self.llm_out,
            "llm_calls": llm_retrieval + (1 if result is not None and result.generated else 0),
            "context_tokens": sum(s.token_count for s in spans),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        return TreeRAGResult(
            query=query,
            strategy=strategy,
            answer=result.answer if result else None,
            citations=list(result.citations) if result else [],
            query_tree=leaves,
            search_tree=tree,
            evidence=evidence,
            context_spans=spans,
            trace=trace,
            stats=stats,
        )

    # ------------------------------------------------------------------ gate reader
    def _gate_read(
        self,
        builder: BlockContextBuilder,
        question: str,
        pool: list[Any],
        budget: int,
        title: str,
        use_llm: bool = True,
    ) -> tuple[list[ContextSpan], dict[str, Any], int]:
        """One confidence-gate reading: (spans, info, LLM calls)."""
        from ..reason.reader import ReasonedReader, reason_spec

        spec = reason_spec("gate")
        assert spec is not None
        rd = ReasonedReader(builder, spec)
        plan = rd.prepare(question, pool, budget)
        calls = 0
        if use_llm and self.llm is not None and not plan.skip:
            try:
                rd.decide(plan, self.llm)
                calls = 1
                self.llm_in += len(plan.prompt) // 4
                self.llm_out += len(plan.reply or "") // 4
            except Exception:  # noqa: BLE001 - provider error: the pool order
                plan.reply = None
        spans, info = rd.assemble(plan, allow_first_overflow=builder.allow_first_overflow)
        self.llm_calls += calls
        return spans, info, calls

    def _verify(self, question: str, spans: list[ContextSpan]) -> dict[str, Any]:
        from ..esearch.core import EVAL_PROMPT, EVAL_SYSTEM

        ev = "\n\n".join(f"[{i + 1}] {sp.content[:900]}" for i, sp in enumerate(spans))
        reply = self._complete(
            EVAL_PROMPT.format(question=question, evidence=ev or "(none)"), EVAL_SYSTEM, 120
        )
        m = re.search(r"\{.*\}", reply, re.S)
        try:
            data = json.loads(m.group(0)) if m else {}
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        cov = data.get("coverage")
        try:
            cov = float(cov) if cov is not None else None
        except (TypeError, ValueError):
            cov = None
        return {
            "answerable": bool(data.get("answerable")),
            "coverage": cov,
            "missing": str(data.get("missing") or "").strip(),
        }

    def _gate_leaf(
        self,
        tree: SearchTree,
        qn: RAGNode,
        search: RAGNode,
        builder: BlockContextBuilder,
        leaf: str,
        anchors: list[Any],
        again: Callable[[str], list[Any]],
        leaf_tokens: int,
        leaf_llm: int,
        order: int,
        title: str,
        trace: list[dict[str, Any]],
    ) -> tuple[list[tuple[Any, ...]], int, int]:
        """A leaf read by the confidence gate: SELECT -> VERIFY -> REWRITE_QUERY -> SELECT."""
        verify = leaf_llm >= 3  # a second selection needs selection + verify + selection
        first_budget = int(leaf_tokens * 0.7) if verify else leaf_tokens
        spans, info, used = self._gate_read(
            builder, leaf, anchors, first_budget, title, use_llm=leaf_llm >= 1
        )
        trace.append({"leaf": leaf, "action": "GATE_SELECT", **_slim(info)})
        n = self._span_nodes(tree, search, spans, order, "SELECT")
        rows = _rows(builder, spans)
        if verify:
            v = self._verify(leaf, spans)
            used += 1
            trace.append({"leaf": leaf, "action": "VERIFY", **v})
            tree.add(
                RAGNode(
                    f"{qn.id}.v1",
                    qn.id,
                    "VERIFY",
                    "VERIFY",
                    leaf,
                    observation=v["missing"],
                    coverage=v["coverage"],
                    answerability=v["answerable"],
                )
            )
            if not v["answerable"] and v["missing"]:
                rw = tree.add(
                    RAGNode(f"{qn.id}.r1", qn.id, "REWRITE", "REWRITE_QUERY", v["missing"])
                )
                read = {b for sp in spans for b in sp.block_ids}
                new = [
                    a for a in again(v["missing"]) if str(getattr(a, "block_id", "")) not in read
                ]
                rest = [a for a in anchors if str(getattr(a, "block_id", "")) not in read]
                pool2 = new[:20] + [a for a in rest if a not in new]
                s2 = tree.add(
                    RAGNode(
                        f"{rw.id}.s",
                        rw.id,
                        "SEARCH",
                        "SEARCH_TEXT+VECTOR",
                        v["missing"],
                        observation=_sources(new),
                    )
                )
                spans2, info2, used2 = self._gate_read(
                    builder,
                    f"{leaf}\n(Still missing: {v['missing']})",
                    pool2,
                    leaf_tokens - first_budget,
                    title,
                )
                used += used2
                trace.append({"leaf": leaf, "action": "GATE_SELECT", "round": 2, **_slim(info2)})
                n += self._span_nodes(tree, s2, spans2, order + n, "SELECT")
                rows += _rows(builder, spans2)
        return rows, used, n

    def _span_nodes(
        self, tree: SearchTree, parent: RAGNode, spans: list[ContextSpan], order: int, how: str
    ) -> int:
        for i, sp in enumerate(spans, 1):
            kind = {"section": "SECTION", "page": "PAGE"}.get(sp.resolution or "", "BLOCK")
            tree.add(
                RAGNode(
                    f"{parent.id}.{how.lower()}{i}",
                    parent.id,
                    kind,
                    f"READ_{how}",
                    observation=sp.content[:300],
                    document_id=sp.document_id,
                    section_id=sp.node_id,
                    block_ids=list(sp.block_ids),
                    page=sp.page_start,
                    cost=sp.token_count,
                    read=order + i,
                )
            )
        return len(spans)

    # ------------------------------------------------------------------ helpers
    def _grow(
        self,
        tree: SearchTree,
        qn: RAGNode,
        search: RAGNode,
        es: EvidenceSearch,
        asked: list[str],
        order: int,
    ) -> int:
        """Add one leaf's VERIFY / REWRITE nodes and its Evidence Tree to the search tree."""
        rewrites: dict[str, RAGNode] = {}
        for i, v in enumerate(es.verifications, 1):
            tree.add(
                RAGNode(
                    f"{qn.id}.v{i}",
                    qn.id,
                    "VERIFY",
                    "VERIFY",
                    qn.query,
                    observation=str(v.get("missing") or ""),
                    coverage=v.get("coverage"),
                    answerability=bool(v.get("answerable")),
                    meta={"read": v.get("read"), "tokens": v.get("tokens")},
                )
            )
        for i, q in enumerate(asked, 1):
            rw = tree.add(RAGNode(f"{qn.id}.r{i}", qn.id, "REWRITE", "REWRITE_QUERY", q))
            rewrites[q] = tree.add(
                RAGNode(f"{rw.id}.s", rw.id, "SEARCH", "SEARCH_TEXT+VECTOR", q, "rewrite")
            )
        reading = {k: i for i, k in enumerate(es._read, 1)}
        prefix = f"{qn.id}."
        # parents before children: units created by an expansion name their parent unit
        pending = list(es.units.values())
        placed: set[str] = set()
        while pending:
            rest = []
            for u in pending:
                if u.parent is not None and u.parent not in placed:
                    rest.append(u)
                    continue
                if u.parent is not None:
                    parent_id = prefix + u.parent
                elif u.origin.startswith("rewrite") and rewrites:
                    parent_id = list(rewrites.values())[-1].id
                else:
                    parent_id = search.id
                ntype, action = UNIT_ACTIONS.get(u.kind, ("BLOCK", u.kind.upper()))
                blocks = es.b.blocks(u.doc)[u.lo : u.hi + 1]
                r = reading.get(u.key)
                tree.add(
                    RAGNode(
                        prefix + u.key,
                        parent_id,
                        ntype,
                        action,
                        observation=u.text[:300],
                        document_id=u.doc,
                        section_id=blocks[0].node_id if blocks else None,
                        block_ids=[x.id for x in blocks],
                        page=blocks[0].page if blocks else None,
                        relevance=round(u.value, 4),
                        prior=round(u.prior, 4),
                        cost=u.tokens,
                        read=(order + r) if r else None,
                        meta={"origin": u.origin},
                    )
                )
                placed.add(u.key)
            if len(rest) == len(pending):  # orphans (parent never scored): attach to search
                for u in rest:
                    u.parent = None
            pending = rest
        # MCTS statistics on the read path
        for st in es.mcts_states(limit=500):
            if len(st["state"]) and not st["stop"]:
                last = prefix + st["state"][-1]
                if last in tree.nodes and tree.nodes[last].visits < st["visits"]:
                    tree.nodes[last].visits = st["visits"]
                    tree.nodes[last].value = st["mean_reward"]
        return order + len(reading)

    def _pack(
        self,
        builder: BlockContextBuilder,
        rows: list[tuple[Any, ...]],
        pools: list[list[Any]],
        budget: int,
    ) -> list[ContextSpan]:
        from ..reason.reader import ReasonedReader

        rows = list(rows)
        pol = resolve_policy("auto", budget)
        for i in range(max((len(p) for p in pools), default=0)):  # round robin over leaves
            for p in pools:
                if i < len(p):
                    a = as_anchors([p[i]])[0]
                    if a.kind == "block" and a.block_id:
                        iv = builder.interval(a, pol)
                        if iv:
                            rows.append((*iv, str(a.block_id), "window"))
        return ReasonedReader(builder)._pack(rows, budget, builder.allow_first_overflow)


def _slim(info: dict[str, Any] | None) -> dict[str, Any]:
    keep = ("top_conf", "accepted", "skip_llm", "choice", "pages", "whole", "excerpts")
    return {k: v for k, v in (info or {}).items() if k in keep}


def _rows(builder: BlockContextBuilder, spans: list[ContextSpan]) -> list[tuple[Any, ...]]:
    """ContextSpans back to (document, first, last block index) reading rows."""
    out = []
    for sp in spans:
        if not sp.block_ids:
            continue
        builder.blocks(sp.document_id)
        idx = [builder._where[b][1] for b in sp.block_ids if b in builder._where]
        if idx:
            res = "section" if (sp.resolution or "").startswith("section") else "window"
            out.append(
                (
                    sp.document_id,
                    min(idx),
                    max(idx),
                    sp.source_evidence_ids[0] if sp.source_evidence_ids else sp.block_ids[0],
                    res,
                )
            )
    return out


def _anchor(x: Any) -> Any:
    """Benchmark evidence carries its anchor in metadata; anchors pass through."""
    md = getattr(x, "metadata", None) or {}
    return md.get("anchor", x) if isinstance(md, dict) else x


def _resolution(kind: str) -> str:
    return "section" if kind in ("page", "parent", "child") else "window"


def _sources(anchors: list[Any]) -> str:
    fts = sum(1 for a in anchors if "fts" in (a.metadata.get("src_ranks") or {}))
    vec = sum(1 for a in anchors if "vector" in (a.metadata.get("src_ranks") or {}))
    if not fts and not vec:
        return f"{len(anchors)} candidates"
    return f"{len(anchors)} candidates (fts {fts}, vector {vec})"


__all__ = ["DECOMPOSE_PROMPT", "TreeRAG", "TreeRAGResult", "parse_subquestions"]
