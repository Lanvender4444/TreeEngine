"""The Retrieval Tree: one hierarchical retrieval space instead of four ranked lists.

    query
      | lexical  (FTS over blocks)        --+
      | semantic (vector over blocks)       +--> block activation
      | relational (Graph-lite entities)  --+        (lateral)
      | node lexical  (BM25 on title path + opening)  --+
      | node semantic (query . subtree centroid)        +--> node activation
      v
    propagation up the document tree:  score(N) = node activation(N) + descendant evidence(N)
      v
    best-first walk from the roots: pop the best node, then
      * readable (fits section_share of the budget) and >= 2 of its parts (children / own
        blocks) hold strong evidence (one of the query's STRONG_BLOCKS best blocks)
                                                   -> NodeAnchor (read the section)
      * otherwise                                  -> push its children, emit its own hit
                                                      blocks as BlockAnchors (stay fine)
      * a node with only its own activation, top-ranked by both title and centroid, readable
                                                   -> NodeAnchor (found from the tree alone)
      * near-tie between siblings, LLM configured, agentic_level > 0 -> the LLM picks
      v
    BlockAnchors and NodeAnchors in one ranking -> ContextBuilder

Every signal is a rank mass ``w / (k + rank)``; raw BM25 / cosine values are never added.
Descendant evidence is the decaying sum ``a1 + a2/2 + a3/4`` of the subtree's best block
activations: a section with several hits outranks its best block alone, but a large section
does not win on size. A flat document (one node) degrades to block retrieval.
"""

from __future__ import annotations

import heapq
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.anchor import Anchor, BlockAnchor, NodeAnchor
from ..core.models import Evidence
from ..core.protocols import LLMProvider, Repository
from .index import DocIndex, NodeIndex

RRF_K = 60
POOL = 100  # block hits read per signal
NODE_POOL = 20  # nodes ranked per node signal and document
DOCS = 5  # documents walked when no document is given
STRONG_BLOCKS = 10  # strong evidence = the query's 10 best activated blocks (rank, not ratio:
# rank masses are flat - rank 1 and rank 4 differ by < 10% - so ratio thresholds pass everything)
STRONG_NODE_RANK = 3  # a node with no block hit is strong when both node signals rank it top 3
DECAY = (1.0, 0.5, 0.25)
TIE = 0.85  # siblings within this ratio of the best are a near-tie (LLM may break it)

TIEBREAK_SYSTEM = (
    "You choose which sections of a document can answer a question. Reply with JSON only: "
    '{"selected": ["c1", ...]} listing the ids of the relevant sections (one or more).'
)


@dataclass
class Activation:
    block: dict[str, float] = field(default_factory=dict)
    block_signals: dict[str, dict[str, int]] = field(default_factory=dict)  # id -> signal: rank
    node: dict[str, float] = field(default_factory=dict)
    node_signals: dict[str, dict[str, int]] = field(default_factory=dict)
    evidence: dict[str, Evidence] = field(default_factory=dict)  # block id -> a hit's evidence


class TreeSignalSearch:
    def __init__(
        self,
        repo: Repository,
        fts: Any,
        *,
        vector: Any = None,  # VectorRetriever (embedder + index)
        graph: Any = None,  # GraphExpander
        llm: LLMProvider | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.repo = repo
        self.fts = fts
        self.vector = vector
        self.graph = graph
        self.llm = llm
        self.index = NodeIndex(repo, getattr(vector, "index", None) if vector else None)

    # ------------------------------------------------------------------ signals
    def _activate(
        self, query: str, document_id: str | None, policy: Any, trace: Any, stats: dict[str, int]
    ) -> tuple[Activation, Any]:
        act = Activation()
        k = RRF_K

        def add(signal: str, hits: list[Evidence], w: float) -> None:
            for r, e in enumerate(hits, 1):
                if not e.block_id:
                    continue
                act.block[e.block_id] = act.block.get(e.block_id, 0.0) + w / (k + r)
                act.block_signals.setdefault(e.block_id, {})[signal] = r
                act.evidence.setdefault(e.block_id, e)

        if policy.fts_weight > 0:
            add(
                "lexical",
                self.fts.fts_search(query, document_id=document_id, limit=POOL, trace=trace),
                policy.fts_weight,
            )
            stats["fts_queries"] += 1
        qv = None
        if policy.vector_weight > 0 and self.vector is not None:
            t0 = time.perf_counter()
            qv = self.vector.embedder.embed_query(query)
            hits = self.vector.index.search(qv, k=POOL, document_id=document_id)
            ev = []
            for bid, sim in hits:
                b = self.repo.get_block(bid)
                if b is not None:
                    ev.append(
                        Evidence(
                            b.document_id,
                            b.node_id,
                            b.id,
                            b.content,
                            "vector",
                            sim,
                            {"page": b.page},
                        )
                    )
            add("semantic", ev, policy.vector_weight)
            stats["vector_queries"] += 1
            if trace is not None:
                trace.add("vector", hits=len(ev), ms=round((time.perf_counter() - t0) * 1000, 2))
        if policy.graph_weight > 0 and self.graph is not None and act.block:
            # lateral activation: blocks one entity hop from the best activated blocks
            seeds = [
                BlockAnchor(act.evidence[b].document_id, b, act.evidence[b].node_id, "seed", r)
                for r, b in enumerate(sorted(act.block, key=lambda b: -act.block[b]), 1)
            ]
            related = self.graph.expand(seeds, policy, trace=trace)
            add(
                "relational",
                [a.evidence for a in related if a.evidence is not None],
                policy.graph_weight,
            )
        return act, qv

    def _node_signals(
        self, act: Activation, di: DocIndex, query: str, qv: Any, policy: Any
    ) -> None:
        w = policy.node_signal_weight
        if w <= 0:
            return
        for signal, ranked in (
            ("node_lexical", self.index.lexical(di, query, NODE_POOL)),
            ("node_semantic", self.index.semantic(di, qv, NODE_POOL)),
        ):
            for r, nid in enumerate(ranked, 1):
                act.node[nid] = act.node.get(nid, 0.0) + w / (RRF_K + r)
                act.node_signals.setdefault(nid, {})[signal] = r

    # ------------------------------------------------------------------ propagation
    @staticmethod
    def _descendant(di: DocIndex, act: Activation) -> dict[str, float]:
        """Decaying sum of each node's best block activations in its subtree."""
        best: dict[str, list[float]] = {}
        for bid, a in act.block.items():
            i = di.where.get(bid)
            if i is None:
                continue
            for nid in di.ancestors(di.blocks[i].node_id):
                lst = best.setdefault(nid, [])
                lst.append(a)
        out: dict[str, float] = {}
        for nid, lst in best.items():
            lst.sort(reverse=True)
            out[nid] = sum(d * a for d, a in zip(DECAY, lst, strict=False))
        return out

    # ------------------------------------------------------------------ walk
    def retrieve(
        self, query: str, document_id: str | None, policy: Any, trace: Any = None
    ) -> tuple[list[Anchor], dict[str, int]]:
        stats = {"fts_queries": 0, "vector_queries": 0, "llm_calls": 0, "visited": 0}
        act, qv = self._activate(query, document_id, policy, trace, stats)
        if document_id is not None:
            docs = [document_id]
        else:  # coarse: the documents with the most block evidence
            per_doc: dict[str, list[float]] = {}
            for bid, a in act.block.items():
                per_doc.setdefault(act.evidence[bid].document_id, []).append(a)
            docs = sorted(
                per_doc,
                key=lambda d: (
                    -sum(
                        x * y for x, y in zip(DECAY, sorted(per_doc[d], reverse=True), strict=False)
                    )
                ),
            )[:DOCS]
        cap = policy.context_budget * policy.section_share
        out: list[tuple[float, Anchor]] = []
        emitted: set[str] = set()
        for doc in docs:
            di = self.index.get(doc, centroids=qv is not None and policy.node_signal_weight > 0)
            if di is None:
                continue
            self._node_signals(act, di, query, qv, policy)
            desc = self._descendant(di, act)
            score = {
                nid: act.node.get(nid, 0.0) + desc.get(nid, 0.0)
                for nid in di.nodes
                if act.node.get(nid, 0.0) + desc.get(nid, 0.0) > 0
            }
            out += self._walk(di, act, score, cap, policy, query, stats, emitted)
        # blocks of documents that were not walked (or not mapped to a node) stay as blocks
        for bid, a in act.block.items():
            if bid not in emitted:
                out.append((a, self._block_anchor(bid, a, act)))
        out.sort(key=lambda x: -x[0])
        anchors = [a for _, a in out[: policy.candidate_pool]]
        anchors = [_rank(a, i) for i, a in enumerate(anchors, 1)]
        if trace is not None:
            trace.add(
                "retrieval_tree",
                documents=len(docs),
                activated_blocks=len(act.block),
                activated_nodes=len(act.node),
                visited=stats["visited"],
                node_anchors=sum(1 for a in anchors if a.kind == "node"),
                block_anchors=sum(1 for a in anchors if a.kind == "block"),
                llm_calls=stats["llm_calls"],
            )
        return anchors, stats

    def _walk(
        self,
        di: DocIndex,
        act: Activation,
        score: dict[str, float],
        cap: float,
        policy: Any,
        query: str,
        stats: dict[str, int],
        emitted: set[str],
    ) -> list[tuple[float, Anchor]]:
        out: list[tuple[float, Anchor]] = []
        heap: list[tuple[float, int, str]] = []
        seq = 0
        for r in di.roots:
            if score.get(r, 0) > 0:
                heapq.heappush(heap, (-score[r], seq, r))
                seq += 1
        llm_ok = self.llm is not None and policy.agentic_level > 0
        top = sorted(act.block, key=lambda b: -act.block[b])[:STRONG_BLOCKS]
        strong_blocks = set(top)
        strong_nodes: set[str] = set()  # nodes holding a strong block, or strong by themselves
        for bid in top:
            i = di.where.get(bid)
            if i is not None:
                strong_nodes.update(di.ancestors(di.blocks[i].node_id))
        for nid, sig in act.node_signals.items():
            if nid in di.nodes and len(sig) == 2 and max(sig.values()) <= STRONG_NODE_RANK:
                strong_nodes.add(nid)
        while heap and len(out) < policy.candidate_pool:
            neg, _, nid = heapq.heappop(heap)
            stats["visited"] += 1
            ni = di.nodes[nid]
            kids = [(score.get(c, 0.0), c) for c in di.children.get(nid, []) if score.get(c, 0) > 0]
            own = [
                (act.block[di.blocks[i].id], di.blocks[i].id)
                for i in ni.own
                if di.blocks[i].id in act.block and di.blocks[i].id not in emitted
            ]
            parts = sorted(
                [(s, "n", x) for s, x in kids] + [(s, "b", x) for s, x in own], reverse=True
            )
            readable = 0 < ni.tokens <= cap and ni.lo >= 0
            if not parts:
                # activated only through its own title / centroid: read it if it fits
                if readable and nid in strong_nodes:
                    out.append(
                        (
                            -neg,
                            self._node_anchor(
                                di, nid, -neg, act, "node signal", cap, policy.context_budget
                            ),
                        )
                    )
                    self._mark(di, nid, emitted)
                continue
            # a part is strong when it holds one of the query's best blocks (or is a strong node
            # itself): weak hits do not add up to a section worth reading
            spread = sum(
                1
                for _, kind, x in parts
                if (x in strong_nodes if kind == "n" else x in strong_blocks)
            )
            if readable and spread >= 2:
                why = f"evidence spread over {spread} parts"
                out.append(
                    (-neg, self._node_anchor(di, nid, -neg, act, why, cap, policy.context_budget))
                )
                self._mark(di, nid, emitted)
                continue
            if llm_ok and stats["llm_calls"] < policy.max_llm_calls and len(kids) >= 2:
                kids.sort(reverse=True)
                if kids[1][0] >= TIE * kids[0][0]:
                    chosen = self._tiebreak(di, query, [c for _, c in kids[:6]], stats)
                    if chosen:
                        boost = 1.0 + policy.agentic_level
                        for c in chosen:
                            score[c] = score.get(c, 0.0) * boost
            for s, kind, x in parts:
                if kind == "n":
                    heapq.heappush(heap, (-score.get(x, s), seq, x))
                    seq += 1
                elif x not in emitted:
                    out.append((s, self._block_anchor(x, s, act)))
                    emitted.add(x)
        return out

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _mark(di: DocIndex, nid: str, emitted: set[str]) -> None:
        for i in di.nodes[nid].blocks:
            emitted.add(di.blocks[i].id)

    def _node_anchor(
        self,
        di: DocIndex,
        nid: str,
        s: float,
        act: Activation,
        why: str,
        cap: float,
        policy_budget: float,
    ) -> Anchor:
        ni = di.nodes[nid]
        hits = [di.blocks[i].id for i in ni.blocks if di.blocks[i].id in act.block]
        focus = max(hits, key=lambda b: act.block[b]) if hits else None
        md: dict[str, Any] = {
            "scope": "subtree",
            "read": "section",
            "node_title": ni.node.title,
            "page": ni.node.page_start,
            "reason": why,
            "hits_inside": len(hits),
            "node_signals": act.node_signals.get(nid, {}),
            "retrieval_paths": sorted(
                {sig for b in hits for sig in act.block_signals.get(b, {})}
                | set(act.node_signals.get(nid, {}))
            ),
            "resolution": "section",
            "share": cap / max(1.0, policy_budget),  # the ContextBuilder reads up to this
            # share of the budget it builds with (the walk judged "readable" at policy budget)
        }
        if focus:
            md["focus_block_id"] = focus
        text = " ".join(di.blocks[i].content for i in ni.blocks[:3])[:2000]
        ev = Evidence(di.document_id, nid, None, text, "tree", round(s, 6), dict(md))
        return NodeAnchor(di.document_id, nid, "tree", 0, round(s, 6), md, ev)

    def _block_anchor(self, bid: str, s: float, act: Activation) -> Anchor:
        e = act.evidence[bid]
        md = {
            **(e.metadata or {}),
            "signals": act.block_signals.get(bid, {}),
            "retrieval_paths": sorted(act.block_signals.get(bid, {})),
            "resolution": "block",
        }
        src = "+".join(sorted(act.block_signals.get(bid, {}))) or "tree"
        ev = Evidence(e.document_id, e.node_id, bid, e.content, src, round(s, 6), md)
        return BlockAnchor(e.document_id, bid, e.node_id, src, 0, round(s, 6), md, ev)

    def _tiebreak(
        self, di: DocIndex, query: str, kids: list[str], stats: dict[str, int]
    ) -> list[str]:
        assert self.llm is not None
        alias = {f"c{i + 1}": c for i, c in enumerate(kids)}
        lines = []
        for a, c in alias.items():
            ni = di.nodes[c]
            head = di.blocks[ni.own[0]].content[:160] if ni.own else ""
            lines.append(f"- {a}: {ni.node.title} | {head}")
        prompt = f"Question: {query}\n\nSections:\n" + "\n".join(lines)
        stats["llm_calls"] += 1
        try:
            reply = self.llm.complete(prompt, system=TIEBREAK_SYSTEM)
            m = re.search(r"\{.*\}", reply, re.S)
            picked = json.loads(m.group(0)).get("selected", []) if m else []
        except Exception:  # noqa: BLE001 - a failed tie-break keeps the deterministic order
            return []
        return [alias[p] for p in picked if p in alias]


def _rank(a: Anchor, r: int) -> Anchor:
    a.rank = r
    return a


__all__ = ["TreeSignalSearch"]
