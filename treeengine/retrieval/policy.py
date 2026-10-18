"""RetrievalPolicy: which retrieval paths may run, how autonomous the controller is, budgets.

Weights are *path preferences*, not score coefficients: BM25, cosine similarity and an LLM's
choice are not on one scale, so they are never added up.

* ``weight == 0`` disables a path, ``weight > 0`` enables it;
* among enabled paths the weight is the path's vote in rank fusion (weighted RRF: only ranks
  are fused) and its priority.

``agentic_level`` is a strategy setting, not a score:

    < 0.3  off       one retrieval pass, no loop
    < 0.6  fallback  if the first pass looks insufficient, one tree-reasoning episode
    < 1.0  guided    tree reasoning step by step while the evidence looks insufficient
    = 1.0  full      the LLM navigates within the budget and decides itself when to stop

Hard budgets (``max_steps``, ``max_llm_calls``, ``max_tree_visits``) always win.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AgenticMode = Literal["off", "fallback", "guided", "full"]


@dataclass(frozen=True)
class RetrievalPolicy:
    fts_weight: float = 1.0
    vector_weight: float = 1.0
    tree_weight: float = 0.0

    agentic_level: float = 0.0

    max_steps: int = 4
    max_llm_calls: int = 2
    max_tree_visits: int = 20

    # V0.9 Graph-lite: one entity hop from the best block anchors (an expansion signal, 0 = off)
    graph_weight: float = 0.0
    graph_seeds: int = 5
    graph_entities_per_seed: int = 5
    graph_blocks_per_entity: int = 3
    graph_max_blocks: int = 10
    # what a tree READ contributes: "node" = the section itself (NodeAnchor, V0.9);
    # "blocks" = FTS inside the section -> BlockAnchors (pre-V0.9 path, for comparison)
    tree_anchor: str = "node"

    # V0.10 how the signals become anchors. "rrf": every path returns a ranked list and the
    # lists are rank-fused (V0.6-V0.9). "tree": the Retrieval Tree (treeengine/rtree) - FTS,
    # Vector and Graph-lite only activate blocks, node titles / centroids activate nodes, the
    # activation propagates up the document tree, a best-first walk decides where to stop and
    # emits BlockAnchors or NodeAnchors (adaptive resolution). Tree reasoning by LLM is not run;
    # with an LLM and agentic_level > 0 the walk asks it only to break near-ties.
    fusion: str = "rrf"
    # V0.15 fusion="gate": weight of the document tree as a third candidate source (sections
    # ranked by title path / subtree centroid, best block each). Off by default: on the V0.15
    # dev sweep any weight > 0 diluted the FTS + Vector ranking (results/v0.15-confidence-gate.md)
    node_source_weight: float = 0.0
    # V0.16 fusion="gate": a local reranker over the top ``rerank_k`` candidates
    # ("ce:<model>" cross-encoder / "colbert:<model>" late interaction; retrieval/rerank.py).
    # Its rank and score become confidence features; ``rerank_order`` also re-sorts those
    # candidates by the reranker.
    rerank: str = ""
    rerank_k: int = 30
    rerank_order: bool = False
    node_signal_weight: float = 1.0  # node title (lexical) + centroid (semantic) activation
    section_share: float = 0.5  # a section may be emitted whole if it fits this budget share
    node_signals: str = "both"  # which node signals: both | lexical | semantic | none
    strong_blocks: int = 10  # strong evidence = the query's N best activated blocks
    min_regions: int = 2  # strong parts a section needs to be read whole

    candidate_pool: int = 50
    context_budget: int = 2000
    context_policy: str = "auto"

    # soft stop: at least this many distinct evidence items covering most query terms
    min_evidence: int = 3
    min_term_coverage: float = 0.6

    def __post_init__(self) -> None:
        for name in (
            "fts_weight",
            "vector_weight",
            "tree_weight",
            "graph_weight",
            "agentic_level",
            "node_source_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.agentic_level > 1:
            raise ValueError("agentic_level must be within [0, 1]")
        if not (self.fts_weight or self.vector_weight or self.tree_weight):
            raise ValueError("at least one retrieval path needs a weight > 0")
        if min(self.max_steps, self.candidate_pool, self.context_budget) <= 0:
            raise ValueError("budgets must be positive")
        if self.tree_anchor not in ("node", "blocks"):
            raise ValueError('tree_anchor must be "node" or "blocks"')
        if self.fusion not in ("rrf", "tree", "gate"):
            raise ValueError('fusion must be "rrf", "tree" or "gate"')
        if self.node_signals not in ("both", "lexical", "semantic", "none"):
            raise ValueError("node_signals must be both | lexical | semantic | none")
        if self.strong_blocks < 1 or self.min_regions < 1:
            raise ValueError("strong_blocks and min_regions must be >= 1")
        if self.node_signal_weight < 0 or not 0 < self.section_share <= 1:
            raise ValueError("node_signal_weight must be >= 0 and section_share in (0, 1]")

    @property
    def mode(self) -> AgenticMode:
        if self.agentic_level >= 1.0:
            return "full"
        if self.agentic_level >= 0.6:
            return "guided"
        if self.agentic_level >= 0.3:
            return "fallback"
        return "off"

    @property
    def fast_paths(self) -> list[str]:
        return [p for p, w in (("fts", self.fts_weight), ("vector", self.vector_weight)) if w > 0]

    @property
    def tree_enabled(self) -> bool:
        return self.tree_weight > 0

    # ---- typical configurations (design doc §4)
    @classmethod
    def fts_only(cls, **kw: object) -> RetrievalPolicy:
        return cls(fts_weight=1.0, vector_weight=0.0, tree_weight=0.0, **kw)  # type: ignore[arg-type]

    @classmethod
    def hybrid(cls, **kw: object) -> RetrievalPolicy:
        return cls(fts_weight=1.0, vector_weight=1.0, tree_weight=0.0, **kw)  # type: ignore[arg-type]

    @classmethod
    def tree_reasoning(cls, **kw: object) -> RetrievalPolicy:
        """PageIndex-like: LLM navigation of the tree only."""
        base: dict[str, object] = {"agentic_level": 1.0, "max_llm_calls": 6, "max_steps": 6}
        base.update(kw)
        return cls(fts_weight=0.0, vector_weight=0.0, tree_weight=1.0, **base)  # type: ignore[arg-type]

    @classmethod
    def hybrid_agentic(cls, **kw: object) -> RetrievalPolicy:
        """FTS + Vector first, tree reasoning only when the evidence looks insufficient. Opt-in:
        on longdoc it did not beat plain ``hybrid()`` (V0.6 results), which stays the default."""
        base: dict[str, object] = {"agentic_level": 0.6, "max_steps": 4, "max_llm_calls": 2}
        base.update(kw)
        return cls(fts_weight=1.0, vector_weight=1.0, tree_weight=0.5, **base)  # type: ignore[arg-type]


__all__ = ["AgenticMode", "RetrievalPolicy"]
