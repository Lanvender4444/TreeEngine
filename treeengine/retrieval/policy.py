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

    candidate_pool: int = 50
    context_budget: int = 2000
    context_policy: str = "auto"

    # soft stop: at least this many distinct evidence items covering most query terms
    min_evidence: int = 3
    min_term_coverage: float = 0.6

    def __post_init__(self) -> None:
        for name in ("fts_weight", "vector_weight", "tree_weight", "agentic_level"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.agentic_level > 1:
            raise ValueError("agentic_level must be within [0, 1]")
        if not (self.fts_weight or self.vector_weight or self.tree_weight):
            raise ValueError("at least one retrieval path needs a weight > 0")
        if min(self.max_steps, self.candidate_pool, self.context_budget) <= 0:
            raise ValueError("budgets must be positive")

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
        """TreeEngine's recommended mode: FTS + Vector first, tree reasoning only when useful."""
        base: dict[str, object] = {"agentic_level": 0.6, "max_steps": 4, "max_llm_calls": 2}
        base.update(kw)
        return cls(fts_weight=1.0, vector_weight=1.0, tree_weight=0.5, **base)  # type: ignore[arg-type]


__all__ = ["AgenticMode", "RetrievalPolicy"]
