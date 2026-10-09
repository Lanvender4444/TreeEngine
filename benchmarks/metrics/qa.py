"""End-to-end QA metrics (Layer B)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .retrieval import _mean, _pct


@dataclass
class QAOutcome:
    query_id: str
    query_type: str
    strategy: str
    category: str | None
    document: str | None
    answer: str
    gold: str
    verdict: str | None  # correct / incorrect / uncertain / None when skipped or excluded
    judge: str  # exact / semantic / human / none
    judge_detail: str = ""
    votes: dict[str, Any] = field(default_factory=dict)
    needs_audit: bool = False
    skipped: str | None = None  # e.g. full_context over budget
    excluded: str | None = None  # human audit: ambiguous / invalid question
    recall5: float | None = None  # retrieval recall@5 when the suite has evidence labels
    context_tokens: int = 0
    retrieval_ms: float = 0.0
    answer_ms: float = 0.0
    embedding_calls: int = 0
    embedding_tokens: int = 0
    retrieval_llm_tokens: int = 0  # LLM navigation / planner, part of the system's cost
    answer_input_tokens: int = 0
    answer_output_tokens: int = 0
    judge_tokens: int = 0  # evaluation cost, never counted as system cost
    usd: float | None = None
    error: str | None = None

    @property
    def scored(self) -> bool:
        return self.skipped is None and self.excluded is None and self.verdict != "uncertain"

    @property
    def system_tokens(self) -> int:
        return self.retrieval_llm_tokens + self.answer_input_tokens + self.answer_output_tokens


def aggregate_qa(outcomes: Sequence[QAOutcome]) -> dict[str, Any]:
    if not outcomes:
        return {"n": 0}
    ran = [o for o in outcomes if o.skipped is None and o.excluded is None]
    scored = [o for o in ran if o.verdict in ("correct", "incorrect")]
    correct = sum(1 for o in scored if o.verdict == "correct")
    lat = [o.retrieval_ms + o.answer_ms for o in ran]
    usd = [o.usd for o in ran if o.usd is not None]
    row: dict[str, Any] = {
        "n": len(outcomes),
        "answered": len(ran),
        "skipped": sum(1 for o in outcomes if o.skipped),
        "excluded": sum(1 for o in outcomes if o.excluded),
        "uncertain": sum(1 for o in ran if o.verdict == "uncertain"),
        "accuracy": correct / len(scored) if scored else None,
        "correct": correct,
        "recall@5": _mean([o.recall5 for o in ran if o.recall5 is not None]),
        "ctx_tokens": _mean([float(o.context_tokens) for o in ran]),
        "tokens/q": _mean([float(o.system_tokens) for o in ran]),
        "embed_tokens/q": _mean([float(o.embedding_tokens) for o in ran]),
        "p50_ms": _pct(lat, 50),
        "p95_ms": _pct(lat, 95),
        "usd/q": (sum(usd) / len(usd)) if usd and len(usd) == len(ran) else None,
        "judge": ", ".join(
            f"{m} {sum(1 for o in ran if o.judge == m)}"
            for m in ("exact", "semantic", "human")
            if any(o.judge == m for o in ran)
        ),
        "errors": sum(1 for o in outcomes if o.error),
    }
    row["usd/correct"] = (
        sum(usd) / correct if usd and correct and row["usd/q"] is not None else None
    )
    row["tokens/correct"] = sum(o.system_tokens for o in ran) / correct if correct else None
    return row


__all__ = ["QAOutcome", "aggregate_qa"]
