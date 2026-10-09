"""Answer judges for end-to-end QA (Layer B), applied in order:

1. ``exact``       deterministic: normalised numbers / dates / short facts / list items.
                   It only ever says "correct"; anything it cannot confirm goes on.
2. ``semantic``    a fixed LLM judge that sees the question, the gold answer and the generated
                   answer - never the source document, so it cannot "fix" an answer by reading.
3. ``human_audit`` uncertain verdicts, judge disagreements and flagged questions are queued for a
                   person; their labels (ambiguous / invalid / multiple-valid, or an answer
                   verdict) override the automatic ones on the next run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CORRECT, INCORRECT, UNCERTAIN = "correct", "incorrect", "uncertain"


@dataclass
class Verdict:
    label: str | None  # correct / incorrect / uncertain / None (no decision)
    method: str  # exact / semantic / human / none
    detail: str = ""
    votes: dict[str, Any] = field(default_factory=dict)
    needs_audit: bool = False


__all__ = ["CORRECT", "INCORRECT", "UNCERTAIN", "Verdict"]
