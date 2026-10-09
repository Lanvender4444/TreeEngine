"""Human audit: a queue out, labels back in.

Each QA run writes ``results/<stamp>-<suite>-audit.jsonl`` with every case that needs a person:
the semantic judge was uncertain, judges disagreed, or the deterministic judge confirmed an
answer the semantic judge rejected. A reviewer copies lines into the suite's ``audit.jsonl``
and adds a ``label``:

    question-level (no "strategy"):  ambiguous | invalid | multiple-valid
        ambiguous / invalid  -> the question is excluded from accuracy (reported separately)
        multiple-valid       -> kept; automatic "incorrect" verdicts for it are re-queued
    answer-level ("strategy" set):   correct | incorrect
        overrides the automatic verdict for that strategy's answer

Labels are applied on every later run, so audits accumulate instead of being redone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import CORRECT, INCORRECT, Verdict

QUESTION_LABELS = ("ambiguous", "invalid", "multiple-valid")
ANSWER_LABELS = (CORRECT, INCORRECT)


@dataclass
class AuditLabels:
    questions: dict[str, str] = field(default_factory=dict)
    answers: dict[tuple[str, str], str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> AuditLabels:
        labels = cls()
        if not path.exists():
            return labels
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            d = json.loads(line)
            label = str(d.get("label", "")).strip().lower()
            if not label:
                continue
            if d.get("strategy"):
                if label not in ANSWER_LABELS:
                    raise ValueError(f"{path}:{i}: answer label must be one of {ANSWER_LABELS}")
                labels.answers[(d["query_id"], d["strategy"])] = label
            else:
                if label not in QUESTION_LABELS:
                    raise ValueError(f"{path}:{i}: question label must be one of {QUESTION_LABELS}")
                labels.questions[d["query_id"]] = label

        return labels

    def excluded(self, query_id: str) -> str | None:
        label = self.questions.get(query_id)
        return label if label in ("ambiguous", "invalid") else None

    def apply(self, query_id: str, strategy: str, verdict: Verdict) -> Verdict:
        human = self.answers.get((query_id, strategy))
        if human:
            return Verdict(human, "human", "audited", verdict.votes)
        if self.questions.get(query_id) == "multiple-valid" and verdict.label == INCORRECT:
            verdict.needs_audit = True
            verdict.detail = (verdict.detail + "; question has multiple valid answers").strip("; ")
        return verdict


def write_queue(path: Path, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, "label": ""}, ensure_ascii=False) + "\n")
    return path


__all__ = ["ANSWER_LABELS", "QUESTION_LABELS", "AuditLabels", "write_queue"]
