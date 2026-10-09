"""LLM judge: question + gold answer + generated answer -> correct / incorrect / uncertain.

The judge never sees the source document or the retrieved evidence, so it cannot repair an
answer by reading. Several judges can be configured; any disagreement goes to human audit.
"""

from __future__ import annotations

from collections.abc import Sequence

from treeengine.core.protocols import LLMProvider
from treeengine.llm.base import extract_json

from . import CORRECT, INCORRECT, UNCERTAIN, Verdict

JUDGE_SYSTEM = (
    "You grade answers to questions about documents. You do not have the documents. "
    "Compare the candidate answer with the gold answer only."
)

JUDGE_PROMPT = """Question:
{question}

Gold answer:
{gold}

Candidate answer:
{answer}

Is the candidate answer correct?
- "correct": it states the gold answer's facts (paraphrase, rounding to the precision asked and
  equivalent units are fine; extra correct detail is fine).
- "incorrect": it contradicts the gold answer, gives a different value, misses required facts,
  or says the information is not available.
- "uncertain": the gold answer itself is ambiguous or you cannot tell.

Reply with JSON only: {{"verdict": "correct" | "incorrect" | "uncertain", "reason": "<short>"}}"""


def ask(llm: LLMProvider, question: str, gold: str, answer: str) -> tuple[str, str]:
    raw = llm.complete(
        JUDGE_PROMPT.format(question=question, gold=gold, answer=answer),
        system=JUDGE_SYSTEM,
        max_tokens=200,
        temperature=0.0,
    )
    try:
        data = extract_json(raw)
        verdict = str(data.get("verdict", "")).strip().lower()
        reason = str(data.get("reason", ""))
    except Exception:
        low = raw.lower()
        verdict = next((v for v in (INCORRECT, CORRECT, UNCERTAIN) if v in low), UNCERTAIN)
        reason = raw[:200]
    if verdict not in (CORRECT, INCORRECT, UNCERTAIN):
        verdict = UNCERTAIN
    return verdict, reason


def judge(
    judges: Sequence[tuple[str, LLMProvider]], question: str, gold: str, answer: str
) -> Verdict:
    votes = {}
    for name, llm in judges:
        try:
            votes[name] = ask(llm, question, gold, answer)
        except Exception as e:  # a failed judge call is an uncertain vote, not a crash
            votes[name] = (UNCERTAIN, f"judge error: {e!r}"[:200])
    labels = {v for v, _ in votes.values()}
    if len(labels) == 1 and UNCERTAIN not in labels:
        label = labels.pop()
        return Verdict(label, "semantic", next(iter(votes.values()))[1], dict(votes))
    detail = "judges disagree" if len(labels) > 1 else "judge uncertain"
    return Verdict(UNCERTAIN, "semantic", detail, dict(votes), needs_audit=True)


__all__ = ["JUDGE_PROMPT", "JUDGE_SYSTEM", "ask", "judge"]
