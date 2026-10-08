"""Answer layer (convenience only): Evidence -> cited answer. Never retrieves by itself."""

from __future__ import annotations

import re

from .core.models import AnswerResult, Citation, Evidence
from .core.protocols import LLMProvider, Repository
from .llm import prompts

_CITE = re.compile(r"\[E(\d+)\]")


class Answerer:
    def __init__(self, repo: Repository, llm: LLMProvider | None = None) -> None:
        self.repo = repo
        self.llm = llm

    def answer(self, question: str, evidence: list[Evidence], query_type: str = "") -> AnswerResult:
        if not evidence:
            return AnswerResult(question, "No relevant evidence found.", [], [], query_type, False)
        if self.llm is None:
            body = "\n".join(f"[E{i}] {e.content}" for i, e in enumerate(evidence, 1))
            cites = [citation(i, e) for i, e in enumerate(evidence, 1)]
            return AnswerResult(question, body, cites, evidence, query_type, generated=False)

        ev_text = "\n\n".join(
            f"[E{i}] ({self.label(e)})\n{e.content}" for i, e in enumerate(evidence, 1)
        )
        answer = self.llm.complete(
            prompts.ANSWER.format(question=question, evidence=ev_text),
            system=prompts.ANSWER_SYSTEM,
            max_tokens=1024,
        ).strip()
        used = sorted({int(m) for m in _CITE.findall(answer) if 1 <= int(m) <= len(evidence)})
        if not used:
            used = list(range(1, len(evidence) + 1))
        cites = [citation(i, evidence[i - 1]) for i in used]
        return AnswerResult(question, answer, cites, evidence, query_type, generated=True)

    def label(self, e: Evidence) -> str:
        doc = self.repo.get_document(e.document_id)
        parts = [doc.title if doc else e.document_id]
        path = e.metadata.get("path") or (
            [e.metadata["node_title"]] if e.metadata.get("node_title") else []
        )
        parts += [p for p in path if p and p != parts[0]]
        if e.metadata.get("page"):
            parts.append(f"p.{e.metadata['page']}")
        return " > ".join(parts)


def citation(i: int, e: Evidence) -> Citation:
    return Citation(
        index=i,
        document_id=e.document_id,
        node_id=e.node_id,
        block_id=e.block_id,
        title=e.metadata.get("node_title"),
        page=e.metadata.get("page"),
        start_offset=e.metadata.get("start_offset"),
        end_offset=e.metadata.get("end_offset"),
    )
