"""Unified answerer: every strategy's Evidence[] becomes the same prompt for the same model.

Same model, same prompt, same temperature (0), same evidence formatting, same top-k: the only
variable left in end-to-end accuracy is what retrieval handed over.
"""

from __future__ import annotations

import time
import urllib.error
from collections.abc import Sequence
from dataclasses import dataclass

from treeengine.core.models import Evidence
from treeengine.core.protocols import LLMProvider
from treeengine.llm.base import MeteredLLM

ANSWER_SYSTEM = "You are a careful analyst answering questions about documents."

ANSWER_PROMPT = """Answer the question only using the evidence below.
If the evidence is insufficient, say so.
Cite supporting evidence as [E#].

Question:
{question}

Evidence:
{evidence}

Answer concisely."""


def _pages(e: Evidence) -> str:
    md = e.metadata or {}
    pages = md.get("pages") or ([md["page"]] if md.get("page") is not None else [])
    if not pages:
        return ""
    if len(pages) == 1:
        return f" (page {pages[0]})"
    return f" (pages {min(pages)}-{max(pages)})"


def format_evidence(evidence: Sequence[Evidence]) -> str:
    """``[E1] (page 12) text`` - page labels only, identical for every strategy (no section
    titles: structural provenance must not leak into the answer prompt for one family only)."""
    return "\n\n".join(f"[E{i}]{_pages(e)} {e.content.strip()}" for i, e in enumerate(evidence, 1))


@dataclass
class Answer:
    text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    error: str | None = None


class Answerer:
    def __init__(self, llm: LLMProvider, max_tokens: int = 512) -> None:
        self.llm = llm if isinstance(llm, MeteredLLM) else MeteredLLM(llm)
        self.max_tokens = max_tokens

    def prompt(self, question: str, evidence: Sequence[Evidence]) -> str:
        return ANSWER_PROMPT.format(question=question, evidence=format_evidence(evidence))

    def answer(self, question: str, evidence: Sequence[Evidence]) -> Answer:
        before = self.llm.usage.snapshot()
        t0 = time.perf_counter()
        try:
            text = self.llm.complete(
                self.prompt(question, evidence),
                system=ANSWER_SYSTEM,
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            err = None
        except Exception as e:  # recorded, scored as incorrect, run continues
            text, err = "", repr(e)
        used = self.llm.usage.since(before)
        return Answer(
            text, (time.perf_counter() - t0) * 1000, used.input_tokens, used.output_tokens, err
        )


class RetryingLLM:
    """Retries transient failures (429 / 5xx / timeouts) with exponential backoff; forwards the
    provider's ``last_usage`` so MeteredLLM still records real token counts."""

    def __init__(self, inner: LLMProvider, retries: int = 5) -> None:
        self.inner = inner
        self.retries = retries
        self.model = getattr(inner, "model", None)

    @property
    def last_usage(self) -> dict[str, int] | None:
        return getattr(self.inner, "last_usage", None)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        for attempt in range(self.retries + 1):
            try:
                return self.inner.complete(
                    prompt, system=system, max_tokens=max_tokens, temperature=temperature
                )
            except Exception as e:
                code = getattr(e, "code", None)
                transient = code in (408, 409, 429, 500, 502, 503, 504) or (
                    code is None
                    and isinstance(e, urllib.error.URLError | TimeoutError | ConnectionError)
                )
                if not transient or attempt == self.retries:
                    raise
                time.sleep(min(60.0, 2.0**attempt))
        raise RuntimeError("unreachable")


def llm_from_env(prefix: str, timeout: float = 300.0) -> LLMProvider | None:
    """``{prefix}_MODEL`` / ``_API_KEY`` / ``_BASE_URL`` -> OpenAI-compatible chat model.

    ``{prefix}_EXTRA_BODY`` (JSON) is merged into every request, e.g. for DeepSeek V4
    ``{"thinking": {"type": "disabled"}}`` so the answer budget is not spent on reasoning."""
    import json
    import os

    from treeengine.llm.base import OpenAICompatibleLLM

    model = os.environ.get(f"{prefix}_MODEL")
    if not model:
        return None
    inner = OpenAICompatibleLLM(
        model=model,
        api_key=os.environ.get(f"{prefix}_API_KEY"),
        base_url=os.environ.get(f"{prefix}_BASE_URL", "https://api.openai.com/v1"),
        timeout=timeout,
        extra_body=json.loads(os.environ.get(f"{prefix}_EXTRA_BODY") or "{}"),
    )
    return RetryingLLM(inner)


__all__ = [
    "ANSWER_PROMPT",
    "ANSWER_SYSTEM",
    "Answer",
    "Answerer",
    "RetryingLLM",
    "format_evidence",
    "llm_from_env",
]
