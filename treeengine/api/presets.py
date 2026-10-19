"""Retrieval presets (TE-022): one option instead of dozens of policy fields.

    lite      FTS + local ContextSpans. Never needs an embedder or an LLM.
    balanced  FTS + vector when an embedder is configured, context reconstruction. (default)
    quality   balanced + local reranker features + one PageIndex-style reading call (LLM) +
              answer self-check retry (``ask``). The V0.16 held-out winner.
    research  the experimental Tree-RAG search (``engine.research()``) with hard budgets.

Missing capabilities never fail silently: with ``strict=False`` (default) the preset falls back
to the closest cheaper one and records each step down in ``degradations`` (also in the v1
result's ``retrieval.degradations`` and ``status="degraded"``); with ``strict=True`` a
``DependencyMissingError`` is raised instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from ..retrieval.policy import RECOMMENDED_RERANKER, RetrievalPolicy
from .errors import DependencyMissingError, InvalidPolicyError


class SearchPreset(StrEnum):
    LITE = "lite"
    BALANCED = "balanced"
    QUALITY = "quality"
    RESEARCH = "research"


@dataclass(frozen=True)
class RetrievalOptions:
    """Simple retrieval options. ``policy`` (a full RetrievalPolicy) overrides the preset for
    experts; ``context_tokens`` is the reading budget (alias of the policy's
    ``context_budget``)."""

    preset: SearchPreset | str = SearchPreset.BALANCED
    context_tokens: int = 2000
    include_trace: bool = False
    strict: bool = False
    policy: RetrievalPolicy | None = None
    max_evidence: int = 20


@dataclass
class ResolvedPreset:
    preset: str
    policy: RetrievalPolicy
    answer_retry: bool = False
    degradations: list[str] = field(default_factory=list)


def _reranker_available() -> bool:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return False
    return True


def resolve(engine: Any, options: RetrievalOptions) -> ResolvedPreset:
    """The policy a preset means for this engine (its embedder / LLM), with degradations."""
    try:
        preset = SearchPreset(options.preset).value
    except ValueError:
        raise InvalidPolicyError(
            f"unknown preset {options.preset!r}", allowed=[p.value for p in SearchPreset]
        ) from None
    if options.context_tokens <= 0:
        raise InvalidPolicyError("context_tokens must be positive")
    budget = int(options.context_tokens)
    if options.policy is not None:
        return ResolvedPreset("custom", replace(options.policy, context_budget=budget))
    has_vec = engine.components.vector is not None
    has_llm = engine.llm is not None
    degr: list[str] = []

    def missing(what: str, step: str) -> None:
        if options.strict:
            raise DependencyMissingError(f"preset {preset!r} needs {what}", preset=preset)
        degr.append(f"{what} not configured: {step}")

    if preset == "lite":
        pol = RetrievalPolicy(fts_weight=1.0, vector_weight=0.0, context_budget=budget)
        return ResolvedPreset(preset, pol)
    vec_w = 1.0 if has_vec else 0.0
    if not has_vec:
        missing("an embedder", "FTS only (no vector path)")
    base = RetrievalPolicy(fts_weight=1.0, vector_weight=vec_w, context_budget=budget)
    if preset == "balanced":
        return ResolvedPreset(preset, base, degradations=degr)
    if not has_llm:
        missing("an LLM", "balanced reading (no reasoned selection, no answer retry)")
        return ResolvedPreset(preset, base, degradations=degr)
    rerank = RECOMMENDED_RERANKER
    if not _reranker_available():
        missing("fastembed (reranker)", "gate confidence from ranks only")
        rerank = ""
    pol = RetrievalPolicy.recommended(vector_weight=vec_w, context_budget=budget, rerank=rerank)
    return ResolvedPreset(preset, pol, answer_retry=True, degradations=degr)


__all__ = ["ResolvedPreset", "RetrievalOptions", "SearchPreset", "resolve"]
