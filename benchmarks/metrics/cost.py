"""Cost accounting: token counting, embedding metering, prices.

Token counts use ``tiktoken``'s ``cl100k_base`` when installed (``pip install tiktoken``) and a
CJK-aware estimate otherwise; the report says which one was used. Prices are USD per 1M tokens
and come from the command line or ``TREEENGINE_PRICE_*`` environment variables; with no price
configured, costs are reported in tokens only.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from treeengine.core.protocols import EmbeddingProvider
from treeengine.llm.base import estimate_tokens


@lru_cache(maxsize=1)
def _encoding() -> Any:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # not installed, or encoding data unavailable offline
        return None


def tokenizer_name() -> str:
    return "tiktoken cl100k_base" if _encoding() is not None else "estimate (chars/4, CJK=1)"


def count_tokens(text: str) -> int:
    enc = _encoding()
    if enc is None:
        return estimate_tokens(text)
    return len(enc.encode(text, disallowed_special=()))


def token_spans(text: str) -> list[tuple[int, int]]:
    """Character span of every token (used by the chunker)."""
    enc = _encoding()
    if enc is None:
        import re

        return [m.span() for m in re.finditer(r"[぀-鿿가-힯]|\w+|[^\w\s]", text)]
    toks = enc.encode(text, disallowed_special=())
    _, offsets = enc.decode_with_offsets(toks)
    spans = []
    for i, start in enumerate(offsets):
        end = offsets[i + 1] if i + 1 < len(offsets) else len(text)
        spans.append((start, max(start, end)))
    return spans


@dataclass
class Prices:
    """USD per 1M tokens."""

    llm_input: float | None = None
    llm_output: float | None = None
    embedding: float | None = None

    @classmethod
    def from_env(cls, **override: float | None) -> Prices:
        def env(name: str) -> float | None:
            v = os.environ.get(name)
            return float(v) if v else None

        p = cls(
            env("TREEENGINE_PRICE_LLM_INPUT"),
            env("TREEENGINE_PRICE_LLM_OUTPUT"),
            env("TREEENGINE_PRICE_EMBEDDING"),
        )
        for k, v in override.items():
            if v is not None:
                setattr(p, k, v)
        return p

    @property
    def configured(self) -> bool:
        return any(x is not None for x in (self.llm_input, self.llm_output, self.embedding))

    def usd(self, llm_in: int = 0, llm_out: int = 0, embed: int = 0) -> float | None:
        if not self.configured:
            return None
        return (
            llm_in * (self.llm_input or 0.0)
            + llm_out * (self.llm_output or 0.0)
            + embed * (self.embedding or 0.0)
        ) / 1e6


@dataclass
class EmbeddingUsage:
    calls: int = 0  # provider requests made by the system (logical, before any cache)
    texts: int = 0
    tokens: int = 0

    def since(self, before: EmbeddingUsage) -> EmbeddingUsage:
        return EmbeddingUsage(
            self.calls - before.calls, self.texts - before.texts, self.tokens - before.tokens
        )

    def snapshot(self) -> EmbeddingUsage:
        return EmbeddingUsage(self.calls, self.texts, self.tokens)


@dataclass
class MeteredEmbedding:
    """Counts what a deployed system would send to the embedding provider.

    Wraps the cache (``Metered(Cached(provider))``) so cached benchmark re-runs still report the
    real cost of indexing and querying."""

    inner: EmbeddingProvider
    batch_size: int = 64
    usage: EmbeddingUsage = field(default_factory=EmbeddingUsage)
    model_name: str = ""

    def __post_init__(self) -> None:
        self.model_name = self.inner.model_name

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.usage.calls += -(-len(texts) // self.batch_size) if texts else 0
        self.usage.texts += len(texts)
        self.usage.tokens += sum(count_tokens(t) for t in texts)
        return self.inner.embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        self.usage.calls += 1
        self.usage.texts += 1
        self.usage.tokens += count_tokens(query)
        return self.inner.embed_query(query)


__all__ = [
    "EmbeddingUsage",
    "MeteredEmbedding",
    "Prices",
    "count_tokens",
    "token_spans",
    "tokenizer_name",
]
