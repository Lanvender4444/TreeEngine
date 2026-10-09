"""Shared, lazily built indexes for one benchmark run.

Resources (each with its own IndexStats):

  corpus          TreeEngine ingest: parse -> tree -> blocks -> FTS5 (built by the runner)
  block_vectors   one embedding per Block (TreeEngine's vector index)
  chunks          traditional RAG: fixed-size token chunks of the raw document text + BM25
  chunk_vectors   one embedding per chunk

Everything is cached across runs: the ingested corpus in ``.cache/corpus-*.db``, embeddings in
``.cache/embeddings.sqlite`` (by model + text hash). Index costs are still reported as if built
from scratch: time from the run that actually computed them is not available on a cache hit,
so ``elapsed_seconds`` is the time of *this* run and the embedding token count is exact.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from treeengine.core.config import EngineConfig
from treeengine.core.protocols import LLMProvider, Repository
from treeengine.llm.base import LLMUsage, MeteredLLM

from ..metrics.cost import EmbeddingUsage, MeteredEmbedding, Prices
from .base import IndexStats


@dataclass
class Workspace:
    repo: Repository
    doc_ids: dict[str, str]  # corpus name -> document id
    config: EngineConfig = field(default_factory=EngineConfig)
    embed_spec: str | None = None
    llm: LLMProvider | None = None
    prices: Prices = field(default_factory=Prices)
    cache_dir: Path | None = None
    chunk_size: int = 600
    chunk_overlap: int = 100
    quiet: bool = True
    ingest_stats: IndexStats | None = None
    _resources: dict[str, Any] = field(default_factory=dict)
    _stats: dict[str, IndexStats] = field(default_factory=dict)
    query_embed_ms: float | None = None

    def __post_init__(self) -> None:
        if self.llm is not None and not isinstance(self.llm, MeteredLLM):
            self.llm = MeteredLLM(self.llm)
        self._embedder: MeteredEmbedding | None = None
        self._cached: Any = None
        if self.ingest_stats is not None:
            self._stats["corpus"] = self.ingest_stats

    # ------------------------------------------------------------------ usage
    def llm_usage(self) -> LLMUsage | None:
        return self.llm.usage.snapshot() if isinstance(self.llm, MeteredLLM) else None

    def embedding_usage(self) -> EmbeddingUsage | None:
        return self._embedder.usage.snapshot() if self._embedder is not None else None

    def log(self, msg: str) -> None:
        if not self.quiet:
            print(msg, file=sys.stderr)

    # ------------------------------------------------------------------ embedder
    @property
    def embedder(self) -> MeteredEmbedding:
        if self._embedder is None:
            if not self.embed_spec:
                raise RuntimeError("this strategy needs an embedder (--embedder / --vector)")
            from treeengine.embeddings import CachedEmbedding, provider_from_spec

            kw: dict[str, Any] = {}
            if self.embed_spec.startswith("fastembed") and self.cache_dir:
                kw["cache_dir"] = str(self.cache_dir / "models")
            if self.embed_spec.startswith("openai"):
                import os

                kw["api_key"] = os.environ.get("TREEENGINE_EMBED_API_KEY")
                kw["base_url"] = os.environ.get(
                    "TREEENGINE_EMBED_BASE_URL", "https://api.openai.com/v1"
                )
            inner = provider_from_spec(self.embed_spec, **kw)
            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._cached = CachedEmbedding(inner, self.cache_dir / "embeddings.sqlite")
                inner = self._cached
            self._embedder = MeteredEmbedding(inner)
        return self._embedder

    def cache_info(self) -> str:
        c = self._cached
        return f"{c.hits} cached, {c.misses} computed" if c is not None else "no cache"

    def warm_queries(self, queries: Sequence[str]) -> None:
        """Embed every query once up front so retrieval latencies of all vector strategies are
        comparable; the per-query embedding latency is measured here and reported separately."""
        if not self.embed_spec:
            return
        emb = self.embedder
        inner = emb.inner
        t0 = time.perf_counter()
        for q in queries:
            inner.embed_query(q)
        self.query_embed_ms = (time.perf_counter() - t0) * 1000 / max(1, len(queries))

    def _vector_index(self, table: str) -> Any:
        try:
            from treeengine.storage.vectors import SQLiteVectorIndex

            return SQLiteVectorIndex(":memory:", table=table)
        except ImportError:
            from treeengine.storage.vectors import MemoryVectorIndex

            return MemoryVectorIndex()

    def _embed_stats(self, name: str, units: int, before: EmbeddingUsage, t0: float, idx: Any):
        used = self.embedder.usage.since(before)
        st = IndexStats(
            name,
            documents=len(self.doc_ids),
            blocks=units,
            elapsed_seconds=time.perf_counter() - t0,
            embedding_calls=used.calls,
            embedding_tokens=used.tokens,
            index_bytes=idx.size_bytes(),
            estimated_cost=self.prices.usd(embed=used.tokens),
            parts=[name],
        )
        self._stats[name] = st
        return st

    # ------------------------------------------------------------------ resources
    def block_vectors(self) -> Any:
        """TreeEngine's VectorRetriever over one embedding per block."""
        if "block_vectors" not in self._resources:
            from treeengine.retrieval.vector import VectorIndexer, VectorRetriever

            index = self._vector_index("block_vectors")
            emb = self.embedder
            before, t0 = emb.usage.snapshot(), time.perf_counter()
            self.log("embedding blocks ...")
            n = VectorIndexer(self.repo, index, emb).reindex_all()
            self._embed_stats("block_vectors", n, before, t0, index)
            self._resources["block_vectors"] = VectorRetriever(self.repo, index, emb, self.config)
        return self._resources["block_vectors"]

    def chunks(self) -> Any:
        if "chunks" not in self._resources:
            from .chunk_rag import ChunkStore

            t0 = time.perf_counter()
            self.log("chunking ...")
            store = ChunkStore(
                self.repo, list(self.doc_ids.values()), self.chunk_size, self.chunk_overlap
            )
            self._stats["chunks"] = IndexStats(
                "chunks",
                documents=len(self.doc_ids),
                blocks=len(store.chunks),
                elapsed_seconds=time.perf_counter() - t0,
                index_bytes=store.size_bytes(),
                parts=["chunks"],
            )
            self._resources["chunks"] = store
        return self._resources["chunks"]

    def chunk_vectors(self) -> Any:
        if "chunk_vectors" not in self._resources:
            store = self.chunks()
            index = self._vector_index("chunk_vectors")
            emb = self.embedder
            before, t0 = emb.usage.snapshot(), time.perf_counter()
            self.log(f"embedding {len(store.chunks)} chunks ...")
            store.index_vectors(index, emb)
            self._embed_stats("chunk_vectors", len(store.chunks), before, t0, index)
            self._resources["chunk_vectors"] = index
        return self._resources["chunk_vectors"]

    def require(self, names: Sequence[str]) -> None:
        for n in names:
            if n == "corpus":
                continue
            getattr(self, n)()

    def resource_stats(self, name: str) -> IndexStats:
        if name not in self._stats:
            self.require([name])
        return self._stats.get(name, IndexStats(name, parts=[name]))

    def all_stats(self) -> dict[str, IndexStats]:
        return dict(self._stats)


__all__ = ["Workspace"]
