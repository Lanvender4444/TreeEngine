"""Embedding providers. Core code depends only on ``core.protocols.EmbeddingProvider``.

* ``FastEmbedProvider``        local ONNX models via ``fastembed`` (extra: ``treeengine[vector]``)
* ``OpenAICompatibleEmbedding`` any ``/v1/embeddings`` endpoint (OpenAI, new-api, vLLM, Ollama)
* ``HashingEmbedding``          dependency-free character n-gram hashing (tests / smoke runs)
* ``CachedEmbedding``           wraps any provider with a persistent SQLite cache
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.protocols import EmbeddingProvider


def l2_normalise(v: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [float(x) / n for x in v]


# Instruction prefixes some models are trained with (query, passage); fastembed does not add them.
_PREFIXES: dict[str, tuple[str, str]] = {
    "google/embeddinggemma-300m": ("task: search result | query: ", "title: none | text: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "nomic-ai/nomic-embed-text-v1.5": ("search_query: ", "search_document: "),
    "nomic-ai/nomic-embed-text-v1.5-Q": ("search_query: ", "search_document: "),
}


class FastEmbedProvider:
    """Local embedding model through fastembed (ONNX runtime, CPU, no torch).

    Default ``jinaai/jina-embeddings-v2-base-zh``: 768-d, trained for mixed Chinese/English.
    Texts are embedded in length-sorted batches (padding waste dominates CPU time otherwise:
    ~3x faster on real block lengths) and returned in input order.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v2-base-zh",
        cache_dir: str | None = None,
        batch_size: int = 16,
        max_chars: int = 8000,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("FastEmbedProvider needs: pip install 'treeengine[vector]'") from e
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.query_prefix, self.passage_prefix = _PREFIXES.get(model_name, ("", ""))
        self._model = TextEmbedding(model_name, cache_dir=cache_dir)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        clipped = [self.passage_prefix + t[: self.max_chars] for t in texts]
        order = sorted(range(len(clipped)), key=lambda i: len(clipped[i]))
        vecs = self._model.embed([clipped[i] for i in order], batch_size=self.batch_size)
        out: list[list[float]] = [[] for _ in clipped]
        for i, v in zip(order, vecs, strict=True):
            out[i] = l2_normalise(v.tolist())
        return out

    def embed_query(self, query: str) -> list[float]:
        if self.query_prefix:
            v = next(iter(self._model.embed([self.query_prefix + query])))
        else:
            v = next(iter(self._model.query_embed([query])))
        return l2_normalise(v.tolist())


class OpenAICompatibleEmbedding:
    """Any ``/v1/embeddings`` endpoint (OpenAI, SiliconFlow, new-api / one-api, vLLM, Ollama).

    Batches by count and by an approximate character budget, retries 429 / 5xx / network errors
    with exponential backoff, and records the provider-reported token usage in ``usage_tokens``.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        batch_size: int = 32,
        max_chars: int = 16000,
        batch_chars: int = 60000,
        retries: int = 6,
        dimensions: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.batch_chars = batch_chars
        self.retries = retries
        self.dimensions = dimensions
        self.usage_tokens = 0
        self.requests = 0

    def _post(self, inputs: list[str]) -> list[list[float]]:
        import time

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body: dict[str, Any] = {"model": self.model_name, "input": inputs}
        if self.dimensions:
            body["dimensions"] = self.dimensions
        payload = json.dumps(body).encode("utf-8")
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/embeddings", data=payload, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if e.code not in (408, 409, 429, 500, 502, 503, 504) or attempt == self.retries:
                    detail = e.read().decode("utf-8", "replace")[:300]
                    raise RuntimeError(f"embedding request failed: HTTP {e.code} {detail}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.retries:
                    raise
            time.sleep(min(60.0, 2.0**attempt))
        self.requests += 1
        self.usage_tokens += int((data.get("usage") or {}).get("total_tokens") or 0)
        rows = sorted(data["data"], key=lambda r: r["index"])
        return [l2_normalise(r["embedding"]) for r in rows]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        batch: list[str] = []
        size = 0
        for t in texts:
            t = t[: self.max_chars] or " "
            if batch and (len(batch) >= self.batch_size or size + len(t) > self.batch_chars):
                out += self._post(batch)
                batch, size = [], 0
            batch.append(t)
            size += len(t)
        if batch:
            out += self._post(batch)
        return out

    def embed_query(self, query: str) -> list[float]:
        return self._post([query[: self.max_chars] or " "])[0]


class HashingEmbedding:
    """Deterministic character-trigram hashing ("bag of n-grams"). Not semantic - it exists so
    tests and offline smoke runs can exercise the vector path without a model download."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.model_name = f"hashing-{dim}"

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        t = f"  {text.lower()}  "
        for i in range(len(t) - 2):
            h = int.from_bytes(hashlib.md5(t[i : i + 3].encode("utf-8")).digest()[:4], "little")
            v[h % self.dim] += 1.0
        return l2_normalise(v)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vec(query)


def _f32(v: Sequence[float]) -> list[float]:
    """Round-trip through float32 so cached and fresh vectors are bit-identical."""
    return list(struct.unpack(f"{len(v)}f", struct.pack(f"{len(v)}f", *v)))


class CachedEmbedding:
    """Persistent cache keyed by (model, sha256(text)); only cache misses reach the provider."""

    def __init__(self, inner: EmbeddingProvider, path: str | Path) -> None:
        self.inner = inner
        self.model_name = inner.model_name
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS emb (model TEXT, kind TEXT, key TEXT, vec BLOB, "
            "PRIMARY KEY (model, kind, key))"
        )
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get(self, kind: str, keys: list[str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            q = ",".join("?" * len(chunk))
            for key, blob in self.conn.execute(
                f"SELECT key, vec FROM emb WHERE model=? AND kind=? AND key IN ({q})",
                [self.model_name, kind, *chunk],
            ):
                out[key] = list(struct.unpack(f"{len(blob) // 4}f", blob))
        return out

    def _put(self, kind: str, rows: list[tuple[str, list[float]]]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO emb VALUES (?, ?, ?, ?)",
            [(self.model_name, kind, k, struct.pack(f"{len(v)}f", *v)) for k, v in rows],
        )
        self.conn.commit()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        have = self._get("passage", list(dict.fromkeys(keys)))
        missing = [(k, t) for k, t in dict(zip(keys, texts, strict=True)).items() if k not in have]
        self.hits += len(keys) - len(missing)
        self.misses += len(missing)
        if missing:
            vecs = self.inner.embed_texts([t for _, t in missing])
            new = [(k, _f32(v)) for (k, _), v in zip(missing, vecs, strict=True)]
            self._put("passage", new)
            have.update(new)
        return [have[k] for k in keys]

    def embed_query(self, query: str) -> list[float]:
        key = self._key(query)
        got = self._get("query", [key])
        if key in got:
            self.hits += 1
            return got[key]
        self.misses += 1
        v = _f32(self.inner.embed_query(query))
        self._put("query", [(key, v)])
        return v


def provider_from_spec(spec: str, **kw: Any) -> EmbeddingProvider:
    """``fastembed:<model>`` | ``openai:<model>`` | ``hashing`` -> provider."""
    kind, _, model = spec.partition(":")
    if kind == "fastembed":
        return FastEmbedProvider(model or "jinaai/jina-embeddings-v2-base-zh", **kw)
    if kind == "openai":
        return OpenAICompatibleEmbedding(model, **kw)
    if kind == "hashing":
        return HashingEmbedding(int(model) if model else 256)
    raise ValueError(f"unknown embedding spec {spec!r}")
