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
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.protocols import EmbeddingProvider


def l2_normalise(v: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [float(x) / n for x in v]


class FastEmbedProvider:
    """Local embedding model through fastembed (ONNX runtime, CPU, no torch).

    Default ``jinaai/jina-embeddings-v2-base-zh``: 768-d, trained for mixed Chinese/English.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v2-base-zh",
        cache_dir: str | None = None,
        batch_size: int = 16,
        max_chars: int = 2000,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("FastEmbedProvider needs: pip install 'treeengine[vector]'") from e
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_chars = max_chars
        self._model = TextEmbedding(model_name, cache_dir=cache_dir)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        clipped = [t[: self.max_chars] for t in texts]
        return [
            l2_normalise(v.tolist())
            for v in self._model.passage_embed(clipped, batch_size=self.batch_size)
        ]

    def embed_query(self, query: str) -> list[float]:
        return l2_normalise(next(iter(self._model.query_embed([query]))).tolist())


class OpenAICompatibleEmbedding:
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size

    def _post(self, inputs: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model_name, "input": inputs}).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        rows = sorted(data["data"], key=lambda r: r["index"])
        return [l2_normalise(r["embedding"]) for r in rows]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out += self._post(list(texts[i : i + self.batch_size]))
        return out

    def embed_query(self, query: str) -> list[float]:
        return self._post([query])[0]


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
