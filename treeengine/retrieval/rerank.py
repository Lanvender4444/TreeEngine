"""V0.16 local rerankers (optional dependency: fastembed).

Two kinds, both scoring (question, block) pairs on the CPU, no LLM:

    "ce:<model>"       cross-encoder (fastembed TextCrossEncoder), e.g.
                       ce:Xenova/ms-marco-MiniLM-L-12-v2, ce:BAAI/bge-reranker-base
    "colbert:<model>"  late interaction (fastembed LateInteractionTextEmbedding): token-level
                       embeddings, score = sum over query tokens of the max similarity, e.g.
                       colbert:answerdotai/answerai-colbert-small-v1

The block text scored is its index title (document | section | page heading, when indexed with
``index_title="context"``) followed by the block content.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

_MODELS: dict[str, Any] = {}
_LOCK = threading.Lock()


def _load(spec: str) -> tuple[str, Any]:
    kind, _, name = spec.partition(":")
    with _LOCK:
        if spec not in _MODELS:
            if kind == "ce":
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                _MODELS[spec] = TextCrossEncoder(name)
            elif kind == "colbert":
                from fastembed import LateInteractionTextEmbedding

                _MODELS[spec] = LateInteractionTextEmbedding(name)
            else:
                raise ValueError(f"unknown reranker {spec!r} (ce:<model> | colbert:<model>)")
    return kind, _MODELS[spec]


_CACHE: dict[tuple[str, str, str], float] = {}
_CACHE_MAX = 200_000


def rerank_scores(spec: str, query: str, texts: Sequence[str]) -> list[float]:
    """One score per text (higher = more relevant). Raises if the model cannot be loaded.
    Scores are cached per (model, query, text): every context policy, budget and search
    algorithm asking about the same unit pays once."""
    if not texts:
        return []
    keys = [(spec, query, t) for t in texts]
    missing = [t for k, t in zip(keys, texts, strict=True) if k not in _CACHE]
    if missing:
        uniq = list(dict.fromkeys(missing))
        for t, sc in zip(uniq, _score(spec, query, uniq), strict=True):
            if len(_CACHE) >= _CACHE_MAX:
                _CACHE.clear()
            _CACHE[(spec, query, t)] = sc
    return [_CACHE[k] for k in keys]


def _score(spec: str, query: str, texts: Sequence[str]) -> list[float]:
    kind, model = _load(spec)
    if kind == "ce":
        return [float(s) for s in model.rerank(query, list(texts), batch_size=16)]
    import numpy as np

    q = np.asarray(next(iter(model.query_embed([query]))), dtype=np.float32)
    out = []
    for d in model.embed(list(texts), batch_size=16):
        d = np.asarray(d, dtype=np.float32)
        out.append(float((q @ d.T).max(axis=1).sum()))
    return out


__all__ = ["rerank_scores"]
