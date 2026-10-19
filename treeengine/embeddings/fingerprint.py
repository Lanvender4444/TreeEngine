"""Embedding-model fingerprints for vector indexes (TE-012).

A vector index records which model wrote its vectors. Opening it with another model (another
name, provider, or embedding-text format) would silently compare incomparable vectors, so it
is refused with ``IndexCompatibilityError`` unless ``EngineConfig.vector_fingerprint`` says
otherwise:

    "error"   (default) refuse a mismatching index
    "reset"   drop the old vectors, record the new model; then run engine.reindex_vectors()
    "ignore"  keep going (the old behaviour; for experiments only)

An index that has vectors but no fingerprint (written before this check existed) is adopted:
the current model is recorded with ``"adopted": true``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..api.errors import IndexCompatibilityError

log = logging.getLogger("treeengine")

# the text each block is embedded as (retrieval/vector.py: block_embedding_text); bump when it
# changes so old vectors are recognised as stale
EMBEDDING_TEXT_FORMAT = "title-newline-content-2000/1"


def embedder_fingerprint(embedder: Any) -> dict[str, Any]:
    inner = getattr(embedder, "inner", embedder)  # CachedEmbedding / metering wrappers
    return {
        "model": str(getattr(embedder, "model_name", "") or getattr(inner, "model_name", "")),
        "provider": type(inner).__name__,
        "text_format": EMBEDDING_TEXT_FORMAT,
        "normalized": True,
    }


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = ("model", "provider", "text_format")
    return all(a.get(k) == b.get(k) for k in keys)


def read_fingerprint(index: Any) -> dict[str, Any] | None:
    raw = index.meta("fingerprint") if hasattr(index, "meta") else None
    return json.loads(raw) if raw else None


def check_index_fingerprint(index: Any, embedder: Any, cfg: Any) -> None:
    if not hasattr(index, "meta") or not hasattr(index, "set_meta"):
        return  # a custom VectorIndex without metadata: nothing to check
    mode = getattr(cfg, "vector_fingerprint", "error")
    want = embedder_fingerprint(embedder)
    have = read_fingerprint(index)
    if have is None:
        if index.count() > 0:
            want = {**want, "adopted": True}
            log.warning("vector index has no model fingerprint; adopting %s", want["model"])
        index.set_meta("fingerprint", json.dumps(want, sort_keys=True))
        return
    if _same(have, want) or mode == "ignore":
        return
    if mode == "reset":
        log.warning(
            "vector index written by %s, engine uses %s: dropping old vectors (reindex next)",
            have.get("model"),
            want["model"],
        )
        index.rebuild()
        index.set_meta("fingerprint", json.dumps(want, sort_keys=True))
        return
    raise IndexCompatibilityError(
        f"the vector index was written by embedding model {have.get('model')!r} "
        f"({have.get('provider')}), this engine uses {want['model']!r} ({want['provider']}). "
        "Use the same model, or open with EngineConfig(vector_fingerprint='reset') and run "
        "engine.reindex_vectors().",
        index_model=have.get("model"),
        engine_model=want["model"],
    )


__all__ = [
    "EMBEDDING_TEXT_FORMAT",
    "check_index_fingerprint",
    "embedder_fingerprint",
    "read_fingerprint",
]
