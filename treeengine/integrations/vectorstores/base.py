"""The contract every vector store adapter keeps.

A record is a vector plus the payload TreeEngine needs to map a hit back: ``document_id``,
``block_id`` (or ``chunk_id``), ``page``, offsets, ``source_revision``, ``uri``. Adapters store
the payload in whatever the database offers (Qdrant payload, Chroma metadata, LanceDB columns,
a FAISS sidecar) and return ``StoreHit``s whose ``to_hit()`` is an external hit that
``engine.build_context`` / ``resolve_hits`` maps exactly.

Scores are similarities: higher is better (cosine for every adapter here).
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

PAYLOAD_KEYS = (
    "document_id",
    "block_id",
    "chunk_id",
    "node_id",
    "page",
    "start_offset",
    "end_offset",
    "source_revision",
    "uri",
)
NAMESPACE = uuid.UUID("6f1b0a52-7d0e-4c55-9a77-1e5e0b7ce001")


@dataclass
class VectorRecord:
    id: str
    vector: Sequence[float]
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def document_id(self) -> str | None:
        v = self.metadata.get("document_id")
        return None if v is None else str(v)


@dataclass
class StoreHit:
    id: str
    score: float
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_hit(self) -> dict[str, Any]:
        """An external hit for ``engine.build_context`` (ids and positions at the top)."""
        h: dict[str, Any] = {
            "text": self.text,
            "score": self.score,
            "metadata": dict(self.metadata),
        }
        for k in PAYLOAD_KEYS:
            if self.metadata.get(k) is not None:
                h[k] = self.metadata[k]
        return h


@runtime_checkable
class VectorStore(Protocol):
    name: str

    def upsert(self, records: Sequence[VectorRecord]) -> None: ...

    def delete(
        self, ids: Sequence[str] | None = None, *, document_id: str | None = None
    ) -> int: ...

    def search(
        self, vector: Sequence[float], k: int = 10, *, document_id: str | None = None
    ) -> list[StoreHit]: ...

    def vectors(self, ids: Sequence[str]) -> dict[str, list[float]]: ...

    def count(self) -> int: ...

    def ids(self, document_id: str | None = None) -> list[str]: ...


def point_uuid(record_id: str) -> str:
    """Stable UUID for databases that only accept UUID / integer ids (Qdrant)."""
    return str(uuid.uuid5(NAMESPACE, record_id))


def int_id(record_id: str) -> int:
    """Stable positive int64 id (FAISS)."""
    return uuid.uuid5(NAMESPACE, record_id).int & ((1 << 63) - 1)


def normalise(v: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(float(x) * float(x) for x in v)) or 1.0
    return [float(x) / n for x in v]


def scalar_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Metadata for stores that only accept str / int / float / bool values (no None)."""
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = "json:" + json.dumps(v, ensure_ascii=False, default=str)
    return out


def unscalar_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if isinstance(v, str) and v.startswith("json:"):
            try:
                out[k] = json.loads(v[5:])
                continue
            except ValueError:
                pass
        out[k] = v
    return out


def batched(xs: Sequence[Any], n: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


__all__ = [
    "PAYLOAD_KEYS",
    "StoreHit",
    "VectorRecord",
    "VectorStore",
    "batched",
    "int_id",
    "normalise",
    "point_uuid",
    "scalar_meta",
    "unscalar_meta",
]
