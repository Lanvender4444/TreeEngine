"""Vector database adapters with one contract (``base.VectorStore``).

* ``InMemoryStore``  dependency-free reference (tests, small corpora)
* ``QdrantStore``    ``treeengine[qdrant]``
* ``ChromaStore``    ``treeengine[chroma]``
* ``LanceDBStore``   ``treeengine[lancedb]``
* ``FaissStore``     ``treeengine[faiss]``

Uses: hits from any of them carry the TreeEngine payload and map exactly in
``engine.build_context`` (``StoreHit.to_hit``); ``sync.BlockSync`` keeps a store in step with
the engine's blocks; ``backend.StoreVectorIndex`` makes a store the engine's own vector index.
The adapters import their client library lazily: ``open_store("qdrant", ...)``.
"""

from __future__ import annotations

from typing import Any

from .base import StoreHit, VectorRecord, VectorStore
from .memory import InMemoryStore

_STORES = {
    "memory": (".memory", "InMemoryStore"),
    "qdrant": (".qdrant", "QdrantStore"),
    "chroma": (".chroma", "ChromaStore"),
    "lancedb": (".lancedb", "LanceDBStore"),
    "faiss": (".faiss_store", "FaissStore"),
}


def store_class(name: str) -> Any:
    import importlib

    if name not in _STORES:
        raise ValueError(f"unknown vector store {name!r}; one of {sorted(_STORES)}")
    mod, cls = _STORES[name]
    return getattr(importlib.import_module(mod, __name__), cls)


def open_store(name: str, *args: Any, **kw: Any) -> Any:
    return store_class(name)(*args, **kw)


STORES = tuple(_STORES)


def __getattr__(name: str) -> Any:  # lazy: BlockSync / StoreVectorIndex
    if name in ("BlockSync", "SyncReport"):
        from . import sync

        return getattr(sync, name)
    if name == "StoreVectorIndex":
        from .backend import StoreVectorIndex

        return StoreVectorIndex
    raise AttributeError(name)


__all__ = [
    "STORES",
    "BlockSync",
    "StoreVectorIndex",
    "SyncReport",
    "InMemoryStore",
    "StoreHit",
    "VectorRecord",
    "VectorStore",
    "open_store",
    "store_class",
]
