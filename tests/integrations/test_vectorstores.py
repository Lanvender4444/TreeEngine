"""One contract for every vector store adapter (skipped when the client library is missing)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from treeengine.integrations.vectorstores import STORES, VectorRecord, open_store

NEEDS = {"qdrant": "qdrant_client", "chroma": "chromadb", "lancedb": "lancedb", "faiss": "faiss"}


def available(name: str) -> bool:
    mod = NEEDS.get(name)
    return mod is None or importlib.util.find_spec(mod) is not None


PARAMS = [
    pytest.param(n, marks=pytest.mark.skipif(not available(n), reason=f"{n} missing"))
    for n in STORES
]


def vec(i: int, dim: int = 8) -> list[float]:
    return [math.cos(i * 0.7 + j) for j in range(dim)]


def recs(doc: str, n: int, start: int = 0) -> list[VectorRecord]:
    return [
        VectorRecord(
            f"blk_{doc}_{i}",
            vec(i if doc != "d2" else i + 50),
            f"text {doc} {i} it's",
            {
                "document_id": doc,
                "block_id": f"blk_{doc}_{i}",
                "page": i % 3,
                "node_ids": ["a", "b"],
            },
        )
        for i in range(start, start + n)
    ]


@pytest.fixture
def store(request: pytest.FixtureRequest, tmp_path: Path):  # type: ignore[no-untyped-def]
    name = request.param
    if name == "lancedb":
        import lancedb

        return open_store(name, lancedb.connect(str(tmp_path / "l")), "t")
    if name == "chroma":
        import uuid

        import chromadb

        return open_store(name, chromadb.EphemeralClient(), "t" + uuid.uuid4().hex[:8])
    return open_store(name)


@pytest.mark.parametrize("store", PARAMS, indirect=True)
def test_contract(store) -> None:  # type: ignore[no-untyped-def]
    assert store.count() == 0 and store.search(vec(1), 3) == [] and store.ids() == []
    store.upsert(recs("d1", 5) + recs("d2", 4))
    assert store.count() == 9
    # nearest neighbour is the identical vector; payload round-trips (lists too)
    top = store.search(vec(2), 3)
    assert top[0].id == "blk_d1_2" and top[0].score == pytest.approx(1.0, abs=1e-4)
    assert top[0].text == "text d1 2 it's"
    assert top[0].metadata["block_id"] == "blk_d1_2" and top[0].metadata["page"] == 2
    assert top[0].metadata["node_ids"] == ["a", "b"]
    assert [h.score for h in top] == sorted((h.score for h in top), reverse=True)
    got = store.vectors(["blk_d1_3", "nope"])
    assert list(got) == ["blk_d1_3"] and len(got["blk_d1_3"]) == 8
    n = math.sqrt(sum(x * x for x in vec(3)))
    assert sum(a * b / n for a, b in zip(got["blk_d1_3"], vec(3), strict=True)) / (
        math.sqrt(sum(x * x for x in got["blk_d1_3"])) or 1
    ) == pytest.approx(1.0, abs=1e-4)  # same direction (stores may normalise)
    hit = top[0].to_hit()
    assert hit["block_id"] == "blk_d1_2" and hit["document_id"] == "d1"
    # document filter is exact
    only = store.search(vec(2), 10, document_id="d2")
    assert only and all(h.metadata["document_id"] == "d2" for h in only) and len(only) == 4
    assert store.ids("d2") == [f"blk_d2_{i}" for i in range(4)]
    # upsert replaces, does not duplicate
    changed = recs("d1", 1)
    changed[0].text = "changed"
    store.upsert(changed)
    assert store.count() == 9 and store.search(vec(0), 1)[0].text == "changed"
    # deletes
    assert store.delete(["blk_d1_0", "missing"]) == 1 and store.count() == 8
    assert store.delete(document_id="d2") == 4 and store.count() == 4
    assert store.search(vec(1), 10, document_id="d2") == []
    assert store.ids() == [f"blk_d1_{i}" for i in range(1, 5)]


def test_unknown_store() -> None:
    with pytest.raises(ValueError):
        open_store("nope")


@pytest.mark.skipif(not available("faiss"), reason="faiss missing")
def test_faiss_save_load(tmp_path: Path) -> None:
    from treeengine.integrations.vectorstores.faiss_store import FaissStore

    st = FaissStore()
    st.upsert(recs("d", 3))
    st.save(tmp_path / "f")
    st2 = FaissStore.load(tmp_path / "f")
    assert st2.count() == 3 and st2.search(vec(1), 1)[0].id == "blk_d_1"
