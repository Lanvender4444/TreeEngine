"""TE-036: LangChain / LlamaIndex adapters against the real frameworks (each CI job installs
only its framework; a skip there is a failure)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from treeengine import TreeEngine

DOC = "# Ops\n\n## Cache\n\nRedis eviction uses allkeys-lru.\n\n## Deploy\n\nRoll out daily.\n"


@pytest.fixture
def engine(tmp_path: Path) -> TreeEngine:
    p = tmp_path / "ops.md"
    p.write_text(DOC, encoding="utf-8")
    eng = TreeEngine(tmp_path / "kb.db")
    eng.ingest(p)
    yield eng
    eng.close()


def _check_meta(meta: dict) -> None:
    assert (
        meta["document_id"] and meta["block_ids"] and meta["source_revision"].startswith("sha256:")
    )
    assert all(isinstance(v, (str, int, float, bool, list, type(None))) for v in meta.values())


def test_langchain_retriever(engine: TreeEngine) -> None:
    pytest.importorskip("langchain_core")
    from treeengine.integrations.langchain import TreeEngineRetriever

    r = TreeEngineRetriever(engine=engine, context_budget=300, preset="lite")
    docs = r.invoke("Redis eviction")
    assert docs and "allkeys-lru" in docs[0].page_content
    _check_meta(docs[0].metadata)
    docs2 = asyncio.run(r.ainvoke("Redis eviction"))  # the async path works too
    assert [d.page_content for d in docs2] == [d.page_content for d in docs]
    assert sum(d.metadata["tokens"] for d in docs) <= 300


def test_llamaindex_retriever(engine: TreeEngine) -> None:
    pytest.importorskip("llama_index.core")
    from treeengine.integrations.llamaindex import TreeEngineLlamaRetriever

    r = TreeEngineLlamaRetriever(engine, context_budget=300, preset="lite")
    nodes = r.retrieve("Redis eviction")
    assert nodes and "allkeys-lru" in nodes[0].node.get_content()
    assert nodes[0].score == 1.0
    _check_meta(nodes[0].node.metadata)


def test_missing_framework_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import importlib
    import sys

    from treeengine.api.errors import DependencyMissingError

    real = builtins.__import__

    def no_llama(name: str, *a: object, **k: object) -> object:
        if name.startswith("llama_index"):
            raise ImportError("no llama")
        return real(name, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", no_llama)
    sys.modules.pop("treeengine.integrations.llamaindex", None)
    with pytest.raises(DependencyMissingError, match="treeengine\\[llamaindex\\]"):
        importlib.import_module("treeengine.integrations.llamaindex")
    sys.modules.pop("treeengine.integrations.llamaindex", None)
