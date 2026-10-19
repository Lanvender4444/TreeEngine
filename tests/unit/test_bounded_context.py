"""TE-029: bounded neighbourhood reads build exactly the same ContextSpans as whole-document
reads, while loading only the pages around the anchors."""

from __future__ import annotations

from pathlib import Path

import pytest

from treeengine import TreeEngine
from treeengine.context.builder import BlockContextBuilder, _LazyBlocks

POLICIES = ["raw", "neighbor1", "neighbor2", "section600", "adaptive600", "auto", "resolution"]


def _doc(n_sections: int = 60, per: int = 12) -> str:
    out = ["# Handbook"]
    for s in range(n_sections):
        out.append(f"## Section {s}")
        out += [
            f"Paragraph {s}.{k}: term{s % 7} detail{k} value {s * per + k}." for k in range(per)
        ]
    return "\n\n".join(out) + "\n"


@pytest.fixture(scope="module")
def engine(tmp_path_factory: pytest.TempPathFactory) -> tuple[TreeEngine, str]:
    p = tmp_path_factory.mktemp("b") / "h.md"
    p.write_text(_doc(), encoding="utf-8")
    te = TreeEngine(":memory:")
    return te, te.ingest(p).id


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("query", ["term3 detail5", "value 500", "Section 41 detail0"])
def test_same_spans(engine: tuple[TreeEngine, str], policy: str, query: str) -> None:
    te, doc = engine
    hits = te.search(query, document_id=doc, limit=8)
    eager = BlockContextBuilder(te.repo, bounded=False).build(hits, token_budget=600, policy=policy)
    lazy = BlockContextBuilder(te.repo).build(hits, token_budget=600, policy=policy)
    assert [(s.block_ids, s.content, s.token_count, s.source_evidence_ids) for s in lazy] == [
        (s.block_ids, s.content, s.token_count, s.source_evidence_ids) for s in eager
    ]


def test_loads_only_pages_near_the_anchors(engine: tuple[TreeEngine, str]) -> None:
    te, doc = engine
    n = te.repo.count_document_blocks(doc)  # type: ignore[attr-defined]
    assert n > 4 * _LazyBlocks.PAGE
    b = BlockContextBuilder(te.repo)
    spans = b.build(te.search("value 500", document_id=doc, limit=1), token_budget=300)
    assert spans
    blocks = b.blocks(doc)
    assert isinstance(blocks, _LazyBlocks) and blocks.loaded <= 2 * _LazyBlocks.PAGE < n


def test_unknown_block_and_eager_repo(engine: tuple[TreeEngine, str]) -> None:
    te, doc = engine
    b = BlockContextBuilder(te.repo)
    assert "blk_nope" not in b._where and b.locate("blk_nope") is None

    class Plain:  # a repository without bounded reads: whole documents, as before
        def __init__(self, inner: object) -> None:
            self.inner = inner

        def __getattr__(self, k: str) -> object:
            if k in ("count_document_blocks", "get_block_range", "get_block_location"):
                raise AttributeError(k)
            return getattr(self.inner, k)

    pb = BlockContextBuilder(Plain(te.repo))  # type: ignore[arg-type]
    assert isinstance(pb.blocks(doc), list)


def test_engine_context_path_is_bounded(tmp_path: Path) -> None:
    p = tmp_path / "h.md"
    p.write_text(_doc(), encoding="utf-8")
    with TreeEngine(tmp_path / "kb.db") as te:
        d = te.ingest(p)
        res = te.retrieve("value 300", document_id=d.id, context_budget=400)
        assert res.context_spans and any("value 300" in s.content for s in res.context_spans)
