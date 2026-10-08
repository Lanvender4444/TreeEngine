from __future__ import annotations

from pathlib import Path

from treeengine import TreeEngine
from treeengine.core.config import EngineConfig
from treeengine.structure.markdown import parse_markdown


def _assert_tree_invariants(te: TreeEngine, doc_id: str) -> None:
    nodes = te.repo.get_document_nodes(doc_id)
    by_id = {n.id: n for n in nodes}
    siblings: dict[str | None, list[int]] = {}
    for n in nodes:
        if n.parent_id is None:
            assert n.depth == 0
        else:
            assert by_id[n.parent_id].depth + 1 == n.depth
        siblings.setdefault(n.parent_id, []).append(n.position)
    for pos in siblings.values():
        assert sorted(pos) == list(range(len(pos)))
    for b in te.repo.get_document_blocks(doc_id):
        assert b.node_id in by_id


def test_annual_report_tree(engine: TreeEngine, fixtures: Path) -> None:
    doc = engine.ingest(fixtures / "annual_report.md")
    assert doc.title == "星河科技 2025 年度报告"
    tree = engine.get_tree(doc.id)
    assert tree["structure_method"] == "native"
    [root] = tree["roots"]
    assert [c["title"] for c in root["children"]] == [
        "一、公司概况",
        "二、财务表现",
        "三、研发进展",
        "四、风险因素",
        "五、未来展望",
    ]
    finance = root["children"][1]
    assert [c["title"] for c in finance["children"]] == ["收入与利润", "毛利率分析", "现金流"]
    assert finance["children"][1]["depth"] == 2
    _assert_tree_invariants(engine, doc.id)


def test_block_offsets_point_into_document(engine: TreeEngine, fixtures: Path) -> None:
    doc = engine.ingest(fixtures / "annual_report.md")
    for b in engine.repo.get_document_blocks(doc.id):
        assert b.start_offset is not None and b.end_offset is not None
        assert doc.text[b.start_offset : b.end_offset] == b.content
    for n in engine.repo.get_document_nodes(doc.id):
        if n.node_type == "section":
            assert doc.text[n.start_offset :].startswith("#")
            assert n.end_offset is not None and n.end_offset > n.start_offset


def test_setext_frontmatter_and_code_fences(engine: TreeEngine, fixtures: Path) -> None:
    doc = engine.ingest(fixtures / "api_guide.md")
    assert doc.title == "Orbit SDK Developer Guide"
    titles = [n.title for n in engine.repo.get_document_nodes(doc.id)]
    assert "Installation" in titles and "Authentication" in titles
    assert not any("comment" in t for t in titles)  # '# this line...' inside ``` is not a heading
    code = [b for b in engine.repo.get_document_blocks(doc.id) if b.block_type == "code"]
    assert any("pip install orbit-sdk" in b.content for b in code)
    kinds = {b.block_type for b in engine.repo.get_document_blocks(doc.id)}
    assert {"paragraph", "code", "list", "table", "quote"} <= kinds
    _assert_tree_invariants(engine, doc.id)


def test_flat_document_gets_single_node(engine: TreeEngine, fixtures: Path) -> None:
    doc = engine.ingest(fixtures / "flat_notes.md")
    tree = engine.get_tree(doc.id)
    assert tree["structure_method"] == "flat"
    assert len(tree["roots"]) == 1 and tree["roots"][0]["block_count"] == 4


def test_skipped_heading_levels_and_preamble() -> None:
    els = parse_markdown("intro\n\n# A\n\n### deep\n\ntext\n\n## B\n")
    assert [(e.kind, e.level) for e in els] == [
        ("block", 0),
        ("heading", 1),
        ("heading", 3),
        ("block", 0),
        ("heading", 2),
    ]
    with TreeEngine(":memory:") as te:
        doc = te.ingest_text("intro\n\n# A\n\n### deep\n\ntext\n\n## B\n", title="T")
        roots = te.repo.get_roots(doc.id)
        assert [r.node_type for r in roots] == ["preamble", "section"]
        a = roots[1]
        assert [c.title for c in te.repo.get_children(a.id)] == ["deep", "B"]


def test_long_paragraph_is_split(tmp_path: Path) -> None:
    para = "。".join(f"第{i}句话包含一些内容" for i in range(200)) + "。"
    with TreeEngine(":memory:", config=EngineConfig(max_block_chars=300)) as te:
        doc = te.ingest_text(f"# T\n\n{para}\n")
        blocks = te.repo.get_document_blocks(doc.id)
        assert len(blocks) > 5
        assert all(len(b.content) <= 300 for b in blocks)
        for b in blocks:
            assert doc.text[b.start_offset : b.end_offset] == b.content


def test_html_comments_are_ignored_but_offsets_kept() -> None:
    src = (
        "# 标题\n\n<!--\n## Hidden English heading\nEnglish text\n-->\n"
        "中文正文。\n\n## 第二节\n\n内容\n"
    )
    els = parse_markdown(src)
    assert [e.text for e in els if e.kind == "heading"] == ["标题", "第二节"]
    body = [e for e in els if e.kind == "block"]
    assert [b.text for b in body] == ["中文正文。", "内容"]
    for b in body:
        assert src[b.start : b.end] == b.text
