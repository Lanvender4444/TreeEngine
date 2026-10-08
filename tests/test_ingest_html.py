from __future__ import annotations

from pathlib import Path

from treeengine import TreeEngine

NOISE = ["噪声", "tracking", "UA-NOISE", "console.log", "font-family", "cookies", "Hidden"]


def test_product_page_tree_and_noise_filter(engine: TreeEngine, fixtures: Path) -> None:
    doc = engine.ingest(fixtures / "product_page.html")
    assert doc.title == "EdgeBox 3 边缘网关"
    assert doc.metadata["html_title"] == "EdgeBox 3 产品文档 - 星河科技"
    for word in NOISE:
        assert word not in doc.text, word
    tree = engine.get_tree(doc.id)
    [root] = tree["roots"]
    assert [c["title"] for c in root["children"]] == ["硬件规格", "软件与协议", "安装与维护"]
    hw = root["children"][0]
    assert [c["title"] for c in hw["children"]] == ["处理器与内存", "接口"]

    blocks = engine.repo.get_document_blocks(doc.id)
    table = [b for b in blocks if b.block_type == "table"]
    assert table and "工作温度 | -40°C 至 85°C" in table[0].content
    code = [b for b in blocks if b.block_type == "code"]
    assert code and "\nedgebox-cli restart" in code[0].content  # whitespace preserved
    lists = [b.content for b in blocks if b.block_type == "list"]
    assert "4 路 RS-485 串口" in lists
    for b in blocks:
        assert doc.text[b.start_offset : b.end_offset] == b.content


def test_page_without_main_uses_role_filter(engine: TreeEngine, fixtures: Path) -> None:
    doc = engine.ingest(fixtures / "blog_post.html")
    assert "Home | Blog" not in doc.text
    for word in NOISE:
        assert word not in doc.text
    titles = [n.title for n in engine.repo.get_document_nodes(doc.id)]
    assert titles == [
        "Understanding SQLite FTS5",
        "Ranking with bm25",
        "Tokenizers",
        "External content tables",
    ]


def test_ingest_html_text(engine: TreeEngine) -> None:
    doc = engine.ingest_text(
        "<h1>A</h1><p>one<br>two</p><h2>B</h2><p>three &amp; four</p>", source_type="html"
    )
    assert "three & four" in doc.text
    roots = engine.repo.get_roots(doc.id)
    assert [r.title for r in roots] == ["A"]
    assert [c.title for c in engine.repo.get_children(roots[0].id)] == ["B"]
