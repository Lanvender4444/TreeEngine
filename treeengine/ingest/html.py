"""HTML adapter built on the standard-library parser (no extra dependency).

* Drops noise: script/style/noscript/nav/footer/aside/form/svg/template/iframe, elements with
  ``hidden`` / ``aria-hidden=true`` and landmark roles navigation/banner/contentinfo/complementary.
* If the page has ``<main>`` or ``<article>``, only content inside them is kept.
* Emits heading (h1-h6) and block elements as native structure hints (``_elements``).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from ..core.ids import new_id
from ..core.models import Document
from .base import Element, read_text_file

_SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "aside",
    "form",
    "svg",
    "template",
    "iframe",
    "button",
    "select",
    "textarea",
    "canvas",
    "object",
    "head",
}
_SKIP_ROLES = {"navigation", "banner", "contentinfo", "complementary", "search"}
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_BLOCK_TAGS = {
    "p": "paragraph",
    "li": "list",
    "dt": "list",
    "dd": "list",
    "pre": "code",
    "blockquote": "quote",
    "figcaption": "paragraph",
    "table": "table",
    "caption": "paragraph",
}
_CONTAINER_TAGS = {
    "div",
    "section",
    "article",
    "main",
    "ul",
    "ol",
    "dl",
    "body",
    "header",
    "figure",
    "details",
    "summary",
    "tr",
    "tbody",
    "thead",
    "tfoot",
    "html",
}
_WS = re.compile(r"\s+")


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.skip_depth = 0  # >0 while inside a skipped subtree
        self.content_depth = 0  # >0 while inside main/article
        self.raw: list[tuple[str, int, str, bool]] = []  # (kind/btype, level, text, in_content)
        self.buf: list[str] = []
        self.mode: tuple[str, int] | None = None  # current heading/block capture
        self.pre_depth = 0
        self.table_rows: list[list[str]] | None = None
        self.cell: list[str] | None = None
        self.title = ""
        self._in_title = False

    # -- helpers
    def _flush(self, btype: str = "paragraph", level: int = 0) -> None:
        text = "".join(self.buf)
        self.buf = []
        text = text.strip("\n") if btype == "code" else _WS.sub(" ", text).strip()
        if text:
            self.raw.append((btype, level, text, self.content_depth > 0))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "title" and not self.stack_contains("body"):
            self._in_title = True
        if tag in _VOID_TAGS:
            if tag == "br" and not self.skip_depth:
                self.buf.append("\n")
            return
        self.stack.append(tag)
        skip = (
            tag in _SKIP_TAGS
            or "hidden" in a
            or (a.get("aria-hidden") or "").lower() == "true"
            or (a.get("role") or "").lower() in _SKIP_ROLES
        )
        if self.skip_depth or skip:
            self.skip_depth += 1
            return
        if tag in ("main", "article"):
            self._flush()
            self.content_depth += 1
        if tag in _HEADINGS:
            self._flush()
            self.mode = ("heading", _HEADINGS[tag])
        elif tag == "table":
            self._flush()
            self.table_rows = []
        elif tag == "tr" and self.table_rows is not None:
            self.table_rows.append([])
        elif tag in ("td", "th") and self.table_rows is not None:
            self.cell = []
        elif tag in _BLOCK_TAGS:
            self._flush()
            if tag == "pre":
                self.pre_depth += 1
        elif tag in _CONTAINER_TAGS:
            self._flush()

    def stack_contains(self, tag: str) -> bool:
        return tag in self.stack

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in _VOID_TAGS or tag not in self.stack:
            return
        # pop up to and including the matching tag (tolerates unclosed children)
        while self.stack:
            top = self.stack.pop()
            self._close(top)
            if top == tag:
                break

    def _close(self, tag: str) -> None:
        if self.skip_depth:
            self.skip_depth -= 1
            return
        if tag in _HEADINGS and self.mode and self.mode[0] == "heading":
            self._flush("heading", self.mode[1])
            self.mode = None
        elif tag in ("td", "th") and self.cell is not None and self.table_rows is not None:
            if not self.table_rows:
                self.table_rows.append([])
            self.table_rows[-1].append(_WS.sub(" ", "".join(self.cell)).strip())
            self.cell = None
        elif tag == "table" and self.table_rows is not None:
            rows = [" | ".join(c for c in r) for r in self.table_rows if any(r)]
            self.table_rows = None
            if rows:
                self.raw.append(("table", 0, "\n".join(rows), self.content_depth > 0))
        elif tag in _BLOCK_TAGS:
            self._flush(_BLOCK_TAGS[tag])
            if tag == "pre":
                self.pre_depth -= 1
        elif tag in _CONTAINER_TAGS:
            self._flush()
        if tag in ("main", "article"):
            self.content_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self.skip_depth:
            return
        if self.cell is not None:
            self.cell.append(data)
        elif self.table_rows is None:
            self.buf.append(data)

    def close(self) -> None:
        super().close()
        while self.stack:
            self._close(self.stack.pop())
        self._flush()


class HTMLAdapter:
    source_type = "html"

    def load(self, source: str) -> Document:
        html = read_text_file(source)
        return self.from_html(
            html, uri=str(Path(source).resolve()), fallback_title=Path(source).stem
        )

    def from_html(
        self, html: str, uri: str | None = None, fallback_title: str = "Untitled"
    ) -> Document:
        p = _Collector()
        p.feed(html)
        p.close()
        raw = p.raw
        if any(in_content for *_, in_content in raw):
            raw = [r for r in raw if r[3]]

        parts: list[str] = []
        elements: list[Element] = []
        pos = 0
        for btype, level, text, _ in raw:
            if parts:
                pos += 2
            start = pos
            parts.append(text)
            pos += len(text)
            if btype == "heading":
                elements.append(Element("heading", text, start, pos, level=level))
            else:
                elements.append(Element("block", text, start, pos, block_type=btype))
        text = "\n\n".join(parts)

        title = _WS.sub(" ", p.title).strip()
        first_h1 = next((e.text for e in elements if e.kind == "heading" and e.level == 1), None)
        title = first_h1 or title or fallback_title
        return Document(
            id=new_id("doc"),
            source_type=self.source_type,
            uri=uri,
            title=title,
            text=text,
            metadata={
                "chars": len(text),
                "html_title": _WS.sub(" ", p.title).strip(),
                "_elements": elements,
            },
        )
