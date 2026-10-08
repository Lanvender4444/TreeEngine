"""Markdown -> Elements. Native structure: ATX (#..######) and setext (===/---) headings.

Headings inside fenced code blocks are ignored. YAML front matter is skipped.
"""

from __future__ import annotations

import re

from ..ingest.base import Element

_ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_HR = re.compile(r"^ {0,3}([-*_])([ \t]*\1){2,}[ \t]*$")
_LIST = re.compile(r"^ {0,3}([-*+]|\d{1,9}[.)])[ \t]+")
_TABLE = re.compile(r"^ {0,3}\|")
_QUOTE = re.compile(r"^ {0,3}>")
_FRONT_TITLE = re.compile(r"^title:\s*[\"']?(.*?)[\"']?\s*$", re.M)
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_EMPH = re.compile(r"(\*\*|__|\*|_|`)(.+?)\1")


def clean_inline(text: str) -> str:
    text = _LINK.sub(r"\1", text)
    text = _EMPH.sub(r"\2", text)
    return text.strip()


def _front_matter_end(text: str) -> int:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            nl = text.find("\n", end + 4)
            return len(text) if nl == -1 else nl + 1
    return 0


def guess_title(text: str) -> str | None:
    fm = _front_matter_end(text)
    if fm:
        m = _FRONT_TITLE.search(text[:fm])
        if m and m.group(1).strip():
            return m.group(1).strip()
    for el in parse_markdown(text):
        if el.kind == "heading" and el.level == 1:
            return el.text
    for el in parse_markdown(text):
        if el.kind == "heading":
            return el.text
    return None


def _classify(line: str) -> str:
    if _LIST.match(line):
        return "list"
    if _TABLE.match(line):
        return "table"
    if _QUOTE.match(line):
        return "quote"
    return "paragraph"


def parse_markdown(text: str) -> list[Element]:
    lines = text.split("\n")
    offsets = []
    pos = 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1

    elements: list[Element] = []
    para: list[int] = []  # indexes of lines in the current block
    para_type = "paragraph"

    def flush() -> None:
        nonlocal para, para_type
        if para:
            start = offsets[para[0]]
            end = offsets[para[-1]] + len(lines[para[-1]])
            content = text[start:end].strip()
            if content:
                lead = len(text[start:end]) - len(text[start:end].lstrip())
                elements.append(
                    Element(
                        "block",
                        content,
                        start + lead,
                        start + lead + len(content),
                        block_type=para_type,
                    )
                )
        para = []
        para_type = "paragraph"

    i = 0
    fm_end = _front_matter_end(text)
    while i < len(lines) and offsets[i] < fm_end:
        i += 1

    while i < len(lines):
        line = lines[i]
        fence = _FENCE.match(line)
        if fence:
            flush()
            marker = fence.group(1)
            j = i + 1
            while j < len(lines) and not lines[j].lstrip().startswith(marker):
                j += 1
            end_line = min(j, len(lines) - 1)
            start = offsets[i]
            end = offsets[end_line] + len(lines[end_line])
            elements.append(
                Element("block", text[start:end].strip(), start, end, block_type="code")
            )
            i = j + 1
            continue

        atx = _ATX.match(line)
        if atx:
            flush()
            title = clean_inline(atx.group(2) or "")
            if title:
                elements.append(
                    Element(
                        "heading",
                        title,
                        offsets[i],
                        offsets[i] + len(line),
                        level=len(atx.group(1)),
                    )
                )
            i += 1
            continue

        # setext heading: a single paragraph line followed by === or ---
        if len(para) == 1 and para_type == "paragraph" and _SETEXT.match(line):
            level = 1 if line.strip().startswith("=") else 2
            h = para[0]
            title = clean_inline(lines[h])
            para = []
            elements.append(
                Element("heading", title, offsets[h], offsets[i] + len(line), level=level)
            )
            i += 1
            continue

        if not line.strip():
            flush()
            i += 1
            continue
        if _HR.match(line):
            flush()
            i += 1
            continue

        kind = _classify(line)
        if (
            para
            and kind != para_type
            and not (para_type == "list" and line.startswith((" ", "\t")))
        ):
            # a new kind of block starts (e.g. paragraph followed directly by a list)
            if kind != "paragraph" or para_type in ("table", "quote"):
                flush()
        if not para:
            para_type = kind
        para.append(i)
        i += 1
    flush()
    return elements


def plain_paragraphs(text: str) -> list[Element]:
    """Blank-line separated paragraphs, no headings (used for plain text sources)."""
    out = []
    for m in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\s*\Z)", text, re.S):
        out.append(Element("block", m.group(0), m.start(), m.end()))
    return out
