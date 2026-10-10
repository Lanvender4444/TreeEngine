"""Line roles: header / footer / page number / watermark / TOC / caption / body.

* header / footer: in the top / bottom band of the page and recurring (digits ignored) on many
  pages, or a bare page number there;
* watermark: very large text recurring across pages;
* toc: every line of a table-of-contents page (titled "Contents", or dominated by entries that
  end in a page number); TOC lines never become headings but are kept as content;
* caption: "Figure 3", "Table 2.1", "Exhibit 10" ...;
* body: everything else (heading detection runs on body lines only).

Header / footer / page number / watermark lines are not structure; ``pdf_elements`` also drops
them from the block stream in the layout modes.
"""

from __future__ import annotations

import re
from collections import Counter

from .model import PDFPage

EDGE = 0.08  # top / bottom share of the page height that may hold running lines
_PAGE_NO = re.compile(
    r"^(page\s*)?[-–—(]?\s*(\d{1,4}|[ivxlc]{1,7})\s*[-–—)]?(\s*(of|/)\s*\d{1,4})?$", re.I
)
# "Risk Factors ........ 12", "Risk Factors 12", "Preface ..... iv" (roman only after leaders)
_TOC_ENTRY = re.compile(
    r"[A-Za-z一-鿿].*?(?:(?:\.{2,}|…+|·{2,})\s*(\d{1,4}|[ivxlc]{1,6})|\s(\d{1,4}))$", re.I
)
_TOC_TITLE = re.compile(r"^(table of contents|contents|index|目\s*录)$", re.I)
_CAPTION = re.compile(r"^(fig(ure)?\.?|table|exhibit|chart|图|表)\s*[\dIVX]+(\.\d+)*\b", re.I)


def _key(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", "", text.lower()))


def classify(pages: list[PDFPage]) -> None:
    n = len(pages)
    edge_keys: Counter[str] = Counter()
    for p in pages:
        seen = set()
        for ln in p.lines:
            if ln.top < EDGE * p.height or ln.bottom > (1 - EDGE) * p.height:
                k = _key(ln.text)
                if k and k not in seen:
                    seen.add(k)
                    edge_keys[k] += 1
    # chapter-specific running heads repeat only within their chapter: 3 pages are enough
    running = {k for k, c in edge_keys.items() if n >= 4 and c >= 3}
    sizes = sorted(ln.size for p in pages for ln in p.lines if ln.text.strip())
    median = sizes[len(sizes) // 2] if sizes else 10.0
    big: Counter[str] = Counter()
    for p in pages:
        for k in {_key(ln.text) for ln in p.lines if ln.size >= 2.5 * median}:
            big[k] += 1
    watermarks = {k for k, c in big.items() if n >= 3 and c >= max(3, int(0.3 * n))}

    for p in pages:
        entries = 0
        numbers: list[int] = []
        titled = False
        body_lines = 0
        for ln in p.lines:
            t = ln.text.strip()
            if not t:
                continue
            edge = ln.top < EDGE * p.height or ln.bottom > (1 - EDGE) * p.height
            k = _key(t)
            if k in watermarks:
                ln.role = "watermark"
            elif edge and _PAGE_NO.match(t):
                ln.role = "page_number"
            elif edge and k in running:
                ln.role = "header" if ln.top < 0.5 * p.height else "footer"
            elif _CAPTION.match(t):
                ln.role = "caption"
            else:
                ln.role = "body"
            if ln.role in ("body", "caption"):
                body_lines += 1
                if _TOC_TITLE.match(t) and ln.top < 0.4 * p.height:
                    titled = True
                else:
                    m = _TOC_ENTRY.search(t)
                    if m and len(t) < 160:
                        num = m.group(1) or m.group(2)
                        if num.isdigit() and int(num) <= n + 10:
                            entries += 1
                            numbers.append(int(num))
                        elif not num.isdigit():
                            entries += 1
        # a TOC lists page numbers that go up; a table of figures ("2018 2017 2016") does not
        rising = sum(1 for a, b in zip(numbers, numbers[1:], strict=False) if b >= a)
        monotonic = rising >= 0.8 * (len(numbers) - 1) if len(numbers) > 1 else True
        # TOCs laid out as a table: titles on the left, a column of bare page numbers right
        bare = [
            int(ln.text)
            for ln in p.lines
            if ln.role == "body" and ln.text.strip().isdigit() and ln.bbox[0] > 0.55 * p.width
        ]
        bare_ok = [x for x in bare if x <= n + 10]
        bare_rising = sum(1 for a, b in zip(bare_ok, bare_ok[1:], strict=False) if b >= a)
        if len(bare_ok) >= 6 and bare_rising >= 0.8 * (len(bare_ok) - 1):
            entries = max(entries, len(bare_ok))
            titled = True
        p.is_toc = body_lines > 0 and (
            (titled and entries >= 3)
            or (entries >= 6 and entries >= 0.5 * body_lines and monotonic)
        )
        if p.is_toc:
            for ln in p.lines:
                if ln.role == "body":
                    ln.role = "toc"


__all__ = ["EDGE", "classify"]
