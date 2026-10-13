"""False-positive audit of detected PDF headings against the FinanceBench reference structure.

    python -m benchmarks.heading_audit [--docs 12] [--modes layout,hybrid] [--out FILE]

For a fixed sample of 10-Ks (the first N by sha256 of the file name: independent of any
benchmark result), every outline entry the layout / hybrid structure produces is matched to
the reference outline (same page, similar title: the rule ``run_structure`` scores with). An
unmatched entry is a false positive *against the reference*; each one is put in one class by
the first rule that fires (the line's own layout features, never an answer or a question):

    TOC_ENTRY        table-of-contents line (leaders / trailing page number, TOC page)
    RUNNING_HEADER   text recurring on 3+ pages near the top   (RUNNING_FOOTER: near the bottom)
    PAGE_ARTIFACT    text recurring on 3+ pages elsewhere, "Table of Contents" links, page marks
    CAPTION          "Figure 3", "Table 2", "Exhibit 10", "Source:" ...
    TABLE_ROW        numeric-heavy, or shares its baseline with other text, or sits in a block of
                     numeric lines
    FINANCIAL_LABEL  a statement line label ("Total ...", "Net income", "Cash flows from ...")
    NUMBERED_BODY    numbered clause / paragraph ("3.5. Holding Period. Shares acquired ...",
                     "(a)", "(i)", "1)") - a number but body text after it
    BOLD_BODY        an emphasised sentence (long, or sentence punctuation)
    HEADING_FRAGMENT the 2nd+ line of a heading split over several lines / letter-spaced text
    SHORT_EMPHASIS   one or two words that do not read as a section ("DATE", "Notes:")
    EXHIBIT_SECTION  a heading inside an exhibit appended after the 10-K body (a real heading of
                     another document; the reference stops at the 10-K's last Item)
    UNREFERENCED     a title-like line followed by running text: most likely a real section
                     heading the reference does not list (the reference holds Items, MD&A
                     subsections and Notes only); counted as FP by the metric, but not an error
    OTHER

The rules are an audit aid; ``--sample`` prints a random sample per class for checking by eye.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from treeengine.pdf import analyze, analyze_structure
from treeengine.pdf.headings import numbering
from treeengine.pdf.model import PDFLine, PDFPage

from .run_structure import reference, similar

HERE = Path(__file__).parent
FILES = HERE / "datasets" / "financebench" / "files"
STRUCTURES = HERE / "datasets" / "financebench" / "structures"
CLASSES = [
    "TOC_ENTRY",
    "RUNNING_HEADER",
    "RUNNING_FOOTER",
    "PAGE_ARTIFACT",
    "CAPTION",
    "TABLE_ROW",
    "FINANCIAL_LABEL",
    "NUMBERED_BODY",
    "BOLD_BODY",
    "HEADING_FRAGMENT",
    "SHORT_EMPHASIS",
    "EXHIBIT_SECTION",
    "UNREFERENCED",
    "OTHER",
]
REAL = {"EXHIBIT_SECTION", "UNREFERENCED"}  # FP against the reference, not structural errors

_CAPTION = re.compile(r"^(fig(ure)?\.?|table|exhibit|chart|source|note:|notes:)\b", re.I)
_TOC_TAIL = re.compile(r"(\.{2,}|…|·{2,})\s*\d{0,4}$|\s\d{1,3}$")
_FIN = re.compile(
    r"^(total\b|net (income|loss|sales|revenue|earnings|cash|change|increase|decrease)\b|"
    r"cash flows? (from|used|provided)|cash and cash equivalents|(gross|operating) (profit|"
    r"income|margin)|income (before|from)|earnings (before|per)|(basic|diluted)\b|assets\b|"
    r"liabilities\b|(current|non-?current|long-term|short-term) (assets|liabilities|debt)|"
    r"(stockholders|shareholders)['’]? equity|revenues?\b|cost of (sales|revenue|goods)|"
    r"(selling|general)\b|(provision|benefit) for income taxes|depreciation|amortization|"
    r"accounts (receivable|payable)|inventor(y|ies)\b|goodwill\b|\(?in (millions|thousands)|"
    r"(fiscal )?years? ended|as of (december|january|june|september|march)|"
    r"three months|six months|nine months|twelve months)",
    re.I,
)
_CLAUSE = re.compile(r"^(\(?[a-z]{1,3}\)|\(?[ivx]{1,5}\)|\d{1,2}\))\s", re.I)
_NUMBERED_SENTENCE = re.compile(r"^\d{1,2}(\.\d{1,2})*\.?\s+[^.]{2,80}\.\s+\S")


@dataclass
class FP:
    doc: str
    mode: str
    level: int
    title: str
    page: int | None
    cls: str
    why: str


def sample_docs(n: int) -> list[str]:
    names = sorted(
        p.stem + ".pdf" for p in STRUCTURES.glob("*.json") if (FILES / (p.stem + ".pdf")).exists()
    )
    names = [d for d in names if (reference(d) or {}).get("outline")]
    return sorted(names, key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:n]


def match(outline: list[Any], ref: list[list[Any]]) -> tuple[list[Any], int]:
    used: set[int] = set()
    fps = []
    for e in outline:
        t, pg = e[1], e[2]
        hit = None
        for i, (_rl, rt, rp, *ra) in enumerate(ref):
            if i in used:
                continue
            if pg == rp and (similar(t, rt) or (ra and similar(t, ra[0]))):
                hit = i
                break
        if hit is None:
            fps.append(e)
        else:
            used.add(hit)
    return fps, len(used)


def _key(s: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\W+", "", s.lower()))


class DocIndex:
    """Line lookup and recurrence counts for one analysed PDF."""

    def __init__(self, pages: list[PDFPage], body_size: float) -> None:
        self.pages = {p.number: p for p in pages}
        self.body_size = body_size
        occ: dict[str, set[int]] = defaultdict(set)
        for p in pages:
            for ln in p.lines:
                k = _key(ln.text)
                if k:
                    occ[k].add(p.number)
        self.occ = occ

    def line(self, page: int | None, top: float | None, text: str) -> tuple[PDFLine | None, int]:
        if page is None or page not in self.pages:
            return None, -1
        p = self.pages[page]
        best, best_i, best_d = None, -1, 1e9
        want = _key(text)[:20]
        for i, ln in enumerate(p.lines):
            k = _key(ln.text)
            if top is not None:
                d = abs(ln.bbox[1] - top)
            else:
                d = 0.0 if want and k.startswith(want) else 1e9
            if want and not (k.startswith(want[:8]) or want.startswith(k[:8])):
                d += 50
            if d < best_d:
                best, best_i, best_d = ln, i, d
        return (best, best_i) if best_d < 60 else (None, -1)


def _numeric_share(t: str) -> float:
    s = t.replace(" ", "")
    return sum(ch.isdigit() or ch in "$%(),.-—" for ch in s) / max(1, len(s))


_COLUMN_HEAD = re.compile(
    r"^(\(?in (millions|thousands|billions)|((fiscal )?years?|three|six|nine|twelve|quarters?|"
    r"weeks?|months?)( months)? ended\b|(as of |at )?(january|february|march|april|may|june|"
    r"july|august|september|october|november|december)\s+\d{1,2},?(\s+\d{4})?$|"
    r"(fiscal\s+)?(19|20)\d\d(\s+(fiscal\s+)?(19|20)\d\d)+$)",
    re.I,
)


def _style(ln: PDFLine) -> tuple[float, bool]:
    return (round(ln.size * 2) / 2, ln.bold)


def classify_fp(
    e: Any, idx: DocIndex, last_ref_page: int, prev_entry: Any | None
) -> tuple[str, str]:
    _, title, page, anchor, top = e
    t = title.strip()
    ln, i = idx.line(page, top, anchor or title)
    p = idx.pages.get(page) if page is not None else None
    words = len(t.split())
    k = _key(t)
    if p is not None and p.is_toc:
        return "TOC_ENTRY", "on a TOC page"
    if k in ("tableofcontents", "index", "contents") or re.match(r"^page\s*\d", t, re.I):
        return "PAGE_ARTIFACT", "navigation link / page mark"
    # running lines: the same text in the same band (top / bottom 10%) on 3+ pages
    if ln is not None and p is not None:
        band = (
            "top"
            if ln.bbox[1] < 0.1 * p.height
            else "bottom"
            if ln.bbox[3] > 0.9 * p.height
            else ""
        )
        if band:
            same = 0
            for q in idx.pages.values():
                for o in q.lines:
                    if _key(o.text) == k and (
                        (band == "top" and o.bbox[1] < 0.1 * q.height)
                        or (band == "bottom" and o.bbox[3] > 0.9 * q.height)
                    ):
                        same += 1
                        break
            if same >= 3:
                cls = "RUNNING_HEADER" if band == "top" else "RUNNING_FOOTER"
                return cls, f"same text in the {band} band of {same} pages"
    if _CAPTION.match(t):
        return "CAPTION", "caption pattern"
    num, _ = numbering(t)
    rest = (t[len(num) :] if num else t).strip(" .:-—–")
    if (len(rest) >= 3 and _numeric_share(rest) >= 0.3) or re.search(r"\$\s?\d", rest):
        return "TABLE_ROW", "numeric / amounts"
    if _COLUMN_HEAD.match(t) or _COLUMN_HEAD.match(re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", t)):
        return "TABLE_ROW", "table column header (period / units)"
    numeric_ctx = 0
    if ln is not None and p is not None:
        mid = (ln.bbox[1] + ln.bbox[3]) / 2
        same_row = [
            o
            for o in p.lines
            if o is not ln and o.column == ln.column and abs((o.bbox[1] + o.bbox[3]) / 2 - mid) < 3
        ]
        if same_row:
            return "TABLE_ROW", "shares its baseline with other text"
        near = p.lines[max(0, i - 3) : i + 4]
        numeric_ctx = sum(1 for o in near if o is not ln and _numeric_share(o.text) >= 0.4)
    if _FIN.match(t) and (numeric_ctx >= 2 or (t.upper() == t and t.lower().startswith("total"))):
        return "FINANCIAL_LABEL", "statement label among numeric lines"
    if numeric_ctx >= 3 and not num:
        return "TABLE_ROW", f"{numeric_ctx} numeric lines around it"
    if _CLAUSE.match(t) or _NUMBERED_SENTENCE.match(t):
        return "NUMBERED_BODY", "numbered clause with body text (run-in)"
    if words > 14 or len(t) > 110 or (words > 8 and t.rstrip().endswith((".", ";", ","))):
        return "BOLD_BODY", f"sentence-like ({words} words)"
    # the 2nd line of one title: same style, right under the previous entry's line
    if prev_entry is not None and prev_entry[2] == page and ln is not None and p is not None:
        pl, _ = idx.line(prev_entry[2], prev_entry[4], prev_entry[3] or prev_entry[1])
        if (
            pl is not None
            and _style(pl) == _style(ln)
            and 0 <= ln.bbox[1] - pl.bbox[3] < 0.6 * (ln.bbox[3] - ln.bbox[1])
        ):
            return "HEADING_FRAGMENT", "continues the previous heading line"
    if words <= 2 and (t.endswith(":") or len(t) <= 5):
        return "SHORT_EMPHASIS", "one or two words"
    if page is not None and last_ref_page and page > last_ref_page + 1:
        return "EXHIBIT_SECTION", "after the 10-K body (an exhibit's own heading)"
    if " " not in t and len(t) > 18 or re.search(r"\b[A-Z](\s[A-Z]){2,}\b", t):
        return "UNREFERENCED", "real heading, glued / letter-spaced text"
    if words <= 14:
        return "UNREFERENCED", "title-like line"
    return "OTHER", ""


def audit(docs: list[str], modes: list[str]) -> list[FP]:
    out: list[FP] = []
    for name in docs:
        path = str(FILES / name)
        ref = (reference(name) or {}).get("outline") or []
        layout = analyze(path)
        idx = DocIndex(layout.pages, layout.body_size)
        last_ref_page = max((int(r[2]) for r in ref if r[2] is not None), default=0)
        for mode in modes:
            res = analyze_structure(path, mode)
            fps, _ = match(res.outline, ref)
            fp_ids = {id(e) for e in fps}
            prev = None
            for e in res.outline:
                if id(e) in fp_ids:
                    if e[4] is None:
                        cls, why = "OTHER", "bookmark entry"
                    else:
                        cls, why = classify_fp(e, idx, last_ref_page, prev)
                    out.append(FP(name, mode, e[0], e[1], e[2], cls, why))
                prev = e
        print(f"audited {name}", file=sys.stderr)
    return out


def summary(
    docs: list[str], modes: list[str], fps: list[FP], sample: int = 3, seed: int = 0
) -> tuple[str, dict[str, Counter[str]]]:
    rng = random.Random(seed)
    lines = []
    dist: dict[str, Counter[str]] = {}
    for mode in modes:
        rows = [f for f in fps if f.mode == mode]
        c = Counter(f.cls for f in rows)
        dist[mode] = c
        n = len(rows) or 1
        real = sum(c[k] for k in REAL)
        lines += [
            f"### {mode}: {len(rows)} entries not in the reference "
            f"({real} = {100 * real / n:.0f}% look like real headings the reference omits)",
            "",
            "| class | count | share | examples (page) | trigger |",
            "|---|---|---|---|---|",
        ]
        for cls in sorted(c, key=lambda k: -c[k]):
            ex = [f for f in rows if f.cls == cls]
            pick = rng.sample(ex, min(sample, len(ex)))
            show = "; ".join(f"“{f.title[:48]}” (p{f.page})" for f in pick).replace("|", "/")
            why = Counter(f.why for f in ex).most_common(1)[0][0]
            lines.append(f"| {cls} | {c[cls]} | {100 * c[cls] / n:.1f}% | {show} | {why} |")
        lines.append("")
    return "\n".join(lines), dist


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.heading_audit")
    ap.add_argument("--docs", type=int, default=12)
    ap.add_argument("--modes", default="layout,hybrid")
    ap.add_argument("--sample", type=int, default=0, help="print N random FPs per class")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    docs = sample_docs(args.docs)
    modes = args.modes.split(",")
    fps = audit(docs, modes)
    text, _ = summary(docs, modes, fps)
    head = f"Sample: {len(docs)} 10-Ks (first by sha256 of the file name): {', '.join(docs)}\n\n"
    if args.out:
        Path(args.out).write_text(head + text, encoding="utf-8")
    print(head + text)
    if args.sample:
        rng = random.Random(1)
        for cls in CLASSES:
            ex = [f for f in fps if f.cls == cls]
            for f in rng.sample(ex, min(args.sample, len(ex))):
                print(f"[{cls}] {f.mode} {f.doc} p{f.page} L{f.level}: {f.title[:110]!r} ({f.why})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
