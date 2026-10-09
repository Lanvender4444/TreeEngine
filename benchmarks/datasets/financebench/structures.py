"""Reference ("oracle") structures for the FinanceBench 10-K filings.

A 10-K follows a structure fixed by SEC Form 10-K, so its true outline can be recovered from the
page text with a small schema instead of a general heading detector:

    Item 1 ... Item 16                          (level 1; canonical Form 10-K item names)
      Item 7  -> MD&A subsections                (level 2: Overview, Results of Operations, ...)
      Item 8  -> auditor's report, the primary   (level 2)
                 statements, Notes
                   Notes -> Note 1 ... Note N    (level 3)

Rules (written against the documents only - FinanceBench questions and evidence pages are never
read here, so the structure cannot leak the answers):

* table-of-contents pages (5+ item headings, or 3+ statement titles with page numbers) are skipped;
* each item / statement / note number is taken at its first standalone occurrence after the
  previous heading, in document order (references inside sentences are not standalone lines);
* the printed line is kept as the anchor, the canonical name as the node title.

Output: ``structures/<document>.json`` = {"source", "reviewed", "outline": [[level, title,
page, anchor], ...]}. ``reviewed`` is set by a person after checking the file; the benchmark
reports reviewed and unreviewed documents separately.

    python -m benchmarks.datasets.financebench.structures      # (re)generate unreviewed files
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "structures"

ITEMS = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer "
    "Purchases of Equity Securities",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related "
    "Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
ITEM_ORDER = list(ITEMS)
ITEM_RE = re.compile(r"^item\s*(\d{1,2}[a-c]?)\s*[.:\-—–]?\s*(.*)$", re.I)

MDNA = [
    ("Overview", r"(executive\s+)?overview"),
    ("Results of Operations", r"(consolidated\s+)?results\s+of\s+operations"),
    (
        "Segment Results",
        r"(segment\s+results|results\s+by\s+(business\s+)?segment|segment\s+information|business\s+segment\s+results)",
    ),
    ("Liquidity and Capital Resources", r"liquidity\s+and\s+capital\s+resources"),
    ("Cash Flows", r"cash\s+flows?"),
    ("Contractual Obligations", r"contractual\s+obligations"),
    ("Off-Balance Sheet Arrangements", r"off[-\s]balance\s+sheet\s+arrangements"),
    (
        "Critical Accounting Estimates",
        r"critical\s+accounting\s+(policies|estimates)(\s+and\s+estimates)?",
    ),
    ("Non-GAAP Financial Measures", r"non[-\s]?gaap\s+(financial\s+)?measures?"),
    (
        "Recent Accounting Pronouncements",
        r"(recent(ly)?\s+(issued\s+)?accounting\s+(pronouncements|standards))",
    ),
]
STATEMENTS = [
    (
        "Report of Independent Registered Public Accounting Firm",
        r"report\s+of\s+independent\s+registered\s+public\s+accounting\s+firm",
    ),
    (
        "Consolidated Statements of Income",
        r"consolidated\s+statements?\s+of\s+(incom|operation|earning)(?!\w*\s+and\s+comprehensive)",
    ),
    (
        "Consolidated Statements of Comprehensive Income",
        r"consolidated\s+statements?\s+of\s+(comprehensive\s+(income|loss|earnings)|(income|operations|earnings)\s+and\s+comprehensive)",
    ),
    (
        "Consolidated Balance Sheets",
        r"consolidated\s+(balance\s+shee|statements?\s+of\s+financial\s+positio)",
    ),
    ("Consolidated Statements of Cash Flows", r"consolidated\s+statements?\s+of\s+cash\s+flo"),
    (
        "Consolidated Statements of Equity",
        r"consolidated\s+statements?\s+of\s+(changes\s+in\s+)?(shareholders'?|stockholders'?|share\s*owners'?|)\s*equit",
    ),
    (
        "Notes to Consolidated Financial Statements",
        r"notes\s+to\s+(the\s+)?consolidated\s+financial\s+statement",
    ),
]
PAGE_REF = re.compile(r"^[ \t]*(\.{2,}[ \t]*)?(page[ \t]*)?\d{1,3}[ \t]*(\n|$)", re.I)
BULLET = re.compile(r"[·•▪●◦]\s*$")
APOS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def _flex(words: str) -> str:
    """Regex for a phrase that tolerates missing / extra whitespace (pypdf glues words)."""
    parts = re.findall(r"[A-Za-z0-9']+", words)
    return r"\W*".join(re.escape(p) for p in parts)


def _split_flex(words: str) -> str:
    """Like ``_flex`` but also tolerates line breaks inside a word ("Busines\\n \\ns.")."""
    parts = re.findall(r"[A-Za-z0-9']+", words)
    return r"\W*".join(r"\s*".join(re.escape(c) for c in p) for p in parts)


def _item_rx(key: str) -> re.Pattern[str]:
    # the first title word only: filings vary the rest ("Exhibits, Financial Statement
    # Schedules", "Principal Accounting Fees")
    title_words = ITEMS[key].replace("'", " ").split()[0]
    return re.compile(
        r"item\s*" + re.escape(key) + r"(?![0-9a-z])[\s.:\-—–]*" + _split_flex(title_words),
        re.I,
    )


SPLIT = re.compile(r"(?<=[A-Za-z]{2})\n[ \t]*\n(?=([a-z]{1,2})(?![A-Za-z]))")
SHORT_WORDS = {"a", "an", "as", "at", "be", "by", "if", "in", "is", "it", "of", "on", "or", "to"}


def _repair(text: str) -> tuple[str, list[int]]:
    """Undo pypdf's split of a word's last letters onto a new line ("Incom\\n \\ne" ->
    "Income"). Returns the repaired text and, for every repaired character, its raw offset."""
    out: list[str] = []
    idx: list[int] = []
    pos = 0
    for m in SPLIT.finditer(text):
        if m.group(1) in SHORT_WORDS:
            continue
        out.append(text[pos : m.start()])
        idx.extend(range(pos, m.start()))
        pos = m.end()
    out.append(text[pos:])
    idx.extend(range(pos, len(text)))
    return "".join(out), idx


class _Doc:
    def __init__(self, pages: list[tuple[int, list[str]]]) -> None:
        self.text = ""  # repaired text: every search runs on this
        self.raw = ""  # page text as TreeEngine sees it: anchors are cut from this
        self.raw_at: list[int] = []
        self.starts: list[tuple[int, int]] = []  # (offset, page)
        for no, lines in pages:
            page = "\n".join(lines).translate(APOS) + "\n"
            fixed, idx = _repair(page)
            self.starts.append((len(self.text), no))
            self.raw_at.extend(len(self.raw) + i for i in idx)
            self.text += fixed
            self.raw += page

    def raw_slice(self, start: int, end: int) -> str:
        if start >= end:
            return ""
        return self.raw[self.raw_at[start] : self.raw_at[end - 1] + 1]

    def page(self, off: int) -> int:
        page = self.starts[0][1]
        for s, no in self.starts:
            if s > off:
                break
            page = no
        return page

    def page_span(self, no: int) -> tuple[int, int]:
        for i, (s, n) in enumerate(self.starts):
            if n == no:
                e = self.starts[i + 1][0] if i + 1 < len(self.starts) else len(self.text)
                return s, e
        return 0, 0


def _toc_pages(d: _Doc) -> set[int]:
    toc = set()
    for _, no in d.starts:
        s, e = d.page_span(no)
        t = d.text[s:e]
        items = sum(1 for k in ITEMS if _item_rx(k).search(t))
        stmts = 0
        for _, rx in STATEMENTS:
            for m in re.finditer(rx, t, re.I):
                if PAGE_REF.match(t[m.end() : m.end() + 12]):
                    stmts += 1
                    break
        if (
            items >= 5
            or stmts >= 3
            or re.search(r"index\s+to\s+(the\s+)?consolidated\s+financial", t, re.I)
        ):
            toc.add(no)
    return toc


def _first(rx: re.Pattern[str], d: _Doc, lo: int, hi: int, toc: set[int]) -> re.Match[str] | None:
    for m in rx.finditer(d.text, lo, hi):
        if d.page(m.start()) in toc:
            continue
        if PAGE_REF.match(d.text[m.end() : m.end() + 12]):
            continue  # an index line ("Consolidated Balance Sheets ..... 45")
        before = d.text[max(0, m.start() - 2) : m.start()]
        if before[-1:] in ('"', "'", "(") or d.text[m.end() : m.end() + 2].startswith(('"', ',"')):
            continue  # a quoted cross-reference: 'see "Item 1A. Risk Factors"'

        if re.search(r"[a-z,]\s$", before) and not before.endswith("\n"):
            continue  # inside a sentence: "... as shown in the Consolidated Balance Sheets"
        if BULLET.search(d.text[max(0, m.start() - 8) : m.start()]):
            continue  # a bullet list of contents ("· Overview · Results of Operations ...")
        return m
    return None


def _top_of_page(
    rx: re.Pattern[str], d: _Doc, lo: int, hi: int, toc: set[int], top: int = 250
) -> re.Match[str] | None:
    """First occurrence near the top of a page (a statement's own page starts with its title;
    an index page or a cross-reference does not); else the first occurrence at all."""
    first = None
    pos = lo
    while True:
        m = _first(rx, d, pos, hi, toc)
        if m is None:
            return first
        first = first or m
        if m.start() - d.page_span(d.page(m.start()))[0] <= top:
            return m
        pos = m.end()


def _anchor(d: _Doc, m: re.Match[str]) -> str:
    """The heading as it appears in TreeEngine's page text (split words included), so the
    builder can place it."""
    return re.sub(r"\s+", " ", d.raw_slice(m.start(), m.end())).strip()


def _notes(d: _Doc, lo: int, hi: int, toc: set[int], style: str) -> list[tuple[int, list]]:
    out: list[tuple[int, list]] = []
    cur = lo
    for n in range(1, 41):
        prefix = r"(?i:note)\s*" if style == "note" else r"(?<![\d$.,%(])"
        rx = re.compile(
            prefix + str(n) + r"(?!\d)\s*[.:\-—–]\s*"
            r"([A-Z][A-Z0-9 ,&'\-()/]{3,80}?(?=[A-Z][a-z]|\n|$)"
            r"|[A-Z][A-Za-z0-9 ,&'\-()/:]{2,90})",
        )
        m = None
        for c in rx.finditer(d.text, cur, hi):
            before = d.text[max(0, c.start() - 2) : c.start()]
            if d.page(c.start()) in toc:
                continue
            if re.search(r"[a-z,]\s$", before) and not before.endswith("\n"):
                continue  # a cross-reference inside a sentence: "see Note 2."
            m = c
            break
        if m is None:
            break
        raw = m.group(1)
        # glued extraction: the title runs into the body text ("Shareholders' EquityWalmart")
        raw = re.split(r"(?<=[a-z)])(?=[A-Z][a-z])|\.\s|\n", raw)[0]
        name = re.sub(r"\s+", " ", raw).strip(" .-—–")
        if len(name) > 80:
            name = name[:80].rsplit(" ", 1)[0]
        name = name.title() if name.isupper() else name
        out.append((m.start(), [3, f"Note {n}. {name}", d.page(m.start()), _anchor(d, m)[:80]]))
        cur = m.end()
    return out


def build_outline(pages: list[tuple[int, list[str]]]) -> list[list]:
    """pages: [(page_no, lines)] -> [[level, title, page, anchor], ...] in document order."""
    d = _Doc(pages)
    toc = _toc_pages(d)
    found: list[tuple[int, list]] = []  # (offset, entry)
    cursor = 0
    item_pos: dict[str, int] = {}
    for key in ITEM_ORDER:
        m = _first(_item_rx(key), d, cursor, len(d.text), toc)
        if m is None:
            continue
        item_pos[key] = m.start()
        cursor = m.end()
        found.append(
            (m.start(), [1, f"Item {key}. {ITEMS[key]}", d.page(m.start()), _anchor(d, m)])
        )

    def span(key: str) -> tuple[int, int] | None:
        if key not in item_pos:
            return None
        after = [p for k, p in item_pos.items() if p > item_pos[key]]
        return item_pos[key], min(after) if after else len(d.text)

    mdna = span("7")
    if mdna:
        lo, hi = mdna
        for title, rx in MDNA:
            pat = re.compile(r"(?<![a-z])" + rx + r"(?![a-z])", re.I)
            m = _first(pat, d, lo, hi, toc)
            # a heading is capitalised and ends its line ("Segment information presented
            # herein ..." is a sentence)
            while m is not None and (
                not m.group(0)[0].isupper()
                or not re.match(r"[ \t:.]*\n", d.text[m.end() : m.end() + 6])
            ):
                m = _first(pat, d, m.end(), hi, toc)
            if m is not None:
                found.append((m.start(), [2, title, d.page(m.start()), _anchor(d, m)]))
    fin = span("8")
    if fin:
        lo, hi = fin
        notes_at = None
        for title, rx in STATEMENTS:
            m = _top_of_page(re.compile(rx, re.I), d, lo, hi, toc)
            if m is not None:
                found.append((m.start(), [2, title, d.page(m.start()), _anchor(d, m)]))
                if title.startswith("Notes to"):
                    notes_at = m.end()
        if notes_at is not None:
            # notes are numbered either "Note 7 — Income Taxes" or "7. Income Taxes"; try both
            # and keep the longer consecutive sequence
            runs: list[list[tuple[int, list]]] = [
                _notes(d, notes_at, hi, toc, style) for style in ("note", "num")
            ]
            best = runs[0] if len(runs[0]) >= len(runs[1]) else runs[1]
            found.extend(best)
    found.sort(key=lambda x: x[0])
    return [e for _, e in found]


def pages_of(path: Path) -> list[tuple[int, list[str]]]:
    from treeengine.ingest.base import get_adapter

    doc = get_adapter("pdf").load(str(path))
    text = doc.text
    return [(int(no), text[s:e].split("\n")) for no, s, e in doc.metadata.get("_pages", [])]


def complete(outline: list[list]) -> bool:
    """Only documents whose schema extraction is complete become reference structures:
    at least 12 items, 5 Item 7 / Item 8 sections and 10 notes. (Decided on the structure alone:
    which questions a document has plays no part.)"""
    levels = [o[0] for o in outline]
    return levels.count(1) >= 12 and levels.count(2) >= 5 and levels.count(3) >= 10


def main(argv: list[str] | None = None) -> int:
    from .. import load_suite

    names = argv if argv is not None else sys.argv[1:]
    suite = load_suite("financebench")
    OUT.mkdir(exist_ok=True)
    docs = [d for d in suite.docs if d.kind.lower().startswith("10k")]
    if names:
        docs = [d for d in docs if d.name in names]
    kept = 0
    for d in docs:
        dest = OUT / (d.name[:-4] + ".json")
        if dest.exists() and json.loads(dest.read_text(encoding="utf-8")).get("reviewed"):
            print(f"keep human-reviewed {dest.name}")
            kept += 1
            continue
        outline = build_outline(pages_of(d.path))
        counts = [sum(1 for o in outline if o[0] == lv) for lv in (1, 2, 3)]
        if not complete(outline):
            dest.unlink(missing_ok=True)
            print(f"{d.name}: incomplete {counts} - skipped")
            continue
        kept += 1
        dest.write_text(
            json.dumps(
                {
                    "source": "10-K schema (semi-automatic, see structures.py)",
                    "reviewed": False,
                    "checked": "automatic consistency checks + spot check by Claude; "
                    "not reviewed by a person",
                    "outline": outline,
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"{d.name}: items {counts[0]}, sections {counts[1]}, notes {counts[2]}")
    print(f"{kept} reference structures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
