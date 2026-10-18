"""Build ``finx``: a fresh FinanceBench-style held-out set over 10-K filings no earlier suite used.

    python -m benchmarks.datasets.finx.build            # needs TREEENGINE_LLM_*

Why: the FinanceBench / longdoc / multires2 test splits were each used by V0.13, V0.14 and
V0.15; the next round needs questions nobody has looked at.

* documents: 24 10-Ks from the FinanceBench repository that no question of any suite uses
  (9 companies never seen, 15 other filings of seen companies), pinned by URL + sha256
* page text: pdfplumber, independent of TreeEngine's own parser; evidence = the page
* two question styles, like FinanceBench's metrics-generated / domain-relevant ones:
    metrics   a financial-statement page; the question uses analyst wording / abbreviations
              (PP&E, COGS, capex, operating margin, ...) instead of the line-item text and may
              need a simple calculation from numbers on that page
    domain    an MD&A / notes / business page; a qualitative analyst question
* verification by the LLM: answerable from the page alone (judge vs gold) AND not answerable
  closed-book (no context) - drops trivia the model already knows
* all questions are "test" (held-out); round-robin over documents, at most PER_DOC each

The system under test never takes part.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from benchmarks.datasets.multires.build import _answers, _json

HERE = Path(__file__).parent
SEED = 23
PER_DOC = 5
REPO = "https://raw.githubusercontent.com/patronus-ai/financebench/cc39aeb4afdf33909ee1412188bf89035950c2eb/pdfs/"
DOCS = [
    "APPLE_2022_10K.pdf",
    "BOSTONPROPERTIES_2015_10K.pdf",
    "EBAY_2022_10K.pdf",
    "FEDEX_2023_10K.pdf",
    "INTEL_2022_10K.pdf",
    "MCDONALDS_2022_10K.pdf",
    "ORACLE_2023_10K.pdf",
    "PG_E_2022_10K.pdf",
    "SALESFORCE_2023_10K.pdf",
    "CVSHEALTH_2021_10K.pdf",
    "BESTBUY_2022_10K.pdf",
    "MGMRESORTS_2021_10K.pdf",
    "BLOCK_2022_10K.pdf",
    "ADOBE_2021_10K.pdf",
    "LOCKHEEDMARTIN_2019_10K.pdf",
    "FOOTLOCKER_2023_10K.pdf",
    "PFIZER_2022_10K.pdf",
    "CORNING_2019_10K.pdf",
    "COSTCO_2022_10K.pdf",
    "WALMART_2023_10K.pdf",
    "KRAFTHEINZ_2022_10K.pdf",
    "PEPSICO_2020_10K.pdf",
    "JPMORGAN_2021_10K.pdf",
    "JOHNSON_JOHNSON_2021_10K.pdf",
]
STATEMENT = re.compile(
    r"consolidated\s+(balance\s+sheets?|statements?\s+of\s+(operations|income|earnings|"
    r"cash\s+flows?|comprehensive\s+income|financial\s+position|(stock|share)holders.\s+equity))",
    re.I,
)

GEN_SYSTEM = "You write evaluation questions for financial-document QA. Reply with JSON only."
GEN = {
    "metrics": """Company filing: {doc}

Page {page}:
{text}

Write ONE question a financial analyst would ask about {company} that is answered by numbers on
this page. Phrase it the way FinanceBench does: name the company and the fiscal year, and use
common analyst wording or abbreviations (for example PP&E, COGS, D&A, capex, free cash flow,
operating margin, gross margin, quick ratio, working capital, effective tax rate) instead of
copying the line-item text. It may require a simple calculation using only numbers on this page;
say the unit and rounding expected (e.g. "in USD millions", "round to one decimal place").
Give the exact answer.
If the page has no usable numbers, reply {{"ok": false}}.
Reply as JSON: {{"ok": true, "question": "...", "answer": "...", "basis": "numbers used"}}
""",
    "domain": """Company filing: {doc}

Page {page}:
{text}

Write ONE question a financial analyst would ask about {company} that this page answers: a
business, segment, risk, legal, accounting-policy or strategy fact. Use the analyst's own words,
not a copy of a sentence; name the company and the fiscal year. The answer must be short and
specific (a name, a number, a short list, or a one-sentence fact).
If the page has nothing specific enough, reply {{"ok": false}}.
Reply as JSON: {{"ok": true, "question": "...", "answer": "..."}}
""",
}


def company(name: str) -> str:
    return name.split("_20")[0].replace("_", " ").title()


def page_texts(path: Path) -> list[str]:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return [(p.extract_text() or "") for p in pdf.pages]


def pick_pages(texts: list[str], rng: random.Random) -> list[tuple[str, int]]:
    stmt = [
        i
        for i, t in enumerate(texts)
        if STATEMENT.search(t[:400]) and sum(c.isdigit() for c in t) > 150
    ]
    dom = [
        i
        for i, t in enumerate(texts)
        if len(t) > 1800
        and sum(c.isalpha() for c in t) > 0.6 * len(t)
        and i > 3
        and "table of contents" not in t[:300].lower()
        and i not in stmt
    ]
    out = [("metrics", i) for i in rng.sample(stmt, min(4, len(stmt)))]
    out += [("domain", i) for i in rng.sample(dom, min(4, len(dom)))]
    return out


def closed_book_knows(llm: Any, q: str, gold: str) -> bool:
    return _answers(llm, q, gold, "(no document text available - answer from memory)")


def main() -> int:
    from benchmarks.answer import llm_from_env

    llm = llm_from_env("TREEENGINE_LLM")
    if llm is None:
        raise SystemExit("needs TREEENGINE_LLM_MODEL / _API_KEY / _BASE_URL")
    files = HERE / "files"
    rng = random.Random(SEED)
    texts: dict[str, list[str]] = {}
    cache = HERE / "pages.json"
    if cache.exists():
        texts = json.loads(cache.read_text())
    for n in DOCS:
        if n not in texts:
            texts[n] = page_texts(files / n)
            print(f"{n}: {len(texts[n])} pages", file=sys.stderr)
    cache.write_text(json.dumps(texts))
    cands = [(n, kind, i) for n in DOCS for kind, i in pick_pages(texts[n], rng)]
    print(f"{len(cands)} candidates", file=sys.stderr)

    def work(c: tuple[str, str, int]) -> dict[str, Any] | None:
        n, kind, i = c
        text = texts[n][i][:6000]
        try:
            out = _json(
                llm.complete(
                    GEN[kind].format(
                        doc=n.rsplit(".", 1)[0], page=i + 1, text=text, company=company(n)
                    ),
                    system=GEN_SYSTEM,
                    max_tokens=400,
                )
            )
            if not out.get("ok") or not out.get("question") or not out.get("answer"):
                return None
            q, a = str(out["question"]), str(out["answer"])
            if not _answers(llm, q, a, text):
                return None
            if closed_book_knows(llm, q, a):
                return None
        except Exception as e:  # noqa: BLE001 - a failed candidate is skipped
            print(f"skip ({e})", file=sys.stderr)
            return None
        return {"doc": n, "type": kind, "page": i + 1, "question": q, "answer": a}

    with ThreadPoolExecutor(8) as ex:
        kept = [g for g in ex.map(work, cands) if g is not None]
    print(f"verified {len(kept)} / {len(cands)}", file=sys.stderr)
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for g in kept:
        by_doc.setdefault(g["doc"], []).append(g)
    rows = []
    for n in DOCS:
        gs = by_doc.get(n, [])
        rng.shuffle(gs)
        # balance the two styles inside a document
        gs.sort(key=lambda g: g["type"])
        mixed = [
            g
            for pair in zip(
                *[[g for g in gs if g["type"] == t] for t in ("metrics", "domain")], strict=False
            )
            for g in pair
        ]
        rest = [g for g in gs if g not in mixed]
        for g in (mixed + rest)[:PER_DOC]:
            rows.append(
                {
                    "id": f"fx-{len(rows) + 1:03d}",
                    "query": g["question"],
                    "document": n,
                    "type": g["type"],
                    "expected_pages": [g["page"]],
                    "answer": g["answer"],
                    "split": "test",
                    "category": "finance",
                }
            )
    entries = []
    for n in DOCS:
        data = (files / n).read_bytes()
        entries.append(
            {
                "name": n,
                "url": REPO + n,
                "sha256": hashlib.sha256(data).hexdigest(),
                "lang": "en",
                "kind": "10k",
                "pages": len(texts[n]),
                "company": company(n),
            }
        )
    manifest = {
        "description": (
            "finx (V0.16): FinanceBench-style analyst questions over 24 10-Ks that no other "
            "suite uses; evidence = the page; verified answerable from the page and not "
            "answerable closed-book. Held-out only."
        ),
        "has_answers": True,
        "documents": entries,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    with (HERE / "queries.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} questions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
