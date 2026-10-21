"""Build the multi-hop benchmark (V0.19): questions that need facts from two places.

    python -m benchmarks.datasets.multihop.build            # needs TREEENGINE_LLM_*

Questions over FinanceBench 10-Ks of the same companies, three to seven fiscal years
apart - three for two pairs (a 10-K reports at most three years of income-statement figures
and two of balance-sheet figures, so neither report contains the other year's numbers):

  xdoc   one passage from the company's older 10-K and the matching passage of its newer 10-K
         (found by BM25 over the newer report, numbers removed from the query); the question
         names the company and both years and needs both passages. Asked over the whole corpus
         (``document`` is null): retrieval must find the right company *and* year.
  calc   (generator kept, not used: CALC_PER_DOC = 0) two passages with figures five pages
         apart in one 10-K, combined into a calculation - dropped after the first build
         produced artificial combinations.

Same pipeline as multires (``benchmarks/datasets/multires/build.py``): candidate passages
(seeded) -> an LLM writes question + short answer -> leave-one-out verification by the LLM
(with both passages the answer must be judged correct; with either passage alone it must
fail) -> needles (distinctive substrings, unique across both reports) -> ids ``mh-x..`` /
``mh-c..``. Dev / test is the runner's ``--hash-split`` (1 in 4 dev). The system under test
never takes part.
"""

from __future__ import annotations

import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from benchmarks.datasets.multires.build import (
    GEN_SYSTEM,
    _answers,
    _json,
    _substantive,
    needle,
)

HERE = Path(__file__).parent
DATASETS = HERE.parent
SEED = 41
PAIRS = [  # (older, newer): 3-7 fiscal years apart
    ("3M_2018_10K.pdf", "3M_2022_10K.pdf"),
    ("ADOBE_2016_10K.pdf", "ADOBE_2022_10K.pdf"),
    ("AMD_2015_10K.pdf", "AMD_2022_10K.pdf"),
    ("BLOCK_2016_10K.pdf", "BLOCK_2020_10K.pdf"),
    ("BOEING_2018_10K.pdf", "BOEING_2022_10K.pdf"),
    ("CVSHEALTH_2018_10K.pdf", "CVSHEALTH_2022_10K.pdf"),
    ("MICROSOFT_2016_10K.pdf", "MICROSOFT_2023_10K.pdf"),
    ("NIKE_2018_10K.pdf", "NIKE_2023_10K.pdf"),
    ("COCACOLA_2017_10K.pdf", "COCACOLA_2022_10K.pdf"),
    ("BESTBUY_2017_10K.pdf", "BESTBUY_2023_10K.pdf"),
    ("MGMRESORTS_2018_10K.pdf", "MGMRESORTS_2022_10K.pdf"),
    ("AMCOR_2020_10K.pdf", "AMCOR_2023_10K.pdf"),  # 3 years apart: no shared income-statement
    ("GENERALMILLS_2019_10K.pdf", "GENERALMILLS_2022_10K.pdf"),  # or balance-sheet year
]
XDOC_PER_PAIR = 30
# calc questions were tried in the first build and dropped: the generator combined unrelated
# figures (sales minus a regulatory threshold) or quoted the inputs in the question
CALC_PER_DOC = 0

GEN_XDOC = (
    "Here are two passages about {company}.\n\n[P1] from its annual report (10-K) for fiscal "
    "{ya}\n{p1}\n\n[P2] from its annual report (10-K) for fiscal {yb}\n{p2}\n\n"
    "Write ONE question whose complete answer needs information from BOTH passages - e.g. how "
    "a figure or a statement changed between fiscal {ya} and fiscal {yb}. Name the company and "
    "both fiscal years explicitly. Neither passage alone may be enough. Give a short gold answer "
    "(<= 40 words) with the figures from both years.\n"
    'JSON: {{"question": "...", "answer": "...", "ok": true}} - set ok false if the two '
    "passages are not about the same item."
)
GEN_CALC = (
    "Here are two passages from {company}'s annual report (10-K) for fiscal {year}.\n\n"
    "[P1] (page {page1})\n{p1}\n\n[P2] (page {page2})\n{p2}\n\n"
    "Write ONE question that can only be answered by combining a figure from P1 with a figure "
    "from P2 (a ratio, a difference, a sum or a share). Name the company and the fiscal year. "
    "The gold answer (<= 40 words) gives the result and the two input figures.\n"
    'JSON: {{"question": "...", "answer": "...", "ok": true}} - set ok false if no natural '
    "calculation combines the two."
)
_MONEY = re.compile(r"\$|%|million|billion|\b\d{1,3}(,\d{3})+\b")


def numeric(text: str) -> bool:
    return _substantive(text) and len(re.findall(r"\d", text)) >= 6 and bool(_MONEY.search(text))


def company_year(name: str) -> tuple[str, str]:
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("_")
    return " ".join(parts[:-2]).title(), parts[-2]


def xdoc_candidates(te: Any, ids: dict[str, str], rng: random.Random) -> list[dict[str, Any]]:
    from treeengine.core.text import query_terms

    out = []
    for old, new in PAIRS:
        blocks_old = te.repo.get_document_blocks(ids[old])
        good = [i for i, b in enumerate(blocks_old) if numeric(b.content)]
        rng.shuffle(good)
        blocks_new = te.repo.get_document_blocks(ids[new])
        index_new = {b.id: i for i, b in enumerate(blocks_new)}
        made = 0
        for i in good:
            if made >= XDOC_PER_PAIR:
                break
            words = re.sub(r"[\d$%,.()]+", " ", blocks_old[i].content)
            terms = query_terms(" ".join(words.split()[:60]))[:14]
            hits = te.repo.fts_query(terms, mode="or", document_id=ids[new], limit=3) if terms else []
            match = next((b for b, _ in hits if numeric(b.content)), None)
            if match is None:
                continue
            out.append({"type": "xdoc", "docs": [old, new], "blocks": [i, index_new[match.id]]})
            made += 1
    return out


def calc_candidates(te: Any, ids: dict[str, str], rng: random.Random) -> list[dict[str, Any]]:
    out = []
    for pair in PAIRS:
        for name in pair:
            blocks = te.repo.get_document_blocks(ids[name])
            good = [i for i, b in enumerate(blocks) if numeric(b.content) and b.page is not None]
            made = tries = 0
            while made < CALC_PER_DOC and tries < 200 and len(good) > 2:
                tries += 1
                a, b = sorted(rng.sample(good, 2))
                if abs(int(blocks[b].page) - int(blocks[a].page)) < 5:
                    continue
                out.append({"type": "calc", "docs": [name, name], "blocks": [a, b]})
                made += 1
    return out


def generate(llm: Any, c: dict[str, Any], blocks: dict[str, list[Any]]) -> dict[str, Any] | None:
    (da, ia), (db, ib) = zip(c["docs"], c["blocks"], strict=True)
    pa, pb = blocks[da][ia], blocks[db][ib]
    company, ya = company_year(da)
    _, yb = company_year(db)
    if c["type"] == "xdoc":
        prompt = GEN_XDOC.format(company=company, ya=ya, yb=yb, p1=pa.content, p2=pb.content)
    else:
        prompt = GEN_CALC.format(company=company, year=ya, p1=pa.content, p2=pb.content,
                                 page1=pa.page, page2=pb.page)  # fmt: skip
    out = _json(llm.complete(prompt, system=GEN_SYSTEM, max_tokens=400))
    if not out.get("ok") or not out.get("question") or not out.get("answer"):
        return None
    return {**c, "question": out["question"], "answer": out["answer"],
            "texts": [pa.content, pb.content], "pages": [pa.page, pb.page]}  # fmt: skip


def verify(llm: Any, g: dict[str, Any]) -> bool:
    a, b = g["texts"]
    labels = ("[from the older report]\n", "[from the newer report]\n")
    if g["type"] == "xdoc":
        a, b = labels[0] + a, labels[1] + b
    if not _answers(llm, g["question"], g["answer"], f"{a}\n\n{b}"):
        return False
    return not _answers(llm, g["question"], g["answer"], a) and not _answers(
        llm, g["question"], g["answer"], b
    )


def main() -> int:
    from benchmarks.answer import llm_from_env
    from treeengine import TreeEngine

    llm = llm_from_env("TREEENGINE_LLM")
    if llm is None:
        raise SystemExit("needs TREEENGINE_LLM_MODEL / _API_KEY / _BASE_URL")
    fb = json.loads((DATASETS / "financebench" / "manifest.json").read_text(encoding="utf-8"))
    by_name = {d["name"]: d for d in fb["documents"]}
    names = [n for p in PAIRS for n in p]
    files = HERE / "files"
    files.mkdir(exist_ok=True)
    (HERE / "__init__.py").touch()
    entries = []
    for n in names:
        entries.append({**by_name[n], "source_suite": "financebench"})
        dst, src = files / n, DATASETS / "financebench" / "files" / n
        if not dst.exists() and src.exists():
            dst.symlink_to(src.resolve())
    te = TreeEngine(":memory:")
    ids = {n: te.ingest(files / n).id for n in names}
    blocks = {n: te.repo.get_document_blocks(ids[n]) for n in names}
    rng = random.Random(SEED)
    cands = xdoc_candidates(te, ids, rng) + calc_candidates(te, ids, rng)
    print(f"{len(cands)} candidates", file=sys.stderr)

    def work(c: dict[str, Any]) -> dict[str, Any] | None:
        try:
            g = generate(llm, c, blocks)
            if g is None or not verify(llm, g):
                return None
        except Exception as e:  # noqa: BLE001 - a failed candidate is skipped
            print(f"skip ({e})", file=sys.stderr)
            return None
        others = [b.content for n in set(g["docs"]) for b in blocks[n]]
        ns = []
        for d, j in zip(g["docs"], g["blocks"], strict=True):
            own = blocks[d][j].content
            rest = list(others)
            rest.remove(own)
            n = needle(own, rest)
            if n is None:
                return None
            ns.append(n)
        g["needles"] = ns
        return g

    with ThreadPoolExecutor(8) as ex:
        kept = [g for g in ex.map(work, cands) if g is not None]
    print(f"verified {len(kept)} / {len(cands)}", file=sys.stderr)
    rows = []
    n = {"xdoc": 0, "calc": 0}
    for g in kept:
        t = g["type"]
        n[t] += 1
        row: dict[str, Any] = {
            "id": f"mh-{t[0]}{n[t]:02d}",
            "query": g["question"],
            "document": None if t == "xdoc" else g["docs"][0],
            "type": t,
            "resolution": "MULTI_DOC" if t == "xdoc" else "MULTI_SECTION",
            "expected_blocks": g["needles"],
            "answer": g["answer"],
            "split": "test",
            "category": "financebench",
            "source": {"docs": g["docs"], "pages": g["pages"]},
        }
        if t == "calc":
            row["expected_pages"] = sorted({int(p) for p in g["pages"] if p is not None})
        rows.append(row)
    (HERE / "queries.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    manifest = {
        "description": "Multi-hop benchmark (V0.19): cross-document questions (same company, "
        "fiscal years 3-7 apart, asked over the whole corpus) over FinanceBench 10-Ks, generated and verified leave-one-out by an LLM; evidence = "
        "text needles of the two gold blocks.",
        "has_answers": True,
        "documents": entries,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} questions: {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
