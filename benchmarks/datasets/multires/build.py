"""Build the Multi-Resolution benchmark (V0.11): questions whose evidence sits at a known
reading resolution.

    python -m benchmarks.datasets.multires.build            # needs TREEENGINE_LLM_*

Three kinds of question, all over the *stable block sequence* (tree independent, V0.7):

  point   (gold resolution BLOCK)          one block answers it
  local   (gold resolution SECTION)        2-3 non-adjacent blocks of one contiguous local
                                           region (<= 900 tokens, <= 2 pages) are all needed
  cross   (gold resolution MULTI_SECTION)  two blocks far apart in the document are both needed

Pipeline: candidate passages (seeded) -> an LLM writes question + short answer from the
passages -> leave-one-out verification by the LLM: with all passages the question must be
answered correctly; for local / cross, *every single passage alone* must fail (otherwise the
question is really a point question and is dropped) -> needles (a distinctive substring of each
gold block, unique in its document) -> balanced selection -> dev / test split (1 : 2, stratified).

The system under test never takes part: the questions come from the documents' blocks and the
generator / verifier model only.
"""

from __future__ import annotations

import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
DATASETS = HERE.parent
SEED = 11
PER_TYPE = 30
DEV_SHARE = 1 / 3

FINANCE = [
    "3M_2018_10K.pdf",
    "ADOBE_2017_10K.pdf",
    "AMAZON_2019_10K.pdf",
    "AMD_2022_10K.pdf",
    "BESTBUY_2019_10K.pdf",
    "BOEING_2018_10K.pdf",
    "COCACOLA_2021_10K.pdf",
    "CORNING_2021_10K.pdf",
    "GENERALMILLS_2020_10K.pdf",
    "MICROSOFT_2023_10K.pdf",
    "NIKE_2019_10K.pdf",
    "PAYPAL_2022_10K.pdf",
]
LONGDOC = [
    "2310.05634v2.pdf",
    "2312.10997v5.pdf",
    "2401.18059v1.pdf",
    "2312.04350v3.pdf",
    "Independents-Report.pdf",
    "PH_2016.06.08_Economy-Final.pdf",
    "fdac8d1e9ef56519371df7e6532df27d.pdf",
    "SAO-StudentSupport_Guidebook-Content.pdf",
]

# multires2 (V0.11b): an independent suite - other documents, other seed, same pipeline -
# for validating changes that were designed while looking at multires
FINANCE2 = [
    "ACTIVISIONBLIZZARD_2019_10K.pdf",
    "AMCOR_2023_10K.pdf",
    "BLOCK_2020_10K.pdf",
    "COSTCO_2021_10K.pdf",
    "CVSHEALTH_2022_10K.pdf",
    "JOHNSON_JOHNSON_2022_10K.pdf",
    "KRAFTHEINZ_2019_10K.pdf",
    "LOCKHEEDMARTIN_2021_10K.pdf",
    "MGMRESORTS_2020_10K.pdf",
    "NETFLIX_2017_10K.pdf",
    "PFIZER_2021_10K.pdf",
    "ULTABEAUTY_2023_10K.pdf",
]
LONGDOC2 = [
    "2310.09158v1.pdf",
    "f1f5242528411b262be447e61e2eb10f.pdf",
    "379f44022bb27aa53efd5d322c7b57bf.pdf",
    "698bba535087fa9a7f9009e172a7f763.pdf",
    "efd88e41c5f2606c57929cac6c1c0605.pdf",
    "0e94b4197b10096b1f4c699701570fbf.pdf",
    "c31e6580d0175ab3f9d99d1ff0bfa000.pdf",
    "Sinopolis-Chengdu.pdf",
]
SUITES = {  # suite -> (finance docs, longdoc docs, seed, id prefix, dev share)
    "multires": (FINANCE, LONGDOC, SEED, "mr", DEV_SHARE),
    "multires2": (FINANCE2, LONGDOC2, 29, "m2", 0.0),  # all test: validation only
}

GEN_SYSTEM = (
    "You write evaluation questions for a document question-answering system. "
    "Reply with JSON only."
)
GEN = {
    "point": (
        "Here is one passage from the document '{doc}'.\n\n[P1]\n{p1}\n\n"
        "Write ONE specific question that this passage answers on its own (a fact, figure, "
        "name, date or short explanation). The question must make sense to someone who has the "
        "whole document but not this passage (name the company / topic, never say 'the "
        "passage'). Give a short gold answer (<= 25 words).\n"
        'JSON: {{"question": "...", "answer": "...", "ok": true}} - set ok false if the '
        "passage has no specific, answerable content."
    ),
    "local": (
        "Here is a contiguous stretch of the document '{doc}'. Three passages in it are "
        "marked.\n\n{region}\n\n"
        "Write ONE question whose complete answer needs information from ALL of {marks} "
        "(e.g. several factors, a figure and its explanation, a list spread over the passages). "
        "No single marked passage may be enough. Name the company / topic, never say 'the "
        "passage'. Give a short gold answer (<= 40 words) that combines them.\n"
        'JSON: {{"question": "...", "answer": "...", "uses": ["P1", ...], "ok": true}} - '
        "list in uses the marked passages the answer needs (at least two); set ok false if "
        "the passages are not about one topic."
    ),
    "cross": (
        "Here are two passages from distant parts of the document '{doc}'.\n\n"
        "[P1] (page {page1})\n{p1}\n\n[P2] (page {page2})\n{p2}\n\n"
        "Write ONE question whose complete answer needs information from BOTH passages (e.g. "
        "relate a figure in one to a statement in the other, or combine two facts). Neither "
        "passage alone may be enough. Name the company / topic, never say 'the passage'. Give "
        "a short gold answer (<= 40 words).\n"
        'JSON: {{"question": "...", "answer": "...", "ok": true}} - set ok false if no '
        "natural question needs both."
    ),
}
ANSWER_SYSTEM = (
    "Answer the question using only the given text. If the text does not contain everything "
    "needed for a complete answer, reply exactly: INSUFFICIENT. Be brief."
)
JUDGE_SYSTEM = "You grade answers. Reply with JSON only."
JUDGE = (
    "Question: {q}\nGold answer: {gold}\nCandidate answer: {cand}\n\n"
    "Is the candidate answer correct and complete with respect to the gold answer (same "
    'facts; wording may differ)? JSON: {{"correct": true|false}}'
)


def _json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return dict(json.loads(m.group(0)))
    except json.JSONDecodeError:
        return {}


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _substantive(text: str) -> bool:
    t = text.strip()
    if not 120 <= len(t) <= 1100:
        return False
    letters = sum(c.isalpha() for c in t)
    if letters < 0.55 * len(t) or t.isupper():
        return False
    return len(t.split()) >= 18


def _norm(text: str) -> str:
    from benchmarks.metrics.retrieval import norm

    return norm(text)


def needle(block: str, others: list[str]) -> str | None:
    """A distinctive 60-90 character substring of ``block`` (word boundaries) that occurs in no
    other block of the document."""
    words = block.split()
    starts = list(range(0, max(1, len(words) - 8)))
    mid = len(words) // 3
    starts.sort(key=lambda i: abs(i - mid))
    others_n = [_norm(o) for o in others]
    for i in starts:
        cand = ""
        for w in words[i:]:
            if len(cand) + len(w) + 1 > 90:
                break
            cand = (cand + " " + w).strip()
        if len(cand) < 60 or not re.search(r"[A-Za-z]{4}", cand):
            continue
        n = _norm(cand)
        if all(n not in o for o in others_n):
            return cand
    return None


def candidates(
    doc: str,
    blocks: list[Any],
    rng: random.Random,
    counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    counts = counts or {"point": 4, "local": 4, "cross": 4}
    good = [i for i, b in enumerate(blocks) if _substantive(b.content)]
    out: list[dict[str, Any]] = []
    # point
    for i in rng.sample(good, min(counts.get("point", 0), len(good))):
        if re.search(r"\d|[A-Z][a-z]+ [A-Z][a-z]+", blocks[i].content):
            out.append({"type": "point", "doc": doc, "blocks": [i]})
    # local: a contiguous region <= 900 tokens, <= 2 pages, with >= 3 substantive blocks
    tries = 0
    made = 0
    while made < counts.get("local", 0) and tries < 400:
        tries += 1
        s = rng.choice(good)
        e, used = s, _tokens(blocks[s].content)
        while e + 1 < len(blocks) and used + _tokens(blocks[e + 1].content) <= 900:
            e += 1
            used += _tokens(blocks[e].content)
        pages = {blocks[j].page for j in range(s, e + 1) if blocks[j].page is not None}
        if pages and max(pages) - min(pages) > 1:
            continue
        sub = [j for j in range(s, e + 1) if j in set(good)]
        picks: list[int] = []
        for j in sub:
            if not picks or j - picks[-1] >= 2:
                picks.append(j)
        if len(picks) < 3:
            continue
        picks = [picks[0], picks[len(picks) // 2], picks[-1]]
        out.append({"type": "local", "doc": doc, "blocks": picks, "region": [s, e]})
        made += 1
    # cross: two blocks far apart
    pages_all = [b.page for b in blocks if b.page is not None]
    span = (max(pages_all) - min(pages_all)) if pages_all else 0
    gap = max(5, int(0.2 * span))
    made = tries = 0
    while made < counts.get("cross", 0) and tries < 200 and len(good) > 2:
        tries += 1
        a, b = sorted(rng.sample(good, 2))
        pa, pb = blocks[a].page, blocks[b].page
        if pa is None or pb is None or pb - pa < gap:
            continue
        out.append({"type": "cross", "doc": doc, "blocks": [a, b]})
        made += 1
    return out


def generate(llm: Any, c: dict[str, Any], blocks: list[Any]) -> dict[str, Any] | None:
    doc = c["doc"].rsplit(".", 1)[0]
    if c["type"] == "point":
        prompt = GEN["point"].format(doc=doc, p1=blocks[c["blocks"][0]].content)
    elif c["type"] == "local":
        s, e = c["region"]
        marks = {j: f"P{k + 1}" for k, j in enumerate(c["blocks"])}
        parts = []
        for j in range(s, e + 1):
            txt = blocks[j].content.strip()
            parts.append(f"[{marks[j]}]\n{txt}\n[/{marks[j]}]" if j in marks else txt)
        prompt = GEN["local"].format(
            doc=doc, region="\n\n".join(parts), marks=", ".join(marks.values())
        )
    else:
        a, b = c["blocks"]
        prompt = GEN["cross"].format(
            doc=doc,
            p1=blocks[a].content,
            p2=blocks[b].content,
            page1=blocks[a].page,
            page2=blocks[b].page,
        )
    out = _json(llm.complete(prompt, system=GEN_SYSTEM, max_tokens=400))
    if not out.get("ok") or not out.get("question") or not out.get("answer"):
        return None
    gold = list(c["blocks"])
    if c["type"] == "local":
        names = {f"P{k + 1}": j for k, j in enumerate(c["blocks"])}
        gold = [names[u] for u in out.get("uses", []) if u in names]
        if len(gold) < 2:
            return None
    return {**c, "gold": gold, "question": out["question"], "answer": out["answer"]}


def _answers(llm: Any, q: str, gold: str, text: str) -> bool:
    cand = llm.complete(f"Text:\n{text}\n\nQuestion: {q}", system=ANSWER_SYSTEM, max_tokens=200)
    if "INSUFFICIENT" in cand.upper():
        return False
    verdict = _json(
        llm.complete(JUDGE.format(q=q, gold=gold, cand=cand), system=JUDGE_SYSTEM, max_tokens=60)
    )
    return bool(verdict.get("correct"))


def verify(llm: Any, g: dict[str, Any], blocks: list[Any]) -> bool:
    texts = [blocks[j].content for j in g["gold"]]
    if not _answers(llm, g["question"], g["answer"], "\n\n".join(texts)):
        return False
    if g["type"] == "point":
        return True
    return not any(_answers(llm, g["question"], g["answer"], t) for t in texts)


def main() -> int:
    """``build`` = everything from scratch; ``build --topup local`` = keep the existing
    questions and generate more of one type (new seed, more candidates per document, the same
    verification), then re-select and re-split that type."""
    import argparse

    from benchmarks.answer import llm_from_env
    from treeengine import TreeEngine

    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=sorted(SUITES), default="multires")
    ap.add_argument("--topup", choices=["point", "local", "cross"], default=None)
    ap.add_argument("--per-doc", type=int, default=12)
    args = ap.parse_args()
    finance, longdoc, seed, prefix, dev_share = SUITES[args.suite]
    out_dir = DATASETS / args.suite
    out_dir.mkdir(exist_ok=True)
    (out_dir / "__init__.py").touch()

    llm = llm_from_env("TREEENGINE_LLM")
    if llm is None:
        raise SystemExit("needs TREEENGINE_LLM_MODEL / _API_KEY / _BASE_URL")
    sources = {
        "financebench": json.loads((DATASETS / "financebench" / "manifest.json").read_text()),
        "longdoc": json.loads((DATASETS / "longdoc" / "manifest.json").read_text()),
    }
    entries = []
    files = out_dir / "files"
    files.mkdir(exist_ok=True)
    for suite, names in (("financebench", finance), ("longdoc", longdoc)):
        by_name = {d["name"]: d for d in sources[suite]["documents"]}
        for n in names:
            d = dict(by_name[n])
            d["source_suite"] = suite
            entries.append(d)
            src = DATASETS / suite / "files" / n
            dst = files / n
            if not dst.exists() and src.exists():
                dst.symlink_to(src.resolve())
    te = TreeEngine(":memory:")
    rng = random.Random(seed if args.topup is None else seed + 1)
    counts = None if args.topup is None else {args.topup: args.per_doc}
    if args.suite != "multires" and args.topup is None:
        # local questions pass verification least often: more candidates from the start
        counts = {"point": 4, "local": args.per_doc, "cross": 4}
    existing: list[dict[str, Any]] = []
    if args.topup is not None:
        existing = [
            json.loads(line)
            for line in (out_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        ]
    blocks_by_doc: dict[str, list[Any]] = {}
    cands: list[dict[str, Any]] = []
    for d in entries:
        doc = te.ingest(files / d["name"])
        blocks = te.repo.get_document_blocks(doc.id)
        blocks_by_doc[d["name"]] = blocks
        cands += candidates(d["name"], blocks, rng, counts)
        print(f"{d['name']}: {len(blocks)} blocks, {len(cands)} candidates", file=sys.stderr)

    def work(c: dict[str, Any]) -> dict[str, Any] | None:
        blocks = blocks_by_doc[c["doc"]]
        try:
            g = generate(llm, c, blocks)
            if g is None or not verify(llm, g, blocks):
                return None
        except Exception as e:  # noqa: BLE001 - a failed candidate is skipped
            print(f"skip ({e})", file=sys.stderr)
            return None
        others = [b.content for k, b in enumerate(blocks)]
        ns = []
        for j in g["gold"]:
            n = needle(blocks[j].content, others[:j] + others[j + 1 :])
            if n is None:
                return None
            ns.append(n)
        g["needles"] = ns
        g["pages"] = sorted({blocks[j].page for j in g["gold"] if blocks[j].page is not None})
        return g

    with ThreadPoolExecutor(8) as ex:
        kept = [g for g in ex.map(work, cands) if g is not None]
    print(f"verified {len(kept)} / {len(cands)}", file=sys.stderr)
    # balanced selection: round-robin over documents within each type
    label = {"point": "BLOCK", "local": "SECTION", "cross": "MULTI_SECTION"}
    rows = []
    for t in ("point", "local", "cross"):
        if args.topup is not None and t != args.topup:
            rows += [r for r in existing if r["type"] == t]
            continue
        pool = [g for g in kept if g["type"] == t]
        if args.topup is not None:  # the questions of this type already accepted come first
            seen = {n for r in existing if r["type"] == t for n in r["expected_blocks"]}
            pool = [g for g in pool if not seen & set(g["needles"])]
            old_rows = [r for r in existing if r["type"] == t]
            pool = [
                {
                    "doc": r["document"],
                    "question": r["query"],
                    "answer": r["answer"],
                    "needles": r["expected_blocks"],
                    "pages": r["expected_pages"],
                    "type": t,
                    "old": True,
                }
                for r in old_rows
            ] + pool
        if args.topup is None:
            rng.shuffle(pool)
        else:  # keep the accepted ones, shuffle only the new ones
            new = [g for g in pool if not g.get("old")]
            rng.shuffle(new)
            pool = new + [g for g in pool if g.get("old")]  # pop() takes from the end first
        by_doc: dict[str, list[dict[str, Any]]] = {}
        for g in pool:
            by_doc.setdefault(g["doc"], []).append(g)
        chosen: list[dict[str, Any]] = []
        while len(chosen) < PER_TYPE and any(by_doc.values()):
            for doc in sorted(by_doc):
                if by_doc[doc] and len(chosen) < PER_TYPE:
                    chosen.append(by_doc[doc].pop())
        n_dev = round(len(chosen) * dev_share)
        for k, g in enumerate(chosen):
            rows.append(
                {
                    "id": f"{prefix}-{t[0]}{k + 1:02d}",
                    "query": g["question"],
                    "document": g["doc"],
                    "type": t,
                    "resolution": label[t],
                    "expected_blocks": g["needles"],
                    "expected_pages": g["pages"],
                    "answer": g["answer"],
                    "split": "dev" if k < n_dev else "test",
                    "category": "financebench" if g["doc"] in finance else "longdoc",
                }
            )
    manifest = {
        "description": (
            "Multi-Resolution benchmark (V0.11): point / local multi-block / cross-section "
            "questions generated from FinanceBench and MMLongBench-Doc documents, verified "
            "leave-one-out by an LLM; evidence = text needles of the gold blocks."
        ),
        "has_answers": True,
        "documents": entries,
    }
    if args.topup is None:
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False)
        )
    with (out_dir / "queries.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {t: sum(1 for r in rows if r["type"] == t) for t in ("point", "local", "cross")}
    print(f"wrote {len(rows)} questions {counts}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
