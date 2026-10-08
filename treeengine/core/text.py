"""Index-agnostic text helpers: query term extraction (latin words + CJK bigrams), lexical
scoring and splitting. Index-specific syntax (FTS5 phrases, CJK segmentation for the
``unicode61`` tokenizer) lives in ``treeengine.storage.fts5``.
"""

from __future__ import annotations

import re
from functools import lru_cache

CJK_CLASS = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af"
CJK_CHAR_RE = re.compile(f"([{CJK_CLASS}])")
_TOKEN_RE = re.compile(f"[{CJK_CLASS}]+|[A-Za-z0-9_][A-Za-z0-9_.+\\-]*[A-Za-z0-9_+]|[A-Za-z0-9_]")

_EN_STOP = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "and",
    "or",
    "by",
    "with",
    "from",
    "as",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "what",
    "which",
    "who",
    "whom",
    "why",
    "how",
    "when",
    "where",
    "does",
    "do",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
    "about",
    "there",
    "their",
    "into",
    "than",
    "then",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "i",
    "tell",
    "explain",
    "please",
    "much",
    "many",
}
_ZH_STOP_CHARS = set("的了是在吗呢吧啊和与及或里中哪么个这那就也都把被给对从向让请你我他她它")
_ZH_STOP_BIGRAMS = {
    "什么",
    "为什",
    "怎么",
    "如何",
    "哪些",
    "哪里",
    "哪个",
    "是多",
    "多少",
    "是否",
    "有没",
    "没有",
    "提到",
    "地方",
    "我们",
    "他们",
    "一下",
    "可以",
    "关于",
    "以及",
    "章节",
    "解释",
    "原因",
    "说明",
    "介绍",
    "一些",
    "有哪",
    "哪儿",
    "有关",
    "所有",
    "相关",
}


def is_cjk(ch: str) -> bool:
    return bool(CJK_CHAR_RE.fullmatch(ch))


def query_terms(query: str) -> list[str]:
    """Extract de-duplicated search terms: latin words/numbers + CJK bigrams."""
    terms: list[str] = []
    for tok in _TOKEN_RE.findall(query):
        if is_cjk(tok[0]):
            chars = [c for c in tok]
            if len(chars) == 1:
                if chars[0] not in _ZH_STOP_CHARS:
                    terms.append(chars[0])
                continue
            for i in range(len(chars) - 1):
                bg = chars[i] + chars[i + 1]
                if (
                    bg in _ZH_STOP_BIGRAMS
                    or chars[i] in _ZH_STOP_CHARS
                    or chars[i + 1] in _ZH_STOP_CHARS
                ):
                    continue
                terms.append(bg)
        else:
            low = tok.lower()
            if low in _EN_STOP:
                continue
            terms.append(low)
    seen: set[str] = set()
    out = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


_SUFFIXES = ("ings", "ing", "edly", "ed", "es", "s")


def stem(term: str) -> str:
    """Tiny English suffix stripper for substring matching ("shuffle" ~ "shuffling",
    "partitions" ~ "partition"). CJK terms and short tokens are returned unchanged."""
    if len(term) <= 4 or not term.isascii() or not term.isalpha():
        return term
    for suf in _SUFFIXES:
        if term.endswith(suf) and len(term) - len(suf) >= 4:
            term = term[: -len(suf)]
            break
    if term.endswith("e") and len(term) > 4:
        term = term[:-1]
    return term


@lru_cache(maxsize=4096)
def _term_pattern(term: str) -> re.Pattern[str] | None:
    """Latin terms must start at a word boundary ("groupkfold" must not match
    "stratifiedgroupkfold"); CJK terms use plain substring matching (no word boundaries)."""
    if is_cjk(term[0]):
        return None
    return re.compile(r"(?<![a-z0-9_])" + re.escape(stem(term)))


def lexical_score(terms: list[str], text: str | None) -> float:
    """Number of distinct query terms that occur in ``text`` (case-insensitive, lightly
    stemmed, word-start anchored for latin terms)."""
    if not text or not terms:
        return 0.0
    low = text.lower()
    n = 0
    for t in terms:
        pat = _term_pattern(t)
        if (t in low) if pat is None else bool(pat.search(low)):
            n += 1
    return float(n)


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])|(?<=[.])\s+")


def split_long_text(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Split text into (relative_offset, piece) chunks of at most ``max_chars``,
    preferring sentence boundaries. Offsets are relative to ``text``."""
    if len(text) <= max_chars:
        return [(0, text)]
    pieces: list[tuple[int, str]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            window = text[start:end]
            cut = -1
            for m in _SENT_SPLIT_RE.finditer(window):
                if m.end() > max_chars // 3:
                    cut = m.end()
            if cut <= 0:
                sp = window.rfind(" ")
                cut = sp + 1 if sp > max_chars // 3 else len(window)
            end = start + cut
        piece = text[start:end]
        if piece.strip():
            lead = len(piece) - len(piece.lstrip())
            pieces.append((start + lead, piece.strip()))
        start = end
    return pieces
