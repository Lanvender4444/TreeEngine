"""SQLite FTS5 specifics.

FTS5's default ``unicode61`` tokenizer treats a run of CJK characters as one token, which makes
Chinese/Japanese text unsearchable. We index CJK text with every CJK character separated by
spaces and turn each query term into a phrase segmented the same way (``"供 应"``).
Latin words, numbers and identifiers are left as they are.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..core.text import CJK_CHAR_RE


def segment_for_index(text: str) -> str:
    """Make CJK characters individual tokens for FTS5's unicode61 tokenizer."""
    return re.sub(r"[ \t]+", " ", CJK_CHAR_RE.sub(r" \1 ", text)).strip()


def fts_phrase(term: str) -> str:
    """Quote a term as an FTS5 phrase, splitting CJK chars the same way as the index."""
    seg = segment_for_index(term).replace('"', '""')
    return f'"{seg}"'


def fts_match_expr(terms: Sequence[str], mode: str = "or") -> str:
    joiner = " AND " if mode == "and" else " OR "
    return joiner.join(fts_phrase(t) for t in terms if t.strip())
