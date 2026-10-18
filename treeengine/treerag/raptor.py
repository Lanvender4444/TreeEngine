"""RAPTOR-style summary tree (optional offline index, V0.17).

    leaves   consecutive block windows of ~``leaf_tokens`` (level 0, no LLM)
    level 1  leaves clustered (k-means on embeddings when an embedder is given, otherwise
             consecutive groups of ``fanout``) and summarised by the LLM
    level n  the same over level n-1 summaries, until at most ``fanout`` nodes remain

Stored in the engine's SQLite database (table ``raptor_nodes``). Evidence Search reads it with
READ_SUMMARY: summary nodes are candidate units (their text is the context they add) whose
children are the lower summaries or, at level 1, the leaf windows (READ_BLOCK).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

SUMMARY_SYSTEM = "You summarise parts of a document for a search index. Be factual and concise."
SUMMARY_PROMPT = """Document: {title}

Summarise the passages below in at most {words} words. Keep the key facts, entities, periods
and figures (with units) that someone might search for. Do not add anything not stated.

{passages}
"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raptor_nodes (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    level INTEGER NOT NULL,
    parent_id TEXT,
    block_lo INTEGER NOT NULL,
    block_hi INTEGER NOT NULL,
    children TEXT NOT NULL,
    summary TEXT NOT NULL,
    tokens INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS raptor_doc ON raptor_nodes (document_id, level);
"""


@dataclass
class SummaryNode:
    id: str
    document_id: str
    level: int
    block_lo: int  # block positions (document order) the node covers
    block_hi: int
    summary: str  # level 0: the leaf text
    tokens: int
    children: list[str] = field(default_factory=list)
    parent_id: str | None = None


class RaptorIndex:
    def __init__(self, repo: Any, llm: Any = None, embedder: Any = None) -> None:
        self.repo = repo
        self.llm = llm
        self.embedder = embedder
        self.conn = repo.conn
        self.conn.executescript(_SCHEMA)
        self._cache: dict[str, list[SummaryNode]] = {}

    # ------------------------------------------------------------------ build
    def build(
        self,
        document_id: str,
        *,
        leaf_tokens: int = 500,
        fanout: int = 6,
        max_levels: int = 4,
        words: int = 120,
    ) -> dict[str, int]:
        """(Re)build one document's summary tree. Needs an LLM; returns node counts by level."""
        if self.llm is None:
            raise RuntimeError("RAPTOR summaries need an LLM")
        from ..context.builder import BlockContextBuilder

        count_tokens = BlockContextBuilder(self.repo).count
        doc = self.repo.get_document(document_id)
        title = doc.title if doc else document_id
        blocks = self.repo.get_document_blocks(document_id)
        leaves: list[SummaryNode] = []
        lo, used = 0, 0
        parts: list[str] = []
        for i, b in enumerate(blocks):
            t = count_tokens(b.content)
            if parts and used + t > leaf_tokens:
                leaves.append(self._leaf(document_id, len(leaves), lo, i - 1, parts, used))
                lo, used, parts = i, 0, []
            parts.append(b.content.strip())
            used += t
        if parts:
            leaves.append(self._leaf(document_id, len(leaves), lo, len(blocks) - 1, parts, used))
        level_nodes, all_nodes, level = leaves, list(leaves), 0
        while len(level_nodes) > fanout and level < max_levels:
            level += 1
            groups = self._cluster(level_nodes, math.ceil(len(level_nodes) / fanout))
            nxt: list[SummaryNode] = []
            for gi, g in enumerate(groups):
                passages = "\n\n".join(f"[{k + 1}] {n.summary[:2500]}" for k, n in enumerate(g))
                text = self.llm.complete(
                    SUMMARY_PROMPT.format(title=title, words=words, passages=passages),
                    system=SUMMARY_SYSTEM,
                    max_tokens=words * 2,
                ).strip()
                node = SummaryNode(
                    f"{document_id}:L{level}:{gi}",
                    document_id,
                    level,
                    min(n.block_lo for n in g),
                    max(n.block_hi for n in g),
                    text,
                    count_tokens(text),
                    [n.id for n in g],
                )
                for n in g:
                    n.parent_id = node.id
                nxt.append(node)
            all_nodes += nxt
            level_nodes = nxt
        self._save(document_id, all_nodes)
        counts: dict[str, int] = {}
        for n in all_nodes:
            counts[f"level{n.level}"] = counts.get(f"level{n.level}", 0) + 1
        return counts

    def _leaf(
        self, doc: str, i: int, lo: int, hi: int, parts: list[str], tokens: int
    ) -> SummaryNode:
        return SummaryNode(f"{doc}:L0:{i}", doc, 0, lo, hi, "\n".join(parts), tokens)

    def _cluster(self, nodes: Sequence[SummaryNode], k: int) -> list[list[SummaryNode]]:
        """k-means on embeddings (RAPTOR clusters by meaning, not position); without an
        embedder, consecutive groups. Clusters keep document order inside."""
        size = math.ceil(len(nodes) / k)
        if self.embedder is None or k <= 1:
            return [list(nodes[i : i + size]) for i in range(0, len(nodes), size)]
        import numpy as np

        x = np.asarray(self.embedder.embed_texts([n.summary[:2000] for n in nodes]), dtype=float)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        centers = x[np.linspace(0, len(x) - 1, k).astype(int)]
        assign = np.zeros(len(x), dtype=int)
        for _ in range(20):
            assign = np.argmax(x @ centers.T, axis=1)
            new = np.stack(
                [
                    x[assign == c].mean(axis=0) if (assign == c).any() else centers[c]
                    for c in range(k)
                ]
            )
            new /= np.linalg.norm(new, axis=1, keepdims=True) + 1e-9
            if np.allclose(new, centers):
                break
            centers = new
        groups = [[n for n, a in zip(nodes, assign, strict=True) if a == c] for c in range(k)]
        out: list[list[SummaryNode]] = []
        for g in groups:  # cap a cluster at 2 * fanout so one summary never reads too much
            out += [g[i : i + 2 * size] for i in range(0, len(g), 2 * size)] if g else []
        return out

    # ------------------------------------------------------------------ storage
    def _save(self, doc: str, nodes: Sequence[SummaryNode]) -> None:
        with self.repo.transaction():
            self.conn.execute("DELETE FROM raptor_nodes WHERE document_id = ?", (doc,))
            self.conn.executemany(
                "INSERT INTO raptor_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        n.id,
                        n.document_id,
                        n.level,
                        n.parent_id,
                        n.block_lo,
                        n.block_hi,
                        json.dumps(n.children),
                        n.summary,
                        n.tokens,
                    )
                    for n in nodes
                ],
            )
        self._cache.pop(doc, None)

    def nodes(self, document_id: str, min_level: int = 0) -> list[SummaryNode]:
        if document_id not in self._cache:
            rows = self.conn.execute(
                "SELECT id, document_id, level, block_lo, block_hi, summary, tokens, children, "
                "parent_id FROM raptor_nodes WHERE document_id = ? ORDER BY level, block_lo",
                (document_id,),
            ).fetchall()
            self._cache[document_id] = [
                SummaryNode(
                    r[0],
                    r[1],
                    int(r[2]),
                    int(r[3]),
                    int(r[4]),
                    r[5],
                    int(r[6]),
                    json.loads(r[7]),
                    r[8],
                )
                for r in rows
            ]
        return [n for n in self._cache[document_id] if n.level >= min_level]

    def has(self, document_id: str) -> bool:
        return bool(self.nodes(document_id, 1))


__all__ = ["RaptorIndex", "SummaryNode"]
