"""Graph-lite expansion: seed blocks -> their entities -> other blocks with the same entity.

One hop, bounded, rank-based. It is an expansion signal, never a primary retriever: the seeds
are the best block anchors of the fast paths, and the related blocks come back as BlockAnchors
(``source="graph"``, ``metadata["seed_block_id"]`` / ``["entity"]``) for rank fusion.

Entity choice per seed: specificity = log(1 + blocks / df) (rare across the corpus = more
informative) x occurrences in the seed; entities found in only the seed itself (df < 2) or in a
large share of the corpus are skipped. Related blocks: same document first, then by the
entity's frequency there; blocks any other path already found are skipped, so the graph only
adds positions and never re-ranks the fast paths' own results. Budgets (``RetrievalPolicy``):
``graph_seeds``, ``graph_entities_per_seed``, ``graph_blocks_per_entity``, ``graph_max_blocks``.
"""

from __future__ import annotations

import math
from typing import Any

from ..core.anchor import Anchor, BlockAnchor
from ..core.models import Evidence
from ..core.protocols import GraphStore, Repository

MAX_DF_SHARE = 0.02  # an entity in more than 2% of all linked blocks is too common to follow


class GraphExpander:
    def __init__(self, repo: Repository) -> None:
        self.repo = repo
        self.store: GraphStore | None = repo if isinstance(repo, GraphStore) else None

    def expand(
        self,
        seeds: list[Anchor],
        policy: Any,
        trace: Any = None,
        exclude: set[str] | None = None,
    ) -> list[Anchor]:
        """Related blocks one entity hop from ``seeds``; never a seed, never a block id in
        ``exclude`` (the controller passes every block another path already found)."""
        if self.store is None:
            return []
        stats = self.store.graph_stats()
        total = max(1, stats.get("blocks", 0))
        if not stats.get("links"):
            if trace is not None:
                trace.add("graph", skipped="graph index is empty (build_graph_index)")
            return []
        seeds = [a for a in seeds if a.kind == "block"][: policy.graph_seeds]
        seed_ids = {str(a.block_id) for a in seeds}
        ents = self.store.block_entities(list(seed_ids))
        max_df = max(3, int(MAX_DF_SHARE * total))
        picked: list[tuple[int, float, Anchor, dict[str, Any]]] = []
        for r, seed in enumerate(seeds):
            cands = [e for e in ents.get(str(seed.block_id), []) if 2 <= e["df"] <= max_df]
            cands.sort(key=lambda e: -(math.log(1 + total / e["df"]) * e["frequency"]))
            for e in cands[: policy.graph_entities_per_seed]:
                spec = math.log(1 + total / e["df"])
                picked.append((r, spec, seed, e))
        picked.sort(key=lambda x: (x[0], -x[1]))
        out: list[Anchor] = []
        seen = set(seed_ids) | set(exclude or ())
        for _, _, seed, e in picked:
            if len(out) >= policy.graph_max_blocks:
                break
            rows = self.store.entity_blocks(
                e["id"], document_id=seed.document_id, limit=policy.graph_blocks_per_entity + 3
            )
            took = 0
            for block_id, doc_id, _ in rows:
                if took >= policy.graph_blocks_per_entity or len(out) >= policy.graph_max_blocks:
                    break
                if block_id in seen:
                    continue
                seen.add(block_id)
                took += 1
                out.append(self._anchor(block_id, doc_id, seed, e, len(out) + 1))
        if trace is not None:
            trace.add(
                "graph",
                seeds=len(seeds),
                entities=len(picked),
                added=len(out),
                index_blocks=total,
            )
        return out

    def _anchor(
        self, block_id: str, doc_id: str, seed: Anchor, ent: dict[str, Any], rank: int
    ) -> Anchor:
        b = self.repo.get_block(block_id)
        node = self.repo.get_node(b.node_id) if b and b.node_id else None
        md = {
            "seed_block_id": seed.block_id,
            "entity": ent["name"],
            "entity_type": ent["type"],
            "page": b.page if b else None,
            "node_title": node.title if node else None,
        }
        ev = Evidence(
            doc_id, b.node_id if b else None, block_id, b.content if b else "", "graph", None, md
        )
        return BlockAnchor(doc_id, block_id, ev.node_id, "graph", rank, None, dict(md), ev)


__all__ = ["MAX_DF_SHARE", "GraphExpander"]
