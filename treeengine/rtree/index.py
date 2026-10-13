"""Node-level index of the Retrieval Tree: what every node of a document knows about itself.

Per document (built on first use, cached, rebuilt when the document's blocks change):

* the hierarchy: children of every node, roots, the block order
* each node's subtree as a block-index range, its blocks, its size (estimated tokens)
* ``text`` for the node's own lexical signal: its title path ("Item 7 > Liquidity") plus the
  opening of its own text - matched with BM25 against the query, among the document's nodes
* ``centroid``: the mean of its subtree's block vectors (the embeddings already stored for
  block retrieval - no extra embedding calls), for the node's semantic signal

Nothing is persisted: a node index is a few hundred rows per document and is derived from the
blocks, the tree and the block vectors that are stored.
"""

from __future__ import annotations

import math
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ..core.models import Block, Node
from ..core.protocols import Repository
from ..core.text import query_terms

HEAD_CHARS = 300  # own text included in a node's lexical text
CACHE_DOCS = 64


@dataclass
class NodeInfo:
    node: Node
    lo: int  # first block index of the subtree (in document order)
    hi: int  # last block index of the subtree
    blocks: list[int] = field(default_factory=list)  # subtree block indices
    own: list[int] = field(default_factory=list)  # blocks whose primary node is this node
    tokens: float = 0.0  # estimated subtree tokens (characters / 4)
    text: str = ""
    centroid: Any = None  # numpy vector or None


@dataclass
class DocIndex:
    document_id: str
    blocks: list[Block]
    nodes: dict[str, NodeInfo]
    children: dict[str, list[str]]  # node -> child node ids, document order
    roots: list[str]
    where: dict[str, int]  # block id -> index
    terms: dict[str, Counter[str]] = field(default_factory=dict)  # node -> term counts
    df: Counter[str] = field(default_factory=Counter)
    avg_len: float = 1.0
    has_centroids: bool = False
    version: str | None = None  # the document hash the index was built from

    def ancestors(self, node_id: str | None) -> list[str]:
        """The node and its ancestors, innermost first."""
        out: list[str] = []
        nid = node_id
        while nid and nid in self.nodes and nid not in out:
            out.append(nid)
            nid = self.nodes[nid].node.parent_id
        return out


class NodeIndex:
    def __init__(self, repo: Repository, vector_index: Any = None) -> None:
        self.repo = repo
        self.vectors = vector_index  # anything with .vectors(block_ids) -> {id: vector}
        self._cache: OrderedDict[str, DocIndex] = OrderedDict()

    def get(self, document_id: str, *, centroids: bool = False) -> DocIndex | None:
        version = self.repo.get_document_hash(document_id)  # cheap staleness check
        di = self._cache.get(document_id)
        if di is None or di.version != version:
            blocks = self.repo.get_document_blocks(document_id)
            if not blocks:
                return None
            di = self._build(document_id, blocks)
            di.version = version
            self._cache[document_id] = di
            while len(self._cache) > CACHE_DOCS:
                self._cache.popitem(last=False)
        self._cache.move_to_end(document_id)
        if centroids and not di.has_centroids:
            self._centroids(di)
        return di

    def _build(self, doc: str, blocks: list[Block]) -> DocIndex:
        nodes = {n.id: n for n in self.repo.get_document_nodes(doc)}
        children: dict[str, list[str]] = {}
        roots: list[str] = []
        for n in sorted(nodes.values(), key=lambda n: (n.start_offset or 0, n.depth)):
            if n.parent_id and n.parent_id in nodes:
                children.setdefault(n.parent_id, []).append(n.id)
            else:
                roots.append(n.id)
        info: dict[str, NodeInfo] = {}
        where = {b.id: i for i, b in enumerate(blocks)}
        for i, b in enumerate(blocks):
            nid = b.node_id
            first = True
            seen: set[str] = set()
            while nid and nid in nodes and nid not in seen:
                seen.add(nid)
                ni = info.get(nid)
                if ni is None:
                    ni = info[nid] = NodeInfo(nodes[nid], i, i)
                ni.lo, ni.hi = min(ni.lo, i), max(ni.hi, i)
                ni.blocks.append(i)
                ni.tokens += len(b.content) / 4
                if first:
                    ni.own.append(i)
                    first = False
                nid = nodes[nid].parent_id
        for nid, n in nodes.items():  # nodes without blocks still get an (empty) entry
            if nid not in info:
                info[nid] = NodeInfo(n, -1, -2)
        di = DocIndex(doc, blocks, info, children, roots, where)
        # lexical text: title path + the opening of the node's own text
        for nid, ni in info.items():
            path = [info[a].node.title for a in reversed(di.ancestors(nid)) if info[a].node.title]
            head = " ".join(blocks[i].content for i in ni.own[:3])[:HEAD_CHARS]
            ni.text = " > ".join(path) + "\n" + head
            tc = Counter(query_terms(ni.text))
            di.terms[nid] = tc
            di.df.update(tc.keys())
        di.avg_len = max(1.0, sum(sum(t.values()) for t in di.terms.values()) / max(1, len(info)))
        return di

    def _centroids(self, di: DocIndex) -> None:
        di.has_centroids = True
        if self.vectors is None or not hasattr(self.vectors, "vectors"):
            return
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            return
        got = self.vectors.vectors([b.id for b in di.blocks])
        if not got:
            return
        dim = len(next(iter(got.values())))
        mat = np.zeros((len(di.blocks), dim), dtype=np.float32)
        have = np.zeros(len(di.blocks), dtype=bool)
        for i, b in enumerate(di.blocks):
            v = got.get(b.id)
            if v is not None:
                mat[i] = v
                have[i] = True
        for ni in di.nodes.values():
            idx = [i for i in ni.blocks if have[i]]
            if not idx:
                continue
            c = mat[idx].mean(axis=0)
            norm = float(np.linalg.norm(c))
            ni.centroid = c / norm if norm > 0 else None

    # ---- node signals (ranked node ids, best first)
    def lexical(self, di: DocIndex, query: str, limit: int) -> list[str]:
        """BM25 of the query against node texts (title path + opening), among this document's
        nodes. k1 = 1.2, b = 0.75."""
        q = query_terms(query)
        if not q:
            return []
        n = max(1, len(di.terms))
        scores: list[tuple[float, str]] = []
        for nid, tc in di.terms.items():
            if di.nodes[nid].lo < 0:
                continue
            length = sum(tc.values()) or 1
            s = 0.0
            for t in q:
                f = tc.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - di.df[t] + 0.5) / (di.df[t] + 0.5))
                s += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * length / di.avg_len))
            if s > 0:
                scores.append((s, nid))
        scores.sort(key=lambda x: -x[0])
        return [nid for _, nid in scores[:limit]]

    def semantic(self, di: DocIndex, query_vec: Any, limit: int) -> list[str]:
        if query_vec is None:
            return []
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        qn = float(np.linalg.norm(q)) or 1.0
        scored = [
            (float(ni.centroid @ q) / qn, nid)
            for nid, ni in di.nodes.items()
            if ni.centroid is not None
        ]
        scored.sort(key=lambda x: -x[0])
        return [nid for _, nid in scored[:limit]]


__all__ = ["DocIndex", "NodeIndex", "NodeInfo"]
