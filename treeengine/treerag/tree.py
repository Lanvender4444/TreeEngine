"""The unified Tree-RAG node model and the search tree it forms.

One tree holds the three trees of the design (docs/v0.17-tree-rag.md):

* the **Query Tree** - QUERY (the question) -> SUBQUERY nodes (DECOMPOSE);
* the **Retrieval Search Tree** - per (sub)query: SEARCH nodes (the fast paths that produced
  candidates), VERIFY nodes (LLM judge), REWRITE nodes (REWRITE_QUERY + its SEARCH);
* the **Evidence Tree** - under a SEARCH node, the reading units it produced (BLOCK windows,
  SECTION heads / children, PAGEs, TABLEs, ENTITY hops, in-section searches) and the units each
  one expanded into. ``read`` gives the reading order; scored but unread units are pruned.

ANSWER closes the tree. ``render()`` draws it as text, ``to_dict()`` serialises it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Evidence Search unit kind -> (node type, action)
UNIT_ACTIONS: dict[str, tuple[str, str]] = {
    "hit": ("BLOCK", "READ_BLOCK"),
    "neighbor": ("BLOCK", "READ_NEIGHBORS"),
    "parent": ("SECTION", "READ_PARENT"),
    "child": ("SECTION", "READ_CHILD"),
    "page": ("PAGE", "READ_PAGE"),
    "table": ("TABLE", "READ_TABLE"),
    "entity": ("ENTITY", "FOLLOW_ENTITY"),
    "section_search": ("BLOCK", "SEARCH_IN_SECTION"),
    "summary": ("SUMMARY", "READ_SUMMARY"),
    "leaf": ("BLOCK", "READ_BLOCK"),
}


@dataclass
class RAGNode:
    id: str
    parent_id: str | None
    node_type: str  # QUERY | SUBQUERY | SEARCH | REWRITE | VERIFY | BLOCK | SECTION | PAGE |
    #                 TABLE | ENTITY | SUMMARY | ANSWER
    action: str = ""  # DECOMPOSE | SEARCH_TEXT+VECTOR | READ_* | FOLLOW_ENTITY | VERIFY | ...
    query: str = ""
    observation: str = ""  # what the action produced (a preview, a judge verdict, the answer)
    children: list[str] = field(default_factory=list)
    document_id: str | None = None
    section_id: str | None = None
    block_ids: list[str] = field(default_factory=list)
    page: int | None = None
    relevance: float | None = None  # node value (reranker, blended with the retrieval prior)
    prior: float | None = None  # retrieval confidence
    coverage: float | None = None  # VERIFY: share of the needed facts present
    answerability: bool | None = None  # VERIFY
    cost: int = 0  # tokens
    read: int | None = None  # reading order (None: scored, not read = pruned)
    visits: int = 0  # MCTS
    value: float | None = None  # MCTS mean reward
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def pruned(self) -> bool:
        return self.node_type in _EVIDENCE and self.read is None


_EVIDENCE = {"BLOCK", "SECTION", "PAGE", "TABLE", "ENTITY", "SUMMARY"}


@dataclass
class SearchTree:
    nodes: dict[str, RAGNode] = field(default_factory=dict)
    root: str | None = None

    def add(self, node: RAGNode) -> RAGNode:
        if node.id in self.nodes:
            return self.nodes[node.id]
        self.nodes[node.id] = node
        if node.parent_id is None:
            self.root = self.root or node.id
        elif node.parent_id in self.nodes:
            self.nodes[node.parent_id].children.append(node.id)
        return node

    def __len__(self) -> int:
        return len(self.nodes)

    def of_type(self, *types: str) -> list[RAGNode]:
        return [n for n in self.nodes.values() if n.node_type in types]

    def read_nodes(self) -> list[RAGNode]:
        return sorted((n for n in self.nodes.values() if n.read), key=lambda n: n.read or 0)

    def path(self, node_id: str) -> list[RAGNode]:
        out = []
        cur: str | None = node_id
        while cur is not None and cur in self.nodes:
            out.append(self.nodes[cur])
            cur = self.nodes[cur].parent_id
        return out[::-1]

    def to_dict(self) -> dict[str, Any]:
        return {"root": self.root, "nodes": [asdict(n) for n in self.nodes.values()]}

    def render(self, *, pruned_per_node: int = 2, max_lines: int = 200, width: int = 70) -> str:
        """The tree as text: read units with their order and value, the best pruned ones."""
        lines: list[str] = []

        def label(n: RAGNode) -> str:
            if n.node_type in ("QUERY", "SUBQUERY"):
                return f"{n.node_type} {n.query}"
            if n.node_type == "SEARCH":
                return f"{n.action}({n.query}) -> {n.observation}"
            if n.node_type == "REWRITE":
                return f"REWRITE_QUERY -> {n.query!r}"
            if n.node_type == "VERIFY":
                cov = "" if n.coverage is None else f" coverage={n.coverage:.2f}"
                miss = f" missing={n.observation!r}" if n.observation else ""
                return f"VERIFY answerable={n.answerability}{cov}{miss}"
            if n.node_type == "ANSWER":
                return f"ANSWER {n.observation[:width]}"
            v = "" if n.relevance is None else f"[{n.relevance:.2f}] "
            state = f"READ #{n.read}" if n.read else "PRUNED"
            where = f" p.{n.page}" if n.page is not None else ""
            text = " ".join(n.observation.split())[:width]
            mc = f" (n={n.visits}, q={n.value:.2f})" if n.visits and n.value is not None else ""
            return f"{v}{state} {n.action}{where}{mc} {text!r}"

        def walk(nid: str, prefix: str, last: bool, top: bool) -> None:
            if len(lines) >= max_lines:
                return
            n = self.nodes[nid]
            lines.append(label(n) if top else f"{prefix}{'└── ' if last else '├── '}{label(n)}")
            kids = [self.nodes[c] for c in n.children]
            shown = [k for k in kids if not k.pruned]
            pruned = sorted((k for k in kids if k.pruned), key=lambda k: -(k.relevance or 0.0))
            shown += pruned[:pruned_per_node]
            hidden = len(kids) - len(shown)
            nxt = prefix if top else prefix + ("    " if last else "│   ")
            for i, k in enumerate(shown):
                walk(k.id, nxt, i == len(shown) - 1 and not hidden, False)
            if hidden and len(lines) < max_lines:
                lines.append(f"{nxt}└── … {hidden} more pruned")

        if self.root:
            walk(self.root, "", True, True)
        return "\n".join(lines)


__all__ = ["UNIT_ACTIONS", "RAGNode", "SearchTree"]
