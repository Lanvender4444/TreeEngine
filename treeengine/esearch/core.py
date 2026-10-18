"""Evidence Search: state, actions, value function and the three search policies."""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from ..context.builder import BlockContextBuilder, Item, as_anchors
from ..context.span import ContextSpan
from ..core.text import query_terms

EVAL_SYSTEM = "You check whether evidence answers a question. Reply with JSON only."
EVAL_PROMPT = """Question: {question}

Evidence read so far:
{evidence}

Can the question be answered fully and exactly from this evidence? Estimate which share of the
facts the question needs is present ("coverage", 0 to 1). If something is missing, write the
single most important missing piece as a short search query in the document's own words (the
statement, table, note or line item that would contain it).
Reply as JSON: {{"answerable": true, "coverage": 1.0, "missing": ""}}
"""


_NUMERIC = re.compile(r"^[\s$(]*-?[\d,.]+%?\)?\s*$")


def _is_table(b: Any) -> bool:
    """A table block (V0.16 table-aware parsing), or a block that is mostly numbers."""
    if getattr(b, "block_type", "") == "table":
        return True
    cells = [c for c in re.split(r"\s{2,}|\n", b.content or "") if c.strip()]
    return len(cells) >= 8 and sum(1 for c in cells if _NUMERIC.match(c)) / len(cells) >= 0.4


@dataclass
class Unit:
    """A reading unit: blocks [lo, hi] of one document, created by an action."""

    key: str
    doc: str
    lo: int
    hi: int
    kind: str  # hit | neighbor | parent | page | rewrite
    origin: str
    parent: str | None
    tokens: int
    text: str
    value: float = 0.0
    prior: float = 0.0  # retrieval confidence (pool) or inherited from the unit it expands


@dataclass(frozen=True)
class SearchSpec:
    algo: str = "best_first"  # best_first | beam | mcts
    reranker: str = "ce:Xenova/ms-marco-MiniLM-L-12-v2"
    seeds: int = 16  # pool candidates turned into units
    window: int = 220  # tokens of a hit unit (section-bounded)
    expand: bool = True  # READ_NEIGHBORS / READ_PARENT / READ_PAGE children
    page_max: int = 900  # a page unit when the page fits in this many tokens
    stop_rel: float = 0.35  # STOP when the best value left < stop_rel * best seed value
    lam: float = 0.15  # token-cost weight (share of the budget)
    fill: bool = False  # after STOP, fill the rest of the budget in pool order
    beam: int = 3
    branch: int = 5
    iters: int = 48
    c_uct: float = 0.8
    seed: int = 7
    prior: float = 0.0  # V0.16b: weight of the retrieval confidence prior in a unit's value
    loop: bool = False  # LLM evaluation (VERIFY) + REWRITE_QUERY
    loop_share: float = 0.7  # budget of the first pass when looping
    # V0.17 Evidence Tree actions beyond neighbours / section head / page, comma separated:
    # child (READ_CHILD: the opening of each sub-section), table (READ_TABLE: table blocks on
    # the same or adjacent page), entity (FOLLOW_ENTITY: Graph-lite, blocks sharing the unit's
    # rarest entities), section (SEARCH_IN_SECTION: BM25 inside the enclosing section)
    actions: str = ""
    max_steps: int = 0  # READ actions per search (0 = budget only)
    loop_rounds: int = 1  # VERIFY -> REWRITE_QUERY rounds when looping


SEARCH_MODES: dict[str, SearchSpec] = {
    "search": SearchSpec(),
    "search_fill": SearchSpec(fill=True),
    "search_beam": SearchSpec(algo="beam"),
    "search_mcts": SearchSpec(algo="mcts"),
    "search_loop": SearchSpec(loop=True, fill=True),
    # value = (1 - prior) * reranker + prior * retrieval confidence; looser stop, cheap cost
    "search_prior": SearchSpec(prior=0.5, stop_rel=0.2, lam=0.05, fill=True),
    "search_mcts_prior": SearchSpec(algo="mcts", prior=0.5, stop_rel=0.2, lam=0.05, fill=True),
    # V0.17: the full Evidence Tree action space (+ child / table / entity / in-section search)
    "search_tree": SearchSpec(
        prior=0.5, stop_rel=0.2, lam=0.05, fill=True, actions="child,table,entity,section"
    ),
    "search_tree_mcts": SearchSpec(
        algo="mcts",
        prior=0.5,
        stop_rel=0.2,
        lam=0.05,
        fill=True,
        actions="child,table,entity,section",
    ),
}

_KEYS: dict[str, tuple[str, Callable[[str], Any]]] = {
    "algo": ("algo", str),
    "rr": ("reranker", str),
    "seeds": ("seeds", int),
    "win": ("window", int),
    "exp": ("expand", lambda v: v not in ("0", "false")),
    "stop": ("stop_rel", float),
    "lam": ("lam", float),
    "fill": ("fill", lambda v: v not in ("0", "false")),
    "beam": ("beam", int),
    "iters": ("iters", int),
    "loop": ("loop", lambda v: v not in ("0", "false")),
    "prior": ("prior", float),
    "act": ("actions", lambda v: v.replace("+", ",")),
    "steps": ("max_steps", int),
    "rounds": ("loop_rounds", int),
}


def parse_search_mode(name: str) -> SearchSpec | None:
    base, _, params = name.partition("@")
    spec = SEARCH_MODES.get(base)
    if spec is None:
        return None
    kw: dict[str, Any] = {}
    for kv in filter(None, params.split(";")):
        k, _, v = kv.partition("=")
        if k.strip() not in _KEYS:
            raise ValueError(f"unknown search parameter {k!r} in {name!r}")
        f, conv = _KEYS[k.strip()]
        kw[f] = conv(v.strip())
    return replace(spec, **kw)


@dataclass
class EvidenceSearchResult:
    """What ``TreeEngine.evidence_search`` returns: the evidence set found (as ContextSpans)
    plus the search itself - the actions taken, the explored search tree and, for MCTS, the
    most visited states."""

    query: str
    algo: str
    spans: list[ContextSpan]
    trace: list[dict[str, Any]]  # actions in the order they were taken
    tree: list[dict[str, Any]]  # explored units: action, parent, value, read order
    states: list[dict[str, Any]]  # MCTS: evidence sets with visits / mean reward
    stats: dict[str, Any]  # read units, tokens, scored, coverage, llm calls, ...
    candidates: int = 0  # retrieval candidates that seeded the search

    @property
    def text(self) -> str:
        return "\n\n".join(s.content for s in self.spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "algo": self.algo,
            "spans": [asdict(s) for s in self.spans],
            "trace": self.trace,
            "tree": self.tree,
            "states": self.states,
            "stats": self.stats,
            "candidates": self.candidates,
        }


@dataclass
class _State:
    read: list[str] = field(default_factory=list)  # unit keys in reading order
    blocks: set[tuple[str, int]] = field(default_factory=set)
    used: int = 0


class EvidenceSearch:
    def __init__(
        self,
        builder: BlockContextBuilder,
        spec: SearchSpec | None = None,
        *,
        retrieve: Callable[[str], Sequence[Any]] | None = None,
        llm: Any = None,
        raptor: Any = None,
    ) -> None:
        self.b = builder
        self.raptor = raptor  # treerag.raptor.RaptorIndex: READ_SUMMARY units (V0.17)
        self.summary_of: dict[str, Any] = {}  # unit key -> SummaryNode
        self.spec = spec or SearchSpec()
        self.retrieve = retrieve
        self.llm = llm
        self.units: dict[str, Unit] = {}
        self.question = ""
        self.terms: list[str] = []
        self.best0 = 1.0
        self.scored = 0
        self.llm_calls = 0
        self.llm_in = 0
        self.llm_out = 0
        self.trace: list[dict[str, Any]] = []
        self._kids: dict[str, list[Unit]] = {}
        self._mcts_stats: dict[frozenset[str], list[float]] = {}
        self._read: list[str] = []
        self.last_coverage: float | None = None
        self.verifications: list[dict[str, Any]] = []
        self._maps: dict[str, tuple[dict[Any, list[int]], dict[Any, list[int]]]] = {}

    def _doc_maps(self, doc: str) -> tuple[dict[Any, list[int]], dict[Any, list[int]]]:
        """(node id -> block indexes, page -> block indexes), once per document."""
        if doc not in self._maps:
            by_node: dict[Any, list[int]] = {}
            by_page: dict[Any, list[int]] = {}
            for k, x in enumerate(self.b.blocks(doc)):
                by_node.setdefault(x.node_id, []).append(k)
                by_page.setdefault(x.page, []).append(k)
            self._maps[doc] = (by_node, by_page)
        return self._maps[doc]

    # ------------------------------------------------------------------ units (actions)
    def _unit(self, doc: str, lo: int, hi: int, kind: str, origin: str, parent: str | None) -> Unit:
        key = f"{doc}:{lo}-{hi}"
        if key in self.units:
            return self.units[key]
        blocks = self.b.blocks(doc)[lo : hi + 1]
        title = str((blocks[0].metadata or {}).get("index_title") or "") if blocks else ""
        body = "\n".join(x.content.strip() for x in blocks)
        text = ((title + "\n") if title else "") + body
        u = Unit(
            key, doc, lo, hi, kind, origin, parent, sum(self.b.tokens(x) for x in blocks), text
        )
        self.units[key] = u
        return u

    def _window(self, doc: str, i: int, size: int) -> tuple[int, int]:
        blocks = self.b.blocks(doc)
        node = blocks[i].node_id
        lo = hi = i
        used = self.b.tokens(blocks[i])
        while used < size:
            grew = False
            if hi + 1 < len(blocks) and blocks[hi + 1].node_id == node:
                t = self.b.tokens(blocks[hi + 1])
                if used + t <= size * 1.25:
                    hi, used, grew = hi + 1, used + t, True
            if used < size and lo - 1 >= 0 and blocks[lo - 1].node_id == node:
                t = self.b.tokens(blocks[lo - 1])
                if used + t <= size * 1.25:
                    lo, used, grew = lo - 1, used + t, True
            if not grew:
                break
        return lo, hi

    def seed_units(self, pool: Sequence[Item], origin: str = "pool") -> list[Unit]:
        out: list[Unit] = []
        for a in as_anchors(pool)[: self.spec.seeds]:
            bid = a.block_id or a.metadata.get("focus_block_id")
            if not bid:
                continue
            self.b.blocks(a.document_id)
            where = self.b._where.get(str(bid))
            if where is None:
                continue
            lo, hi = self._window(a.document_id, where[1], self.spec.window)
            u = self._unit(a.document_id, lo, hi, "hit", origin + ":" + (a.source or ""), None)
            from ..reason.gate import conf_of

            c = conf_of(a)
            u.prior = max(u.prior, c if c > 0 else 1.0 / (1 + len(out)))
            if u not in out:
                out.append(u)
        if self.raptor is not None and origin == "pool":
            for doc in dict.fromkeys(a.document_id for a in as_anchors(pool)):
                if self.raptor.has(doc):
                    top = max(n.level for n in self.raptor.nodes(doc, 1))
                    for n in self.raptor.nodes(doc, max(1, top - 1)):
                        out.append(self._summary_unit(n, None))
        return out

    def _summary_unit(self, n: Any, parent: str | None) -> Unit:
        key = f"{n.document_id}:S:{n.id}"
        if key not in self.units:
            u = Unit(
                key,
                n.document_id,
                n.block_lo,
                n.block_hi,
                "summary",
                f"L{n.level}",
                parent,
                n.tokens,
                n.summary,
            )
            u.prior = 0.3
            self.units[key] = u
            self.summary_of[key] = n
        return self.units[key]

    def children(self, u: Unit) -> list[Unit]:
        """READ_NEIGHBORS (after / before), READ_PARENT (section head), READ_PAGE."""
        if not self.spec.expand:
            return []
        if u.key in self._kids:
            return self._kids[u.key]
        if u.kind == "summary":  # READ_SUMMARY: lower summaries, or the leaf windows
            sn = self.summary_of[u.key]
            by_id = {n.id: n for n in self.raptor.nodes(u.doc)}
            kids = []
            for cid in sn.children:
                c = by_id.get(cid)
                if c is None:
                    continue
                if c.level >= 1:
                    kids.append(self._summary_unit(c, u.key))
                else:
                    kids.append(self._unit(u.doc, c.block_lo, c.block_hi, "leaf", c.id, u.key))
            for c in kids:
                c.prior = max(c.prior, u.prior)
            self._kids[u.key] = kids
            return kids
        blocks = self.b.blocks(u.doc)
        by_node, by_page = self._doc_maps(u.doc)
        out: list[Unit] = []
        if u.hi + 1 < len(blocks):
            lo, hi = self._window(u.doc, u.hi + 1, self.spec.window)
            lo = max(lo, u.hi + 1)
            out.append(self._unit(u.doc, lo, hi, "neighbor", "next", u.key))
        if u.lo - 1 >= 0:
            lo, hi = self._window(u.doc, u.lo - 1, self.spec.window)
            hi = min(hi, u.lo - 1)
            out.append(self._unit(u.doc, lo, hi, "neighbor", "prev", u.key))
        node = blocks[u.lo].node_id
        idx = by_node.get(node, []) if node else []
        if idx and idx[0] < u.lo:
            lo, hi = self._window(u.doc, idx[0], self.spec.window)
            out.append(self._unit(u.doc, lo, min(hi, u.lo - 1), "parent", "section_head", u.key))
        page = blocks[u.lo].page
        if page is not None:
            pidx = by_page.get(page, [])
            ptok = sum(self.b.tokens(blocks[k]) for k in pidx)
            if pidx and ptok <= self.spec.page_max and (pidx[0], pidx[-1]) != (u.lo, u.hi):
                out.append(self._unit(u.doc, pidx[0], pidx[-1], "page", "page", u.key))
        acts = {a.strip() for a in self.spec.actions.split(",") if a.strip()}
        if acts:
            out += self._more_children(u, acts, blocks, by_node, by_page)
        seen: set[str] = set()
        kids = []
        for c in out:
            if c.lo <= c.hi and c.key != u.key and c.key not in seen:
                seen.add(c.key)
                kids.append(c)
        for c in kids:
            c.prior = max(c.prior, 0.8 * u.prior)
        self._kids[u.key] = kids
        return kids

    def _more_children(
        self,
        u: Unit,
        acts: set[str],
        blocks: Sequence[Any],
        by_node: dict[Any, list[int]],
        by_page: dict[Any, list[int]],
    ) -> list[Unit]:
        """V0.17 Evidence Tree actions: READ_CHILD, READ_TABLE, FOLLOW_ENTITY,
        SEARCH_IN_SECTION. Each child unit is a window around one block."""
        repo: Any = self.b.repo  # SQLite repository: Graph-lite and section-scoped BM25
        out: list[Unit] = []
        inside = set(range(u.lo, u.hi + 1))
        node = blocks[u.lo].node_id

        def win(i: int, kind: str, origin: str) -> None:
            if i in inside:
                return
            lo, hi = self._window(u.doc, i, self.spec.window)
            if blocks[i].block_type == "table":
                lo = hi = i  # a table block is read whole, alone
            out.append(self._unit(u.doc, lo, hi, kind, origin, u.key))

        if "child" in acts and node:
            for ch in repo.get_children(str(node))[:4]:
                idx = by_node.get(ch.id)
                if idx:
                    win(idx[0], "child", ch.title or "child")
        if "table" in acts:
            page = blocks[u.lo].page
            if page is not None:
                near = [k for p in (page, page + 1, page - 1) for k in by_page.get(p, [])]
                tables = [k for k in near if _is_table(blocks[k])][:3]
                for k in tables:
                    win(k, "table", f"p{blocks[k].page}")
        if "entity" in acts and hasattr(repo, "block_entities"):
            ids = [blocks[k].id for k in range(u.lo, u.hi + 1)]
            ents = [e for v in repo.block_entities(ids).values() for e in v]
            ents = sorted({e["id"]: e for e in ents}.values(), key=lambda e: e["df"])
            where = self.b._where
            for e in ents[:2]:
                for bid, doc, _ in repo.entity_blocks(e["id"], document_id=u.doc, limit=4):
                    loc = where.get(bid)
                    if doc == u.doc and loc is not None:
                        win(loc[1], "entity", str(e["name"]))
        if "section" in acts and node and self.terms and hasattr(repo, "fts_query"):
            n = repo.get_node(str(node))
            scope = n.parent_id if n is not None and n.parent_id else str(node)
            ids = [scope, *repo.get_subtree_ids(scope)]
            where = self.b._where
            for b, _ in repo.fts_query(self.terms, document_id=u.doc, node_ids=ids, limit=4):
                loc = where.get(b.id)
                if loc is not None:
                    win(loc[1], "section_search", "bm25")
        return out

    # ------------------------------------------------------------------ value function
    def score(self, units: Sequence[Unit]) -> None:
        todo = [u for u in units if u.value == 0.0 and u.text.strip()]
        if not todo:
            return
        from ..retrieval.rerank import rerank_scores

        raw = rerank_scores(self.spec.reranker, self.question, [u.text[:1500] for u in todo])
        self.scored += len(todo)
        kind = self.spec.reranker.partition(":")[0]
        for u, s in zip(todo, raw, strict=True):
            if kind == "ce":
                u.value = 1.0 / (1.0 + math.exp(-s))
                if self.spec.prior > 0:
                    u.value = (1 - self.spec.prior) * u.value + self.spec.prior * u.prior
            else:  # late interaction: unbounded, per query-term average similarity
                u.value = max(1e-6, s / max(1, len(self.terms) + 2))
        if kind != "ce":  # put ColBERT scores on [0, 1] by the best seen so far
            top = max(x.value for x in self.units.values()) or 1.0
            for x in self.units.values():
                if x.value > 0:
                    x.value = min(1.0, x.value / top)

    def _blocks_of(self, u: Unit) -> set[tuple[str, int]]:
        if u.kind == "summary":  # a summary adds its own text, it does not read the blocks
            return set()
        return {(u.doc, k) for k in range(u.lo, u.hi + 1)}

    def _new_tokens(self, u: Unit, st: _State) -> int:
        if u.kind == "summary":
            return 0 if u.key in st.read else u.tokens
        blocks = self.b.blocks(u.doc)
        return sum(
            self.b.tokens(blocks[k]) for k in range(u.lo, u.hi + 1) if (u.doc, k) not in st.blocks
        )

    def priority(self, u: Unit, st: _State, budget: int) -> float:
        nb = self._blocks_of(u)
        novelty = 1.0 - len(nb & st.blocks) / max(1, len(nb))
        return u.value * novelty - self.spec.lam * self._new_tokens(u, st) / max(1, budget)

    def coverage(self, st: _State) -> float:
        if not self.terms:
            return 0.0
        text = " ".join(self.units[k].text.lower() for k in st.read)
        return sum(1 for t in self.terms if t in text) / len(self.terms)

    def reward(self, st: _State, budget: int) -> float:
        """Relevance (diminishing) + coverage - cost - redundancy, on ~[0, 1]."""
        if not st.read:
            return 0.0
        vals = sorted((self.units[k].value for k in st.read), reverse=True)
        rel = sum(v * (0.6**i) for i, v in enumerate(vals)) / (self.best0 * 2.5 or 1.0)
        size = sum(self.units[k].tokens for k in st.read)
        redundancy = max(0.0, 1.0 - st.used / max(1, size))
        return (
            rel
            + 0.3 * self.coverage(st)
            - self.spec.lam * st.used / max(1, budget)
            - 0.2 * redundancy
        )

    # ------------------------------------------------------------------ transitions
    def _apply(self, st: _State, u: Unit) -> _State:
        nt = self._new_tokens(u, st)
        return _State(st.read + [u.key], st.blocks | self._blocks_of(u), st.used + nt)

    def _frontier(self, st: _State, seeds: Sequence[Unit]) -> list[Unit]:
        cand = {u.key: u for u in seeds}
        for k in st.read:
            for c in self.children(self.units[k]):
                cand.setdefault(c.key, c)
        out = [u for u in cand.values() if u.key not in st.read]
        self.score(out)
        return out

    def _ok(self, u: Unit, st: _State, budget: int) -> bool:
        if self.spec.max_steps and len(st.read) >= self.spec.max_steps:
            return False
        nt = self._new_tokens(u, st)
        return nt > 0 and st.used + nt <= budget and u.value >= self.spec.stop_rel * self.best0

    def _ranked(self, st: _State, seeds: Sequence[Unit], budget: int) -> list[Unit]:
        cands = [u for u in self._frontier(st, seeds) if self._ok(u, st, budget)]
        return sorted(cands, key=lambda u: -self.priority(u, st, budget))

    # ------------------------------------------------------------------ policies
    def best_first(self, st: _State, seeds: Sequence[Unit], budget: int) -> _State:
        while True:
            ranked = self._ranked(st, seeds, budget)
            if not ranked:
                return st
            st = self._apply(st, ranked[0])
            self.trace.append(
                {"read": ranked[0].key, "kind": ranked[0].kind, "v": round(ranked[0].value, 3)}
            )

    def beam_search(self, st: _State, seeds: Sequence[Unit], budget: int) -> _State:
        beams = [st]
        done: list[_State] = []
        for _ in range(64):
            nxt: list[_State] = []
            for s in beams:
                ranked = self._ranked(s, seeds, budget)[: self.spec.branch]
                if not ranked:
                    done.append(s)
                    continue
                nxt += [self._apply(s, u) for u in ranked]
            if not nxt:
                break
            uniq = {frozenset(s.read): s for s in nxt}
            beams = sorted(uniq.values(), key=lambda s: -self.reward(s, budget))[: self.spec.beam]
        done += beams
        return max(done, key=lambda s: self.reward(s, budget))

    def mcts(self, st: _State, seeds: Sequence[Unit], budget: int) -> _State:
        rng = random.Random(self.spec.seed)
        stats: dict[frozenset[str], list[float]] = {}  # state -> [visits, total reward]
        self._mcts_stats = stats
        tried: dict[frozenset[str], list[str]] = {}
        STOP = "__stop__"

        def actions(s: _State) -> list[str]:
            return [u.key for u in self._ranked(s, seeds, budget)[: self.spec.branch]] + [STOP]

        def rollout(s: _State) -> float:
            while True:
                ranked = self._ranked(s, seeds, budget)[:3]
                if not ranked or rng.random() < 0.1:
                    return self.reward(s, budget)
                s = self._apply(s, rng.choice(ranked))

        for _ in range(self.spec.iters):
            s, path = st, [frozenset(st.read)]
            # selection / expansion
            while True:
                key = frozenset(s.read)
                acts = tried.setdefault(key, actions(s))
                untried = [
                    a
                    for a in acts
                    if (a == STOP and key | {STOP} not in stats)
                    or (a != STOP and frozenset(s.read + [a]) not in stats)
                ]
                if untried:
                    a = untried[0]
                    if a == STOP:
                        path.append(key | {STOP})
                        r = self.reward(s, budget)
                        break
                    s = self._apply(s, self.units[a])
                    path.append(frozenset(s.read))
                    r = rollout(s)
                    break
                n = stats.get(key, [1, 0])[0]

                def uct(a: str, s: _State = s, n: float = n) -> float:
                    k = (frozenset(s.read) | {STOP}) if a == STOP else frozenset(s.read + [a])
                    v, w = stats.get(k, [1e-9, 0.0])
                    return w / v + self.spec.c_uct * math.sqrt(math.log(n + 1) / v)

                a = max(acts, key=uct)
                if a == STOP:
                    path.append(key | {STOP})
                    r = self.reward(s, budget)
                    break
                s = self._apply(s, self.units[a])
                path.append(frozenset(s.read))
            # backpropagation
            for k in path:
                v = stats.setdefault(k, [0, 0.0])
                v[0] += 1
                v[1] += r
        # final: follow the most visited actions
        s = st
        while True:
            key = frozenset(s.read)
            acts = tried.get(key, [])
            best, best_n = None, 0.0
            for a in acts:
                k = (key | {STOP}) if a == STOP else frozenset(s.read + [a])
                n = stats.get(k, [0, 0])[0]
                if n > best_n:
                    best, best_n = a, n
            if best == STOP:
                return s
            if best is None:
                # the tree is only a few levels deep after `iters` simulations: continue from
                # its most visited leaf with the default (greedy) policy, as the rollouts did
                return self.best_first(s, seeds, budget)
            s = self._apply(s, self.units[best])
            self.trace.append({"read": best, "visits": best_n})

    # ------------------------------------------------------------------ evaluation loop
    def evaluate(self, st: _State) -> tuple[bool, str]:
        ev = "\n\n".join(f"[{i + 1}] {self.units[k].text[:700]}" for i, k in enumerate(st.read))
        prompt = EVAL_PROMPT.format(question=self.question, evidence=ev or "(none)")
        reply = self.llm.complete(prompt, system=EVAL_SYSTEM, max_tokens=120)
        self.llm_calls += 1
        self.llm_in += len(prompt) // 4
        self.llm_out += len(reply) // 4
        m = re.search(r"\{.*\}", reply or "", re.S)
        try:
            import json

            data = json.loads(m.group(0)) if m else {}
        except ValueError:
            data = {}
        ok = bool(data.get("answerable")) if isinstance(data, dict) else False
        missing = str(data.get("missing") or "").strip() if isinstance(data, dict) else ""
        try:
            cov = data.get("coverage") if isinstance(data, dict) else None
            self.last_coverage = float(cov) if cov is not None else None
        except (TypeError, ValueError):
            self.last_coverage = None
        return ok, missing

    # ------------------------------------------------------------------ run
    def run(
        self, question: str, pool: Sequence[Item], budget: int
    ) -> tuple[list[ContextSpan], dict[str, Any]]:
        self.question = question
        self.terms = [t for t in query_terms(question) if len(t) > 2]
        seeds = self.seed_units(pool)
        self.score(seeds)
        self.best0 = max((u.value for u in seeds), default=1.0) or 1.0
        algo = {"best_first": self.best_first, "beam": self.beam_search, "mcts": self.mcts}[
            self.spec.algo
        ]
        looping = self.spec.loop and self.llm is not None
        rounds = max(1, self.spec.loop_rounds) if looping else 0
        share = self.spec.loop_share
        first = int(budget * share) if looping else budget
        st = algo(_State(), seeds, first)
        info: dict[str, Any] = {"algo": self.spec.algo}
        self.verifications = []
        for r in range(rounds):
            ok, missing = self.evaluate(st)
            self.verifications.append(
                {
                    "round": r + 1,
                    "answerable": ok,
                    "coverage": self.last_coverage,
                    "missing": missing,
                    "read": len(st.read),
                    "tokens": st.used,
                }
            )
            self.trace.append({"verify": ok, "missing": missing})
            info["answerable"] = ok
            if ok:
                break
            if missing and self.retrieve is not None:
                info.setdefault("rewrites", []).append(missing)
                info["rewrite"] = missing
                self.trace.append({"rewrite": missing})
                more = self.seed_units(list(self.retrieve(missing))[:10], origin="rewrite")
                self.score(more)
                seeds = list(seeds) + more
            last = r == rounds - 1
            cap = budget if last else int(budget * (share + (1 - share) * (r + 1) / rounds))
            st = algo(st, seeds, cap)
        if looping:
            info["verify"] = self.verifications
        units = [self.units[k] for k in st.read]
        self._read = list(st.read)
        spans = self._pack(units, pool, budget)
        info.update(
            {
                "read": len(units),
                "kinds": {
                    k: sum(1 for u in units if u.kind == k)
                    for k in ("hit", "neighbor", "parent", "page")
                },
                "tokens": st.used,
                "scored": self.scored,
                "coverage": round(self.coverage(st), 3),
                "llm_calls": self.llm_calls,
                "selector_input_tokens": self.llm_in,
                "selector_output_tokens": self.llm_out,
                "best0": round(self.best0, 3),
            }
        )
        return spans, info

    # ------------------------------------------------------------------ inspection
    def search_tree(self, limit: int = 200) -> list[dict[str, Any]]:
        """The explored retrieval search tree after ``run``: every unit that was created and
        scored, with the action that created it (``kind``), the unit it expanded (``parent``),
        its value and whether it was read (and in which order)."""
        order = {k: i for i, k in enumerate(self._read, 1)}
        rows = []
        for u in self.units.values():
            rows.append(
                {
                    "unit": u.key,
                    "parent": u.parent,
                    "action": u.kind,
                    "origin": u.origin,
                    "value": round(u.value, 4),
                    "prior": round(u.prior, 4),
                    "tokens": u.tokens,
                    "read": order.get(u.key),
                    "preview": u.text[:120],
                }
            )
        rows.sort(key=lambda r: (r["read"] is None, int(r["read"] or 0), -float(r["value"] or 0.0)))
        return rows[:limit]

    def mcts_states(self, limit: int = 50) -> list[dict[str, Any]]:
        """MCTS only: the most visited states (evidence sets) with visits and mean reward."""
        out = []
        for key, (n, w) in self._mcts_stats.items():
            if n <= 0:
                continue
            out.append(
                {
                    "state": sorted(k for k in key if k != "__stop__"),
                    "stop": "__stop__" in key,
                    "visits": int(n),
                    "mean_reward": round(w / n, 4),
                }
            )
        out.sort(key=lambda r: -int(r["visits"]))  # type: ignore[call-overload]
        return out[:limit]

    def _pack(self, units: Sequence[Unit], pool: Sequence[Item], budget: int) -> list[ContextSpan]:
        from ..context.builder import resolve_policy
        from ..reason.reader import ReasonedReader

        reader = ReasonedReader(self.b)
        summaries = [self.summary_span(u) for u in units if u.kind == "summary"]
        budget -= sum(sp.token_count for sp in summaries)
        rows = [
            (
                u.doc,
                u.lo,
                u.hi,
                f"es:{u.key}",
                "section" if u.kind in ("page", "parent") else "window",
            )
            for u in units
            if u.kind != "summary"
        ]
        if self.spec.fill:
            pol = resolve_policy("auto", budget)
            for a in as_anchors(pool):
                if a.kind == "block" and a.block_id:
                    iv = self.b.interval(a, pol)
                    if iv:
                        rows.append((*iv, str(a.block_id), "window"))
        return summaries + reader._pack(rows, budget, self.b.allow_first_overflow)

    def summary_span(self, u: Unit) -> ContextSpan:
        """A read summary as a pass-through ContextSpan (no blocks of its own)."""
        blocks = self.b.blocks(u.doc)[u.lo : u.hi + 1]
        pages = sorted({b.page for b in blocks if b.page is not None})
        sn = self.summary_of.get(u.key)
        head = f"[Summary, level {sn.level if sn else '?'}"
        head += f", pages {pages[0]}-{pages[-1]}]" if pages else "]"
        return ContextSpan(
            document_id=u.doc,
            node_id=None,
            block_ids=[],
            content=f"{head}\n{u.text}",
            page_start=pages[0] if pages else None,
            page_end=pages[-1] if pages else None,
            start_offset=None,
            end_offset=None,
            source_evidence_ids=[f"es:{u.key}"],
            token_count=u.tokens + 12,
            pages=pages,
            resolution="summary",
        )


__all__ = [
    "EVAL_PROMPT",
    "EvidenceSearchResult",
    "EVAL_SYSTEM",
    "SEARCH_MODES",
    "EvidenceSearch",
    "SearchSpec",
    "Unit",
    "parse_search_mode",
]
