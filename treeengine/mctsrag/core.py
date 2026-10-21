"""MCTS-RAG: Monte Carlo tree search over retrieval *operators*, with LLM thinking as actions.

Tree-RAG (V0.17) searched over reading units of one candidate list. MCTS-RAG searches one level
up: over what to retrieve next. A state is the sequence of operators applied so far; replaying
it gives the question set (the question, sub-questions, rewritten queries), the candidate pool
and what the LLM concluded. Actions:

    SEARCH_BM25(q)    FTS5 BM25 over blocks for one question of the set
    SEARCH_VECTOR(q)  embedding search
    SEARCH_TREE(q)    the document tree: sections located by their titles / content
    SEARCH_FUSED(q)   the engine's fused + gated fast path (the recommended policy)
    EXPAND            neighbours of the most valuable block not expanded yet
    THINK_DECOMPOSE   LLM: the question -> sub-questions (new targets and queries)
    THINK_VERIFY      LLM judge over the best evidence of the state: answerable / coverage /
                      the missing fact -> a REWRITE query; its coverage becomes the state's value

Value of a state (no rollouts - a learned / cheap estimate at the leaf, as in AlphaZero-style
search): for every target question (the question + sub-questions) the best cross-encoder score
of the pool (sigmoid), the question weighted ``w_main``, the sub-questions sharing the rest;
blended with the LLM judge's coverage where one ran (``w_judge``); minus a cost per action and
per LLM call. Selection is UCT with action priors (progressive widening: the untried action
with the highest prior first). LLM results are memoised across branches - a decomposition is
paid once - and limited by ``max_llm`` distinct calls.

The best state's pool is then read once by the confidence gate (accepted candidates + one
PageIndex-style LLM selection over the catalog and the page outline), with the sub-questions
and the missing fact in the reading question. Answer-level retry stays a separate step.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..context import BlockContextBuilder
from ..context.builder import as_anchors
from ..context.span import ContextSpan
from ..core.anchor import BlockAnchor

Operator = Callable[[str, str | None], Sequence[Any]]

SEARCH_OPS = ("fused", "bm25", "vector", "tree")
PRIORS = {
    "fused": 0.9,
    "bm25": 0.55,
    "vector": 0.55,
    "tree": 0.45,
    "expand": 0.35,
    "decompose": 0.7,
    "verify": 0.6,
}


@dataclass
class MCTSRAGSpec:
    iters: int = 24
    depth: int = 4  # actions after the root
    c_uct: float = 0.8
    max_llm: int = 2  # THINK_* calls (the gate reading and the answer are extra)
    per_search: int = 10  # candidates kept from one SEARCH
    pool_max: int = 80
    reranker: str = "ce:Xenova/ms-marco-MiniLM-L-12-v2"
    score_top: int = 12  # new candidates scored by the reranker per SEARCH
    w_main: float = 0.6
    w_judge: float = 0.5
    lam: float = 0.01  # cost per action
    llm_cost: float = 0.03  # extra cost per THINK action
    ops: tuple[str, ...] = ("fused", "bm25", "vector", "tree", "expand", "decompose", "verify")
    reader: str = "gate"  # gate | order (the pool in value order, no LLM)
    max_subqueries: int = 3

    def __post_init__(self) -> None:
        bad = set(self.ops) - set(PRIORS)
        if bad:
            raise ValueError(f"unknown MCTS-RAG operators {sorted(bad)}; one of {sorted(PRIORS)}")


@dataclass(frozen=True)
class Action:
    op: str
    query: str = ""

    def label(self) -> str:
        name = {
            "decompose": "THINK_DECOMPOSE",
            "verify": "THINK_VERIFY",
            "expand": "EXPAND",
        }.get(self.op, f"SEARCH_{self.op.upper()}")
        return f"{name}({self.query[:60]})" if self.query else name


@dataclass
class State:
    actions: tuple[Action, ...]
    targets: list[str]  # questions the evidence must cover (question first)
    queries: list[str]  # queries that can be searched
    pool: list[Any]  # anchors, in arrival order
    keys: set[str]
    expanded: set[str]
    judge: dict[str, Any] | None = None
    llm: int = 0


@dataclass
class _Node:
    state: State
    parent: _Node | None
    action: Action | None
    untried: list[Action] = field(default_factory=list)
    children: list[_Node] = field(default_factory=list)
    visits: int = 0
    total: float = 0.0
    value: float = 0.0

    @property
    def mean(self) -> float:
        return self.total / self.visits if self.visits else 0.0


DECOMPOSE_SYSTEM = (
    "You split questions about documents into the sub-questions that must be looked up. "
    "Reply with JSON only."
)
DECOMPOSE_PROMPT = """Question: {question}
{title}
If answering needs several separate facts found in different places of the document (different
statements, years, notes or sections), list one short search question per fact, in the
document's own vocabulary (at most {n}). If one lookup is enough, return an empty list.
Reply as JSON: {{"sub_questions": ["...", "..."]}}
"""


class MCTSRAG:
    """``operators``: name -> ``fn(query, document_id) -> ranked anchors / evidence`` for any of
    fused / bm25 / vector / tree (missing ones are simply not offered)."""

    def __init__(
        self,
        repo: Any,
        operators: dict[str, Operator],
        llm: Any = None,
        *,
        spec: MCTSRAGSpec | None = None,
        answerer: Any = None,
        to_evidence: Callable[[list[ContextSpan]], list[Any]] | None = None,
        title_of: Callable[[str | None], str] | None = None,
        builder: BlockContextBuilder | None = None,
    ) -> None:
        self.repo = repo
        self.ops = dict(operators)
        self.llm = llm
        self.spec = spec or MCTSRAGSpec()
        self.answerer = answerer
        self.to_evidence = to_evidence or (lambda spans: [sp.to_evidence() for sp in spans])
        self.title_of = title_of or (lambda _d: "")
        self.builder = builder or BlockContextBuilder(repo, policy="auto")
        self.llm_calls = self.llm_in = self.llm_out = 0

    @classmethod
    def from_engine(
        cls, engine: Any, spec: MCTSRAGSpec | None = None, policy: Any = None
    ) -> MCTSRAG:
        from dataclasses import replace

        from ..retrieval.policy import RetrievalPolicy

        c = engine.components
        pol = policy or RetrievalPolicy.recommended()
        seed = replace(pol, context_policy="auto", context_budget=1)
        ops: dict[str, Operator] = {
            "fused": lambda q, d: list(
                c.controller.retrieve(q, document_id=d, policy=seed).anchors
            ),
            "bm25": lambda q, d: c.fts.search(q, document_id=d, limit=20),
            "tree": lambda q, d: c.tree.search(q, document_id=d, limit=10),
        }
        if c.vector is not None:
            ops["vector"] = lambda q, d: c.vector.search(q, document_id=d, limit=20)
        return cls(
            engine.repo,
            ops,
            engine.llm,
            spec=spec,
            answerer=c.answerer,
            to_evidence=engine._span_evidence,
            title_of=c.controller._title,
        )

    # ------------------------------------------------------------------ run
    def run(
        self,
        query: str,
        document_id: str | None = None,
        *,
        budget: int = 2000,
        seed: Sequence[Any] | None = None,
        answer: bool = True,
    ) -> Any:
        from ..treerag.search import TreeRAGResult

        t0 = time.perf_counter()
        sp = self.spec
        self.llm_calls = self.llm_in = self.llm_out = 0
        self.doc = document_id
        self.question = query
        self.title = self.title_of(document_id)
        self._search: dict[tuple[str, str], list[Any]] = {}
        self._decomp: list[str] | None = None
        self._verify: dict[tuple[str, ...], dict[str, Any]] = {}
        self._values: dict[tuple[Action, ...], float] = {}
        self.trace: list[dict[str, Any]] = []

        root_pool = [_anchor(x) for x in (seed if seed is not None else self._op("fused", query))]
        self._search[("fused", query)] = root_pool
        root_state = State((), [query], [query], [], set(), set())
        self._add(root_state, root_pool, limit=sp.pool_max)
        root = _Node(root_state, None, None)
        root.untried = self._actions(root_state)
        root.value = self._value(root_state)

        for _ in range(sp.iters):
            node = root
            # selection
            while not node.untried and node.children:
                node = max(node.children, key=lambda ch: self._uct(node, ch))
            # expansion
            if node.untried and len(node.state.actions) < sp.depth:
                act = node.untried.pop(0)
                st = self._apply(node.state, act)
                if st is None:  # unavailable (LLM budget spent): drop it
                    continue
                child = _Node(st, node, act)
                child.untried = self._actions(st)
                child.value = self._value(st)
                node.children.append(child)
                node = child
            # evaluation + backpropagation (the leaf's value estimate, no rollout)
            v = node.value
            while node is not None:
                node.visits += 1
                node.total += v
                node = node.parent  # type: ignore[assignment]

        best = self._best(root)
        st = best.state
        spans, read_info, read_calls = self._read(st, budget)
        # Evidence and the tree
        tree = self._tree(root, best, spans)
        result = None
        evidence = self.to_evidence(spans)
        if answer and self.answerer is not None:
            q = query
            subs = st.targets[1:]
            if subs:
                q += "\n(Sub-questions looked up: " + "; ".join(subs) + ")"
            result = self.answerer.answer(q, evidence)
        stats = {
            "iters": sp.iters,
            "nodes": _count(root),
            "best_path": [a.label() for a in st.actions],
            "best_value": round(best.value, 4),
            "pool": len(st.pool),
            "targets": st.targets,
            "llm_calls_think": self.llm_calls - read_calls,
            "llm_calls_retrieval": self.llm_calls,
            "llm_input_tokens_retrieval": self.llm_in,
            "llm_output_tokens_retrieval": self.llm_out,
            "searches": len(self._search),
            "context_tokens": sum(s.token_count for s in spans),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "read": read_info,
        }
        self.best_state = st
        return TreeRAGResult(
            query=query,
            strategy="mcts_rag",
            answer=result.answer if result else None,
            citations=list(result.citations) if result else [],
            query_tree=st.targets,
            search_tree=tree,
            evidence=evidence,
            context_spans=spans,
            trace=self.trace,
            stats=stats,
        )

    # ------------------------------------------------------------------ actions
    def _actions(self, st: State) -> list[Action]:
        sp = self.spec
        done = set(st.actions)
        acts: list[Action] = []
        for q in st.queries:
            for op in SEARCH_OPS:
                if op not in sp.ops or op not in self.ops:
                    continue
                if op == "fused" and q == self.question:
                    continue  # the root's own retrieval
                a = Action(op, q)
                if a not in done:
                    acts.append(a)
        if "expand" in sp.ops and self._expandable(st):
            acts.append(Action("expand"))
        llm_ok = self.llm is not None and st.llm < sp.max_llm
        if llm_ok and "decompose" in sp.ops and not any(a.op == "decompose" for a in st.actions):
            acts.append(Action("decompose"))
        if llm_ok and "verify" in sp.ops and sum(a.op == "verify" for a in st.actions) < 2:
            acts.append(Action("verify"))

        def prior(a: Action) -> float:
            p = PRIORS[a.op]
            if a.query and a.query != self.question:
                p += 0.1  # a new question: its first search is worth more
            return p

        return sorted(acts, key=lambda a: -prior(a))

    def _apply(self, st: State, act: Action) -> State | None:
        new = State(
            st.actions + (act,),
            list(st.targets),
            list(st.queries),
            list(st.pool),
            set(st.keys),
            set(st.expanded),
            dict(st.judge) if st.judge else None,
            st.llm,
        )
        if act.op in SEARCH_OPS:
            self._add(new, self._op(act.op, act.query))
        elif act.op == "expand":
            b = self._expandable(new)
            if b is None:
                return None
            new.expanded.add(b)
            self._add(new, self._neighbours(b))
        elif act.op == "decompose":
            subs = self._decompose()
            if subs is None:
                return None
            new.llm += 1
            for s in subs:
                if s not in new.targets:
                    new.targets.append(s)
                    new.queries.append(s)
        elif act.op == "verify":
            j = self._judge(new)
            if j is None:
                return None
            new.llm += 1
            new.judge = j
            miss = j.get("missing") or ""
            if miss and not j.get("answerable") and miss not in new.queries:
                new.queries.append(miss)
        return new

    def _op(self, op: str, q: str) -> list[Any]:
        key = (op, q)
        if key not in self._search:
            try:
                got = [_anchor(x) for x in self.ops[op](q, self.doc)]
            except Exception as e:  # noqa: BLE001 - an operator failing is a dead branch
                self.trace.append({"op": op, "query": q, "error": str(e)[:200]})
                got = []
            self._search[key] = got
            self.trace.append({"op": op, "query": q, "candidates": len(got)})
        return self._search[key]

    def _add(self, st: State, items: Sequence[Any], limit: int | None = None) -> None:
        limit = limit or self.spec.per_search
        n = 0
        for a in items:
            if getattr(a, "kind", "block") != "block" or not getattr(a, "block_id", None):
                continue
            k = str(a.block_id)
            if k in st.keys:
                continue
            st.keys.add(k)
            st.pool.append(a)
            n += 1
            if n >= limit or len(st.pool) >= self.spec.pool_max:
                break

    def _expandable(self, st: State) -> str | None:
        best = self._best_scores(st, st.pool)
        ranked = sorted(st.pool, key=lambda a: -best[str(a.block_id)])[:5]
        for a in ranked:
            if str(a.block_id) not in st.expanded:
                return str(a.block_id)
        return None

    def _neighbours(self, block_id: str) -> list[Any]:
        loc = self.builder.locate(block_id)
        if loc is None:
            return []
        doc, i = loc
        blocks = self.builder.blocks(doc)
        out = []
        for j in (i - 1, i + 1, i - 2, i + 2):
            if 0 <= j < len(blocks):
                out.append(BlockAnchor(doc, blocks[j].id, blocks[j].node_id, "expand"))
        return out

    # ------------------------------------------------------------------ LLM thinking
    def _complete(self, prompt: str, system: str, max_tokens: int) -> str:
        reply = self.llm.complete(prompt, system=system, max_tokens=max_tokens)
        self.llm_calls += 1
        self.llm_in += len(prompt) // 4
        self.llm_out += len(reply or "") // 4
        return str(reply or "")

    def _decompose(self) -> list[str] | None:
        if self._decomp is None:
            if self._distinct_llm() >= self.spec.max_llm:
                return None
            from ..treerag.search import parse_subquestions

            reply = self._complete(
                DECOMPOSE_PROMPT.format(
                    question=self.question,
                    title=f"Document: {self.title}\n" if self.title else "",
                    n=self.spec.max_subqueries,
                ),
                DECOMPOSE_SYSTEM,
                200,
            )
            subs = parse_subquestions(reply, self.spec.max_subqueries)
            self._decomp = subs if len(subs) > 1 else []
            self.trace.append({"op": "decompose", "sub_questions": self._decomp})
        return self._decomp

    def _judge(self, st: State) -> dict[str, Any] | None:
        top = self._top(st, 8)
        key = tuple(str(a.block_id) for a in top)
        if key not in self._verify:
            if self._distinct_llm() >= self.spec.max_llm:
                return None
            from ..esearch.core import EVAL_PROMPT, EVAL_SYSTEM

            ev = "\n\n".join(f"[{i + 1}] {self._text(a)[:700]}" for i, a in enumerate(top))
            q = self.question
            if st.targets[1:]:
                q += "\n(It needs: " + "; ".join(st.targets[1:]) + ")"
            reply = self._complete(EVAL_PROMPT.format(question=q, evidence=ev or "(none)"),
                                   EVAL_SYSTEM, 120)  # fmt: skip
            self._verify[key] = _parse_judge(reply)
            self.trace.append({"op": "verify", **self._verify[key]})
        return self._verify[key]

    def _distinct_llm(self) -> int:
        return (1 if self._decomp is not None else 0) + len(self._verify)

    # ------------------------------------------------------------------ value
    def _text(self, a: Any) -> str:
        ev = getattr(a, "evidence", None)
        if ev is not None and getattr(ev, "content", None):
            return str(ev.content)
        loc = self.builder.locate(str(a.block_id))
        if loc is None:
            return ""
        doc, i = loc
        return self.builder.blocks(doc)[i].content

    def _scores(self, target: str, items: Sequence[Any]) -> list[float]:
        from ..retrieval.rerank import rerank_scores

        texts = [self._text(a)[:1200] for a in items]
        try:
            raw = rerank_scores(self.spec.reranker, target, texts)
        except Exception:  # noqa: BLE001 - no reranker: lexical overlap
            from ..core.text import lexical_score, query_terms

            terms = query_terms(target)
            return [min(1.0, lexical_score(terms, t) / 4) for t in texts]
        return [1.0 / (1.0 + math.exp(-s)) for s in raw]

    def _best_score(self, st: State, a: Any) -> float:
        return self._best_scores(st, [a])[str(a.block_id)]

    def _best_scores(self, st: State, items: Sequence[Any]) -> dict[str, float]:
        """Best probability over the state's targets, per block (one batched reranker call per
        target: single-pair calls are what makes a cross-encoder slow)."""
        out = {str(a.block_id): 0.0 for a in items}
        for t in st.targets:
            for a, sc in zip(items, self._scores(t, items), strict=True):
                k = str(a.block_id)
                out[k] = max(out[k], sc)
        return out

    def _value(self, st: State) -> float:
        key = st.actions
        if key in self._values:
            return self._values[key]
        sp = self.spec
        cand = st.pool[: sp.pool_max]
        rel = []
        for t in st.targets:
            sc = self._scores(t, cand) if cand else [0.0]
            rel.append(max(sc))
        main = rel[0]
        subs = rel[1:]
        v = sp.w_main * main + (1 - sp.w_main) * (sum(subs) / len(subs) if subs else main)
        if st.judge and st.judge.get("coverage") is not None:
            cov = float(st.judge["coverage"])
            if st.judge.get("answerable"):
                cov = max(cov, 0.9)
            v = (1 - sp.w_judge) * v + sp.w_judge * cov
        v -= sp.lam * len(st.actions) + sp.llm_cost * st.llm
        self._values[key] = v
        return v

    def _uct(self, parent: _Node, ch: _Node) -> float:
        if ch.visits == 0:
            return float("inf")
        return ch.mean + self.spec.c_uct * math.sqrt(math.log(parent.visits + 1) / ch.visits)

    def _best(self, root: _Node) -> _Node:
        best, stack = root, [root]
        while stack:
            n = stack.pop()
            if n.visits and n.value > best.value:
                best = n
            stack += n.children
        return best

    def _top(self, st: State, n: int) -> list[Any]:
        best = self._best_scores(st, st.pool)
        scored = sorted(st.pool, key=lambda a: -best[str(a.block_id)])
        return scored[:n]

    # ------------------------------------------------------------------ reading
    def _read(self, st: State, budget: int) -> tuple[list[ContextSpan], dict[str, Any], int]:
        from ..reason.gate import conf_of

        # the question's own fused candidates keep their order and gate confidence; what the
        # search added follows in value order, with its reranker probability as confidence
        own = {
            str(getattr(a, "block_id", "")) for a in self._search.get(("fused", self.question), [])
        }
        head = [a for a in st.pool if str(a.block_id) in own]
        extra = [a for a in st.pool if str(a.block_id) not in own]
        best = self._best_scores(st, extra) if extra else {}
        extra.sort(key=lambda a: -best[str(a.block_id)])
        for a in extra:
            if conf_of(a) == 0.0:
                c = round(0.8 * best[str(a.block_id)], 4)
                a.metadata = {**(a.metadata or {}), "conf": c}
                if a.evidence is not None:
                    a.evidence.metadata = {**(a.evidence.metadata or {}), "conf": c}
        pool = head + extra
        if self.spec.reader == "order" or self.llm is None:
            spans = self.builder.build(pool, token_budget=budget, policy="auto")
            return spans, {"reader": "order"}, 0
        from ..reason.reader import ReasonedReader, reason_spec

        q = self.question
        notes = []
        if st.targets[1:]:
            notes.append("It needs: " + "; ".join(st.targets[1:]))
        if st.judge and st.judge.get("missing") and not st.judge.get("answerable"):
            notes.append("Still missing: " + str(st.judge["missing"]))
        if notes:
            q += "\n(" + " | ".join(notes) + ")"
        spec = reason_spec("gate")
        rd = ReasonedReader(self.builder, spec)
        plan = rd.prepare(q, pool, budget)
        calls = 0
        if not plan.skip:
            try:
                rd.decide(plan, self.llm)
                calls = 1
                self.llm_calls += 1
                self.llm_in += len(plan.prompt) // 4
                self.llm_out += len(plan.reply or "") // 4
            except Exception:  # noqa: BLE001 - provider error: the pool order
                plan.reply = None
        spans, info = rd.assemble(plan, allow_first_overflow=self.builder.allow_first_overflow)
        keep = ("top_conf", "accepted", "skip_llm", "choice", "pages")
        return spans, {k: v for k, v in (info or {}).items() if k in keep}, calls

    # ------------------------------------------------------------------ tree
    def _tree(self, root: _Node, best: _Node, spans: list[ContextSpan]) -> Any:
        from ..treerag.tree import RAGNode, SearchTree

        tree = SearchTree()
        on_path = set()
        n: _Node | None = best
        while n is not None:
            on_path.add(id(n))
            n = n.parent
        ids: dict[int, str] = {}

        def add(node: _Node, pid: str | None, nid: str) -> None:
            ids[id(node)] = nid
            st = node.state
            a = node.action
            if a is None:
                ntype, action, q = "QUERY", "MCTS_RAG", self.question
            elif a.op == "decompose":
                ntype, action, q = "SUBQUERY", "THINK_DECOMPOSE", "; ".join(st.targets[1:])
            elif a.op == "verify":
                ntype, action, q = (
                    "VERIFY",
                    "THINK_VERIFY",
                    str((st.judge or {}).get("missing", "")),
                )
            elif a.op == "expand":
                ntype, action, q = "SEARCH", "EXPAND", ""
            else:
                ntype, action, q = "SEARCH", f"SEARCH_{a.op.upper()}", a.query
            tree.add(
                RAGNode(
                    nid,
                    pid,
                    ntype,
                    action,
                    q,
                    observation=(q if ntype == "VERIFY" else f"pool {len(st.pool)}")
                    + (" *best*" if node is best else ""),
                    relevance=round(node.value, 4),
                    coverage=(st.judge or {}).get("coverage"),
                    answerability=(st.judge or {}).get("answerable") if st.judge else None,
                    visits=node.visits,
                    value=round(node.mean, 4),
                    meta={"on_best_path": id(node) in on_path},
                )
            )
            for i, ch in enumerate(sorted(node.children, key=lambda c: -c.visits), 1):
                add(ch, nid, f"{nid}.{i}")

        add(root, None, "q")
        bid = ids[id(best)]
        for i, s in enumerate(spans, 1):
            tree.add(
                RAGNode(
                    f"{bid}.read{i}",
                    bid,
                    {"section": "SECTION", "page": "PAGE"}.get(s.resolution or "", "BLOCK"),
                    "READ_SELECT",
                    observation=s.content[:300],
                    document_id=s.document_id,
                    section_id=s.node_id,
                    block_ids=list(s.block_ids),
                    page=s.page_start,
                    cost=s.token_count,
                    read=i,
                )
            )
        return tree


def _parse_judge(reply: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", reply or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    cov = data.get("coverage")
    try:
        cov = None if cov is None else max(0.0, min(1.0, float(cov)))
    except (TypeError, ValueError):
        cov = None
    return {
        "answerable": bool(data.get("answerable")),
        "coverage": cov,
        "missing": " ".join(str(data.get("missing") or "").split())[:200],
    }


def _anchor(x: Any) -> Any:
    md = getattr(x, "metadata", None) or {}
    a = md.get("anchor", x) if isinstance(md, dict) else x
    if not hasattr(a, "kind"):
        a = as_anchors([a])[0]
    return a


def _count(n: _Node) -> int:
    return 1 + sum(_count(c) for c in n.children)


def parse_mcts_mode(mode: str) -> MCTSRAGSpec:
    """ "mctsrag@iters=24;depth=4;llm=2;ops=fused+bm25+vector;rd=gate" -> spec."""
    _, _, params = mode.replace("+retry", "").partition("@")
    keys: dict[str, tuple[str, Any]] = {
        "iters": ("iters", int),
        "depth": ("depth", int),
        "c": ("c_uct", float),
        "llm": ("max_llm", int),
        "per": ("per_search", int),
        "lam": ("lam", float),
        "wj": ("w_judge", float),
        "wm": ("w_main", float),
        "rr": ("reranker", str),
        "rd": ("reader", str),
        "ops": ("ops", lambda v: tuple(x for x in v.split("+") if x)),
    }
    kw: dict[str, Any] = {}
    for item in filter(None, params.split(";")):
        k, _, v = item.partition("=")
        if k not in keys:
            raise ValueError(f"unknown mctsrag parameter {k!r} in {mode!r}")
        f, conv = keys[k]
        kw[f] = conv(v)
    return MCTSRAGSpec(**kw)


__all__ = ["MCTSRAG", "Action", "MCTSRAGSpec", "parse_mcts_mode"]
