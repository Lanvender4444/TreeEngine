"""V0.16 Evidence Search: retrieval as a budgeted search for a sufficient evidence set.

Traditional RAG ranks passages once and takes the top k. Evidence Search treats reading as a
search problem over states:

    State   = evidence chosen so far + the current queries + visited units + budget left
    Actions = SEARCH_TEXT / SEARCH_VECTOR (the fast paths, also for rewritten queries)
              READ (a candidate unit) / READ_NEIGHBORS / READ_PARENT (section head) /
              READ_PAGE (a whole page: a statement with its header) / REWRITE_QUERY / STOP
    Value   = a local reranker's relevance of a unit to the question (the node value
              function), times its novelty against what is already read, minus its token cost

Search policies (``SearchSpec.algo``):

    best_first  a priority queue of units; read the best, expand it, re-score, repeat
    beam        states = evidence sets; keep the ``beam`` best sets per step
    mcts        UCT over the same actions; rollouts complete a set greedily with noise; the
                reward of a finished set (relevance + coverage - cost - redundancy) is
                back-propagated to every action on the path
    + loop      (any algo) after the first pass an LLM judges whether the evidence answers the
                question; if not, its "missing" query is searched (REWRITE_QUERY) and the
                search continues with the rest of the budget

The reranker is the only learned component: no LLM call unless ``loop`` is on.
"""

from .core import (
    SEARCH_MODES,
    EvidenceSearch,
    SearchSpec,
    Unit,
    parse_search_mode,
)

__all__ = ["SEARCH_MODES", "EvidenceSearch", "SearchSpec", "Unit", "parse_search_mode"]
