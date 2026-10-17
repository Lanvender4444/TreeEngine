"""V0.15 confidence gate: multi-source candidates, calibrated confidence, PageIndex for the rest.

Every candidate block carries its rank in each source that found it:

    fts      BM25 over blocks
    vector   cosine over block embeddings
    node     the document tree: sections ranked by BM25 over their title path + opening and by
             the centroid of their subtree's vectors; each top section contributes its best block

``confidence(ranks)`` turns those ranks into the probability that the block is evidence for the
question - a logistic model over rank features, fitted once on the dev split (``DEFAULT_COEF``,
``benchmarks/fit_gate.py``). Raw BM25 / cosine values are not comparable across queries and
documents; ranks and cross-source agreement are.

The reader then (``ReasonSpec.gate``):

    accept   candidates with confidence >= accept_tau go straight into the context (no LLM)
    skip     a question whose best candidate is >= skip_tau is answered from the accepted
             candidates + the usual fill - no LLM call at all
    reason   otherwise ONE PageIndex-style call: the remaining candidates as a catalog *plus* the
             document's page outline, in the same prompt; the LLM may pick excerpts, sections
             or whole pages ("p57") for the rest of the budget
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SOURCES = ("fts", "vector", "node")
FEATURES = (
    "inv_fts",  # 1 / rank in FTS (0 when absent)
    "inv_vector",
    "inv_node",
    "log_fused",  # log of the fused (RRF) rank
    "fts_and_vector",  # found by both block paths in their top 10
    "n_sources",  # number of sources that found it
    "top1_any",  # rank 1 in any source
)


def features(ranks: Mapping[str, int], fused_rank: int) -> dict[str, float]:
    def inv(s: str) -> float:
        r = ranks.get(s)
        return 1.0 / r if r else 0.0

    f, v = ranks.get("fts"), ranks.get("vector")
    return {
        "inv_fts": inv("fts"),
        "inv_vector": inv("vector"),
        "inv_node": inv("node"),
        "log_fused": math.log(max(1, fused_rank)),
        "fts_and_vector": 1.0 if (f and v and f <= 10 and v <= 10) else 0.0,
        "n_sources": float(sum(1 for s in SOURCES if ranks.get(s))),
        "top1_any": 1.0 if any(ranks.get(s) == 1 for s in SOURCES) else 0.0,
    }


# fitted on the V0.13 dev split (FinanceBench 35 + longdoc 23 + multires2 20) and multires (90);
# see benchmarks/fit_gate.py and results/v0.15-confidence-gate.md
DEFAULT_COEF: dict[str, float] = {
    "bias": -3.445,
    "inv_fts": 2.1337,
    "inv_vector": 1.1422,
    "inv_node": 0.1352,
    "log_fused": -0.1614,
    "fts_and_vector": 0.7424,
    "n_sources": 0.13,
    "top1_any": 0.1222,
}


def confidence(
    ranks: Mapping[str, int], fused_rank: int, coef: Mapping[str, float] | None = None
) -> float:
    c = coef or DEFAULT_COEF
    x = features(ranks, fused_rank)
    z = c.get("bias", 0.0) + sum(c.get(k, 0.0) * v for k, v in x.items())
    return 1.0 / (1.0 + math.exp(-z))


def annotate(anchors: Sequence[Any], lists: Mapping[str, Sequence[Any]]) -> None:
    """Write each fused block anchor's source ranks and confidence into its metadata (and its
    evidence's), so a reader downstream can gate on them."""
    pos: dict[str, dict[str, int]] = {}
    for src, lst in lists.items():
        if src not in SOURCES:
            continue
        for r, a in enumerate(lst, 1):
            if getattr(a, "block_id", None):
                pos.setdefault(str(a.block_id), {}).setdefault(src, r)
    for fr, a in enumerate(anchors, 1):
        if getattr(a, "kind", "block") != "block" or not a.block_id:
            continue
        ranks = pos.get(str(a.block_id), {})
        conf = round(confidence(ranks, fr), 4)
        a.metadata["src_ranks"] = dict(ranks)
        a.metadata["conf"] = conf
        if a.evidence is not None:
            a.evidence.metadata = {
                **(a.evidence.metadata or {}),
                "src_ranks": dict(ranks),
                "conf": conf,
            }


def conf_of(item: Any) -> float:
    md = getattr(item, "metadata", None) or {}
    if "conf" in md:
        return float(md["conf"])
    ev = getattr(item, "evidence", None)
    if ev is not None and ev.metadata and "conf" in ev.metadata:
        return float(ev.metadata["conf"])
    return 0.0


@dataclass(frozen=True)
class GateDecision:
    accepted: list[Any]
    rest: list[Any]
    skip_llm: bool
    top_conf: float


def split(pool: Sequence[Any], accept_tau: float, skip_tau: float, accept_max: int) -> GateDecision:
    """Accept the confident head of the pool; decide whether the LLM is needed at all."""
    confs = [conf_of(a) for a in pool]
    top = max(confs, default=0.0)
    order = sorted(range(len(pool)), key=lambda i: -confs[i])
    acc_idx = [i for i in order if confs[i] >= accept_tau][:accept_max]
    accepted = [pool[i] for i in acc_idx]
    rest = [a for i, a in enumerate(pool) if i not in set(acc_idx)]
    return GateDecision(accepted, rest, top >= skip_tau, top)


__all__ = [
    "DEFAULT_COEF",
    "FEATURES",
    "SOURCES",
    "GateDecision",
    "annotate",
    "conf_of",
    "confidence",
    "features",
    "split",
]
