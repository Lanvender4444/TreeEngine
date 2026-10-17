"""V0.13 Reasoned Reading: PageIndex-style reasoning over a retrieval-induced table of contents.

PageIndex lets an LLM read a document's table of contents (titles + summaries + page ranges) and
reason about *where* to read, instead of trusting a similarity ranking. TreeEngine keeps its
cheap, high-recall FTS + Vector pool and spends one LLM call on the decision the fixed reading
policies (raw / auto / section600 / adaptive) cannot make: which candidate regions to read, and
whether to read a matching excerpt or the whole region (a table with its header, a section with
its definitions).

    pool (50 anchors) -> RegionCatalog (regions = sections or pages, with excerpts)
                      -> one LLM call: ordered list of excerpts / whole regions
                      -> packed into the token budget; the rest of the pool fills what is left

Every step degrades to the plain ``auto`` context when the LLM is missing or its reply is
unusable, so a reasoned context is never worse-formed than the default one.
"""

from .catalog import Excerpt, Region, RegionCatalog, build_catalog
from .reader import (
    REASON_ALIASES,
    REASON_MODES,
    ReasonedReader,
    ReasonPlan,
    ReasonSpec,
    build_context,
    parse_choice,
    reason_spec,
)

__all__ = [
    "REASON_ALIASES",
    "REASON_MODES",
    "build_context",
    "reason_spec",
    "Excerpt",
    "ReasonPlan",
    "ReasonSpec",
    "ReasonedReader",
    "Region",
    "RegionCatalog",
    "build_catalog",
    "parse_choice",
]
