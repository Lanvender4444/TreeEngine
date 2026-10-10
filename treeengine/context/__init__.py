"""Context reconstruction: retrieve fine, read coherent.

``Evidence`` says *why* something was retrieved (a block hit). ``ContextSpan`` is *what the
answer model reads*: a continuous run of blocks rebuilt at query time around the retrieved
anchors, under a token budget. It is not a chunk — chunks are cut at index time, blindly.

    builder = BlockContextBuilder(repo)
    spans = builder.build(evidence, token_budget=2000)               # policy "auto"
    spans = builder.build(evidence, token_budget=2000, policy="section")
"""

from .builder import POLICIES, BlockContextBuilder, ContextBuilder, resolve_policy
from .span import ContextSpan

__all__ = ["POLICIES", "BlockContextBuilder", "ContextBuilder", "ContextSpan", "resolve_policy"]
