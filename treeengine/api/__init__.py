"""Stable public API layer (TE-019 / TE-021 / TE-022 / TE-023).

* ``errors``   typed exceptions with stable codes
* ``dto``      versioned, JSON-serialisable request / result records (schema_version "1")
* ``presets``  lite / balanced / quality / research
* ``schemas/`` JSON Schemas of the v1 records (``treeengine.api.schema(name)``)

Everything importable from ``treeengine.api`` is covered by the compatibility policy in
docs/api/compatibility.md.
"""

from . import dto
from .dto import (
    SCHEMA_VERSION,
    AnswerV1,
    BlockV1,
    CitationV1,
    ContextSpanV1,
    DocumentV1,
    EvidenceV1,
    MappingV1,
    NodeV1,
    ResearchResultV1,
    RetrievalResultV1,
)
from .errors import *  # noqa: F403
from .errors import ERRORS
from .errors import __all__ as _errors_all
from .presets import RetrievalOptions, SearchPreset, resolve
from .schemas import schema

__all__ = list(_errors_all) + [
    "ERRORS",
    "SCHEMA_VERSION",
    "AnswerV1",
    "BlockV1",
    "CitationV1",
    "ContextSpanV1",
    "DocumentV1",
    "EvidenceV1",
    "MappingV1",
    "NodeV1",
    "ResearchResultV1",
    "RetrievalOptions",
    "RetrievalResultV1",
    "SearchPreset",
    "dto",
    "resolve",
    "schema",
]
