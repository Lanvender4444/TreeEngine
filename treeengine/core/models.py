"""Core data model: Document -> Node -> Block, plus Evidence (unified retriever output)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, TypeVar

T = TypeVar("T")


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in data.items() if k in names})


def dumps_meta(meta: dict[str, Any] | None) -> str:
    return json.dumps(meta or {}, ensure_ascii=False, sort_keys=True)


def loads_meta(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    value = json.loads(raw)
    return value if isinstance(value, dict) else {}


class _Serializable:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        return _from_dict(cls, data)


@dataclass
class Document(_Serializable):
    id: str
    source_type: str  # markdown | html | pdf | text
    uri: str | None
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Node(_Serializable):
    id: str
    document_id: str
    parent_id: str | None
    depth: int
    position: int
    title: str
    summary: str | None = None
    text: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    node_type: str = "section"


@dataclass
class Block(_Serializable):
    id: str
    document_id: str
    node_id: str | None
    position: int
    block_type: str  # paragraph | code | list | table | quote
    content: str
    page: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence(_Serializable):
    document_id: str
    node_id: str | None
    block_id: str | None
    content: str
    source: str  # tree | fts | vector | web | graph
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation(_Serializable):
    index: int
    document_id: str
    node_id: str | None
    block_id: str | None
    title: str | None = None
    page: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None


@dataclass
class AnswerResult(_Serializable):
    question: str
    answer: str
    citations: list[Citation]
    evidence: list[Evidence]
    query_type: str
    generated: bool  # False when no LLM was available (extractive fallback)
