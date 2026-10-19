"""JSON Schemas (draft 2020-12) of the v1 records in ``treeengine.api.dto``.

``schema("retrieval_result")`` loads the shipped file; ``build_schema(cls)`` derives it from the
dataclass (``tools/gen_schemas.py`` regenerates the files; a test keeps them in sync, so a
change to a public record is always a visible diff).
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from importlib import resources
from typing import Any

_BASE = "https://github.com/Lanvender4444/TreeEngine/schemas/v1/"


def _type(t: Any, defs: dict[str, Any]) -> dict[str, Any]:
    origin = typing.get_origin(t)
    args = typing.get_args(t)
    if origin in (typing.Union, types.UnionType):
        opts = [_type(a, defs) for a in args if a is not type(None)]
        base = opts[0] if len(opts) == 1 else {"anyOf": opts}
        if type(None) in args:
            return {"anyOf": [base, {"type": "null"}]}
        return base
    if origin in (list, tuple, set):
        return {"type": "array", "items": _type(args[0], defs) if args else {}}
    if origin is dict or t is dict:
        return {"type": "object"}
    if t is Any:
        return {}
    if t is str:
        return {"type": "string"}
    if t is bool:
        return {"type": "boolean"}
    if t is int:
        return {"type": "integer"}
    if t is float:
        return {"type": "number"}
    if dataclasses.is_dataclass(t) and isinstance(t, type):
        name = t.__name__
        if name not in defs:
            defs[name] = {}  # recursion guard
            defs[name] = _object(t, defs)
        return {"$ref": f"#/$defs/{name}"}
    raise TypeError(f"no JSON type for {t!r}")


def _object(cls: Any, defs: dict[str, Any]) -> dict[str, Any]:
    hints = typing.get_type_hints(cls)
    props: dict[str, Any] = {}
    required: list[str] = []
    for f in dataclasses.fields(cls):
        props[f.name] = _type(hints[f.name], defs)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            required.append(f.name)
    out: dict[str, Any] = {"type": "object", "properties": props, "additionalProperties": True}
    if required:
        out["required"] = required
    return out


def build_schema(name: str) -> dict[str, Any]:
    from ..dto import RECORDS

    cls = RECORDS[name]
    defs: dict[str, Any] = {}
    body = _object(cls, defs)
    out = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{_BASE}{name}.json",
        "title": cls.__name__,
        "description": (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else "",
        **body,
    }
    if defs:
        out["$defs"] = defs
    return out


def schema(name: str) -> dict[str, Any]:
    """The shipped JSON Schema of a v1 record (``"retrieval_result"``, ``"context_span"``,
    ``"error"`` ...)."""
    path = resources.files("treeengine.api.schemas").joinpath(f"{name}.json")
    return dict(json.loads(path.read_text("utf-8")))


ERROR_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": f"{_BASE}error.json",
    "title": "ErrorV1",
    "description": "The payload of every TreeEngine error (HTTP body, MCP error content).",
    "type": "object",
    "required": ["schema_version", "status", "error"],
    "properties": {
        "schema_version": {"const": "1"},
        "status": {"const": "error"},
        "error": {
            "type": "object",
            "required": ["code", "message", "retryable"],
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
                "retryable": {"type": "boolean"},
                "details": {"type": "object"},
            },
        },
    },
}

__all__ = ["ERROR_SCHEMA", "build_schema", "schema"]
