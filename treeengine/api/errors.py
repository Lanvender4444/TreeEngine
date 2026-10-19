"""Public exception types (TE-023). Catch ``TreeEngineError`` for anything raised on purpose by
TreeEngine; the original low-level exception (sqlite3, provider, MCP ...) stays available as
``__cause__`` for logs.

    TreeEngineError
    ├── IngestionError
    │   ├── UnsupportedFormatError
    │   └── OCRRequiredError
    ├── IndexCompatibilityError
    ├── StorageError
    ├── SourceNotFoundError
    ├── DocumentNotFoundError
    ├── NodeNotFoundError
    ├── InvalidPolicyError
    ├── ResourceLimitError
    ├── MappingError
        └── DependencyMissingError

Every error has a stable ``code`` (used by the HTTP / MCP error payloads), an HTTP ``status``
and ``retryable``.
"""

from __future__ import annotations

from typing import Any


class TreeEngineError(Exception):
    code = "treeengine_error"
    status = 500
    retryable = False

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        """The v1 error payload (``schema_version`` 1)."""
        return {
            "schema_version": "1",
            "status": "error",
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
                "details": {k: v for k, v in self.details.items() if _safe(v)},
            },
        }


def _safe(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool, list, dict))


class IngestionError(TreeEngineError):
    code = "ingestion_failed"
    status = 422


class UnsupportedFormatError(IngestionError, ValueError):
    code = "unsupported_format"
    status = 415


class OCRRequiredError(IngestionError):
    code = "ocr_required"
    status = 422


class IndexCompatibilityError(TreeEngineError):
    code = "index_incompatible"
    status = 409


class StorageError(TreeEngineError):
    code = "storage_error"
    status = 500
    retryable = True


class SourceNotFoundError(TreeEngineError, FileNotFoundError):
    code = "source_not_found"
    status = 404


class DocumentNotFoundError(TreeEngineError, KeyError):
    code = "document_not_found"
    status = 404

    def __str__(self) -> str:  # KeyError would repr() the message
        return self.message


class NodeNotFoundError(TreeEngineError, KeyError):
    code = "node_not_found"
    status = 404

    def __str__(self) -> str:
        return self.message


class InvalidPolicyError(TreeEngineError, ValueError):
    code = "invalid_request"
    status = 422


class ResourceLimitError(TreeEngineError, ValueError):
    code = "resource_limit"
    status = 413


class MappingError(TreeEngineError, ValueError):
    code = "mapping_failed"
    status = 422


class IntegrationError(TreeEngineError):
    code = "integration_error"
    status = 502
    retryable = True


class DependencyMissingError(IntegrationError, ImportError):
    code = "dependency_missing"
    status = 501
    retryable = False


class AccessDeniedError(TreeEngineError, PermissionError):
    code = "access_denied"
    status = 403


ERRORS: tuple[type[TreeEngineError], ...] = (
    TreeEngineError,
    IngestionError,
    UnsupportedFormatError,
    OCRRequiredError,
    IndexCompatibilityError,
    StorageError,
    SourceNotFoundError,
    DocumentNotFoundError,
    NodeNotFoundError,
    InvalidPolicyError,
    ResourceLimitError,
    MappingError,
    IntegrationError,
    DependencyMissingError,
    AccessDeniedError,
)

__all__ = [e.__name__ for e in ERRORS] + ["ERRORS"]
