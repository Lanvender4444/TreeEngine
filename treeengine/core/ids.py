from __future__ import annotations

import hashlib
import uuid


def new_id(prefix: str) -> str:
    """Opaque, sortable-enough id such as ``doc_3f2a...``."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
