"""Backwards-compatible entry point: ``python -m benchmarks.run`` = ``benchmarks.run_retrieval``.

See ``run_retrieval`` (Layer A, retrieval only) and ``run_qa`` (Layer B, end-to-end QA).
"""

from __future__ import annotations

from .loader import load_queries
from .run_retrieval import main

__all__ = ["load_queries", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
