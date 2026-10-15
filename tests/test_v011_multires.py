"""V0.11 multi-resolution metrics: chosen resolution and error classes from context spans."""

from __future__ import annotations

from benchmarks.metrics.retrieval import Query
from benchmarks.run_multires import classify
from treeengine.context import ContextSpan


def _q(t: str, needles: list[str]) -> Query:
    gold = {"point": "BLOCK", "local": "SECTION", "cross": "MULTI_SECTION"}[t]
    return Query.from_dict(
        {"id": "q", "query": "?", "type": t, "expected_blocks": needles, "resolution": gold}
    )


def _span(text: str, kind: str) -> ContextSpan:
    return ContextSpan("d", None, ["b"], text, None, None, None, None, [], 100, resolution=kind)


def test_point_window_is_correct_and_section_is_over_expand() -> None:
    q = _q("point", ["revenue grew 12 percent"])
    r = classify(q, [_span("In 2023 revenue grew 12 percent.", "window")])
    assert r["chosen"] == "BLOCK" and r["correct_resolution"] and r["error"] == "ok"
    r = classify(q, [_span("Intro. Revenue grew 12 percent. More.", "section")])
    assert r["chosen"] == "SECTION" and r["error"] == "over_expand"
    assert r["density"] == 10.0  # 1 needle per 100 tokens


def test_local_needs_one_span_with_everything() -> None:
    q = _q("local", ["first factor", "second factor"])
    one = classify(q, [_span("the first factor and the second factor", "section")])
    assert one["chosen"] == "SECTION" and one["error"] == "ok" and one["correct_resolution"]
    split = classify(q, [_span("first factor", "window"), _span("second factor", "window")])
    assert split["chosen"] == "MULTI" and split["error"] == "under_expand"
    missing = classify(q, [_span("only the first factor", "window")])
    assert missing["coverage"] == 0.5 and missing["error"] == "under_expand"


def test_cross_and_misses() -> None:
    q = _q("cross", ["risk factor text", "md and a text"])
    both = classify(q, [_span("risk factor text", "window"), _span("md and a text", "section")])
    assert both["chosen"] == "MULTI" and both["correct_resolution"] and both["error"] == "ok"
    collapsed = classify(q, [_span("risk factor text here", "section")])
    assert collapsed["error"] == "collapse"
    miss = classify(q, [_span("nothing relevant", "window")])
    assert miss["chosen"] == "MISS" and miss["error"] == "wrong_branch"
