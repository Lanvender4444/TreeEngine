"""Structure quality gate: is a detected outline believable enough to build the tree from?"""

from __future__ import annotations

import re

from .model import StructureQuality
from .outline import Entry


def structure_quality(outline: list[Entry], page_count: int) -> StructureQuality:
    n = len(outline)
    pages = max(1, page_count)
    if n == 0:
        return StructureQuality(0.0, 0, 0.0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, True, ["no headings"])
    with_page = [e[2] for e in outline if e[2] is not None]
    first = min(with_page) if with_page else 1
    coverage = (pages - first + 1) / pages
    depth = max(e[0] for e in outline)
    rising = sum(1 for a, b in zip(with_page, with_page[1:], strict=False) if b >= a)
    monotonic = rising / (len(with_page) - 1) if len(with_page) > 1 else 1.0
    orphans = sum(1 for a, b in zip(outline, outline[1:], strict=False) if b[0] - a[0] > 1)
    orphan_ratio = orphans / n
    tiny = 0
    for a, b in zip(outline, outline[1:], strict=False):
        if a[2] == b[2] and a[4] is not None and b[4] is not None and 0 <= b[4] - a[4] < 30:
            tiny += 1  # two headings with no room for body text between them
    tiny_ratio = tiny / n
    density = n / pages
    norm = [re.sub(r"\W+", "", e[1]).lower() for e in outline]
    duplicate = 1 - len(set(norm)) / n
    reasons = []
    if n < 2:
        reasons.append("fewer than 2 headings")
    if density > 6:
        reasons.append(f"{density:.1f} headings per page")
    if orphan_ratio > 0.3:
        reasons.append(f"{orphan_ratio:.0%} level jumps")
    if tiny_ratio > 0.4:
        reasons.append(f"{tiny_ratio:.0%} empty sections")
    if monotonic < 0.8:
        reasons.append(f"pages out of order ({monotonic:.0%} rising)")
    if duplicate > 0.3:
        reasons.append(f"{duplicate:.0%} duplicate titles")
    if coverage < 0.5:
        reasons.append(f"first heading on page {first} of {pages}")
    score = max(0.0, 1.0 - 0.2 * len(reasons))
    return StructureQuality(
        score=score,
        heading_count=n,
        coverage=coverage,
        max_depth=depth,
        monotonic_ratio=monotonic,
        orphan_ratio=orphan_ratio,
        tiny_node_ratio=tiny_ratio,
        heading_density=density,
        duplicate_ratio=duplicate,
        suspicious=bool(reasons),
        reasons=reasons,
    )


__all__ = ["structure_quality"]
