"""Static report figures (SVG via matplotlib, optional: ``pip install matplotlib``).

Retrieval:  recall@5 vs context tokens        - what a strategy finds for what it makes you read
            recall@5 by document length         - where the long-document advantage shows up
QA:         accuracy vs $/query (or tokens)     - the PageIndex-style headline figure
            accuracy vs context tokens
            document length vs query cost

Color encodes the strategy *family* (TreeEngine / traditional RAG / baseline): three
categorical slots from the reference palette, validated for all-pairs use in scatter plots.
Every point is also direct-labelled with its strategy name, so identity never relies on color.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

FAMILY_COLORS = {"treeengine": "#2a78d6", "traditional": "#eb6834", "baseline": "#1baf7a"}
FAMILY_LABELS = {
    "treeengine": "TreeEngine",
    "traditional": "Traditional RAG",
    "baseline": "Full context",
}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "svg.fonttype": "none",
        }
    )
    return plt


def _family(name: str) -> str:
    if name == "full_context":
        return "baseline"
    return "traditional" if name.startswith("rag_") else "treeengine"


def _direct_labels(fig: Any, ax: Any, points: Sequence[tuple[str, float, float]]) -> None:
    """Label every point; nudge labels that would collide, flip them left near the right edge."""
    fig.canvas.draw()
    to_px = ax.transData.transform
    x_max = ax.get_window_extent().x1
    placed: list[tuple[float, float]] = []
    for name, x, y in sorted(points, key=lambda p: -p[2]):
        px, py = to_px((x, y))
        left = px > x_max - 110
        dy = 4.0
        while any(abs(py + dy - qy) < 11 and abs(px - qx) < 90 for qx, qy in placed):
            dy -= 11
        placed.append((px, py + dy))
        ax.annotate(
            name,
            (x, y),
            xytext=(-6 if left else 6, dy),
            textcoords="offset points",
            ha="right" if left else "left",
            fontsize=8,
            color=INK,
        )


def scatter(
    points: Sequence[tuple[str, float, float]],
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    logx: bool = False,
) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    seen = set()
    for name, x, y in points:
        fam = _family(name)
        ax.scatter(
            [x],
            [y],
            s=64,
            color=FAMILY_COLORS[fam],
            edgecolors=SURFACE,
            linewidths=2,
            zorder=3,
            label=FAMILY_LABELS[fam] if fam not in seen else None,
        )
        seen.add(fam)
    if logx:
        ax.set_xscale("log")
    ax.margins(x=0.12, y=0.08)
    _direct_labels(fig, ax, points)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    if len(seen) > 1:
        ax.legend(frameon=False, loc="lower right", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def grouped_lines(
    series: dict[str, list[tuple[str, float | None]]],
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> Path:
    """One line per strategy across ordered buckets; families share hue, direct labels at end."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labels: list[str] = []
    for pts in series.values():
        for b, _ in pts:
            if b not in labels:
                labels.append(b)
    styles = ["-", "--", ":", "-."]
    per_family: dict[str, int] = defaultdict(int)
    seen = set()
    for name, pts in series.items():
        fam = _family(name)
        xs = [labels.index(b) for b, v in pts if v is not None]
        ys = [v for _, v in pts if v is not None]
        if not xs:
            continue
        style = styles[per_family[fam] % len(styles)]
        per_family[fam] += 1
        ax.plot(xs, ys, style, color=FAMILY_COLORS[fam], linewidth=2, marker="o", markersize=5)
        ax.annotate(
            name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK
        )
        seen.add(fam)
    ax.set_xticks(range(len(labels)), labels)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.margins(x=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def retrieval_charts(payload: dict[str, Any], stem: Path) -> list[Path]:
    _plt()  # ImportError -> caller skips charts
    summary = payload["summary"]
    pts = [
        (s, r["ctx_tokens@5"], r["recall@5"] * 100)
        for s, r in summary.items()
        if r.get("recall@5") is not None and r.get("ctx_tokens@5")
    ]
    out = []
    if pts:
        out.append(
            scatter(
                pts,
                "context tokens in the top 5 (log)",
                "recall@5 (%)",
                f"Recall vs context size — {payload['meta'].get('suite')}",
                stem.with_name(stem.name + "-recall-vs-context.svg"),
                logx=True,
            )
        )
    return out


def qa_charts(payload: dict[str, Any], stem: Path) -> list[Path]:
    _plt()
    summary = payload["summary"]
    out = []
    priced = all(r.get("usd/q") is not None for r in summary.values())
    xkey, xlabel = (
        ("usd/q", "$ per query (log)") if priced else ("tokens/q", "tokens per query (log)")
    )
    pts = [
        (s, r[xkey], r["accuracy"] * 100)
        for s, r in summary.items()
        if r.get("accuracy") is not None and r.get(xkey)
    ]
    if pts:
        out.append(
            scatter(
                pts,
                xlabel,
                "QA accuracy (%)",
                "Accuracy vs cost",
                stem.with_name(stem.name + "-accuracy-vs-cost.svg"),
                logx=True,
            )
        )
    pts = [
        (s, r["ctx_tokens"], r["accuracy"] * 100)
        for s, r in summary.items()
        if r.get("accuracy") is not None and r.get("ctx_tokens")
    ]
    if pts:
        out.append(
            scatter(
                pts,
                "average context tokens (log)",
                "QA accuracy (%)",
                "Accuracy vs context size",
                stem.with_name(stem.name + "-accuracy-vs-context.svg"),
                logx=True,
            )
        )
    by_len = payload.get("cost_by_length") or {}
    if by_len:
        out.append(
            grouped_lines(
                {s: [(b, v) for b, v in rows] for s, rows in by_len.items()},
                "document length",
                xlabel.replace(" (log)", ""),
                "Query cost vs document length",
                stem.with_name(stem.name + "-cost-vs-length.svg"),
            )
        )
    return out


__all__ = ["grouped_lines", "qa_charts", "retrieval_charts", "scatter"]
