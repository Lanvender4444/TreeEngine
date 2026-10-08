"""Regenerate generated fixtures: handbook_large.md (78 nodes) and the two PDFs.

python tests/fixtures/make_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent

CHAPTERS = {
    "Networking": {
        "Firewall": ["Inbound rules", "Outbound rules", "Audit logging"],
        "DNS": ["Internal zones", "Caching resolvers", "DNSSEC"],
        "Load balancing": ["Health checks", "Sticky sessions", "TLS termination"],
    },
    "Storage": {
        "Block volumes": ["Snapshots", "Encryption at rest", "Resizing"],
        "Object storage": ["Lifecycle policies", "Versioning", "Presigned URLs"],
        "Backups": ["Retention", "Restore drills", "Offsite copies"],
    },
    "Security": {
        "Identity": ["Single sign-on", "Service accounts", "Password policy"],
        "Secrets": ["Vault usage", "Rotation", "Break-glass access"],
        "Incident response": ["Severity levels", "On-call escalation", "Postmortems"],
    },
    "Observability": {
        "Metrics": ["Retention windows", "Cardinality limits", "Dashboards"],
        "Logging": ["Log levels", "PII scrubbing", "Shipping agents"],
        "Tracing": ["Sampling", "Span attributes", "Trace storage"],
    },
    "Deployment": {
        "CI pipeline": ["Build cache", "Test sharding", "Artifact signing"],
        "Release process": ["Canary releases", "Rollback", "Change freeze"],
        "Environments": ["Staging parity", "Ephemeral previews", "Production access"],
    },
    "Databases": {
        "PostgreSQL": ["Connection pooling", "Vacuum tuning", "Replication lag"],
        "Redis": ["Eviction policy", "Persistence", "Cluster mode"],
        "Migrations": ["Online schema changes", "Review checklist", "Backfills"],
    },
}


def leaf_fact(ci: int, si: int, li: int, chapter: str, section: str, leaf: str) -> str:
    code = f"{ci + 1}{si + 1}{li + 1}"
    return (
        f"The {leaf.lower()} standard for {section.lower()} in the {chapter.lower()} area is "
        f"policy HB-{code}. Teams must review it every {30 + ci * 10 + si * 3 + li} days and "
        f"record approvals in ticket queue Q{code}."
    )


def make_handbook() -> str:
    out = [
        "# Platform Engineering Handbook",
        "",
        "This handbook collects the operational standards of the platform team.",
        "",
    ]
    for ci, (chapter, sections) in enumerate(CHAPTERS.items()):
        out += [f"## {chapter}", "", f"Standards for the {chapter.lower()} area.", ""]
        for si, (section, leaves) in enumerate(sections.items()):
            out += [f"### {section}", "", f"Overview of {section.lower()}.", ""]
            for li, leaf in enumerate(leaves):
                out += [f"#### {leaf}", "", leaf_fact(ci, si, li, chapter, section, leaf), ""]
    return "\n".join(out)


def make_pdfs() -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    sections = [
        (
            "1 Introduction",
            1,
            [
                "Helios is a solar inverter monitoring system.",
                "It was first deployed in 2021 at the Ningbo plant.",
            ],
        ),
        ("2 Architecture", 1, ["The system has three tiers: collectors, broker and dashboard."]),
        ("2.1 Collectors", 2, ["Each collector polls inverters every 15 seconds over Modbus TCP."]),
        ("2.2 Message broker", 2, ["Readings are published to a NATS cluster with 3 nodes."]),
        ("3 Operations", 1, ["Operators review alarms twice per shift."]),
        ("3.1 Alarm thresholds", 2, ["An over-temperature alarm fires at 78 degrees Celsius."]),
        ("3.2 Maintenance", 2, ["Firmware is updated quarterly during the Sunday window."]),
    ]

    def draw(path: Path, bookmarks: bool) -> None:
        c = canvas.Canvas(str(path), pagesize=A4)
        c.setTitle("Helios Operations Manual")
        for i, (title, level, paras) in enumerate(sections):
            y = 780
            c.setFont("Helvetica-Bold", 16 if level == 1 else 13)
            c.drawString(72, y, title)
            if bookmarks:
                key = f"s{i}"
                c.bookmarkPage(key)
                c.addOutlineEntry(title, key, level=level - 1)
            c.setFont("Helvetica", 11)
            for p in paras:
                y -= 30
                c.drawString(72, y, p)
            c.showPage()
        c.save()

    draw(HERE / "manual_bookmarks.pdf", bookmarks=True)
    draw(HERE / "manual_plain.pdf", bookmarks=False)


if __name__ == "__main__":
    (HERE / "handbook_large.md").write_text(make_handbook(), encoding="utf-8")
    make_pdfs()
    print("fixtures regenerated")
