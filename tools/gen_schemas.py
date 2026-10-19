"""Regenerate treeengine/api/schemas/*.json from the v1 dataclasses (TE-021).

python tools/gen_schemas.py          # write
python tools/gen_schemas.py --check  # fail if the files are stale (CI / tests)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from treeengine.api.dto import RECORDS  # noqa: E402
from treeengine.api.schemas import ERROR_SCHEMA, build_schema  # noqa: E402

OUT = ROOT / "treeengine" / "api" / "schemas"


def rendered() -> dict[str, str]:
    files = {f"{n}.json": build_schema(n) for n in RECORDS}
    files["error.json"] = ERROR_SCHEMA
    return {k: json.dumps(v, indent=2, ensure_ascii=False) + "\n" for k, v in files.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    stale = []
    for name, text in rendered().items():
        path = OUT / name
        if args.check:
            if not path.exists() or path.read_text("utf-8") != text:
                stale.append(name)
        else:
            path.write_text(text, encoding="utf-8")
    if stale:
        print("stale schemas (run python tools/gen_schemas.py):", ", ".join(stale))
        return 1
    print("schemas", "up to date" if args.check else f"written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
