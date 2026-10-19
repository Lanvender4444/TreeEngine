"""Build wheel + sdist, check them, install the wheel into a fresh venv and smoke-test it
(TE-004 / TE-007). Usage: ``python tools/check_wheel.py [--extras vector,mcp] [--keep]``.

Fails when the wheel needs the source tree, when metadata is invalid (``twine check``), or
when forbidden files (databases, benchmark data, secrets) are packaged.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (".db", ".db-wal", ".db-shm", ".sqlite", ".env", ".pdf", "/benchmarks/", "/tests/")
MAX_WHEEL_BYTES = 2_000_000


def run(cmd: list[str], **kw: object) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)  # type: ignore[call-overload]


def audit(dist: Path) -> None:
    wheel = next(dist.glob("*.whl"))
    sdist = next(dist.glob("*.tar.gz"))
    names = zipfile.ZipFile(wheel).namelist()
    names += tarfile.open(sdist).getnames()
    bad = [n for n in names if any(f in "/" + n for f in FORBIDDEN)]
    if bad:
        raise SystemExit(f"forbidden files in the distribution: {bad[:10]}")
    for need in ("treeengine/storage/schema.sql", "treeengine/py.typed"):
        if need not in names:
            raise SystemExit(f"{need} missing from the wheel")
    if not any(n.endswith("LICENSE") for n in names):
        raise SystemExit("LICENSE missing from the distribution")
    size = wheel.stat().st_size
    if size > MAX_WHEEL_BYTES:
        raise SystemExit(f"wheel is {size} bytes (> {MAX_WHEEL_BYTES})")
    print(f"audit OK: {wheel.name} {size} bytes, {len(names)} entries")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extras", default="", help="extras to install with the wheel")
    ap.add_argument("--keep", action="store_true", help="keep the temporary venv")
    ap.add_argument("--dist", default=None, help="use an existing dist/ directory")
    ap.add_argument("--out", default=None, help="build into this directory (kept: the release)")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="te-wheel-"))
    dist = Path(args.dist or args.out or tmp / "dist").resolve()
    if not args.dist:
        run([sys.executable, "-m", "build", "--outdir", str(dist), str(ROOT)])
    run([sys.executable, "-m", "twine", "check", "--strict", *map(str, dist.glob("*"))])
    audit(dist)
    env_dir = tmp / "venv"
    venv.EnvBuilder(with_pip=True, clear=True).create(env_dir)
    bindir = env_dir / ("Scripts" if os.name == "nt" else "bin")
    py = str(bindir / ("python.exe" if os.name == "nt" else "python"))
    wheel = str(next(dist.glob("*.whl")))
    spec = f"{wheel}[{args.extras}]" if args.extras else wheel
    run([py, "-m", "pip", "install", "-q", spec])
    clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    run(
        [py, str(ROOT / "tests" / "packaging" / "smoke_installed.py")],
        cwd=str(tmp),  # never the source tree: imports must resolve to site-packages
        env=clean_env,
    )
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
