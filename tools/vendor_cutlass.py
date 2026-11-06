"""Show how MegaBake's CUTLASS GEMM bodies differ from pinned upstream files.

Each body's upstream snapshot and checksum are specified in
``src/megabake/bodies/vendor_sources.py``.

    .venv/bin/python tools/vendor_cutlass.py          # diff upstream -> fork
    .venv/bin/python tools/vendor_cutlass.py --stat   # changed line counts only
    .venv/bin/python tools/vendor_cutlass.py --check  # verify the upstream copy's hash

Upgrading CUTLASS: replace the snapshot and its recorded hash, preserving any
documented prefix, then merge upstream changes into the fork by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = runpy.run_path(str(ROOT / "src" / "megabake" / "bodies" / "vendor_sources.py"))[
    "references"
]()


def check_upstream() -> None:
    """Fail unless every upstream copy has its recorded prefix and hash."""
    for reference in REFERENCES:
        original = reference["original"]
        source = original.read_bytes()
        header = reference["header"]
        if not source.startswith(header):
            raise SystemExit(f"upstream {original} lint header was modified")
        actual = hashlib.sha256(source[len(header) :]).hexdigest()
        expected = reference["sha256"]
        if actual != expected:
            raise SystemExit(f"upstream {original} was modified ({actual} != {expected})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only verify the upstream copy")
    parser.add_argument("--stat", action="store_true", help="show changed line counts only")
    args = parser.parse_args()
    check_upstream()
    if args.check:
        return 0
    for reference in REFERENCES:
        subprocess.run(
            [
                "git",
                "diff",
                "--no-index",
                *(["--stat"] if args.stat else []),
                str(reference["original"].relative_to(ROOT)),
                str(reference["adaptation"].relative_to(ROOT)),
            ],
            check=False,
            cwd=ROOT,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
