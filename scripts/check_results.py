"""Re-check the sha256 of every published result file against results.sha256.

    python scripts/check_results.py            # exit 1 if a listed file changed or is missing, or a result file is unlisted
    python scripts/check_results.py --write    # rewrite results.sha256 (only when adding a new, versioned result file)

Covered: every file under chapters/*/results/, each chapter's PREREGISTRATION.md, and chapter 13's
plan_overrides.json.  Published result files are never edited; a correction goes into a new file.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATTERNS = ("chapters/*/results/**/*", "chapters/*/PREREGISTRATION.md", "chapters/13-*/plan_overrides.json")


def covered(root: Path) -> list[str]:
    files = {p for pattern in PATTERNS for p in root.glob(pattern) if p.is_file()}
    return sorted(p.relative_to(root).as_posix() for p in files)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            out[rel] = digest
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="rewrite the manifest from the files on disk")
    ap.add_argument("--root", type=Path, default=REPO, help="tree to check (default: this repository)")
    ap.add_argument("--manifest", type=Path, default=None, help="default: <root>/results.sha256")
    a = ap.parse_args(argv)
    manifest = a.manifest or a.root / "results.sha256"
    files = covered(a.root)
    if a.write:
        manifest.write_text("".join(f"{sha256(a.root / rel)}  {rel}\n" for rel in files))
        print(f"wrote {manifest.name}: {len(files)} files")
        return 0
    recorded = read_manifest(manifest)
    problems = [f"missing  {rel}" for rel in sorted(set(recorded) - set(files))]
    problems += [f"unlisted {rel}" for rel in files if rel not in recorded]
    problems += [f"changed  {rel}" for rel in files if rel in recorded and sha256(a.root / rel) != recorded[rel]]
    for line in problems:
        print(line)
    print(f"{len(recorded)} listed, {len(files)} on disk, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
