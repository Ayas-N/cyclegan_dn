#!/usr/bin/env python3
"""
Split NCT-CRC-HE-100K images in a flat folder into class-based subfolders.

Usage:
  python split_nct_by_class.py \
      --src /path/to/flat_folder \
      --dst /path/to/output_root \
      [--copy] [--dry-run] [--strict] [--extensions .png .jpg .jpeg .tif .tiff]
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Optional, List

# Class codes for NCT-CRC-HE-100K
CLASS_CODES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

def compile_patterns(codes: List[str]) -> List[re.Pattern]:
    """
    Build regex patterns that match each class code as a standalone token,
    allowing separators like -, _, space, or boundaries. Case-insensitive.
    """
    patterns = []
    for code in codes:
        # Match when code appears with non-letter boundaries or at start/end
        # e.g., "..._ADI_", "ADI-xxxx", "(BACK)", "....TUM.png"
        pat = re.compile(rf"(?i)(?<![A-Za-z]){code}(?![A-Za-z])")
        patterns.append(pat)
    return patterns

PATTERNS = compile_patterns(CLASS_CODES)

def detect_class(filename: str) -> Optional[str]:
    """
    Return the detected class code (e.g., 'ADI') from the filename, or None.
    Break ties by the first class matched (rare unless the filename is weird).
    """
    for code, pat in zip(CLASS_CODES, PATTERNS):
        if pat.search(filename):
            return code
    return None

def main():
    ap = argparse.ArgumentParser(description="Split NCT images into class folders based on filename tokens.")
    ap.add_argument("--src", required=True, type=Path, help="Path to flat folder containing generated images.")
    ap.add_argument("--dst", required=True, type=Path, help="Destination root folder where class subfolders will be created.")
    ap.add_argument("--copy", action="store_true", help="Copy files instead of moving them (default: move).")
    ap.add_argument("--dry-run", action="store_true", help="Show what would happen without writing any files.")
    ap.add_argument("--strict", action="store_true", help="If set, skip files that don't match a known class instead of putting them in UNKNOWN.")
    ap.add_argument("--extensions", nargs="+", default=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
                    help="File extensions to include (case-insensitive).")
    args = ap.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    exts = set([e.lower() for e in args.extensions])

    if not src.is_dir():
        raise SystemExit(f"[ERROR] Source is not a directory: {src}")

    # Create destination root (and class subfolders lazily)
    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    # Pre-create class directories to avoid repeated checks
    class_dirs = {c: dst / c for c in CLASS_CODES}
    if not args.dry_run:
        for d in class_dirs.values():
            d.mkdir(parents=True, exist_ok=True)


    moved = 0
    skipped = 0
    unknown = 0

    for entry in src.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in exts:
            continue

        cls = detect_class(entry.name)
        if cls is None:
            if args.strict:
                print(f"[SKIP: no class] {entry.name}")
                skipped += 1
                continue
            target_dir = unknown_dir
            unknown += 1
        else:
            target_dir = class_dirs[cls]

        target_path = target_dir / entry.name

        if args.dry_run:
            action = "COPY" if args.copy else "MOVE"
            print(f"[{action}] {entry} -> {target_path}")
            moved += 1
        else:
            if args.copy:
                shutil.copy2(str(entry), str(target_path))
            else:
                shutil.move(str(entry), str(target_path))
            moved += 1

    print("\n=== Summary ===")
    print(f"Processed folder: {src}")
    print(f"Output root     : {dst}")
    print(f"Action          : {'COPY' if args.copy else 'MOVE'}")
    print(f"Moved/Copied    : {moved}")
    print(f"Unknown (no class matched): {unknown}" if not args.strict else "")
    print(f"Skipped (strict only)     : {skipped}" if args.strict else "")

if __name__ == "__main__":
    main()
