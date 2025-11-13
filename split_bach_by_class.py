#!/usr/bin/env python3
import os
import re
import shutil
import argparse

# Map short code to folder name
CLASS_MAP = {
    "b": "Benign",
    "is": "In-Situ",
    "iv": "Invasive",
    "n": "Normal",
}

# Matches ..._b005.png, ..._is012.png, etc. at the end of the filename
CLASS_PATTERN = re.compile(r"_(b|is|iv|n)\d+\.(png|jpg|jpeg|tif|tiff)$", re.IGNORECASE)


def split_bach_images(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = input_dir

    os.makedirs(output_dir, exist_ok=True)

    counts = {k: 0 for k in CLASS_MAP.keys()}
    unmatched = 0

    for fname in os.listdir(input_dir):
        src_path = os.path.join(input_dir, fname)

        if not os.path.isfile(src_path):
            continue

        m = CLASS_PATTERN.search(fname)
        if not m:
            unmatched += 1
            print(f"[WARN] Could not infer class for: {fname}")
            continue

        code = m.group(1).lower()
        class_folder_name = CLASS_MAP[code]
        dst_dir = os.path.join(output_dir, class_folder_name)
        os.makedirs(dst_dir, exist_ok=True)

        dst_path = os.path.join(dst_dir, fname)
        shutil.move(src_path, dst_path)
        counts[code] += 1

    print("Done.")
    for code, num in counts.items():
        print(f"{CLASS_MAP[code]} ({code}): {num} images moved")
    if unmatched:
        print(f"Unmatched files (skipped): {unmatched}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split BACH images into class folders.")
    parser.add_argument("input_dir", help="Folder containing BACH images")
    parser.add_argument(
        "--output_dir",
        help="Folder to create class subfolders in (default: input_dir)",
        default=None,
    )
    args = parser.parse_args()

    split_bach_images(args.input_dir, args.output_dir)
