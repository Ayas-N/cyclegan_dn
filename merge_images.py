# merge_images.py
import argparse, shutil
from pathlib import Path

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

def iter_images(root: Path):
    for p in root.rglob('*'):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            yield p

def merge(srcs, dst, suffixes=None):
    dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    if suffixes and len(suffixes) != len(srcs):
        raise SystemExit("len(--suffix) must equal number of --src dirs")

    for i, src in enumerate(map(Path, srcs)):
        tag = (suffixes[i] if suffixes else f"from{i+1}")
        for p in iter_images(src):
            rel = p.relative_to(src)              # preserves class folders, etc.
            out_dir = (dst / rel.parent)
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / p.name
            if out_path.exists():
                stem, ext = p.stem, p.suffix.lower()
                k = 1
                cand = out_dir / f"{stem}_{tag}{ext}"
                while cand.exists():
                    cand = out_dir / f"{stem}_{tag}_{k}{ext}"; k += 1
                out_path = cand

            shutil.copy2(p, out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', nargs='+', required=True, help='source image dirs (e.g. .../train/images .../test/images)')
    ap.add_argument('--dst', required=True, help='destination dir')
    ap.add_argument('--suffix', nargs='*', help='optional per-source suffix for name clashes (e.g. train test)')
    args = ap.parse_args()
    merge(args.src, args.dst, args.suffix)

if __name__ == '__main__':
    main()
