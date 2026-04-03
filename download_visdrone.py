#!/usr/bin/env python3
"""
download_visdrone.py — Download and convert VisDrone2019-DET to YOLO format.

Usage:
    python download_visdrone.py --path /mnt/mydrive/ajith/data_set/VisDrone
    python download_visdrone.py --path /dist_home/ak_ajithkumar/ondemand/ajith_work/VisDrone
    python download_visdrone.py --path ./datasets/VisDrone --skip-download  # convert only

Downloads ~2.3 GB. Splits: train (6471), val (548), test-dev (1610).
"""

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

SPLITS = [
    "VisDrone2019-DET-train",
    "VisDrone2019-DET-val",
    "VisDrone2019-DET-test-dev",
    "VisDrone2019-DET-test-challenge",
]

BASE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"


# ── Download helpers ──────────────────────────────────────────────────────────

class _ProgressBar:
    def __init__(self, filename):
        self._pbar = tqdm(unit='B', unit_scale=True, unit_divisor=1024,
                          desc=filename, ncols=80) if HAS_TQDM else None
        self._seen = 0

    def __call__(self, _block_num, block_size, total_size):
        if self._pbar is not None:
            if self._pbar.total is None:
                self._pbar.total = total_size
            self._pbar.update(block_size)
        else:
            self._seen += block_size
            mb = self._seen / 1024 / 1024
            if total_size > 0:
                pct = 100 * self._seen / total_size
                print(f"\r  {mb:.1f} MB / {total_size/1024/1024:.1f} MB  ({pct:.0f}%)", end="", flush=True)

    def close(self):
        if self._pbar is not None:
            self._pbar.close()
        else:
            print()


def download_split(split: str, dest_dir: Path) -> Path:
    zip_name = f"{split}.zip"
    zip_path = dest_dir / zip_name
    url = f"{BASE_URL}/{zip_name}"

    if zip_path.exists():
        if not zipfile.is_zipfile(zip_path):
            print(f"  {zip_name} is corrupt, re-downloading...")
            zip_path.unlink()
        else:
            print(f"  {zip_name} already downloaded, skipping.")
            return zip_path

    print(f"Downloading {zip_name} ...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    pb = _ProgressBar(zip_name)
    try:
        urlretrieve(url, zip_path, reporthook=pb)
    finally:
        pb.close()
    return zip_path


def extract_zip(zip_path: Path, dest_dir: Path):
    split_dir = dest_dir / zip_path.stem
    if split_dir.exists() and any(split_dir.iterdir()):
        print(f"  {zip_path.stem}/ already extracted, skipping.")
        return
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)
    print(f"  -> {split_dir}")


# ── Annotation conversion ─────────────────────────────────────────────────────

def convert_box(img_w: int, img_h: int, box: tuple) -> tuple:
    """VisDrone (x, y, w, h) pixel → YOLO (cx, cy, w, h) normalised."""
    x, y, w, h = box
    return (x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h


def convert_split(split_dir: Path):
    """Convert VisDrone annotations to YOLO label format."""
    ann_dir = split_dir / "annotations"
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    if not ann_dir.exists():
        print(f"  WARNING: no annotations dir in {split_dir}, skipping conversion.")
        return

    lbl_dir.mkdir(exist_ok=True)

    ann_files = sorted(ann_dir.glob("*.txt"))
    iterator = tqdm(ann_files, desc=f"Converting {split_dir.name}", ncols=80) \
        if HAS_TQDM else ann_files

    converted = skipped = 0
    for ann_file in iterator:
        img_file = (img_dir / ann_file.name).with_suffix(".jpg")
        lbl_file = lbl_dir / ann_file.name

        if lbl_file.exists():
            skipped += 1
            continue

        # get image dimensions without loading full image
        try:
            from PIL import Image
            img_w, img_h = Image.open(img_file).size
        except Exception as e:
            print(f"  WARNING: cannot open {img_file}: {e}")
            continue

        lines = []
        for row in ann_file.read_text().strip().splitlines():
            parts = row.split(",")
            if len(parts) < 6:
                continue
            if parts[4].strip() == "0":          # score=0 → ignored region
                continue
            cls = int(parts[5].strip()) - 1      # VisDrone classes are 1-indexed
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            if w == 0 or h == 0:
                continue
            cx, cy, nw, nh = convert_box(img_w, img_h, (x, y, w, h))
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        lbl_file.write_text("".join(lines))
        converted += 1

    print(f"  {split_dir.name}: {converted} converted, {skipped} already done.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download & convert VisDrone2019-DET")
    parser.add_argument("--path", required=True,
                        help="Destination directory (dataset root)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, only run annotation conversion")
    parser.add_argument("--skip-convert", action="store_true",
                        help="Skip annotation conversion (download + extract only)")
    parser.add_argument("--splits", nargs="+",
                        default=["VisDrone2019-DET-train",
                                 "VisDrone2019-DET-val",
                                 "VisDrone2019-DET-test-dev"],
                        help="Which splits to process (default: train val test-dev)")
    args = parser.parse_args()

    root = Path(args.path).expanduser().resolve()
    print(f"Dataset root : {root}")
    print(f"Splits       : {args.splits}")
    print()

    # Download + extract
    if not args.skip_download:
        for split in args.splits:
            zip_path = download_split(split, root)
            extract_zip(zip_path, root)
            # Remove zip to save space
            zip_path.unlink()
            print(f"  Removed {zip_path.name}")
        print()

    # Convert annotations
    if not args.skip_convert:
        for split in args.splits:
            split_dir = root / split
            if not split_dir.exists():
                print(f"  WARNING: {split_dir} not found, skipping.")
                continue
            convert_split(split_dir)
        print()

    # Summary
    print("=" * 50)
    for split in args.splits:
        split_dir = root / split
        n_img = len(list((split_dir / "images").glob("*.jpg"))) if (split_dir / "images").exists() else 0
        n_lbl = len(list((split_dir / "labels").glob("*.txt"))) if (split_dir / "labels").exists() else 0
        print(f"  {split}: {n_img} images, {n_lbl} labels")
    print("=" * 50)
    print("Done. Update data/VisDrone.yaml path if needed:")
    print(f"  path: {root}")


if __name__ == "__main__":
    main()
