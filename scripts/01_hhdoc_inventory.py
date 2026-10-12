"""01 - HHDoc inventory -> processed/hhdoc_manifest.csv

For every image in raw.hhdoc_root/{1..24}: folder, filename, rel_path, bytes,
PIL mode, width, height, md5, pHash, ink_ratio, aspect_ratio.
Idempotent, prints a summary. Seed 42 (no randomness here, kept for consistency).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modi.utils import console  # noqa: F401,E402  (force utf-8 stdout)


from modi.data.imstats import full_image_record, is_image_file  # noqa: E402
from modi.utils.parallel import default_workers, pmap  # noqa: E402
from modi.utils.paths import P  # noqa: E402

SEED = 42
OUT = P.processed / "hhdoc_manifest.csv"


def _worker(task):
    """Top-level so it is picklable by the process pool. task = (folder, abs_path, rel)."""
    folder, fpath, rel = task
    fpath = Path(fpath)
    try:
        rec = full_image_record(fpath)
        err = None
    except Exception as e:  # unreadable -> flag, never drop silently
        rec = {"bytes": fpath.stat().st_size, "md5": None, "mode": None, "width": None,
               "height": None, "phash": None, "ink_ratio": None, "aspect_ratio": None,
               "gray_mean": None, "gray_std": None, "ink_ratio_fg": None}
        err = repr(e)
    return {"folder": folder, "filename": fpath.name, "rel_path": rel, **rec}, err


def iter_hhdoc_images():
    root = P.hhdoc_root
    for folder in sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
        if not folder.is_dir():
            continue
        for f in sorted(folder.iterdir()):
            if is_image_file(f):
                yield folder.name, f


def main() -> None:
    root = P.hhdoc_root
    print(f"HHDoc root: {root}")
    files = list(iter_hhdoc_images())
    print(f"Found {len(files)} image files across folders.")
    tasks = [(folder, str(f), f.relative_to(root).as_posix()) for folder, f in files]
    nw = default_workers()
    print(f"Using {nw} worker processes.")
    out = pmap(_worker, tasks, workers=nw, desc="inventory")
    rows = [r for r, _ in out]
    skipped = [{"folder": r["folder"], "filename": r["filename"],
                "rel_path": r["rel_path"], "error": e}
               for r, e in out if e is not None]

    df = pd.DataFrame(rows)
    # sort by numeric folder then filename for stable, idempotent output
    df["_fnum"] = pd.to_numeric(df["folder"], errors="coerce")
    df = df.sort_values(["_fnum", "folder", "filename"]).drop(columns="_fnum").reset_index(drop=True)
    df.to_csv(OUT, index=False, encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(f"Total images: {len(df)}")
    print(f"Unreadable (flagged): {len(skipped)}")
    if skipped:
        for s in skipped:
            print("   !", s["rel_path"], s["error"])
    print("\nPer-folder counts:")
    print(df.groupby("folder").size().reindex(
        sorted(df["folder"].unique(), key=lambda x: int(x) if x.isdigit() else 1e9)).to_string())
    print("\nPIL mode distribution:")
    print(df["mode"].value_counts(dropna=False).to_string())
    print("\nSize (width x height) — describe:")
    print(df[["width", "height", "aspect_ratio", "ink_ratio"]].describe().to_string())
    print(f"\nWrote {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
