"""01b - patch step: adds edge_density column to hhdoc_manifest.csv (see imstats.edge_density).

Added after discovering that the brief's ink_ratio (global Otsu on raw intensity)
degenerates on low-contrast aged-paper scans: visual inspection of flag_blank's
contact sheet showed the vast majority of "blank"-flagged pages actually contain
clear handwriting. edge_density (gradient-magnitude Otsu) is robust to background
tone and does not have this failure mode. Idempotent; only computes for rows
missing the column or re-run entirely (cheap relative to md5/phash pass).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modi.utils import console  # noqa: F401,E402  (force utf-8 stdout)


from modi.data.imstats import edge_density  # noqa: E402
from modi.utils.parallel import default_workers, pmap  # noqa: E402
from modi.utils.paths import P  # noqa: E402

MANIFEST = P.processed / "hhdoc_manifest.csv"


def _worker(path_or_none):
    """Top-level so it is picklable by the process pool."""
    if path_or_none is None:
        return None
    try:
        with Image.open(path_or_none) as im:
            im.load()
            return edge_density(im)
    except Exception:
        return None


def main():
    df = pd.read_csv(MANIFEST, dtype={"folder": str})
    root = P.hhdoc_root
    tasks = [None if pd.isna(r["md5"]) else str(root / r["rel_path"])
             for _, r in df.iterrows()]
    nw = default_workers()
    print(f"Using {nw} worker processes.")
    df["edge_density"] = pmap(_worker, tasks, workers=nw, desc="edge_density")
    df.to_csv(MANIFEST, index=False, encoding="utf-8")
    ed = pd.to_numeric(df["edge_density"], errors="coerce")
    print("edge_density distribution:")
    print(ed.describe(percentiles=[0.01, 0.02, 0.05, 0.1, 0.25, 0.5]).to_string())
    print(f"\nUpdated {MANIFEST} with edge_density")


if __name__ == "__main__":
    main()
