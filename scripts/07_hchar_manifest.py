"""07 - HChar manifest -> processed/hchar_manifest.csv

Columns: category (consonants/vowels/digits), class, rel_path, width, height, mode, md5.
Reports class counts + imbalance, unreadable files, exact duplicates (within and
across classes; across-class dups -> review).
Has a built-in timing estimate on a sample before the full pass (rule 8).
Idempotent. Seed 42.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modi.utils import console  # noqa: F401,E402  (force utf-8 stdout)


from modi.data.imstats import is_image_file, md5_of_file  # noqa: E402
from modi.utils.parallel import default_workers, pmap  # noqa: E402
from modi.utils.paths import P  # noqa: E402
from PIL import Image  # noqa: E402

SEED = 42
OUT = P.processed / "hchar_manifest.csv"
REVIEW = P.processed / "hchar_crossclass_dups.csv"
CATS = ["consonants", "vowels", "digits"]


def iter_files():
    root = P.hchar_root
    for cat in CATS:
        cdir = root / cat
        if not cdir.exists():
            continue
        for cls in sorted(cdir.iterdir()):
            if not cls.is_dir():
                continue
            for f in sorted(cls.iterdir()):
                if is_image_file(f):
                    yield cat, cls.name, f


def _worker(task):
    """Top-level so it is picklable by the process pool. task = (cat, cls, abs_path)."""
    cat, cls, fpath = task
    return record(cat, cls, Path(fpath))


def record(cat, cls, f):
    rel = f.relative_to(P.hchar_root).as_posix()
    try:
        with Image.open(f) as im:
            im.load()
            mode, w, h = im.mode, im.width, im.height
        md5 = md5_of_file(f)
        return {"category": cat, "class": cls, "rel_path": rel, "width": w, "height": h,
                "mode": mode, "md5": md5}, None
    except Exception as e:
        return ({"category": cat, "class": cls, "rel_path": rel, "width": None, "height": None,
                 "mode": None, "md5": None}, repr(e))


def main():
    files = list(iter_files())
    n = len(files)
    print(f"HChar image files: {n}")

    nw = default_workers()
    tasks = [(cat, cls, str(f)) for cat, cls, f in files]

    # --- timing estimate on a parallel sample (rule 8) ---
    sample_n = min(4000, n)
    t0 = time.time()
    pmap(_worker, tasks[:sample_n], workers=nw, desc="timing sample")
    dt = time.time() - t0
    per = dt / max(1, sample_n)
    est_h = per * n / 3600.0
    print(f"timing sample: {sample_n} files in {dt:.1f}s ({per*1000:.2f} ms/file, "
          f"{nw} workers) -> full est {est_h:.2f} h")
    if est_h >= 3.0:
        print("ABORT per rule 8: estimated full run >= 3 h. Stopping.")
        return

    out = pmap(_worker, tasks, workers=nw, desc="hchar")
    rows = [rec for rec, _ in out]
    unreadable = [{"rel_path": rec["rel_path"], "error": err}
                  for rec, err in out if err]

    df = pd.DataFrame(rows).sort_values(["category", "class", "rel_path"]).reset_index(drop=True)
    df.to_csv(OUT, index=False, encoding="utf-8")

    # ---- class counts + imbalance ----
    counts = df.groupby(["category", "class"]).size()
    print("\n=== HCHAR SUMMARY ===")
    print(f"total files: {len(df)}  unreadable: {len(unreadable)}")
    print("\nper-category class counts:")
    for cat in CATS:
        sub = counts.loc[cat] if cat in counts.index.get_level_values(0) else pd.Series(dtype=int)
        if len(sub):
            print(f"  {cat}: {len(sub)} classes, total {int(sub.sum())}, "
                  f"min {int(sub.min())} ({sub.idxmin()}), max {int(sub.max())} ({sub.idxmax()}), "
                  f"imbalance max/min = {sub.max()/max(1,sub.min()):.2f}")
    if unreadable:
        print("\nunreadable files:")
        for u in unreadable[:20]:
            print("   !", u["rel_path"], u["error"])

    # ---- exact duplicates ----
    valid = df[df["md5"].notna()]
    dup = valid[valid.duplicated("md5", keep=False)]
    n_groups = dup["md5"].nunique()
    print(f"\nexact-duplicate md5 groups: {n_groups}  (rows involved: {len(dup)})")
    # across-class duplicates -> review
    cross = []
    for md5, g in dup.groupby("md5"):
        classes = g[["category", "class"]].drop_duplicates()
        if len(classes) > 1:
            cross.append({"md5": md5, "n": len(g),
                          "classes": "; ".join(f"{r.category}/{r['class']}" for _, r in classes.iterrows()),
                          "files": " | ".join(g["rel_path"].head(6))})
    cr = pd.DataFrame(cross)
    cr.to_csv(REVIEW, index=False, encoding="utf-8")
    print(f"ACROSS-class duplicate groups (review): {len(cr)} -> {REVIEW}")
    within = n_groups - len(cr)
    print(f"within-class duplicate groups: {within}")
    print(f"\nwrote {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
