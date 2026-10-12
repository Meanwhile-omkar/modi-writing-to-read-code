"""03 - HHDoc quality flags -> adds flag_blank, flag_non_modi_printed,
flag_unusual_shape, review_note to the manifest. Makes contact sheets per flag.

Thresholds are chosen from the data distribution and justified in the printed
summary (which 09 folds into the report). No silent exclusion: flags only.
Idempotent. Seed 42.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modi.utils import console  # noqa: F401,E402  (force utf-8 stdout)


from modi.utils.contact_sheet import make_contact_sheet  # noqa: E402
from modi.utils.paths import P  # noqa: E402

SEED = 42
MANIFEST = P.processed / "hhdoc_manifest.csv"

# folder 18 is known printed Devanagari (per task brief)
KNOWN_PRINTED_FOLDER = "18"
ASPECT_HI, ASPECT_LO = 2.5, 0.4


def main() -> None:
    df = pd.read_csv(MANIFEST, dtype={"folder": str})
    n = len(df)
    print(f"Loaded manifest: {n} rows")

    # ---------- flag_blank ----------
    # The brief's ink_ratio (global Otsu on raw intensity) is background-polarity-
    # dependent (dark-bg manuscripts ~1.0, white-bg pages ~0.0), AND we found by
    # visual inspection of the contact sheet that even the polarity-robust
    # ink_ratio_fg/gray_std combo (tried first) degenerates on these low-contrast
    # aged-paper scans: it flagged ~26% of pages as "blank" when the large majority
    # visibly contain handwriting. Root cause: global intensity Otsu finds a
    # degenerate split (almost all mass on one side) when ink and paper tone are
    # close. We instead use edge_density (gradient-magnitude Otsu, computed in
    # 01b_hhdoc_add_edge_density.py), which is robust to background tone because
    # ink strokes create local edges regardless of absolute brightness.
    if "edge_density" not in df.columns:
        raise SystemExit(
            "edge_density column missing — run scripts/01b_hhdoc_add_edge_density.py first."
        )
    ed = pd.to_numeric(df["edge_density"], errors="coerce")
    ink = pd.to_numeric(df["ink_ratio"], errors="coerce")  # kept for reference only

    print("\nbrief ink_ratio distribution (polarity-dependent; NOT used for flag_blank):")
    print(ink.describe(percentiles=[0.01, 0.02, 0.05, 0.1, 0.5]).to_string())
    print("\nedge_density distribution (gradient-magnitude Otsu; used for flag_blank):")
    print(ed.describe(percentiles=[0.01, 0.02, 0.05, 0.1, 0.25, 0.5]).to_string())
    print("edge_density low-tail histogram:")
    bins = [0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 1.0]
    print(pd.cut(ed, bins=bins, right=False).value_counts().sort_index().to_string())

    # Threshold: the 2nd percentile of edge_density defines a REVIEW CANDIDATE POOL,
    # not a confident blank verdict.
    #
    # Visual inspection of the lowest-edge_density pages (outputs/figures/
    # _lowest_edge_density.jpg) showed that genuinely blank backing sheets
    # (e.g. 4/page104, 4/page81, 4/page77 — empty olive sheets) and genuinely
    # handwritten but SPARSE pages (e.g. 21/page51, 20/page90 — 1-3 lines of Modi)
    # occupy the SAME edge_density range (~0.030). No threshold on this feature can
    # separate them. Per project rule 6 we therefore do NOT decide: flag_blank marks
    # the bottom tail as candidates for a human to confirm via the contact sheet.
    ed_p02 = float(ed.quantile(0.02))
    blank_thr = ed_p02
    df["flag_blank"] = ed < blank_thr
    print(f"\nblank rule (REVIEW CANDIDATES, not a verdict): edge_density < "
          f"{blank_thr:.5f} (2nd pctile); flagged {int(df['flag_blank'].sum())} pages")
    print("NOTE: blank sheets and sparse-but-real handwriting overlap in this feature "
          "(verified visually). Nothing here may be excluded from training without "
          "human confirmation of outputs/figures/flag_blank.jpg.")

    # ---------- flag_non_modi_printed ----------
    # whole folder 18 flagged; also look for other 1-bit pages with similar stats.
    df["flag_book_folder18"] = df["folder"] == KNOWN_PRINTED_FOLDER
    f18 = df[df["folder"] == KNOWN_PRINTED_FOLDER]
    print(f"\nfolder {KNOWN_PRINTED_FOLDER}: {len(f18)} pages flagged as printed Devanagari.")
    print(f"  folder {KNOWN_PRINTED_FOLDER} modes: {f18['mode'].value_counts().to_dict()}")
    # candidate look-alikes elsewhere: mode in {'1'} OR (bilevel-ish) not in folder 18
    mode_is_1bit = df["mode"].astype(str).eq("1")
    lookalike = mode_is_1bit & (df["folder"] != KNOWN_PRINTED_FOLDER)
    df["_printed_lookalike"] = lookalike
    print(f"  1-bit (mode='1') pages OUTSIDE folder {KNOWN_PRINTED_FOLDER} (review-only): "
          f"{int(lookalike.sum())}")
    print("  1-bit pages per folder (all folders):")
    print(df[mode_is_1bit].groupby("folder").size().to_string())

    # ---------- flag_unusual_shape ----------
    ar = pd.to_numeric(df["aspect_ratio"], errors="coerce")
    df["flag_unusual_shape"] = (ar > ASPECT_HI) | (ar < ASPECT_LO)
    print(f"\nunusual shape (AR>{ASPECT_HI} or AR<{ASPECT_LO}): "
          f"{int(df['flag_unusual_shape'].sum())} pages")

    # ---------- review_note ----------
    def note(row):
        notes = []
        if row["flag_blank"]:
            notes.append("blank?")
        if row["_printed_lookalike"]:
            notes.append("1bit-printed-lookalike?")
        if row["flag_unusual_shape"]:
            notes.append("unusual-shape")
        if pd.isna(row["md5"]):
            notes.append("unreadable")
        return ";".join(notes)

    df["review_note"] = df.apply(note, axis=1)
    df.drop(columns="_printed_lookalike").to_csv(MANIFEST, index=False, encoding="utf-8")

    # ---------- contact sheets ----------
    root = P.hhdoc_root
    def sheet_for(mask, name, cap_cols=("folder", "filename")):
        sub = df[mask]
        if sub.empty:
            print(f"  (no pages for {name})")
            return
        items = [(root / r["rel_path"], f"{r['folder']}/{r['filename']}") for _, r in sub.iterrows()]
        out = P.figures / name
        k = make_contact_sheet(items, out, max_items=300)
        print(f"  contact sheet: {out.name} ({k} of {len(sub)} pages)")

    print("\nBuilding contact sheets ->", P.figures)
    sheet_for(df["flag_blank"], "flag_blank.jpg")
    sheet_for(df["flag_book_folder18"], "flag_book_folder18.jpg")
    sheet_for(df["mode"].astype(str).eq("1") & (df["folder"] != KNOWN_PRINTED_FOLDER),
              "flag_printed_lookalike_review.jpg")
    sheet_for(df["flag_unusual_shape"], "flag_unusual_shape.jpg")

    print("\n=== QUALITY FLAG SUMMARY ===")
    for col in ["flag_blank", "flag_book_folder18", "flag_unusual_shape"]:
        print(f"  {col}: {int(df[col].sum())}")
    print(f"  rows with any review_note: {int((df['review_note'].str.len() > 0).sum())}")
    print(f"\nUpdated {MANIFEST}")


if __name__ == "__main__":
    main()
