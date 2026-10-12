"""10 - build a stratified GROUND-TRUTH labelling set for HHDoc page quality.

Why this exists
---------------
Every quality threshold in step 03 is currently justified only by "the number
looks plausible". That is not defensible, and it is exactly the failure mode that
produced two wrong metrics during development (ink_ratio measured paper colour,
not ink; Otsu-on-gradients collapsed to ~0). Neither was a tuning problem, and no
threshold search could have found either.

A small hand-labelled sample fixes this: once labels exist, any candidate metric
can be scored by precision/recall against real answers instead of plausibility,
and thresholds reported in the write-up become measurable claims.

What it does
------------
Draws a STRATIFIED sample across the edge_density range (plus deliberate
over-sampling of the low tail, folder 18 and odd shapes, where the interesting
cases live), renders numbered contact sheets, and writes a CSV with one blank
`label` column per sampled page for a human to fill in.

Label vocabulary (put one in the `label` column):
    normal      - ordinary handwritten Modi page
    sparse      - genuinely handwritten but only a few lines
    blank       - no writing at all (backing sheet, empty page)
    printed     - printed text, not handwriting (e.g. folder 18)
    separator   - scanner artefact / START-END card / ruler / colour chart
    unusable    - damaged, out of focus, or otherwise unreadable
    unsure      - cannot tell; needs a second opinion

Idempotent (same sample for the same seed). Seed 42.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modi.utils import console  # noqa: F401,E402
from modi.utils.contact_sheet import make_contact_sheet  # noqa: E402
from modi.utils.paths import P  # noqa: E402

SEED = 42
N_PER_BAND = 40          # ordinary edge_density bands
N_LOW_TAIL = 80          # over-sample the low tail (blank/sparse live here)
N_SPECIAL = 40           # folder 18 + unusual shapes
COLS = 8
THUMB = 200

OUT_CSV = P.processed / "hhdoc_label_sample.csv"


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(P.processed / "hhdoc_manifest.csv", dtype={"folder": str})
    df = df[df["md5"].notna()].copy()
    ed = pd.to_numeric(df["edge_density"], errors="coerce")
    df["edge_density"] = ed

    picked = {}

    def take(sub, n, reason):
        sub = sub[~sub.index.isin(picked)]
        if len(sub) == 0:
            return
        n = min(n, len(sub))
        idx = rng.choice(sub.index.values, n, replace=False)
        for i in idx:
            picked[i] = reason

    # 1. the low tail, where blank / sparse / separator cases concentrate
    take(df.nsmallest(200, "edge_density"), N_LOW_TAIL, "low_tail")

    # 2. even coverage across the rest of the range, so "normal" is represented
    qs = df["edge_density"].quantile([0.1, 0.3, 0.5, 0.7, 0.9]).tolist()
    bounds = [df["edge_density"].min()] + qs + [df["edge_density"].max()]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        band = df[(df["edge_density"] >= lo) & (df["edge_density"] < hi)]
        take(band, N_PER_BAND, f"band_{lo:.3f}_{hi:.3f}")

    # 3. deliberate over-sampling of the known-special groups
    take(df[df["folder"] == "18"], N_SPECIAL // 2, "folder18_printed")
    if "flag_unusual_shape" in df.columns:
        take(df[df["flag_unusual_shape"] == True], N_SPECIAL // 2, "unusual_shape")  # noqa: E712

    sample = df.loc[sorted(picked)].copy()
    sample["sample_reason"] = [picked[i] for i in sample.index]
    sample = sample.sort_values("edge_density").reset_index(drop=True)
    sample.insert(0, "label_id", range(1, len(sample) + 1))
    sample["label"] = ""          # <- human fills this in
    sample["label_note"] = ""

    cols = ["label_id", "label", "label_note", "folder", "filename", "rel_path",
            "edge_density", "aspect_ratio", "mode", "sample_reason"]
    sample[cols].to_csv(OUT_CSV, index=False, encoding="utf-8")

    # contact sheets, numbered to match label_id
    root = P.hhdoc_root
    per_sheet = COLS * 8        # 64 per sheet keeps each image legible
    n_sheets = 0
    for s in range(0, len(sample), per_sheet):
        chunk = sample.iloc[s:s + per_sheet]
        items = [(root / r["rel_path"], f"#{r['label_id']} {r['folder']}/{r['filename']}")
                 for _, r in chunk.iterrows()]
        out = P.figures / f"label_sheet_{s//per_sheet + 1:02d}.jpg"
        make_contact_sheet(items, out, thumb=THUMB, cols=COLS, max_items=per_sheet)
        n_sheets += 1

    print("=== LABEL SAMPLE ===")
    print(f"sampled pages: {len(sample)}")
    print(sample["sample_reason"].value_counts().to_string())
    print(f"\nedge_density range covered: {sample['edge_density'].min():.4f} "
          f"- {sample['edge_density'].max():.4f}")
    print(f"contact sheets written: {n_sheets} (outputs/figures/label_sheet_*.jpg)")
    print(f"label file (fill the `label` column): {OUT_CSV}")
    print("\nlabels: normal | sparse | blank | printed | separator | unusable | unsure")


if __name__ == "__main__":
    main()
