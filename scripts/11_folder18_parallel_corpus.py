"""11 - folder 18 is a PARALLEL CORPUS, not printed junk.

Discovery
---------
The task brief states folder 18 is "known to be printed Devanagari (1-bit pages)"
and says to flag the whole folder. Visual inspection shows that is wrong and
costly. Folder 18 is a scanned scholarly edition titled "ILLUSTRATIVE MODI
DOCUMENTS" in which each historical document appears TWICE:

  page11  handwritten Modi facsimile
  page12  printed Devanagari transcription of that same text (+ English notes)
  page13  handwritten Modi facsimile
  page14  "No. 3] RADHABAI [7-5-1733" + English summary + Devanagari transcription

So roughly half of folder 18 is genuine handwritten Modi, and the printed half is
a page-level Modi -> Devanagari TRANSCRIPTION with dates and document titles.
HHDoc is otherwise completely unlabelled, so this is the only page-level ground
truth in the whole 3352-page collection. Flagging the folder away would discard
both the handwritten pages and the only transcription pairing available.

This script separates the two page types so a human can confirm, and writes the
candidate pairing. It does NOT decide anything: per project rule 6 the result is
written to review artifacts and DATA_DECISIONS_TODO, not applied as a filter.

Method
------
Printed type has a very regular horizontal line rhythm; handwritten Modi does not.
We take the row ink-projection profile and measure its periodicity by
autocorrelation, plus the share of fully-blank rows (printed pages have wide
margins and clean interline gaps). Both are polarity-safe and cheap.

Idempotent. Seed 42.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modi.utils import console  # noqa: F401,E402
from modi.utils.contact_sheet import make_contact_sheet  # noqa: E402
from modi.utils.parallel import default_workers, pmap  # noqa: E402
from modi.utils.paths import P  # noqa: E402

SEED = 42
OUT = P.processed / "folder18_page_types.csv"
TARGET_H = 1200


def _profile_features(path_str):
    """Row-profile periodicity + blank-row share. Top-level for the process pool."""
    try:
        with Image.open(path_str) as im:
            im = im.convert("L")
            w, h = im.size
            if h > TARGET_H:
                s = TARGET_H / h
                im = im.resize((max(1, int(w * s)), TARGET_H), Image.BILINEAR)
            a = np.asarray(im, dtype=np.float32)
    except Exception as e:
        return {"err": repr(e)}

    # ink = darker than the page's own midpoint between min and max (polarity safe
    # for these 1-bit / near-bilevel scans)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    thr = (lo + hi) / 2.0
    ink = (a < thr).astype(np.float32)

    prof = ink.sum(axis=1)
    blank_row_share = float((prof <= prof.max() * 0.01).mean()) if prof.max() > 0 else 1.0

    # autocorrelation of the detrended profile -> line rhythm regularity
    p = prof - prof.mean()
    denom = float((p * p).sum())
    if denom > 0:
        ac = np.correlate(p, p, mode="full")[len(p) - 1:] / denom
        lo_lag, hi_lag = 8, min(120, len(ac) - 1)
        periodicity = float(ac[lo_lag:hi_lag].max()) if hi_lag > lo_lag else 0.0
    else:
        periodicity = 0.0

    # Horizontal ink run lengths. This is the discriminating feature: handwritten
    # Modi is written with long sweeping horizontal strokes that run across a large
    # fraction of the text width, while printed Devanagari breaks into
    # character-sized marks. Periodicity alone does NOT separate the two (verified:
    # it mixed the title page and the table of contents in with the handwriting).
    W = ink.shape[1]
    runs = []
    for row in ink[:: max(1, ink.shape[0] // 300)]:          # subsample rows for speed
        if row.max() <= 0:
            continue
        d = np.diff(np.concatenate(([0], row, [0])))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        if len(starts):
            runs.append((ends - starts).max())
    max_run_frac = float(np.percentile(runs, 95) / W) if runs else 0.0

    return {"periodicity": periodicity, "blank_row_share": blank_row_share,
            "ink_share": float(ink.mean()), "max_run_frac": max_run_frac, "err": None}


def main():
    man = pd.read_csv(P.processed / "hhdoc_manifest.csv", dtype={"folder": str})
    f18 = man[man["folder"] == "18"].copy()
    if f18.empty:
        print("folder 18 not found in manifest")
        return
    paths = [str(P.hhdoc_root / r) for r in f18["rel_path"]]
    feats = pmap(_profile_features, paths, workers=default_workers(), desc="folder18")
    for k in ("periodicity", "blank_row_share", "ink_share", "max_run_frac"):
        f18[k] = [f.get(k) for f in feats]

    # page number, to inspect the alternating structure
    f18["page_no"] = (f18["filename"].str.extract(r"page(\d+)", expand=False)
                      .astype(float))

    mrf = pd.to_numeric(f18["max_run_frac"], errors="coerce")
    print("max_run_frac distribution (higher = long sweeping strokes = handwritten):")
    print(mrf.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())
    print("\nhistogram:")
    print(pd.cut(mrf, bins=[0, .05, .1, .15, .2, .25, .3, .4, .5, 1.0],
                 right=False).value_counts().sort_index().to_string())

    # NO AUTOMATIC SPLIT. Two different features were tried and both failed:
    #   * periodicity + median split  -> put the title page and the table of
    #     contents in with the handwriting (the median forced a 50/50 answer).
    #   * max_run_frac + largest-gap  -> found only 6 handwritten pages, far too few.
    # Continuing to tune a threshold here would be optimising against "the number
    # looks about right", with no ground truth to check it against - the exact
    # failure mode that produced the two broken metrics earlier in this audit.
    # Folder 18 is only 88 pages, so hand-labelling is cheap and gives a real answer.
    # We therefore emit the features as EVIDENCE and a numbered sheet for labelling.
    f18["page_type_guess"] = "NEEDS_HUMAN_LABEL"
    f18 = f18.sort_values("page_no").reset_index(drop=True)
    f18.insert(0, "label_id", range(1, len(f18) + 1))
    f18["label"] = ""     # handwritten_modi | printed_devanagari | both | other
    print("\nNO automatic split applied - see comment in this script. "
          "All 88 pages emitted for hand labelling.")

    cols = ["label_id", "label", "filename", "page_no", "rel_path",
            "max_run_frac", "periodicity", "ink_share", "blank_row_share"]
    f18[cols].to_csv(OUT, index=False, encoding="utf-8")

    # numbered sheets, in page order, so the facing-page structure is visible
    root = P.hhdoc_root
    per_sheet = 24          # large thumbnails: the Modi/Devanagari call needs detail
    n_sheets = 0
    for s in range(0, len(f18), per_sheet):
        chunk = f18.iloc[s:s + per_sheet]
        items = [(root / r["rel_path"],
                  f"#{r['label_id']} {r['filename']} r={r['max_run_frac']:.2f}")
                 for _, r in chunk.iterrows()]
        out = P.figures / f"folder18_label_{s//per_sheet + 1:02d}.jpg"
        make_contact_sheet(items, out, thumb=400, cols=4, max_items=per_sheet)
        n_sheets += 1

    print(f"\nwrote {OUT}")
    print(f"labelling sheets: {n_sheets} (outputs/figures/folder18_label_*.jpg)")
    print("labels: handwritten_modi | printed_devanagari | both | other")
    print("\nWHY THIS MATTERS: folder 18 is 'ILLUSTRATIVE MODI DOCUMENTS', a scholarly "
          "edition pairing each handwritten Modi facsimile with its printed Devanagari "
          "transcription plus an English title and date (verified on pages 11-14). "
          "HHDoc is otherwise entirely unlabelled, so this is the only page-level "
          "Modi->Devanagari ground truth in the collection. The task brief says to "
          "flag the whole folder away as printed junk; doing so would discard both "
          "the handwritten pages and the only transcription pairing available.")


if __name__ == "__main__":
    main()
