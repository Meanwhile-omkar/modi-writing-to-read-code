"""12 - folder 18 label template + flag rename (addendum item 1).

Folder 18 is a published book: "Illustrative Modi Documents" (Maharashtra State
Archives) - handwritten Modi facsimile pages followed by printed Devanagari
transcriptions. It is EVALUATION DATA, not junk.

The automatic page-type classifier written in step 11 is unreliable (pages 2, 6,
16, 18, 20 are printed, not handwritten, per human review), so its labels are NOT
used. This script emits a template with filenames only, for a human to fill in.

Also renames flag_non_modi_printed -> flag_book_folder18 in the manifest. The old
flag was true for folder 18 and nothing else (step 03 found zero 1-bit look-alikes
in other folders), so this is a clean rename, not a change of meaning.

Idempotent: an existing template is never overwritten, so human edits are safe.
Seed 42.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modi.utils import console  # noqa: F401,E402
from modi.utils.paths import P  # noqa: E402

MANIFEST = P.processed / "hhdoc_manifest.csv"
TEMPLATE = P.processed / "folder18_labels_TEMPLATE.csv"

PAGE_TYPES = ["modi_facsimile", "devanagari_transcription",
              "english_or_front_matter", "mixed", "blank"]


def main():
    df = pd.read_csv(MANIFEST, dtype={"folder": str})

    # ---- flag rename ----
    if "flag_non_modi_printed" in df.columns:
        old = df["flag_non_modi_printed"].fillna(False).astype(bool)
        f18 = df["folder"] == "18"
        mismatch = int((old != f18).sum())
        df["flag_book_folder18"] = f18
        df = df.drop(columns=["flag_non_modi_printed"])
        print(f"renamed flag_non_modi_printed -> flag_book_folder18 "
              f"(rows where old flag != (folder==18): {mismatch}, expected 0)")
    elif "flag_book_folder18" not in df.columns:
        df["flag_book_folder18"] = df["folder"] == "18"
        print("created flag_book_folder18")
    else:
        print("flag_book_folder18 already present")
    df.to_csv(MANIFEST, index=False, encoding="utf-8")
    print(f"folder-18 pages flagged: {int(df['flag_book_folder18'].sum())}")

    # ---- label template (filenames only; never clobber human edits) ----
    if TEMPLATE.exists():
        print(f"template already exists, left untouched: {TEMPLATE}")
        cur = pd.read_csv(TEMPLATE)
        filled = int((cur["page_type"].fillna("").astype(str).str.strip() != "").sum())
        print(f"  rows: {len(cur)}, filled page_type: {filled}")
        return

    f18 = df[df["folder"] == "18"].copy()
    f18["page_no"] = f18["filename"].str.extract(r"page(\d+)", expand=False).astype(float)
    f18 = f18.sort_values("page_no")
    out = pd.DataFrame({
        "filename": f18["filename"].values,
        "page_type": "",      # one of PAGE_TYPES - human fills in
        "pair_id": "",        # same id for a facsimile and its transcription
        "notes": "",
    })
    out.to_csv(TEMPLATE, index=False, encoding="utf-8")
    print(f"wrote {TEMPLATE} ({len(out)} rows, filenames only)")
    print(f"page_type vocabulary: {' | '.join(PAGE_TYPES)}")
    print("pair_id: give a facsimile page and its transcription page the same id")
    print("labelling sheets to view: outputs/figures/folder18_label_*.jpg")


if __name__ == "__main__":
    main()
