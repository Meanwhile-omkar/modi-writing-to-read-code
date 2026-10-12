"""16 - folder-18 page labels -> processed/folder18_labels.csv

Folder 18 is the book "Illustrative Modi Documents" (Maharashtra State Archives, ed.
B. G. Kunte, 1978). Labels come from visual inspection of all 88 pages
(outputs/figures/folder18_label_*.jpg), encoded here as the book's structure:

  pages 1-7    front matter (cover, title, imprint, preface, contents)
  pages 8-83   17 letters. EVEN page = printed Devanagari transcription,
               ODD page = handwritten Modi facsimile. Each letter starts on an even
               page carrying an English header "No. N".
  pages 84-88  back matter (about the archives, list of publications)

The transcription is re-flowed, so page breaks in the print do NOT coincide with
page breaks in the facsimile. Ground truth is reliable per LETTER (doc_no), not per
page: evaluate by concatenating a letter's facsimile pages against its full
transcription.

ood_tier for facsimile pages:
  excluded  - the page itself is a MoDeTrans source page (step 14 match)
  partial   - another page of the same letter is in MoDeTrans (same writer)
  strict    - no page of the letter appears in MoDeTrans  -> cleanest OOD test
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modi.utils import console  # noqa: F401,E402
from modi.utils.paths import P  # noqa: E402

# letter number -> first (even) page; each letter runs until the next one starts.
LETTER_START = {1: 8, 2: 10, 3: 14, 4: 16, 5: 20, 6: 22, 7: 30, 8: 40, 9: 42,
                10: 46, 11: 50, 12: 58, 13: 66, 14: 70, 15: 76, 16: 78, 17: 80}
LAST_LETTER_PAGE = 83


def doc_of(page: int):
    if page < 8 or page > LAST_LETTER_PAGE:
        return None
    return max(n for n, s in LETTER_START.items() if s <= page)


def main():
    tpl = pd.read_csv(P.processed / "folder18_labels_TEMPLATE.csv")
    tpl["page"] = tpl["filename"].str.extract(r"(\d+)").astype(int)
    rows = []
    for _, r in tpl.sort_values("page").iterrows():
        p, d = r["page"], doc_of(r["page"])
        if d is None:
            ptype = "english_or_front_matter"
        elif p % 2 == 0:
            ptype = "devanagari_transcription"
        else:
            ptype = "modi_facsimile"
        rows.append({"filename": r["filename"], "page": p, "page_type": ptype,
                     "doc_no": d, "pair_id": f"doc{d:02d}" if d else "",
                     "overlaps_modetrans": bool(r["overlaps_modetrans"])})
    lab = pd.DataFrame(rows)

    hit_docs = set(lab.loc[lab["overlaps_modetrans"], "doc_no"].dropna().astype(int))
    def tier(r):
        if r["page_type"] != "modi_facsimile":
            return ""
        if r["overlaps_modetrans"]:
            return "excluded"
        return "partial" if r["doc_no"] in hit_docs else "strict"
    lab["ood_tier"] = lab.apply(tier, axis=1)
    lab["notes"] = ""
    lab.loc[lab["page"] == 1, "notes"] = "cover shows a cropped Modi facsimile; not used"
    lab.to_csv(P.processed / "folder18_labels.csv", index=False, encoding="utf-8")

    print(lab["page_type"].value_counts().to_string())
    fac = lab[lab["page_type"] == "modi_facsimile"]
    print("\nfacsimile pages by ood_tier:", fac["ood_tier"].value_counts().to_dict())
    print("letters touched by MoDeTrans:", sorted(hit_docs))
    print("strict letters:", sorted(set(fac.loc[fac["ood_tier"] == "strict", "doc_no"].astype(int))))
    # sanity: every letter has >=1 transcription and >=1 facsimile page
    for d in LETTER_START:
        g = lab[lab["doc_no"] == d]["page_type"]
        assert {"devanagari_transcription", "modi_facsimile"} <= set(g), d


if __name__ == "__main__":
    main()
