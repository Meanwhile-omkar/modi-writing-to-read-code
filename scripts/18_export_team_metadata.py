"""18 - export the small derived files teammates need, so `git clone` is enough.

Copies (never modifies) into the repo:
  assets/metadata/   splits.json, modetrans_text.csv, charset.json, modetrans_docs.csv,
                     folder18_labels.csv   (about 1.8 MB, all text)
Builds next to the data folder (NOT in git, for a shared drive):
  FYP-modi-script-data/folder18_pages.zip   the 88 page images of the book, 7 MB

Raw MoDeTrans images and HHDoc pages are never copied into the repo; teammates download them
from the public links in the README.
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modi.utils import console  # noqa: F401,E402
from modi.utils.paths import P  # noqa: E402

FILES = ["splits.json", "modetrans_text.csv", "charset.json", "modetrans_docs.csv", "folder18_labels.csv"]
REPO = Path(__file__).resolve().parents[1]

README = """# Team metadata (derived files, safe to commit)

| File | What it is |
|---|---|
| `splits.json` | which MoDeTrans patch is train / val / test (seed 42); also HHDoc and HChar assignments |
| `modetrans_text.csv` | expert transcriptions: `text_raw` (original), `text_norm`, `text_train` (what models learn) |
| `charset.json` | the 86 symbols + `<FRAC>`, `<blank>`, `<unk>` |
| `modetrans_docs.csv` | patch -> source-document grouping |
| `folder18_labels.csv` | page type, letter number and test tier for the 88 pages of the book (folder 18) |

Images are NOT here. MoDeTrans images come from Hugging Face (`historyHulk/MoDeTrans`);
folder-18 pages come from the HHDoc download or `folder18_pages.zip`.
Regenerate with `python scripts/18_export_team_metadata.py`.
"""


def main():
    out = REPO / "assets" / "metadata"
    out.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copyfile(P.processed / f, out / f)
    (out / "README.md").write_text(README, encoding="utf-8")
    print(f"copied {len(FILES)} files to {out} ({sum((out / f).stat().st_size for f in FILES) / 1e6:.1f} MB)")

    zpath = P.data_root / "folder18_pages.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for img in sorted((P.hhdoc_root / "18").glob("*.jpg")):
            z.write(img, f"18/{img.name}")
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
