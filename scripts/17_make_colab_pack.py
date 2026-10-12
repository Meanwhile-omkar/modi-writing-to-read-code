"""17 - build the small data pack that Colab cannot download by itself.

Colab gets the big data straight from the source, never from this PC:
  - MoDeTrans  -> datasets.load_dataset("historyHulk/MoDeTrans")  (sha256 of the parquet
                  is identical to our local copy, checked Oct 2026)
  - MODI-HHDoc -> Mendeley Data direct link (7.2 GB .rar), only needed after the baseline
Everything else the experiments need is small and lives in this pack:
  metadata/  splits.json, modetrans_text.csv (text_train), charset.json,
             modetrans_docs.csv, folder18_labels.csv, hhdoc_manifest.csv
  folder18/  the 88 page images of the folder-18 book (7 MB, OOD test set)
  MANIFEST.json  sha256 of every file + the expected MoDeTrans parquet hash

Output: ../FYP-modi-script-data/colab_pack.zip  (upload this one file to Google Drive)
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modi.utils import console  # noqa: F401,E402
from modi.utils.paths import P  # noqa: E402

META = ["splits.json", "modetrans_text.csv", "charset.json", "modetrans_docs.csv",
        "folder18_labels.csv", "hhdoc_manifest.csv"]
MODETRANS_PARQUET_SHA256 = "b51ea5f49fd50d094fd878793f4de539c0ecd64c0bd1b65ebe39af27b14868ec"
HHDOC_URL = ("https://data.mendeley.com/public-files/datasets/sg337vf6wn/files/"
             "1a3db10c-80b3-4593-9bc0-694950530b5a/file_downloaded")
HHDOC_RAR_SHA256 = "f4b07bdf0c6c4502398bb7797b78e6c57b741b4e4c72a3f38cd919fdc5020523"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = P.data_root / "colab_pack.zip"
    files = {f"metadata/{n}": P.processed / n for n in META}
    for img in sorted((P.hhdoc_root / "18").glob("*.jpg")):
        files[f"folder18/{img.name}"] = img
    missing = [k for k, v in files.items() if not v.exists()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")

    manifest = {
        "files": {k: sha256(v) for k, v in files.items()},
        "modetrans_hf_dataset": "historyHulk/MoDeTrans",
        "modetrans_parquet_sha256": MODETRANS_PARQUET_SHA256,
        "hhdoc_url": HHDOC_URL,
        "hhdoc_rar_sha256": HHDOC_RAR_SHA256,
    }
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, src in files.items():
            z.write(src, arc)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {len(files)} files)")


if __name__ == "__main__":
    main()
