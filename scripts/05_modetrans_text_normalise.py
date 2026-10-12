"""05 - MoDeTrans text normalise -> processed/modetrans_text.csv and processed/charset.json

Columns: image_id, text_raw (unchanged), text_norm, changes (list), needs_review (bool).
All normalisation is reversible because text_raw is preserved.
Idempotent. Seed 42.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modi.utils import console  # noqa: F401,E402  (force utf-8 stdout)


from modi.utils.paths import P  # noqa: E402

SEED = 42
OUT_CSV = P.processed / "modetrans_text.csv"
OUT_CHARSET = P.processed / "charset.json"

SIGMA = "σ"
FRAC_TOKEN = "<FRAC>"
ZWNJ = "‌"
ZWJ = "‍"
EDITORIAL = set("[]?")  # editorial marks that force review
ASCII_DIGITS = set("0123456789")


def normalise(raw: str):
    changes = []
    needs_review = False
    review_reasons = []
    t = raw

    # Unicode NFC
    nfc = unicodedata.normalize("NFC", t)
    if nfc != t:
        changes.append("nfc")
    t = nfc

    # tabs -> single space
    if "\t" in t:
        t = t.replace("\t", " ")
        changes.append("tab->space")

    # strip trailing newlines then leading/trailing whitespace
    if t != t.strip():
        changes.append("strip")
    t = t.strip("\n").strip()

    # collapse multiple spaces (not newlines)
    if re.search(r"  +", t):
        t = re.sub(r" {2,}", " ", t)
        changes.append("collapse_spaces")
    # also collapse spaces around internal newlines but keep newline
    t = re.sub(r"[ ]*\n[ ]*", "\n", t)

    # σ -> <FRAC> token
    if SIGMA in t:
        t = t.replace(SIGMA, FRAC_TOKEN)
        changes.append("sigma->FRAC")

    # ZWNJ/ZWJ: keep, but count/flag (do not remove)
    nz = t.count(ZWNJ) + t.count(ZWJ)
    if nz:
        changes.append(f"has_zw({nz})")

    # ASCII digits: do NOT convert -> review
    if any(ch in ASCII_DIGITS for ch in t):
        needs_review = True
        review_reasons.append("ascii_digit")

    # editorial marks [ ] ? : keep -> review
    if any(ch in EDITORIAL for ch in t):
        needs_review = True
        review_reasons.append("editorial_mark")

    return t, changes, needs_review, review_reasons


def main() -> None:
    df = pd.read_parquet(P.modetrans_parquet)
    rows = []
    charset = set()
    n_review = 0
    reason_counts = {}
    for _, r in df.iterrows():
        raw = "" if pd.isna(r["text"]) else str(r["text"])
        norm, changes, nr, reasons = normalise(raw)
        charset.update(ch for ch in norm if ch not in ("\n",))  # keep newline out of symbol list
        if nr:
            n_review += 1
        for rsn in reasons:
            reason_counts[rsn] = reason_counts.get(rsn, 0) + 1
        rows.append({
            "image_id": r["filename"],
            "text_raw": raw,
            "text_norm": norm,
            "changes": json.dumps(changes, ensure_ascii=False),
            "needs_review": nr,
            "review_reasons": ";".join(reasons),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # charset.json: sorted symbols in text_norm (excluding the <FRAC> placeholder
    # chars themselves) + special tokens.
    # Remove the characters that only come from the <FRAC> token so the symbol list
    # reflects true script symbols; represent <FRAC> as a special token instead.
    symbols = sorted(s for s in charset if s not in set(FRAC_TOKEN))
    charset_obj = {
        "symbols": symbols,
        "n_symbols": len(symbols),
        "special_tokens": ["<FRAC>", "<blank>", "<unk>"],
        "notes": {
            "<FRAC>": "replaces Greek small letter sigma (U+03C3) used to mark fractions in source",
            "<blank>": "CTC blank token",
            "<unk>": "unknown/out-of-vocabulary symbol",
            "newline": "internal newlines are preserved in text_norm but excluded from the symbol list",
        },
    }
    OUT_CHARSET.write_text(json.dumps(charset_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== TEXT NORMALISE SUMMARY ===")
    print(f"rows: {len(out)}")
    print(f"needs_review: {n_review}")
    print(f"review reasons: {reason_counts}")
    print(f"symbols in charset: {len(symbols)}")
    # change-type frequency
    from collections import Counter
    cc = Counter()
    for c in out["changes"]:
        for x in json.loads(c):
            cc[x.split("(")[0]] += 1
    print(f"change types applied: {dict(cc)}")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_CHARSET}")


if __name__ == "__main__":
    main()
