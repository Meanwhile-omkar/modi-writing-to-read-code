"""04 - MoDeTrans text audit -> outputs/reports/modetrans_text_audit.md
and processed/modetrans_charset_raw.json

Checks parquet rows <-> image files <-> metadata rows 1:1. Produces character
frequency table (with Unicode names), rows with suspicious characters, newline
rows, identical-text groups, length stats. Read-only. Idempotent. Seed 42.
"""
from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modi.utils import console  # noqa: F401,E402  (force utf-8 stdout)


from modi.utils.paths import P  # noqa: E402

SEED = 42
REPORT = P.reports / "modetrans_text_audit.md"
CHARSET_RAW = P.processed / "modetrans_charset_raw.json"

SIGMA = "\u03c3"          # σ  -> fraction marker per brief
ZWNJ = "\u200c"
ZWJ = "\u200d"
BRACKETS = set("[]()")
QMARK = "?"


def uname(ch: str) -> str:
    try:
        return unicodedata.name(ch)
    except ValueError:
        return f"U+{ord(ch):04X}-UNNAMED"


def main() -> None:
    df = pd.read_parquet(P.modetrans_parquet)
    meta = pd.read_csv(P.modetrans_meta, encoding="utf-8-sig")
    img_dir = P.modetrans_images
    disk = {f.name for f in img_dir.iterdir() if f.is_file()}

    pq_files = set(df["filename"])
    meta_files = set(meta["filename"])

    L = []  # report lines
    def w(s=""):
        L.append(s)

    w("# MoDeTrans Text Audit")
    w("")
    w(f"- parquet rows: **{len(df)}**")
    w(f"- metadata rows: **{len(meta)}**")
    w(f"- image files on disk: **{len(disk)}**")
    w("")
    w("## 1:1 consistency (parquet <-> images <-> metadata)")
    w(f"- parquet filenames not on disk: {len(pq_files - disk)} {sorted(pq_files - disk)[:10]}")
    w(f"- disk files not in parquet: {len(disk - pq_files)} {sorted(disk - pq_files)[:10]}")
    w(f"- parquet filenames not in metadata: {len(pq_files - meta_files)} {sorted(pq_files - meta_files)[:10]}")
    w(f"- metadata filenames not in parquet: {len(meta_files - pq_files)} {sorted(meta_files - pq_files)[:10]}")
    dup_pq = df["filename"].value_counts()
    w(f"- duplicate filenames in parquet: {int((dup_pq > 1).sum())}")
    ok = (pq_files == disk == meta_files) and (dup_pq.max() == 1)
    w(f"- **1:1 match overall: {'YES' if ok else 'NO — see above'}**")
    w("")

    # ---- text source: use parquet 'text' (ground truth for images) ----
    text = df["text"].fillna("").astype(str)

    # ---- character frequency ----
    cnt = Counter()
    for t in text:
        cnt.update(t)
    charset_raw = sorted(cnt.keys())
    CHARSET_RAW.write_text(
        json.dumps(
            {
                "n_rows": len(df),
                "n_unique_chars": len(charset_raw),
                "chars": [
                    {"char": c, "codepoint": f"U+{ord(c):04X}", "name": uname(c), "count": cnt[c]}
                    for c in sorted(cnt, key=lambda c: -cnt[c])
                ],
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    w("## Character frequency (full table)")
    w("")
    w("| char | codepoint | unicode name | count |")
    w("|------|-----------|--------------|-------|")
    for c in sorted(cnt, key=lambda c: -cnt[c]):
        disp = c
        if c == "\n":
            disp = "\\n"
        elif c == "\t":
            disp = "\\t"
        elif c == " ":
            disp = "(space)"
        elif c in (ZWNJ, ZWJ):
            disp = "(ZW)"
        w(f"| `{disp}` | U+{ord(c):04X} | {uname(c)} | {cnt[c]} |")
    w("")

    # ---- suspicious-row buckets ----
    def rows_with(pred):
        idx = [i for i, t in enumerate(text) if pred(t)]
        return idx

    buckets = {
        "contains σ (SIGMA / fraction)": rows_with(lambda t: SIGMA in t),
        "contains TAB": rows_with(lambda t: "\t" in t),
        "contains ZWNJ/ZWJ": rows_with(lambda t: ZWNJ in t or ZWJ in t),
        "contains ASCII digit": rows_with(lambda t: any(ch.isascii() and ch.isdigit() for ch in t)),
        "contains ASCII letter": rows_with(lambda t: any(ch.isascii() and ch.isalpha() for ch in t)),
        "contains bracket [ ] ( )": rows_with(lambda t: any(ch in BRACKETS for ch in t)),
        "contains '?'": rows_with(lambda t: QMARK in t),
        "contains Latin char": rows_with(
            lambda t: any("LATIN" in uname(ch) for ch in t if ch.isalpha())),
        "has internal newline": rows_with(lambda t: "\n" in t.strip("\n")),
    }
    w("## Suspicious-character row counts")
    w("")
    w("| bucket | n_rows |")
    w("|--------|--------|")
    for k, idx in buckets.items():
        w(f"| {k} | {len(idx)} |")
    w("")

    # detailed listings (filename + raw text) for the trickiest buckets
    for key in ["contains σ (SIGMA / fraction)", "contains ASCII digit",
                "contains bracket [ ] ( )", "contains '?'", "contains Latin char",
                "has internal newline"]:
        idx = buckets[key]
        w(f"### Rows: {key} ({len(idx)})")
        w("")
        for i in idx[:50]:
            raw = text.iloc[i].replace("\n", "\\n").replace("\t", "\\t")
            w(f"- `{df['filename'].iloc[i]}`: {raw[:160]}")
        if len(idx) > 50:
            w(f"- ... and {len(idx) - 50} more")
        w("")

    # ---- identical-text groups ----
    grp = df.groupby(text.values)
    dup_groups = [(t, list(g["filename"])) for t, g in grp if len(g) > 1]
    dup_groups.sort(key=lambda x: -len(x[1]))
    w(f"## Identical-text groups: {len(dup_groups)} groups covering "
      f"{sum(len(g) for _, g in dup_groups)} rows")
    w("")
    for t, files in dup_groups[:40]:
        disp = t.replace("\n", "\\n")[:80]
        w(f"- ({len(files)}) `{disp}` -> {files[:8]}{' ...' if len(files) > 8 else ''}")
    if len(dup_groups) > 40:
        w(f"- ... and {len(dup_groups) - 40} more groups")
    w("")

    # ---- length stats ----
    lens = text.str.len()
    words = text.str.split().apply(len)
    w("## Length statistics")
    w("")
    w("```")
    w("char length:\n" + lens.describe().to_string())
    w("\nword count:\n" + words.describe().to_string())
    w("```")
    w("")

    # ---- metadata columns description ----
    w("## Metadata columns")
    w("")
    for col in meta.columns:
        dt = str(meta[col].dtype)
        nun = meta[col].nunique(dropna=True)
        sample = meta[col].dropna().astype(str).head(2).tolist()
        w(f"- **{col}** ({dt}, {nun} unique) e.g. {sample}")
    w("")

    REPORT.write_text("\n".join(L), encoding="utf-8")

    # ---- console summary ----
    print("=== MODETRANS TEXT AUDIT ===")
    print(f"1:1 match: {'YES' if ok else 'NO'}")
    print(f"unique chars: {len(charset_raw)}")
    for k, idx in buckets.items():
        print(f"  {k}: {len(idx)}")
    print(f"identical-text groups: {len(dup_groups)}")
    print(f"wrote {REPORT}")
    print(f"wrote {CHARSET_RAW}")


if __name__ == "__main__":
    main()
