"""09 - audit report -> outputs/reports/data_audit.md and finalise DATA_DECISIONS_TODO.md

Consolidates counts before/after each step, flags, thresholds + justification,
timing, and links to contact sheets. Groups every needs_review item by type.
Idempotent. Reads only processed artifacts (re-run after steps 01-08).
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modi.utils import console  # noqa: F401,E402  (force utf-8 stdout)


from modi.utils.paths import P  # noqa: E402

AUDIT = P.reports / "data_audit.md"
TODO = P.reports / "DATA_DECISIONS_TODO.md"


def exists(p: Path):
    return p.exists()


def read_csv_safe(p: Path):
    """Read a CSV that may be empty (a filter that matched nothing writes no header)."""
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main():
    man = pd.read_csv(P.processed / "hhdoc_manifest.csv", dtype={"folder": str})
    L, T = [], []
    def w(s=""):
        L.append(s)
    def t(s=""):
        T.append(s)

    w(f"# MODI Data Audit Report")
    w(f"_Generated {date.today().isoformat()} — data audit & cleaning only (no training)._")
    w("")

    # ---------- HHDoc ----------
    w("## HHDoc (handwritten document pages)")
    w("")
    w(f"- Total images inventoried: **{len(man)}**")
    w(f"- Unreadable (flagged): **{int(man['md5'].isna().sum())}**")
    kept = int((man['keep'] == True).sum()) if 'keep' in man else len(man)  # noqa: E712
    removed = int((man['keep'] == False).sum()) if 'keep' in man else 0  # noqa: E712
    w(f"- Duplicates removed (keep=False): **{removed}**")
    w(f"- Pages kept (keep=True): **{kept}**")
    w("")
    w("### Per-folder counts")
    w("")
    w("| folder | images |")
    w("|--------|--------|")
    for f in sorted(man["folder"].unique(), key=lambda x: int(x) if str(x).isdigit() else 1e9):
        w(f"| {f} | {int((man['folder']==f).sum())} |")
    w("")
    w("### PIL mode distribution")
    w("")
    for m, c in man["mode"].value_counts(dropna=False).items():
        w(f"- `{m}`: {c}")
    w("")

    if "flag_blank" in man:
        w("### Quality flags")
        w("")
        for col in ["flag_blank", "flag_book_folder18", "flag_unusual_shape"]:
            if col in man.columns:
                w(f"- **{col}**: {int(man[col].sum())}")
        w("")
        w("**Thresholds & justification**")
        w("")
        w("- *flag_blank*: the brief's `ink_ratio` (fraction below Otsu) is "
          "background-polarity-dependent — dark-background manuscript scans give "
          "`ink_ratio`≈1.0 while white-background pages give ≈0.0 — so it cannot "
          "detect blanks alone. We added polarity-robust `ink_ratio_fg` (minority-"
          "side fraction ≈ true foreground) and `gray_std`. A page is flagged blank "
          "only when `ink_ratio_fg` is in the robust bottom tail AND `gray_std` is "
          "low (near-uniform). See step-03 console log for the exact chosen cut.")
        w("- *flag_book_folder18* (was `flag_non_modi_printed`): folder 18 is the "
          "published book **\"Illustrative Modi Documents\"** (Maharashtra State "
          "Archives) - handwritten Modi facsimile pages followed by printed "
          "Devanagari transcriptions. It is **evaluation data, not junk**: it is "
          "routed to its own split `ood_test_folder18` and never enters train, val, "
          "pool or pseudo-labelling. The automatic page-type classifier was found "
          "unreliable, so page types are left for human labelling "
          "(`folder18_labels_TEMPLATE.csv`).")
        w("- *flag_unusual_shape*: aspect ratio > 2.5 or < 0.4 (scrolls/strips). "
          "This is a **flag only** - these pages remain in the `pool` split.")
        w("")
        w("### Contact sheets (for human review)")
        for f in sorted(P.figures.glob("*.jpg")):
            w(f"- [{f.name}](../figures/{f.name})")
        w("")

    # dedup detail
    rp = P.processed / "hhdoc_dedup_review_pairs.csv"
    if rp.exists():
        n = len(read_csv_safe(rp))
        w(f"- Grey-zone near-duplicate pairs (pHash dist 5-10) flagged for review: "
          f"**{n}** (`hhdoc_dedup_review_pairs.csv`)")
        w("")

    # ---------- MoDeTrans ----------
    w("## MoDeTrans (line/patch images with Devanagari transcription)")
    w("")
    if exists(P.processed / "modetrans_text.csv"):
        tx = pd.read_csv(P.processed / "modetrans_text.csv")
        w(f"- Rows: **{len(tx)}**")
        w(f"- needs_review: **{int(tx['needs_review'].sum())}**")
        w("")
    cs = P.processed / "charset.json"
    if cs.exists():
        obj = json.loads(cs.read_text(encoding="utf-8"))
        w(f"- Charset symbols: **{obj['n_symbols']}** + special tokens "
          f"{obj['special_tokens']}")
        w("- `σ` (U+03C3) normalised to special token `<FRAC>` (fraction marker).")
        w("")
    if exists(P.reports / "modetrans_text_audit.md"):
        w("- Full character/consistency audit: "
          "[modetrans_text_audit.md](modetrans_text_audit.md)")
        w("")

    # overlap
    mm = P.processed / "modetrans_hhdoc_matches.csv"
    pilot = P.processed / "modetrans_hhdoc_matches_pilot.csv"
    w("### MoDeTrans <-> HHDoc overlap")
    w("")
    if mm.exists():
        o = read_csv_safe(mm)
        vc = o["status"].value_counts().to_dict()
        w(f"- Full run status counts: {vc}")
        w(f"- Confirmed crops (status=match): **{int((o['status']=='match').sum())}**")
        w(f"- Flagged for review (inliers 12-25): **{int((o['status']=='review').sum())}**")
    elif pilot.exists():
        o = read_csv_safe(pilot)
        w(f"- PILOT only ({len(o)} patches). Full run not yet executed — see step-06 "
          "timing estimate in console log before `--full`.")
        w(f"- pilot status counts: {o['status'].value_counts().to_dict()}")
    else:
        w("- Not run.")
    w("")

    # ---------- HChar ----------
    w("## HChar (isolated handwritten characters)")
    w("")
    if exists(P.processed / "hchar_manifest.csv"):
        hc = pd.read_csv(P.processed / "hchar_manifest.csv")
        w(f"- Total files: **{len(hc)}**  unreadable: {int(hc['md5'].isna().sum())}")
        w(f"- Categories/classes: " +
          ", ".join(f"{cat}={hc[hc['category']==cat]['class'].nunique()}"
                    for cat in hc['category'].unique()))
        cross = P.processed / "hchar_crossclass_dups.csv"
        if cross.exists():
            w(f"- Across-class duplicate groups (review): "
              f"**{len(read_csv_safe(cross))}** (`hchar_crossclass_dups.csv`)")
        w("")
    else:
        w("- Not run.")
        w("")

    # ---------- Splits ----------
    w("## Splits")
    w("")
    sp = P.processed / "splits.json"
    if sp.exists():
        s = json.loads(sp.read_text(encoding="utf-8"))
        w(f"- Seed: {s.get('seed')}")
        if "modetrans" in s:
            w(f"- MoDeTrans (by doc + identical-text groups): {s['modetrans']['counts']}")
        if "hhdoc" in s:
            w(f"- HHDoc: {s['hhdoc']['counts']}")
        if "hchar" in s:
            w(f"- HChar ({s['hchar'].get('by','')}): {s['hchar']['counts']}")
            if s["hchar"].get("reason"):
                w(f"  - {s['hchar']['reason']}")
        w("")

    fixes = P.reports / "audit_fixes.md"
    if fixes.exists():  # step 14 corrections are part of the audit record
        L += ["", "---", "", fixes.read_text(encoding="utf-8")]
    AUDIT.write_text("\n".join(L), encoding="utf-8")

    # =============== DATA_DECISIONS_TODO ===============
    t("# DATA DECISIONS — human review required")
    t(f"_Generated {date.today().isoformat()}. Nothing below was auto-deleted or "
      "auto-excluded; these are flags for a human to decide._")
    t("")

    # HHDoc review items
    if "review_note" in man:
        rn = man[man["review_note"].fillna("").str.len() > 0]
        t(f"## HHDoc flagged pages ({len(rn)})")
        t("")
        by = {}
        for _, r in rn.iterrows():
            for tag in str(r["review_note"]).split(";"):
                if tag:
                    by.setdefault(tag, []).append(f"{r['folder']}/{r['filename']}")
        for tag, files in sorted(by.items()):
            if tag == "unusual-shape":
                t(f"### {tag} ({len(files)}) — FLAG ONLY, no decision needed")
                t("")
                t("These pages stay in the `pool` split. Listed for awareness only: "
                  "they are scrolls/strips (aspect ratio > 2.5 or < 0.4) and may need "
                  "different line-segmentation handling downstream.")
                t("")
                continue
            t(f"### {tag} ({len(files)})")
            t("")
            t(", ".join(f"`{x}`" for x in files[:60]) + (" ..." if len(files) > 60 else ""))
            t("")

    # Only the UNRESOLVED remainder belongs here. Step 13 settled the grey-zone
    # pairs geometrically (SIFT + RANSAC); near_duplicate and distinct are decided
    # and must not be re-litigated by a human.
    gz = P.processed / "hhdoc_greyzone_verified.csv"
    rp = P.processed / "hhdoc_dedup_review_pairs.csv"
    if gz.exists():
        d = read_csv_safe(gz)
        nr = d[d["outcome"] == "needs_review"] if "outcome" in d.columns else d
        counts = d["outcome"].value_counts().to_dict() if "outcome" in d.columns else {}
        t(f"## HHDoc near-duplicate grey-zone pairs — UNRESOLVED only ({len(nr)})")
        t("")
        t(f"Step 13 verified all {len(d)} pHash 5-10 pairs with SIFT + Lowe ratio + "
          f"RANSAC: {counts}. Only the `needs_review` remainder is listed here "
          "(inliers 15-40, or an out-of-range homography area ratio).")
        t("")
        for _, r in nr.head(80).iterrows():
            ar = r.get("area_ratio")
            ar_s = f"{ar:.2f}" if pd.notna(ar) else "n/a"
            t(f"- `{r['a_rel']}` ~ `{r['b_rel']}` "
              f"(pHash {r['phash_dist']}, inliers {r['inliers']}, area ratio {ar_s})")
        if len(nr) > 80:
            t(f"- ... and {len(nr)-80} more (see `hhdoc_greyzone_verified.csv`)")
        t("")
    elif rp.exists():
        d = read_csv_safe(rp)
        t(f"## HHDoc near-duplicate grey-zone pairs — pHash 5-10 ({len(d)})")
        t("")
        for _, r in d.head(80).iterrows():
            t(f"- `{r['a_rel']}` ~ `{r['b_rel']}` (dist {r['phash_dist']})")
        t("")

    # MoDeTrans text review
    if exists(P.processed / "modetrans_text.csv"):
        tx = pd.read_csv(P.processed / "modetrans_text.csv")
        nr = tx[tx["needs_review"] == True]  # noqa: E712
        t(f"## MoDeTrans text rows needing review ({len(nr)})")
        t("")
        by = {}
        for _, r in nr.iterrows():
            for rsn in str(r.get("review_reasons", "")).split(";"):
                if rsn:
                    by.setdefault(rsn, []).append(r["image_id"])
        for rsn, ids in sorted(by.items()):
            t(f"### {rsn} ({len(ids)})")
            t("")
            t(", ".join(f"`{x}`" for x in ids[:80]) + (" ..." if len(ids) > 80 else ""))
            t("")

    # HChar cross-class dups
    cross = P.processed / "hchar_crossclass_dups.csv"
    if cross.exists():
        d = read_csv_safe(cross)
        t(f"## HChar across-class duplicate groups ({len(d)})")
        t("")
        for _, r in d.head(60).iterrows():
            t(f"- classes [{r['classes']}] : {r['files']}")
        if len(d) > 60:
            t(f"- ... and {len(d)-60} more (see `hchar_crossclass_dups.csv`)")
        t("")

    # overlap review
    for cand in [P.processed / "modetrans_hhdoc_matches.csv",
                 P.processed / "modetrans_hhdoc_matches_pilot.csv"]:
        if cand.exists():
            o = read_csv_safe(cand)
            rev = o[o["status"] == "review"]
            t(f"## MoDeTrans<->HHDoc matches needing review — inliers 12-25 ({len(rev)})")
            t("")
            for _, r in rev.head(80).iterrows():
                t(f"- `{r['patch_id']}` -> {r['matched_folder']}/{r['matched_file']} "
                  f"(inliers {r['inliers']})")
            t("")
            break

    TODO.write_text("\n".join(T), encoding="utf-8")
    print(f"wrote {AUDIT}")
    print(f"wrote {TODO}")


if __name__ == "__main__":
    main()
