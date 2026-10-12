"""08 - make splits -> processed/splits.json

Seed 42. Split by SOURCE DOCUMENT, never by patch.
- MoDeTrans: split by modetrans_doc_id, 70/15/15 by patch count.
  Identical-text duplicate groups (step 04/05) are kept in ONE split (units are
  connected components over {shared doc_id} U {identical text_norm}).
- HHDoc: pages matched to a MoDeTrans val/test patch -> "heldout" (never train).
  Other keep=True, unflagged pages -> "pool".
- HChar: stratified 80/10/10 per class.
Prints leakage checks. Idempotent. Seed 42.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modi.utils import console  # noqa: F401,E402  (force utf-8 stdout)


from modi.utils.paths import P  # noqa: E402

SEED = 42
rng = np.random.default_rng(SEED)
OUT = P.processed / "splits.json"


def connected_components(nodes, edges):
    parent = {n: n for n in nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    comp = defaultdict(list)
    for n in nodes:
        comp[find(n)].append(n)
    return list(comp.values())


def assign_7015(units_with_sizes, ratios=(0.70, 0.15, 0.15)):
    """Greedy largest-first bin packing to hit target fractions by item count."""
    total = sum(s for _, s in units_with_sizes)
    targets = [r * total for r in ratios]
    cur = [0, 0, 0]
    names = ["train", "val", "test"]
    assignment = {}
    order = sorted(units_with_sizes, key=lambda x: -x[1])
    for unit, size in order:
        # choose split with largest remaining deficit
        deficit = [targets[i] - cur[i] for i in range(3)]
        j = int(np.argmax(deficit))
        assignment[unit] = names[j]
        cur[j] += size
    return assignment, cur, total


def main():
    splits = {"seed": SEED}

    # ================= MoDeTrans =================
    pm = pd.read_parquet(P.modetrans_parquet)[["filename"]]
    docs_path = P.processed / "modetrans_docs.csv"
    text_path = P.processed / "modetrans_text.csv"
    if docs_path.exists():
        docs = pd.read_csv(docs_path)
        doc_of = dict(zip(docs["image_id"], docs["modetrans_doc_id"]))
    else:
        print("WARN: modetrans_docs.csv missing (step 06 --full not run). "
              "Falling back to per-patch doc ids.")
        doc_of = {fn: f"patch::{fn}" for fn in pm["filename"]}

    patches = pm["filename"].tolist()
    # identical-text groups
    text_edges = []
    if text_path.exists():
        tx = pd.read_csv(text_path)
        tnorm = dict(zip(tx["image_id"], tx["text_norm"].fillna("")))
        by_text = defaultdict(list)
        for fn in patches:
            by_text[tnorm.get(fn, "")].append(fn)
        id_text_groups = 0
        for t, fns in by_text.items():
            if t.strip() == "":
                continue
            if len(fns) > 1:
                id_text_groups += 1
                for k in range(1, len(fns)):
                    text_edges.append((fns[0], fns[k]))
    else:
        id_text_groups = 0

    # edges from shared doc_id
    doc_groups = defaultdict(list)
    for fn in patches:
        doc_groups[doc_of.get(fn, f"patch::{fn}")].append(fn)
    doc_edges = []
    for d, fns in doc_groups.items():
        for k in range(1, len(fns)):
            doc_edges.append((fns[0], fns[k]))

    comps = connected_components(patches, doc_edges + text_edges)
    units = [(f"comp{i}", len(c)) for i, c in enumerate(comps)]
    assign, cur, total = assign_7015(units)
    comp_members = {f"comp{i}": c for i, c in enumerate(comps)}
    mt_split = {}
    for uid, sp in assign.items():
        for fn in comp_members[uid]:
            mt_split[fn] = sp

    splits["modetrans"] = {
        "by": "connected components over shared modetrans_doc_id + identical text_norm",
        "n_patches": len(patches),
        "n_units": len(units),
        "n_identical_text_groups_merged": id_text_groups,
        "counts": {s: int(sum(1 for v in mt_split.values() if v == s)) for s in ["train", "val", "test"]},
        "assignment": mt_split,
    }

    # ================= HHDoc heldout / pool =================
    man = pd.read_csv(P.processed / "hhdoc_manifest.csv", dtype={"folder": str})
    keep = man[(man["keep"] == True) & (man["md5"].notna())].copy()  # noqa: E712
    # flag_unusual_shape is a FLAG ONLY: those pages stay in the pool. Only blank
    # candidates are held out of the pool, and folder 18 is routed to its own split.
    # flag_true_blank (step 14) replaces flag_blank: most blank candidates carry faint
    # or sparse handwriting and belong in the pool.
    blank_col = "flag_true_blank" if "flag_true_blank" in keep.columns else "flag_blank"
    flag_cols = [c for c in [blank_col] if c in keep.columns]
    keep["any_flag"] = keep[flag_cols].any(axis=1) if flag_cols else False

    matches_path = P.processed / "modetrans_hhdoc_matches.csv"
    heldout_pages = set()
    f18_overlap = set()  # folder-18 pages that also appear as MoDeTrans patches
    if matches_path.exists():
        mt = pd.read_csv(matches_path)
        mt = mt[mt["status"] == "match"]
        valtest = {fn for fn, s in mt_split.items() if s in ("val", "test")}

        def _folder_str(v):
            # matched_folder round-trips through CSV as a float (1.0, 18.0) because
            # the column is numeric with NaNs. Rebuild the original "1" / "18" form,
            # otherwise the "<folder>/<file>" key never matches the manifest.
            if pd.isna(v):
                return None
            if isinstance(v, float) and v.is_integer():
                return str(int(v))
            return str(v)

        for _, r in mt.iterrows():
            f = _folder_str(r["matched_folder"])
            if f == "18":
                f18_overlap.add(f"18/{r['matched_file']}")
            if r["patch_id"] in valtest and f is not None:
                heldout_pages.add(f"{f}/{r['matched_file']}")
    else:
        print("WARN: modetrans_hhdoc_matches.csv missing (step 06 --full not run). "
              "No HHDoc heldout can be derived; all usable pages go to pool.")

    hhdoc_assign = {}
    for _, r in keep.iterrows():
        key = f"{r['folder']}/{r['filename']}"
        if r["rel_path"] in f18_overlap:
            # Seen by MoDeTrans (any split): cannot be an independent test page.
            hhdoc_assign[r["rel_path"]] = "excluded_overlaps_modetrans"
        elif r["folder"] == "18":
            # Folder 18 is the published book "Illustrative Modi Documents":
            # evaluation data, never training or pseudo-labelling material.
            hhdoc_assign[r["rel_path"]] = "ood_test_folder18"
        elif key in heldout_pages:
            hhdoc_assign[r["rel_path"]] = "heldout"
        elif not r["any_flag"]:
            hhdoc_assign[r["rel_path"]] = "pool"
        else:
            hhdoc_assign[r["rel_path"]] = "excluded_flagged"
    splits["hhdoc"] = {
        "note": ("heldout = matched to a MoDeTrans val/test patch; never train/pseudo-label. "
                 "ood_test_folder18 = the published book, held out entirely as OOD evaluation. "
                 "excluded_overlaps_modetrans = folder-18 pages that also appear as MoDeTrans "
                 "patches, removed from the OOD test. "
                 "flag_unusual_shape is a flag only and does NOT remove a page from pool."),
        "counts": {s: int(sum(1 for v in hhdoc_assign.values() if v == s))
                   for s in ["pool", "heldout", "ood_test_folder18",
                             "excluded_overlaps_modetrans", "excluded_flagged"]},
        "assignment": hhdoc_assign,
    }

    # ================= HChar stratified 80/10/10 =================
    # HChar is NOT randomly split. Class counts are near-identical (10,098-10,110),
    # which indicates the set was augmented from a smaller pool of originals. A
    # random split would put different augmentations of the SAME source character
    # into train and test, so any score measured on it would be inflated. The whole
    # set is therefore marked pretrain_only and never used for reported evaluation.
    hc_path = P.processed / "hchar_manifest.csv"
    if hc_path.exists():
        hc = pd.read_csv(hc_path)
        hc = hc[hc["md5"].notna()]
        counts = hc.groupby(["category", "class"]).size()
        splits["hchar"] = {
            "by": "pretrain_only (no evaluation split)",
            "reason": ("class counts are near-identical "
                       f"({int(counts.min())}-{int(counts.max())}), which suggests "
                       "augmentation; random splits would leak, so HChar is not used "
                       "for reported evaluation"),
            "counts": {"pretrain_only": int(len(hc))},
            "assignment_omitted_in_json": True,
            "n": int(len(hc)),
        }
        pd.DataFrame({"rel_path": hc["rel_path"].values, "split": "pretrain_only"}
                     ).to_csv(P.processed / "hchar_splits.csv", index=False, encoding="utf-8")
    else:
        print("WARN: hchar_manifest.csv missing (step 07 not run).")

    OUT.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")

    # ================= leakage checks =================
    print("=== SPLIT SUMMARY ===")
    print("MoDeTrans:", splits["modetrans"]["counts"], "units:", len(units),
          "identical-text groups merged:", id_text_groups)
    print("HHDoc:", splits["hhdoc"]["counts"])
    if "hchar" in splits:
        print("HChar:", splits["hchar"]["counts"])

    print("\n--- leakage checks ---")
    # 1. no doc_id across splits
    doc_split = defaultdict(set)
    for fn, sp in mt_split.items():
        doc_split[doc_of.get(fn, fn)].add(sp)
    bad_doc = {d: s for d, s in doc_split.items() if len(s) > 1}
    print(f"doc_ids spanning >1 split: {len(bad_doc)} (expect 0)")

    # 2. no identical text group across splits
    if text_path.exists():
        bad_text = 0
        for t, fns in by_text.items():
            if t.strip() == "" or len(fns) < 2:
                continue
            set_s = {mt_split[f] for f in fns}
            if len(set_s) > 1:
                bad_text += 1
        print(f"identical-text groups spanning >1 split: {bad_text} (expect 0)")

    # 3. no matched HHDoc page in both heldout and pool / train
    held = {k for k, v in hhdoc_assign.items() if v == "heldout"}
    pool = {k for k, v in hhdoc_assign.items() if v == "pool"}
    print(f"HHDoc pages in both heldout and pool: {len(held & pool)} (expect 0)")

    # 4. NO folder-18 file may appear outside ood_test_folder18
    stray = [rel for rel, sp in hhdoc_assign.items()
             if rel.split("/")[0] == "18"
             and sp not in ("ood_test_folder18", "excluded_overlaps_modetrans")]
    print(f"folder-18 pages outside ood_test_folder18: {len(stray)} (expect 0)")
    if stray:
        print(f"   !! {stray[:10]}")
    n_ood = sum(1 for v in hhdoc_assign.values()
                if v in ("ood_test_folder18", "excluded_overlaps_modetrans"))
    n18_keep = int(((keep["folder"] == "18")).sum())
    print(f"folder-18 kept pages: {n18_keep}, assigned to ood_test_folder18: {n_ood} "
          f"({'OK' if n_ood == n18_keep else 'MISMATCH'})")

    # 5. flag_unusual_shape is a flag only -> those pages must still be in pool
    if "flag_unusual_shape" in man.columns:
        us = keep[keep["flag_unusual_shape"] == True]  # noqa: E712
        us_pool = sum(1 for _, r in us.iterrows()
                      if hhdoc_assign.get(r["rel_path"]) == "pool")
        us_other = {}
        for _, r in us.iterrows():
            sp = hhdoc_assign.get(r["rel_path"])
            if sp != "pool":
                us_other[sp] = us_other.get(sp, 0) + 1
        print(f"\nflag_unusual_shape kept pages: {len(us)}; in pool: {us_pool}; "
              f"elsewhere: {us_other if us_other else 'none'}")
        print("unusual-shape breakdown by folder:")
        print(us.groupby("folder").size().sort_index(
            key=lambda s: [int(x) for x in s]).to_string())

    # 6. no md5 duplicate crossing MoDeTrans splits (patch-level sanity)
    # 7. no folder-18 OOD test page may be a MoDeTrans source page
    leak18 = {k for k, v in hhdoc_assign.items() if v == "ood_test_folder18"} & f18_overlap
    print(f"folder-18 OOD test pages that appear in MoDeTrans: {len(leak18)} (expect 0)")
    # 8. every kept page has a split
    missing = set(keep["rel_path"]) - set(hhdoc_assign)
    print(f"kept HHDoc pages without a split: {len(missing)} (expect 0)")
    print(f"\nMoDeTrans patches assigned: {len(mt_split)} / {len(patches)}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
