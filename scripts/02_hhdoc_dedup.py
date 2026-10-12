"""02 - HHDoc dedup -> adds dup_group, dup_type (exact|near), keep (bool) to manifest.

- Exact duplicates by md5.
- Near duplicates by pHash Hamming distance <= 6 (reports full distance histogram
  of nearest-neighbour pairs; pairs in 5-10 are flagged for review, not auto-merged).
- Keep rule: within a duplicate group keep the copy in the LOWER folder number
  (per page, not per folder).
Idempotent. Seed 42.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modi.utils import console  # noqa: F401,E402  (force utf-8 stdout)


from modi.utils.paths import P  # noqa: E402

SEED = 42
MANIFEST = P.processed / "hhdoc_manifest.csv"
NEAR_THR = 6          # pHash Hamming distance <= 6 => near-duplicate
REVIEW_LO, REVIEW_HI = 5, 10  # grey zone to flag for human review


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def folder_rank(folder: str) -> int:
    try:
        return int(folder)
    except (ValueError, TypeError):
        return 10**9


def main() -> None:
    df = pd.read_csv(MANIFEST, dtype={"folder": str})
    n = len(df)
    print(f"Loaded manifest: {n} rows")

    df["dup_group"] = pd.NA
    df["dup_type"] = pd.NA

    # ---- union-find over exact (md5) and near (phash<=6) edges ----
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # exact by md5
    exact_edges = 0
    for md5, idx in df.groupby("md5").groups.items():
        if md5 is None or (isinstance(md5, float)):
            continue
        idx = list(idx)
        for j in idx[1:]:
            union(idx[0], j)
            exact_edges += 1
    exact_pairs_mask = defaultdict(set)  # record exact membership
    for md5, idx in df.groupby("md5").groups.items():
        if len(idx) > 1 and isinstance(md5, str):
            for i in idx:
                exact_pairs_mask[i] = idx

    # near by pHash: brute-force pairwise over valid phashes.
    ph = df["phash"].tolist()
    valid = [i for i in range(n) if isinstance(ph[i], str) and ph[i]]
    print(f"Images with valid pHash: {len(valid)}")
    nn_dist = []          # nearest-neighbour distance per image (for histogram)
    near_edges = 0
    review_pairs = []
    best = {i: 10**9 for i in valid}
    V = valid
    for ii in range(len(V)):
        i = V[ii]
        hi = ph[i]
        for jj in range(ii + 1, len(V)):
            j = V[jj]
            d = hamming_hex(hi, ph[j])
            if d < best[i]:
                best[i] = d
            if d < best[j]:
                best[j] = d
            if d <= NEAR_THR:
                # only union if not already exact-identical md5 (still fine to union)
                union(i, j)
                near_edges += 1
            if REVIEW_LO <= d <= REVIEW_HI:
                review_pairs.append((i, j, d))
    nn_dist = [best[i] for i in valid if best[i] < 10**9]

    # ---- assign groups ----
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    gid = 0
    for root, members in groups.items():
        if len(members) <= 1:
            continue
        gid += 1
        # determine type: exact if all share one md5, else near (mixed)
        md5s = set(df.loc[members, "md5"])
        gtype = "exact" if len(md5s) == 1 else "near"
        for m in members:
            df.at[m, "dup_group"] = gid
            df.at[m, "dup_type"] = gtype

    # ---- keep rule: within group keep lower folder number; tie -> lower filename ----
    df["keep"] = True
    for g, sub in df[df["dup_group"].notna()].groupby("dup_group"):
        order = sorted(sub.index, key=lambda i: (folder_rank(df.at[i, "folder"]), str(df.at[i, "filename"])))
        keep_idx = order[0]
        for i in order:
            df.at[i, "keep"] = (i == keep_idx)

    df.to_csv(MANIFEST, index=False, encoding="utf-8")

    # ---- report ----
    print("\n=== DEDUP SUMMARY ===")
    print(f"Exact-duplicate union edges: {exact_edges}")
    print(f"Near-duplicate union edges (pHash<= {NEAR_THR}): {near_edges}")
    ndup_groups = df["dup_group"].nunique(dropna=True)
    print(f"Duplicate groups: {ndup_groups}")
    print(df["dup_type"].value_counts(dropna=True).to_string())
    removed = int((~df["keep"]).sum())
    print(f"Rows marked keep=False (duplicates removed): {removed}")
    print(f"Rows kept: {int(df['keep'].sum())}")

    # nearest-neighbour phash distance histogram
    print("\npHash nearest-neighbour distance histogram (0..20+):")
    s = pd.Series(nn_dist)
    bins = list(range(0, 21)) + [1000]
    hist = pd.cut(s, bins=bins, right=False).value_counts().sort_index()
    print(hist.to_string())

    # grey-zone review pairs
    print(f"\nGrey-zone near pairs (phash dist {REVIEW_LO}-{REVIEW_HI}) flagged for review: {len(review_pairs)}")
    rp = pd.DataFrame(
        [{"a_rel": df.at[i, "rel_path"], "b_rel": df.at[j, "rel_path"], "phash_dist": d}
         for i, j, d in review_pairs]
    )
    rp_path = P.processed / "hhdoc_dedup_review_pairs.csv"
    rp.to_csv(rp_path, index=False, encoding="utf-8")
    print(f"  -> {rp_path}")

    # cross-folder exact-dup overview (sanity vs expected 13<->15, 20<->22)
    print("\nCross-folder exact/near duplicate folder-pairs (count of grouped images):")
    pairs = defaultdict(int)
    for g, sub in df[df["dup_group"].notna()].groupby("dup_group"):
        fs = sorted(set(sub["folder"]), key=folder_rank)
        if len(fs) >= 2:
            key = " <-> ".join(fs)
            pairs[key] += len(sub)
    for k, v in sorted(pairs.items(), key=lambda x: -x[1]):
        print(f"   {k}: {v} images in shared groups")

    print(f"\nUpdated {MANIFEST}")


if __name__ == "__main__":
    main()
