"""13 - settle the grey-zone near-duplicate pairs with SIFT + RANSAC (addendum item 2).

Step 02 left 931 pairs in the pHash 5-10 band undecided, because pHash distance
alone cannot tell "same page, rescanned" from "different page, same layout".
Geometric verification can: if two images are the same physical page, a homography
between them has many inliers AND is close to the identity in scale.

Decision rule (from the addendum):
    inliers >= 40 AND 0.8 <= homography area ratio <= 1.25  -> near_duplicate
    inliers < 15                                            -> distinct
    otherwise                                               -> needs_review

For a near_duplicate pair the copy in the LOWER folder number is kept, matching the
rule already used in step 02. Only the needs_review remainder is handed to a human.

Both pages are downscaled to 1600 px long side, as in step 06.
Idempotent. Seed 42.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modi.utils import console  # noqa: F401,E402
from modi.data.overlap import load_gray_scaled, sift_detect  # noqa: E402
from modi.utils.parallel import default_workers, pmap  # noqa: E402
from modi.utils.paths import P  # noqa: E402

SEED = 42
LONG = 1600
NFEAT = 1500
RATIO = 0.75
INLIERS_DUP = 40
INLIERS_DISTINCT = 15
AREA_LO, AREA_HI = 0.8, 1.25

PAIRS = P.processed / "hhdoc_dedup_review_pairs.csv"
MANIFEST = P.processed / "hhdoc_manifest.csv"
OUT = P.processed / "hhdoc_greyzone_verified.csv"


def _quad_area(pts):
    """Shoelace area of a 4-point polygon."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _worker(task):
    a_rel, b_rel = task
    try:
        ga, _ = load_gray_scaled(P.hhdoc_root / a_rel, LONG)
        gb, _ = load_gray_scaled(P.hhdoc_root / b_rel, LONG)
        pa, da = sift_detect(ga, NFEAT)
        pb, db = sift_detect(gb, NFEAT)
        if da.shape[0] < 4 or db.shape[0] < 4:
            return {"inliers": 0, "area_ratio": None, "err": "too few descriptors"}
        bf = cv2.BFMatcher(cv2.NORM_L2)
        good = []
        for mm in bf.knnMatch(da, db, k=2):
            if len(mm) == 2 and mm[0].distance < RATIO * mm[1].distance:
                good.append(mm[0])
        if len(good) < 4:
            return {"inliers": len(good), "area_ratio": None, "err": None}
        src = np.float32([pa[m.queryIdx] for m in good]).reshape(-1, 1, 2)
        dst = np.float32([pb[m.trainIdx] for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            return {"inliers": 0, "area_ratio": None, "err": None}
        inl = int(mask.sum())
        h, w = ga.shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        src_area = float(w * h)
        dst_area = float(_quad_area(proj))
        ratio = dst_area / src_area if src_area > 0 else None
        return {"inliers": inl, "area_ratio": ratio, "err": None}
    except Exception as e:
        return {"inliers": 0, "area_ratio": None, "err": repr(e)}


def folder_rank(f):
    try:
        return int(f)
    except (ValueError, TypeError):
        return 10 ** 9


def main():
    pairs = pd.read_csv(PAIRS)
    print(f"grey-zone pairs to verify: {len(pairs)}")
    tasks = list(zip(pairs["a_rel"], pairs["b_rel"]))
    res = pmap(_worker, tasks, workers=default_workers(), desc="verify pairs")

    pairs["inliers"] = [r["inliers"] for r in res]
    pairs["area_ratio"] = [r["area_ratio"] for r in res]
    pairs["verify_error"] = [r["err"] for r in res]

    def decide(r):
        ar = r["area_ratio"]
        if r["inliers"] >= INLIERS_DUP and ar is not None and AREA_LO <= ar <= AREA_HI:
            return "near_duplicate"
        if r["inliers"] < INLIERS_DISTINCT:
            return "distinct"
        return "needs_review"

    pairs["outcome"] = pairs.apply(decide, axis=1)
    pairs.to_csv(OUT, index=False, encoding="utf-8")

    counts = pairs["outcome"].value_counts()
    print("\n=== GREY-ZONE OUTCOMES ===")
    for k in ("near_duplicate", "distinct", "needs_review"):
        print(f"  {k}: {int(counts.get(k, 0))}")
    print("\ninlier histogram:")
    print(pd.cut(pairs["inliers"], bins=[0, 5, 10, 15, 25, 40, 60, 100, 10**6],
                 right=False).value_counts().sort_index().to_string())

    # ---- apply near_duplicate decisions to the manifest ----
    man = pd.read_csv(MANIFEST, dtype={"folder": str})
    idx_of = {r: i for i, r in enumerate(man["rel_path"])}
    nd = pairs[pairs["outcome"] == "near_duplicate"]

    # union-find over verified near-duplicate edges, seeded with existing groups
    parent = list(range(len(man)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # keep existing dup_group membership intact
    for g, sub in man[man["dup_group"].notna()].groupby("dup_group"):
        members = list(sub.index)
        for m in members[1:]:
            union(members[0], m)
    applied = 0
    for _, r in nd.iterrows():
        ia, ib = idx_of.get(r["a_rel"]), idx_of.get(r["b_rel"])
        if ia is not None and ib is not None:
            union(ia, ib)
            applied += 1

    from collections import defaultdict
    groups = defaultdict(list)
    for i in range(len(man)):
        groups[find(i)].append(i)

    newly_removed = 0
    gid = 0
    man["dup_group"] = pd.NA
    for root, members in groups.items():
        if len(members) <= 1:
            continue
        gid += 1
        for m in members:
            man.at[m, "dup_group"] = gid
        # mark verified type where this group gained a verified edge
        order = sorted(members, key=lambda i: (folder_rank(man.at[i, "folder"]),
                                               str(man.at[i, "filename"])))
        keep_idx = order[0]
        for i in order:
            was_keep = bool(man.at[i, "keep"])
            man.at[i, "keep"] = (i == keep_idx)
            if was_keep and i != keep_idx:
                newly_removed += 1

    verified_rels = set(nd["a_rel"]) | set(nd["b_rel"])
    man.loc[man["rel_path"].isin(verified_rels), "dup_type"] = "near_verified"
    man.to_csv(MANIFEST, index=False, encoding="utf-8")

    print(f"\nverified near-duplicate edges applied: {applied}")
    print(f"additional pages marked keep=False by verification: {newly_removed}")
    print(f"pages kept overall: {int(man['keep'].sum())} / {len(man)}")
    print(f"\nwrote {OUT}")
    print(f"updated {MANIFEST}")


if __name__ == "__main__":
    main()
