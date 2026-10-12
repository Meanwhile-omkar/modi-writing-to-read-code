"""06 - MoDeTrans<->HHDoc overlap -> processed/modetrans_hhdoc_matches.csv
and processed/modetrans_docs.csv

Find which MoDeTrans patches are crops of HHDoc pages (suspected ~900).
Retrieval: SIFT -> MiniBatchKMeans visual vocab (k~1000) -> BoVW TF-IDF -> top-5.
Verify: Lowe 0.75 + RANSAC homography, inliers>=25 (12-25 flagged for review).
Only HHDoc pages with keep=True are used.

Usage:
  python scripts/06_modetrans_hhdoc_overlap.py --pilot 50     # timing pilot (rule 8)
  python scripts/06_modetrans_hhdoc_overlap.py --full         # full run
Idempotent per mode. Seed 42.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modi.utils import console  # noqa: F401,E402  (force utf-8 stdout)


from modi.data.overlap import (  # noqa: E402
    PAGE_LONG, PAGE_NFEAT, PATCH_NFEAT, bbox_in_page, bovw_histogram,
    load_gray_scaled, sift_detect, verify_match,
)
from modi.utils.parallel import default_workers, pmap  # noqa: E402
from modi.utils.paths import P  # noqa: E402

SEED = 42
np.random.seed(SEED)

MANIFEST = P.processed / "hhdoc_manifest.csv"
OUT_MATCHES = P.processed / "modetrans_hhdoc_matches.csv"
OUT_DOCS = P.processed / "modetrans_docs.csv"
CACHE = P.processed / "feat_cache"
CACHE.mkdir(parents=True, exist_ok=True)
PAGE_FEAT = CACHE / "hhdoc_page_feats.npz"
VOCAB = CACHE / "vocab_k1000.npy"

K = 1000
TOPK = 5
MIN_INLIERS = 25
REVIEW_LO, REVIEW_HI = 12, 25


def _page_worker(rel):
    """Top-level so it is picklable. Returns (rel, pts, desc, scale)."""
    gray, scale = load_gray_scaled(P.hhdoc_root / rel, PAGE_LONG)
    pts, desc = sift_detect(gray, PAGE_NFEAT)
    return rel, pts.astype(np.float16), desc.astype(np.uint8), np.float32(scale)


def build_page_features(pages_df):
    """Compute (or load) SIFT feats for each keep=True page. Cached to NPZ.

    Returns (index, feats, scales) consistently on both the cache-hit and
    cache-miss paths.
    """
    if PAGE_FEAT.exists():
        data = np.load(PAGE_FEAT, allow_pickle=True)
        index = data["index"].tolist()
        feats = {rel: (data[f"pts_{i}"], data[f"desc_{i}"]) for i, rel in enumerate(index)}
        scales = {rel: float(data[f"scale_{i}"]) for i, rel in enumerate(index)}
        return index, feats, scales

    rels = pages_df["rel_path"].tolist()
    nw = default_workers()
    print(f"page SIFT using {nw} worker processes over {len(rels)} pages")
    out = pmap(_page_worker, rels, workers=nw, desc="page SIFT")

    index, arrays, feats, scales = [], {}, {}, {}
    for i, (rel, pts, desc, scale) in enumerate(out):
        index.append(rel)
        arrays[f"pts_{i}"] = pts
        arrays[f"desc_{i}"] = desc
        arrays[f"scale_{i}"] = scale
        feats[rel] = (pts, desc)
        scales[rel] = float(scale)
    np.savez_compressed(PAGE_FEAT, index=np.array(index, dtype=object), **arrays)
    mb = PAGE_FEAT.stat().st_size / 1e6
    print(f"page feature cache: {mb:.0f} MB (budget 1000 MB)")
    return index, feats, scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=50)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    man = pd.read_csv(MANIFEST, dtype={"folder": str})
    pages = man[(man["keep"] == True) & (man["md5"].notna())].reset_index(drop=True)  # noqa: E712
    print(f"keep=True HHDoc pages: {len(pages)}")

    # ---- page features (cache) ----
    t0 = time.time()
    index, feats, scales = build_page_features(pages)
    print(f"page features ready: {len(index)} pages in {time.time()-t0:.1f}s")

    # ---- vocab (MiniBatchKMeans on a descriptor sample) ----
    if VOCAB.exists():
        centers = np.load(VOCAB)
    else:
        rng = np.random.default_rng(SEED)
        sample = []
        per = max(1, 200000 // max(1, len(index)))
        for rel in index:
            d = feats[rel][1].astype(np.float32)
            if d.shape[0] == 0:
                continue
            take = min(per, d.shape[0])
            sample.append(d[rng.choice(d.shape[0], take, replace=False)])
        sample = np.vstack(sample) if sample else np.zeros((1, 128), np.float32)
        print(f"vocab training sample: {sample.shape}")
        km = MiniBatchKMeans(n_clusters=K, random_state=SEED, batch_size=4096, n_init=3, max_iter=100)
        km.fit(sample)
        centers = km.cluster_centers_.astype(np.float32)
        np.save(VOCAB, centers)
    print(f"vocab ready: {centers.shape}")

    # ---- page BoVW + TF-IDF ----
    page_hist = np.zeros((len(index), K), dtype=np.float32)
    for i, rel in enumerate(tqdm(index, desc="page BoVW")):
        page_hist[i] = bovw_histogram(feats[rel][1].astype(np.float32), centers)
    df_count = (page_hist > 0).sum(0)
    idf = np.log((1 + len(index)) / (1 + df_count)) + 1.0
    def tfidf(h):
        tf = h / (h.sum() + 1e-9)
        v = tf * idf
        n = np.linalg.norm(v) + 1e-9
        return v / n
    page_tfidf = np.vstack([tfidf(page_hist[i]) for i in range(len(index))])

    # ---- patches ----
    pm = pd.read_parquet(P.modetrans_parquet)
    patch_files = pm["filename"].tolist()
    if not args.full:
        rng = np.random.default_rng(SEED)
        sel = rng.choice(len(patch_files), min(args.pilot, len(patch_files)), replace=False)
        patch_files = [patch_files[i] for i in sorted(sel)]
        print(f"PILOT mode: {len(patch_files)} patches")
    else:
        print(f"FULL mode: {len(patch_files)} patches")

    img_dir = P.modetrans_images
    results = []
    inlier_hist = []
    t_start = time.time()
    for fn in tqdm(patch_files, desc="patch match"):
        gray, pscale = load_gray_scaled(img_dir / fn, PAGE_LONG)
        ppts, pdesc = sift_detect(gray, PATCH_NFEAT)
        if pdesc.shape[0] == 0:
            results.append({"patch_id": fn, "matched_folder": None, "matched_file": None,
                            "inliers": 0, "bbox_x": None, "bbox_y": None, "bbox_w": None,
                            "bbox_h": None, "status": "no_features"})
            continue
        q = tfidf(bovw_histogram(pdesc.astype(np.float32), centers))
        sims = page_tfidf @ q
        cand = np.argsort(-sims)[:TOPK]
        best = (0, None, None)  # inliers, rel, H
        for ci in cand:
            rel = index[ci]
            gpts = feats[rel][0].astype(np.float32)
            gdesc = feats[rel][1].astype(np.float32)
            inl, H, _ = verify_match(ppts, pdesc, gpts, gdesc,
                                     ratio=0.75, min_inliers=MIN_INLIERS)
            if inl > best[0]:
                best = (inl, rel, H)
        inl, rel, H = best
        inlier_hist.append(inl)
        if rel is not None and H is not None and inl >= REVIEW_LO:
            pscale_page = scales.get(rel, 1.0)
            bb = bbox_in_page(H, gray.shape, pscale_page)
            folder = rel.split("/")[0]
            fname = rel.split("/")[-1]
            status = "match" if inl >= MIN_INLIERS else "review"
            results.append({"patch_id": fn, "matched_folder": folder, "matched_file": fname,
                            "inliers": inl, "bbox_x": round(bb[0], 1), "bbox_y": round(bb[1], 1),
                            "bbox_w": round(bb[2], 1), "bbox_h": round(bb[3], 1), "status": status})
        else:
            results.append({"patch_id": fn, "matched_folder": None, "matched_file": None,
                            "inliers": inl, "bbox_x": None, "bbox_y": None, "bbox_w": None,
                            "bbox_h": None, "status": "no_match"})

    elapsed = time.time() - t_start
    per = elapsed / max(1, len(patch_files))
    res = pd.DataFrame(results)

    print("\n=== OVERLAP SUMMARY ===")
    print(f"patches processed: {len(res)}  elapsed: {elapsed:.1f}s  per-patch: {per:.2f}s")
    print(res["status"].value_counts().to_string())
    ih = pd.Series(inlier_hist)
    if len(ih):
        print("\ninlier histogram (verified candidate best):")
        bins = [0, 1, 5, 10, 12, 15, 20, 25, 35, 50, 100, 10000]
        print(pd.cut(ih, bins=bins, right=False).value_counts().sort_index().to_string())

    if not args.full:
        full_est = per * len(pm) / 3600.0
        print(f"\nESTIMATED full run ({len(pm)} patches): {full_est:.2f} h")
        print("Per rule 8: re-run with --full only if estimate < 3 h.")
        res.to_csv(P.processed / "modetrans_hhdoc_matches_pilot.csv", index=False, encoding="utf-8")
        print(f"wrote pilot results -> {P.processed/'modetrans_hhdoc_matches_pilot.csv'}")
        return

    res.to_csv(OUT_MATCHES, index=False, encoding="utf-8")
    print(f"wrote {OUT_MATCHES}")
    build_docs(res, pm)


def build_docs(res, pm):
    """Derive modetrans_doc_id: patches matched to the same HHDoc page share a doc id."""
    matched = res[res["status"] == "match"].copy()
    page_key = matched["matched_folder"].astype(str) + "/" + matched["matched_file"].astype(str)
    key_to_doc = {}
    doc_ids = {}
    nid = 0
    for fn, key in zip(matched["patch_id"], page_key):
        if key not in key_to_doc:
            key_to_doc[key] = f"hhdoc::{key}"
        doc_ids[fn] = key_to_doc[key]
    # unmatched: try a source/sequence field in metadata, else own id
    meta = pd.read_csv(P.modetrans_meta, encoding="utf-8-sig")
    meta_cols = set(meta.columns)
    src_col = None
    for cand in ["source", "document", "doc", "page", "parquet_path", "original_filename"]:
        if cand in meta_cols and meta[cand].notna().any() and meta[cand].nunique() < len(meta):
            src_col = cand
            break
    meta_idx = meta.set_index("filename")
    rows = []
    for fn in pm["filename"]:
        if fn in doc_ids:
            did = doc_ids[fn]
        elif src_col is not None and fn in meta_idx.index and pd.notna(meta_idx.loc[fn, src_col]):
            did = f"src::{src_col}::{meta_idx.loc[fn, src_col]}"
        else:
            did = f"patch::{fn}"
        rows.append({"image_id": fn, "modetrans_doc_id": did,
                     "matched": fn in doc_ids})
    docs = pd.DataFrame(rows)
    docs.to_csv(OUT_DOCS, index=False, encoding="utf-8")
    print(f"doc ids: {docs['modetrans_doc_id'].nunique()} unique "
          f"(src_col used for unmatched: {src_col})")
    print(f"wrote {OUT_DOCS}")


if __name__ == "__main__":
    main()
