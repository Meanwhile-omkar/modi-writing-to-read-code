"""Helpers for MoDeTrans<->HHDoc crop matching (SIFT BoVW retrieval + RANSAC verify)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PAGE_LONG = 1600
PATCH_LONG = 1600  # patches are small already; scale factor handled separately
PAGE_NFEAT = 1500
PATCH_NFEAT = 1000


def load_gray_scaled(path: Path, long_side: int):
    """Return (gray_uint8_array, scale) where scale maps scaled->original (orig = scaled/scale)."""
    with Image.open(path) as im:
        im = im.convert("L")
        w, h = im.size
        m = max(w, h)
        scale = 1.0 if m <= long_side else long_side / float(m)
        if scale != 1.0:
            im = im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
    return arr, scale


def sift_detect(gray: np.ndarray, nfeatures: int):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kps, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    pts = np.array([k.pt for k in kps], dtype=np.float32)
    return pts, desc.astype(np.float32)


def bovw_histogram(desc: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Hard-assignment BoVW count histogram."""
    k = centers.shape[0]
    if desc.shape[0] == 0:
        return np.zeros(k, dtype=np.float32)
    # assign each descriptor to nearest center (chunked to bound memory)
    idx = np.empty(desc.shape[0], dtype=np.int32)
    step = 2000
    c2 = (centers ** 2).sum(1)
    for s in range(0, desc.shape[0], step):
        d = desc[s:s + step]
        dist = (d ** 2).sum(1, keepdims=True) - 2 * d @ centers.T + c2[None, :]
        idx[s:s + step] = np.argmin(dist, axis=1)
    h = np.bincount(idx, minlength=k).astype(np.float32)
    return h


def verify_match(patch_pts, patch_desc, page_pts, page_desc,
                 ratio=0.75, min_inliers=25):
    """BF knn match + Lowe ratio + RANSAC homography. Returns (n_inliers, H, good_pts)."""
    if patch_desc.shape[0] < 4 or page_desc.shape[0] < 4:
        return 0, None, None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(patch_desc, page_desc, k=2)
    good = []
    for mm in matches:
        if len(mm) < 2:
            continue
        a, b = mm
        if a.distance < ratio * b.distance:
            good.append(a)
    if len(good) < 4:
        return len(good), None, None
    src = np.float32([patch_pts[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([page_pts[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return 0, None, None
    inliers = int(mask.sum())
    return inliers, H, None


def bbox_in_page(H, patch_shape, page_scale):
    """Project patch corners through H into scaled page coords, then to ORIGINAL page coords."""
    ph, pw = patch_shape
    corners = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    # scaled page coords -> original page coords (orig = scaled / page_scale)
    proj_orig = proj / page_scale
    x0, y0 = proj_orig.min(0)
    x1, y1 = proj_orig.max(0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)
