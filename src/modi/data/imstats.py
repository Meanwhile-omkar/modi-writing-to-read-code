"""Shared image-statistics helpers for the data audit.

All functions are read-only on raw images. Nothing here writes or copies
full-resolution images.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import imagehash
except Exception:  # pragma: no cover
    imagehash = None

# Pillow: do not choke on truncated files, but we still flag failures upstream.
Image.MAX_IMAGE_PIXELS = None  # these are scanned pages; disable decompression-bomb cap


def md5_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _downscale_long_side(img: Image.Image, long_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= long_side:
        return img
    s = long_side / float(m)
    return img.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.BILINEAR)


def phash_gray(img: Image.Image, long_side: int = 256) -> Optional[str]:
    """pHash on a grayscale downscaled copy. Returns hex string or None."""
    if imagehash is None:
        return None
    g = _downscale_long_side(img.convert("L"), long_side)
    return str(imagehash.phash(g))


def ink_ratio_otsu(img: Image.Image, long_side: int = 1024) -> float:
    """Fraction of dark (ink) pixels via Otsu threshold on grayscale downscaled copy.

    NOTE (brief definition): literally fraction of pixels BELOW the Otsu threshold.
    For dark-background scans this is dominated by background, not ink. Use the
    polarity-robust `ink_ratio_fg` for blank/foreground detection.
    """
    g = _downscale_long_side(img.convert("L"), long_side)
    arr = np.asarray(g, dtype=np.uint8)
    thr = _otsu_threshold(arr)
    # ink = dark pixels (below threshold)
    return float((arr < thr).mean())


def edge_density(img: Image.Image, long_side: int = 1024, grad_thr: float = 8.0) -> float:
    """Fraction of pixels with above-threshold local gradient magnitude.

    Unlike a global-intensity Otsu split, gradient magnitude is robust to the
    overall background tone of aged/sepia paper: ink strokes create local edges
    regardless of absolute brightness, so this does not degenerate the way
    `ink_ratio_otsu` can on low-contrast scans (see `ink_ratio_fg` docstring note).

    A FIXED threshold is used deliberately, not a per-image Otsu split: gradient
    magnitude on these scans is strongly unimodal/skewed (most pixels are smooth
    background with a long thin tail of edge pixels), and Otsu assumes
    bimodality — on this data it picks a threshold near the per-image max,
    collapsing edge_density to ~0 for every page regardless of content (verified
    empirically; see outputs/reports/DATA_DECISIONS_TODO.md). grad_thr=8.0 (on a
    0-255 grayscale image downscaled to `long_side`) was chosen by inspecting the
    gradient histogram of a known handwritten page, where it isolates genuine ink
    strokes from JPEG/compression noise. Blank/empty pages have almost no pixels
    above this fixed threshold; handwritten pages have several percent.
    """
    g = _downscale_long_side(img.convert("L"), long_side)
    arr = np.asarray(g, dtype=np.float32)
    gx = np.abs(np.diff(arr, axis=1, prepend=arr[:, :1]))
    gy = np.abs(np.diff(arr, axis=0, prepend=arr[:1, :]))
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float((mag > grad_thr).mean())


def gray_signals(img: Image.Image, long_side: int = 1024) -> dict:
    """Polarity-robust foreground/blankness signals on a downscaled grayscale copy.

    - gray_mean, gray_std: raw luminance stats (blank pages have tiny std).
    - ink_ratio_fg: fraction of pixels on the MINORITY side of the Otsu split
      (i.e. the foreground/ink, regardless of whether the page has a light or dark
      background). Near 0 for blank/uniform pages; ~0.03-0.35 for text pages.
    """
    g = _downscale_long_side(img.convert("L"), long_side)
    arr = np.asarray(g, dtype=np.uint8)
    thr = _otsu_threshold(arr)
    below = float((arr < thr).mean())
    return {
        "gray_mean": float(arr.mean()),
        "gray_std": float(arr.std()),
        "ink_ratio_fg": float(min(below, 1.0 - below)),
    }


def _otsu_threshold(arr: np.ndarray) -> int:
    hist, _ = np.histogram(arr.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128
    omega = np.cumsum(hist)
    mu = np.cumsum(hist * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (total - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / denom
    sigma_b = np.nan_to_num(sigma_b, nan=0.0, posinf=0.0, neginf=0.0)
    return int(np.argmax(sigma_b))


def basic_image_info(path: Path) -> dict:
    """Open an image and return mode/size; raises on failure (caller flags)."""
    with Image.open(path) as img:
        img.load()
        return {"mode": img.mode, "width": img.width, "height": img.height}


def full_image_record(path: Path, phash_long: int = 256, ink_long: int = 1024) -> dict:
    """All per-image stats used by the HHDoc manifest."""
    rec = {"bytes": path.stat().st_size}
    rec["md5"] = md5_of_file(path)
    with Image.open(path) as img:
        img.load()
        rec["mode"] = img.mode
        rec["width"] = img.width
        rec["height"] = img.height
        rec["phash"] = phash_gray(img, phash_long)
        rec["ink_ratio"] = ink_ratio_otsu(img, ink_long)
        rec.update(gray_signals(img, ink_long))
    w, h = rec["width"], rec["height"]
    rec["aspect_ratio"] = float(w) / float(h) if h else float("nan")
    return rec


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS
