"""20 - visual audit of everything done to an image before the SPAN model sees it.

The SPAN pipeline edits an image in exactly two ways: it scales the whole patch to 320 x 880 once,
and during training it applies SPAN's augmentations. Nothing is cut, split, reordered or dropped.
This script renders both so they can be checked by eye, and measures what the augmentations do to
contrast, because an augmentation that leaves a page unreadable teaches the model nothing.

Output: notebooks/outputs-of-notebook/image-processing-audit-span/
Run:    python scripts/20_span_input_audit.py
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO / "notebooks" / "span"))
import span_lib as L  # noqa: E402

META = REPO / "assets" / "metadata"
PARQUET = ROOT / "FYP-modi-script-data" / "MoDeTrans-dataset" / "train-00000-of-00001.parquet"
OUT = REPO / "notebooks" / "outputs-of-notebook" / "image-processing-audit-span"
H, W = 320, 880

for _f in ("Nirmala UI", "Mangal", "Noto Sans Devanagari"):      # Devanagari captions otherwise render as boxes
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.family"] = [_f, "DejaVu Sans"]
        break
    except Exception:
        pass


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pq = pd.read_parquet(PARQUET)
    row = {r.filename: r for r in pq.itertuples()}
    text = (pd.read_csv(META / "modetrans_text.csv", keep_default_na=False)
            .set_index("image_id")["text_train"].to_dict())
    cs = json.loads((META / "charset.json").read_text(encoding="utf-8"))
    vocab = ["<blank>"] + cs["symbols"] + ["<FRAC>", "<unk>"]
    stoi = {c: i for i, c in enumerate(vocab)}
    tok = re.compile(r"<FRAC>|.", re.S)
    nlab = lambda n: len(tok.findall(text[n]))

    # a clean patch, a dense one, a heavily speckled one
    picks = [("1156.jpg", "a typical patch"), ("32.jpg", "dense, 7 lines"), ("1261.jpg", "heavy speckle noise")]
    raw = {n: np.array(Image.open(io.BytesIO(row[n].image["bytes"])).convert("L")) for n, _ in picks}
    small = {n: cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA) for n, a in raw.items()}
    rh, rw = L.reduced_shape(H, W)

    # ---- 1. the single edit, with the output grid drawn on it
    fig, ax = plt.subplots(len(picks), 2, figsize=(17, 3.4 * len(picks)), gridspec_kw={"width_ratios": [1268, 880]})
    for i, (n, why) in enumerate(picks):
        ax[i][0].imshow(raw[n], cmap="gray"); ax[i][0].axis("off")
        ax[i][0].set_title(f"{n} ({why}): as stored, {raw[n].shape[1]}x{raw[n].shape[0]}", fontsize=9, loc="left")
        ax[i][1].imshow(small[n], cmap="gray"); ax[i][1].axis("off")
        for r in range(1, rh):
            ax[i][1].axhline(r * H / rh - 0.5, color="tab:red", lw=0.5, alpha=0.55)
        ax[i][1].set_title(f"what the model gets: {W}x{H}, scaled once. Red = the {rh} rows of the output grid "
                           f"({rh}x{rw} = {rh*rw} slots for {nlab(n)} characters = {rh*rw/nlab(n):.1f} per character)",
                           fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "01_the_one_edit_with_output_grid.png", dpi=95); plt.close(fig)

    # ---- 2. what training does to a page, at SPAN's own rate (each augmentation at p = 0.2)
    rng = np.random.default_rng(7)
    for n, why in picks:
        fig, ax = plt.subplots(4, 3, figsize=(17, 9))
        ax = ax.ravel()
        ax[0].imshow(small[n], cmap="gray", vmin=0, vmax=255); ax[0].set_title(f"{n}: original", fontsize=8); ax[0].axis("off")
        for k in range(1, 12):
            a = L.augment(small[n].copy(), rng, 0.2)
            ax[k].imshow(a, cmap="gray", vmin=0, vmax=255)
            ax[k].set_title(f"sample {k} at p=0.2   (size {a.shape[1]}x{a.shape[0]}, contrast {a.std():.0f} vs {small[n].std():.0f})", fontsize=8)
            ax[k].axis("off")
        fig.suptitle(f"{n} ({why}) - what SPAN's augmentation does at the published rate", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / f"02_augmentations_at_published_rate_{Path(n).stem}.png", dpi=85); plt.close(fig)

    # ---- 3. stress test: every augmentation forced on, to see the worst case
    n = picks[0][0]
    fig, ax = plt.subplots(3, 2, figsize=(15, 8)); ax = ax.ravel()
    rng = np.random.default_rng(3)
    for k in range(6):
        a = L.augment(small[n].copy(), rng, 1.0)
        ax[k].imshow(a, cmap="gray", vmin=0, vmax=255); ax[k].axis("off")
        ax[k].set_title(f"every augmentation forced on (p=1.0), sample {k+1}", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "03_stress_test_all_augmentations_on.png", dpi=85); plt.close(fig)

    # ---- 4. measured: how often does an augmented page come out nearly unreadable?
    rng = np.random.default_rng(0)
    base = {n: float(small[n].std()) for n, _ in picks}
    ratios, sizes, ink_frac = [], [], []
    for n, _ in picks:
        for _ in range(300):
            a = L.augment(small[n].copy(), rng, 0.2)
            ratios.append(float(a.std()) / base[n]); sizes.append(a.shape)
    ratios = np.array(ratios)
    resized = np.mean([s != (H, W) for s in sizes])
    report = {
        "samples": int(len(ratios)),
        "share_resized_by_dpi_augmentation": round(float(resized), 3),
        "contrast_ratio_median": round(float(np.median(ratios)), 3),
        "share_with_contrast_below_25pct_of_original": round(float((ratios < 0.25).mean()), 3),
        "share_with_contrast_below_10pct_of_original": round(float((ratios < 0.10).mean()), 3),
    }
    (OUT / "augmentation_stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    (OUT / "README.md").write_text(f"""# SPAN input audit

The SPAN pipeline edits an image in exactly two ways. Both are rendered here.

| file | what it shows |
|---|---|
| `01_the_one_edit_with_output_grid.png` | the whole patch scaled once to {W}x{H}, with the {rh} rows of the model's output grid drawn on it. Nothing is cut, split, reordered or discarded. |
| `02_augmentations_at_published_rate_*.png` | eleven training samples per patch, each augmentation at SPAN's probability of 0.2 |
| `03_stress_test_all_augmentations_on.png` | every augmentation forced on at once, to see the worst case |
| `augmentation_stats.json` | measured over {len(ratios)} augmented samples: how often an augmentation leaves a page low-contrast |

Why this replaced the line-cutting pipeline of notebooks 02 and 03: that pipeline had to find the
text lines first, and it failed in three separate ways (merged lines crushed into one band, header
lines dropped, ink thrown away) before any of them could be seen. SPAN needs no segmentation, so
that whole class of failure cannot occur.
""", encoding="utf-8")
    print("wrote", len(list(OUT.glob('*.png'))), "images to", OUT)


if __name__ == "__main__":
    main()
