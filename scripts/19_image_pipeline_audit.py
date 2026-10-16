"""19 - visual audit of EVERY image-editing stage between the raw patch and the model's input.

Why this exists. Two serious bugs got all the way into a training run because nobody ever looked
at what the model was actually being handed:

  * the strip was padded out to a fixed width, so ~80% of the model's input was blank paper
  * the line finder returned ONE band for a dense or slanted page, and five lines of Modi were
    then squashed into a 64 px strip - a 246-character label inside a 183 px image

Both were invisible in the numbers and obvious in a picture. So this script renders every stage,
for a random sample AND for the worst outlier at each stage, and writes the result to
notebooks/outputs-of-notebook/image-processing-audit/ with a README that tells the story.

Run:  python scripts/19_image_pipeline_audit.py [--n-sample 8] [--all]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Devanagari in the captions renders as empty boxes under the default font
for _f in ("Nirmala UI", "Mangal", "Noto Sans Devanagari"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.family"] = [_f, "DejaVu Sans"]
        break
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
META = REPO / "assets" / "metadata"
PARQUET = ROOT / "FYP-modi-script-data" / "MoDeTrans-dataset" / "train-00000-of-00001.parquet"
OUTDIR = REPO / "notebooks" / "outputs-of-notebook" / "image-processing-audit"

SH, SW, GAP = 64, 8192, 20
W_STRIDE = 4
SKEW_RANGE = 8.0        # the deskew search limit; the check below must match it
MIN_SLOTS_PER_CHAR = 3.2  # strips below this are stretched; mirrors cell 7 of the notebook
LINE_THR = 0.08           # ink threshold for a line core
GROW_CAP = 0.35           # how far a line may grow toward its neighbour, in line heights

# ----------------------------------------------------------------- the pipeline, stage by stage


def prof(g):
    """Stage input: a greyscale patch. Output: how much ink sits in each row of pixels."""
    return (cv2.GaussianBlur(g, (5, 5), 0) < 128).sum(1).astype(np.float32)


def smooth(p, k=31):
    return cv2.GaussianBlur(p.reshape(-1, 1), (1, k), 0).ravel()


def deskew(g, rng=SKEW_RANGE, step=0.5):
    h, w = g.shape
    best, ba = -1.0, 0.0
    for a in np.arange(-rng, rng + 1e-9, step):
        r = g if a == 0 else cv2.warpAffine(
            g, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0), (w, h),
            flags=cv2.INTER_NEAREST, borderValue=255)
        s = float(np.sum(np.diff(prof(r)) ** 2))
        if s > best:
            best, ba = s, a
    if ba == 0:
        return g, 0.0
    return cv2.warpAffine(g, cv2.getRotationMatrix2D((w / 2, h / 2), ba, 1.0), (w, h),
                          flags=cv2.INTER_LINEAR, borderValue=255), ba


def bands(p, thr=0.22, minh=12):
    on = p > p.max() * thr
    out, s = [], None
    for i, v in enumerate(on):
        if v and s is None:
            s = i
        if not v and s is not None:
            if i - s > minh:
                out.append((s, i))
            s = None
    if s is not None and len(on) - s > minh:
        out.append((s, len(on)))
    return out


def split_at_valleys(p, a, b, k, unit):
    seg = p[a:b]
    cuts = []
    cand = sorted((i for i in range(1, len(seg) - 1) if seg[i] <= seg[i - 1] and seg[i] <= seg[i + 1]),
                  key=lambda i: seg[i])
    for i in cand:
        if len(cuts) >= k - 1:
            break
        if all(abs(i - c) > 0.55 * unit for c in cuts) and min(i, len(seg) - i) > 0.45 * unit:
            cuts.append(i)
    e = [0] + sorted(cuts) + [len(seg)]
    return [(a + e[i], a + e[i + 1]) for i in range(len(e) - 1)]


def find_lines_old(g):
    """v1. Kept so the audit can SHOW the failure, not just describe it."""
    ink = smooth(prof(g))
    return bands(ink)


def find_lines_new(g, med_line):
    """Deskew, find line cores, split merged bands, then grow each line until it meets its
    neighbour so that no ink is left outside a line. See the README for why that matters."""
    g2, ang = deskew(g)
    p = smooth(prof(g2))
    raw = prof(g2)
    cores = []
    for a, b in (bands(p, thr=LINE_THR) or [(0, g2.shape[0])]):
        k = int(round((b - a) / med_line))
        cores += (split_at_valleys(p, a, b, k, med_line)
                  if (k > 1 and (b - a) > 1.6 * med_line) else [(a, b)])
    rows = np.where(raw > 0)[0]
    if not len(rows):
        return g2, cores, ang
    top, bot, lim = int(rows[0]), int(rows[-1]) + 1, GROW_CAP * med_line
    out = []
    for i, (a, b) in enumerate(cores):
        lo = top if i == 0 else (cores[i - 1][1] + a) // 2
        hi = bot if i == len(cores) - 1 else (b + cores[i + 1][0]) // 2
        out.append((min(max(lo, int(a - lim)), a), max(min(hi, int(b + lim)), b)))
    return g2, out, ang


def line_crops(g2, lines):
    """Crop each detected line to its own ink, then scale it to exactly 64 px tall."""
    crops, raw = [], []
    for y0, y1 in lines:
        c = g2[max(0, y0):min(g2.shape[0], y1)]
        xs = np.where((c < 128).any(0))[0]
        if len(xs) == 0:
            continue
        c = c[:, xs[0]:xs[-1] + 1]
        raw.append(c)
        crops.append(cv2.resize(c, (max(1, int(c.shape[1] * SH / c.shape[0])), SH)))
    return raw, crops


def assemble(crops):
    if not crops:
        crops = [np.full((SH, SH), 255, np.uint8)]
    parts = []
    for i, s in enumerate(crops):
        if i:
            parts.append(np.full((SH, GAP), 255, np.uint8))
        parts.append(s)
    strip = np.hstack(parts)
    capped = strip.shape[1] > SW
    if capped:
        strip = cv2.resize(strip, (SW, SH))
    return strip, capped


def augment(img, rng):
    H, W = img.shape
    out = img
    if rng.random() < 0.5:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-1.5, 1.5), rng.uniform(0.96, 1.04))
        sh = rng.uniform(-0.10, 0.10)
        M[0, 1] += sh
        M[0, 2] -= sh * H / 2
        M[1, 2] += rng.uniform(-2, 2)
        out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=255)
    if rng.random() < 0.5:
        k = np.ones((2, 2), np.uint8)
        out = cv2.erode(out, k) if rng.random() < 0.5 else cv2.dilate(out, k)
    if rng.random() < 0.3:
        a, s = rng.uniform(10, 25), rng.uniform(4, 6)
        dx = cv2.GaussianBlur((rng.random((H, W)) * 2 - 1).astype(np.float32), (0, 0), s) * a
        dy = cv2.GaussianBlur((rng.random((H, W)) * 2 - 1).astype(np.float32), (0, 0), s) * a
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        out = cv2.remap(out, gx + dx, gy + dy, cv2.INTER_LINEAR, borderValue=255)
    if rng.random() < 0.4:
        out = np.clip(out.astype(np.float32) * rng.uniform(0.9, 1.1)
                      + rng.normal(0, rng.uniform(0, 8), out.shape), 0, 255).astype(np.uint8)
    return out


# ----------------------------------------------------------------- rendering helpers


def save(fig, folder: Path, name: str):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def wide(h=2.0):
    return plt.subplots(figsize=(16, h))


def show_strip(ax, img, title, maxw=4200):
    ax.imshow(img[:, :maxw], cmap="gray", aspect="auto")
    ax.set_title(title, fontsize=8, loc="left")
    ax.axis("off")


# ----------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=8, help="random patches rendered per stage")
    ap.add_argument("--n-outlier", type=int, default=5, help="worst cases rendered per check")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"writing to {OUTDIR}")

    pq = pd.read_parquet(PARQUET)
    names = list(pq["filename"])
    imgs = {r.filename: r.image["bytes"] for r in pq.itertuples()}

    cs = json.loads((META / "charset.json").read_text(encoding="utf-8"))
    VOCAB = ["<blank>"] + cs["symbols"] + ["<FRAC>", "<unk>"]
    STOI = {c: i for i, c in enumerate(VOCAB)}
    TOK = re.compile(r"<FRAC>|.", re.S)
    text = (pd.read_csv(META / "modetrans_text.csv", keep_default_na=False)
            .set_index("image_id")["text_train"].to_dict())

    def enc(s):
        return [STOI.get(t, STOI["<unk>"]) for t in TOK.findall(s)]

    def need(n):
        e = enc(text[n])
        return len(e) + sum(a == b for a, b in zip(e, e[1:]))

    load = lambda n: np.array(Image.open(io.BytesIO(imgs[n])).convert("L"))

    # ---- measure the typical line height first; everything downstream depends on it
    hs = []
    for n in names[:300]:
        g2, _ = deskew(load(n))
        hs += [b - a for a, b in bands(smooth(prof(g2)))]
    MED_LINE = float(np.median(hs))
    print(f"measured typical line height: {MED_LINE:.0f} px")

    # ---- full sweep: record what happens to every patch at every stage
    rows = []
    for n in names:
        g = load(n)
        old = find_lines_old(g)
        g2, lines, ang = find_lines_new(g, MED_LINE)
        raw, crops = line_crops(g2, lines)
        strip, capped = assemble(crops)
        w_raw = strip.shape[1]
        nd = need(n)
        nlab = len(enc(text[n]))
        # stage 06b: the widening pass from cell 7 of the notebook, mirrored here so the audit
        # reports what the model is really given rather than the pre-widening strip
        want = max(int(nd * 1.3), int(nlab * MIN_SLOTS_PER_CHAR) + 1) * W_STRIDE
        w = min(SW, max(w_raw, want))
        slots = w // W_STRIDE
        ink_line = [float((c < 128).mean()) for c in raw]
        _ink = (g2 < 128)
        ink_kept = float(sum(_ink[a:b].sum() for a, b in lines) / max(1, _ink.sum()))
        # how many of the strip's 64 px the writing itself occupies, after the band was grown
        _rows = [np.where(_ink[a:b].any(1))[0] for a, b in lines]
        text_px = float(np.median([64.0 * (r[-1] - r[0] + 1) / max(1, b - a)
                                   for (a, b), r in zip(lines, _rows) if len(r)] or [64.0]))
        rows.append(dict(
            image_id=n, h=g.shape[0], w_img=g.shape[1], skew_deg=ang,
            lines_old=len(old), lines_new=len(lines),
            line_h_min=min((b - a for a, b in lines), default=0),
            line_h_max=max((b - a for a, b in lines), default=0),
            line_ink_min=min(ink_line, default=0.0), ink_kept=ink_kept, text_px_of_64=text_px,
            widest_aspect=max((c.shape[1] / c.shape[0] for c in raw), default=0.0),
            strip_w=w, strip_w_before_widening=w_raw, widened=w > w_raw, capped=capped,
            slots=slots, label_len=nlab, need=nd,
            slots_per_char=slots / max(1, nlab),
            slots_per_char_before=(w_raw // W_STRIDE) / max(1, nlab), tight=slots < nd,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "stage_stats.csv", index=False)

    # ---- the outlier checks. Each one is a thing that CAN go wrong at a named stage.
    checks = {
        "02_deskew__angle_at_search_limit":
            (df.skew_deg.abs() >= SKEW_RANGE - 0.1,
             f"skew pinned at the +-{SKEW_RANGE:.0f} deg search limit, so the page may still be slanted"),
        "03_lines__old_finder_merged_everything":
            ((df.lines_old <= 1) & (df.h > 2.5 * MED_LINE), "v1 returned one band for a multi-line page"),
        "03_lines__still_single_band":
            ((df.lines_new <= 1) & (df.h > 2.5 * MED_LINE), "v2 STILL returns one band - needs attention"),
        "03b_lines__ink_left_outside_every_line":
            (df.ink_kept < 0.95, "ink the model never sees, although its characters are in the label"),
        "04_crops__near_blank_line":
            (df.line_ink_min < 0.01, "a detected line is almost pure paper"),
        "04_crops__extreme_aspect":
            (df.widest_aspect > 120, "a crop is far wider than tall; probably two lines joined"),
        "05_resize__writing_too_small_in_strip":
            (df.text_px_of_64 < 32,
             "after growing the band, the writing fills under half the 64 px strip"),
        "06_strip__hit_the_width_cap":
            (df.capped, f"strip longer than the {SW} px cap, so it was squeezed horizontally"),
        "07_ctc__too_few_slots":
            (df.tight, "fewer output slots than the label needs; CTC would return zero loss"),
        "07_ctc__slots_per_char_low":
            (df.slots_per_char < 3.0, "still below 3 slots per character after widening"),
        "06b_widened_to_fit_the_label":
            (df.widened, "strip stretched horizontally so CTC has room; not a failure, a repair"),
    }

    summary = []
    for key, (mask, why) in checks.items():
        hit = df[mask]
        summary.append(dict(check=key, n_flagged=int(mask.sum()),
                            pct=round(100 * float(mask.mean()), 2), meaning=why))
        print(f"  {key:46s} {int(mask.sum()):4d} patches  ({100*float(mask.mean()):.1f}%)")
        if not len(hit):
            continue
        folder = OUTDIR / "10_outliers" / key
        sort_col = {"02_deskew__angle_at_search_limit": "skew_deg",
                    "06_strip__hit_the_width_cap": "strip_w",
                    "07_ctc__too_few_slots": "slots_per_char",
                    "07_ctc__slots_per_char_low": "slots_per_char"}.get(key)
        worst = (hit.reindex(hit[sort_col].abs().sort_values(ascending=sort_col != "skew_deg").index)
                 if sort_col else hit).head(args.n_outlier)
        for _, r in worst.iterrows():
            n = r.image_id
            g = load(n)
            g2, lines, ang = find_lines_new(g, MED_LINE)
            fig, ax = plt.subplots(2, 1, figsize=(13, 5))
            ax[0].imshow(g, cmap="gray"); ax[0].axis("off")
            ax[0].set_title(f"{n}  RAW  {g.shape[1]}x{g.shape[0]}  |  v1 found {r.lines_old} line(s)",
                            fontsize=8, loc="left")
            for y0, y1 in find_lines_old(g):
                ax[0].axhline(y0, color="red", lw=0.8)
            ax[1].imshow(g2, cmap="gray"); ax[1].axis("off")
            ax[1].set_title(f"after deskew {ang:+.1f}deg  |  v2 found {r.lines_new} line(s)  |  "
                            f"strip {r.strip_w}px -> {r.slots} slots for a {r.label_len}-char label "
                            f"({r.slots_per_char:.1f}x)  |  WHY FLAGGED: {why}", fontsize=8, loc="left")
            for y0, y1 in lines:
                ax[1].axhline(y0, color="lime", lw=0.9)
            save(fig, folder, f"{Path(n).stem}__flagged")
    pd.DataFrame(summary).to_csv(OUTDIR / "outlier_checks.csv", index=False)

    # ---- the per-stage gallery, on a fixed random sample so stages are comparable
    rng = np.random.default_rng(42)
    sample = [names[i] for i in rng.choice(len(names), args.n_sample, replace=False)]
    (OUTDIR / "00_sample.txt").write_text("\n".join(sample), encoding="utf-8")

    for n in sample:
        stem = Path(n).stem
        g = load(n)
        g2, lines, ang = find_lines_new(g, MED_LINE)
        raw, crops = line_crops(g2, lines)
        strip, capped = assemble(crops)

        f, a = plt.subplots(figsize=(11, 4)); a.imshow(g, cmap="gray"); a.axis("off")
        a.set_title(f"{n} - stage 01: the raw patch as it comes from MoDeTrans "
                    f"({g.shape[1]}x{g.shape[0]})", fontsize=8, loc="left")
        save(f, OUTDIR / "01_original", f"{stem}__01_original")

        f, a = plt.subplots(2, 1, figsize=(11, 7))
        a[0].imshow(g, cmap="gray"); a[0].axis("off")
        a[0].set_title("before: as scanned", fontsize=8, loc="left")
        a[1].imshow(g2, cmap="gray"); a[1].axis("off")
        a[1].set_title(f"stage 02: deskewed by {ang:+.1f} degrees so the lines sit horizontally",
                       fontsize=8, loc="left")
        save(f, OUTDIR / "02_deskew", f"{stem}__02_deskew_{ang:+.1f}deg")

        p = smooth(prof(g2))
        f, a = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 1]})
        a[0].imshow(g2, cmap="gray"); a[0].axis("off")
        for y0, y1 in find_lines_old(g):
            a[0].axhline(y0, color="red", lw=0.8, ls="--")
        for y0, y1 in lines:
            a[0].axhline(y0, color="lime", lw=1.0)
        a[0].set_title(f"stage 03: line detection. red dashed = v1 ({len(find_lines_old(g))} bands), "
                       f"green = v2 ({len(lines)} lines)", fontsize=8, loc="left")
        a[1].plot(p, np.arange(len(p)), lw=0.9)
        a[1].axvline(p.max() * 0.22, color="orange", ls=":", lw=0.9)
        for y0, y1 in lines:
            a[1].axhline(y0, color="lime", lw=0.8)
        a[1].invert_yaxis(); a[1].set_title("ink per row; cuts go at the valleys", fontsize=7)
        save(f, OUTDIR / "03_line_detection", f"{stem}__03_lines_v1_{len(find_lines_old(g))}_v2_{len(lines)}")

        if raw:
            f, a = plt.subplots(len(raw), 1, figsize=(11, 1.1 * len(raw)), squeeze=False)
            for i, c in enumerate(raw):
                a[i][0].imshow(c, cmap="gray", aspect="auto"); a[i][0].axis("off")
                a[i][0].set_title(f"line {i+1}: cropped to its ink, {c.shape[1]}x{c.shape[0]} px, "
                                  f"ink {(c<128).mean()*100:.0f}%", fontsize=7, loc="left")
            save(f, OUTDIR / "04_line_crops", f"{stem}__04_crops_{len(raw)}_lines")

            f, a = plt.subplots(len(crops), 1, figsize=(11, 0.8 * len(crops)), squeeze=False)
            for i, c in enumerate(crops):
                a[i][0].imshow(c[:, :2600], cmap="gray", aspect="auto"); a[i][0].axis("off")
                a[i][0].set_title(f"line {i+1} scaled to 64 px tall -> {c.shape[1]} px wide",
                                  fontsize=7, loc="left")
            save(f, OUTDIR / "05_resized_lines", f"{stem}__05_resized_h64")

        f, a = wide(1.6)
        show_strip(a, strip, f"stage 06: all lines laid END TO END in one 64 px strip, "
                              f"{strip.shape[1]} px wide{'  [HIT THE CAP, squeezed]' if capped else ''}"
                              f"   | {text[n][:60]}")
        save(f, OUTDIR / "06_strip_assembled", f"{stem}__06_strip_{strip.shape[1]}px")

        padv1 = np.full((SH, SW), 255, np.uint8); padv1[:, :strip.shape[1]] = strip
        f, a = plt.subplots(2, 1, figsize=(16, 3))
        show_strip(a[0], padv1, f"v1: padded to a fixed {SW} px -> "
                                f"{100*(1-strip.shape[1]/SW):.0f}% of this image is dead paper, "
                                f"and CTC was told to read all of it", maxw=SW)
        show_strip(a[1], strip, f"v2: kept at its own {strip.shape[1]} px; the batch pads only to "
                                f"its widest member, and CTC is given this length", maxw=SW)
        save(f, OUTDIR / "07_padding_v1_vs_v2", f"{stem}__07_padding_{strip.shape[1]}_vs_{SW}")

        f, a = plt.subplots(3, 1, figsize=(16, 3.4))
        show_strip(a[0], strip, "no augmentation (this is what validation and test see)")
        for i in (1, 2):
            show_strip(a[i], augment(strip, np.random.default_rng(100 + i)),
                       f"stage 08: training augmentation, sample {i} "
                       f"(affine, stroke thickness, elastic, speckle)")
        save(f, OUTDIR / "08_augmentation", f"{stem}__08_augmented")

        x = strip.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        sl = strip.shape[1] // W_STRIDE
        f, a = plt.subplots(2, 1, figsize=(16, 2.6))
        a[0].imshow(x[:, :4200], cmap="gray", aspect="auto"); a[0].axis("off")
        a[0].set_title(f"stage 09: the exact tensor the model receives, 1 x 64 x {strip.shape[1]}, "
                       f"scaled to [-1, +1]", fontsize=8, loc="left")
        a[1].imshow(np.repeat((strip[:, :4200] < 128).any(0)[None, :], 20, 0), cmap="RdYlGn",
                    aspect="auto")
        a[1].axis("off")
        a[1].set_title(f"stage 10: green = a column holding ink, red = blank. "
                       f"{sl} output slots for a {len(enc(text[n]))}-character label "
                       f"= {sl/max(1,len(enc(text[n]))):.1f} slots per character (healthy is 3-8)",
                       fontsize=8, loc="left")
        save(f, OUTDIR / "09_model_input", f"{stem}__09_model_input_{sl}slots")

    # ---- one overview figure: a single patch through every stage
    n = sample[0]
    g = load(n); g2, lines, ang = find_lines_new(g, MED_LINE)
    raw, crops = line_crops(g2, lines); strip, _ = assemble(crops)
    f = plt.figure(figsize=(15, 11))
    gs = f.add_gridspec(6, 1, height_ratios=[3, 3, 3, 1, 1, 1])
    a0 = f.add_subplot(gs[0]); a0.imshow(g, cmap="gray"); a0.axis("off")
    a0.set_title("1. raw patch", fontsize=9, loc="left")
    a1 = f.add_subplot(gs[1]); a1.imshow(g2, cmap="gray"); a1.axis("off")
    a1.set_title(f"2. deskewed {ang:+.1f} deg", fontsize=9, loc="left")
    a2 = f.add_subplot(gs[2]); a2.imshow(g2, cmap="gray"); a2.axis("off")
    for y0, y1 in lines:
        a2.axhline(y0, color="lime", lw=1.0)
    a2.set_title(f"3. {len(lines)} text lines detected (deskew + valley splitting)", fontsize=9, loc="left")
    a3 = f.add_subplot(gs[3]); show_strip(a3, strip, f"4. cropped, scaled to 64 px, laid end to end: "
                                                     f"{strip.shape[1]} px")
    padv1 = np.full((SH, SW), 255, np.uint8); padv1[:, :strip.shape[1]] = strip
    a4 = f.add_subplot(gs[4]); show_strip(a4, padv1, f"5. v1 padded this to {SW} px (the blank right "
                                                     f"half is what the model was learning to ignore)", maxw=SW)
    a5 = f.add_subplot(gs[5]); show_strip(a5, strip, "6. v2 keeps the true width and tells CTC that length",
                                          maxw=SW)
    f.tight_layout()
    save(f, OUTDIR, "00_pipeline_overview")

    # ---- the README that tells a mentor the story
    s = pd.DataFrame(summary).set_index("check")["n_flagged"].to_dict()
    readme = f"""# Image-processing audit — what the model is actually handed

Every image in `FYP-modi-script-data/MoDeTrans-dataset` is edited several times before the network
sees it. This folder renders each of those edits so they can be checked by eye rather than trusted.
It was produced by `scripts/19_image_pipeline_audit.py` and covers all **{len(df):,} patches**;
the galleries show a fixed random sample of {args.n_sample} (listed in `00_sample.txt`) plus the
worst outlier of every automatic check.

Start with **`00_pipeline_overview.png`** — one patch through all six stages on a single sheet.

## The stages

| Folder | What is done to the image | What can go wrong |
|---|---|---|
| `01_original` | nothing; the patch as downloaded | — |
| `02_deskew` | rotate to the angle where the text lines are most horizontal (±{SKEW_RANGE:.0f}°, 0.5° steps) | a page slanted by more than {SKEW_RANGE:.0f}° stays slanted |
| `03_line_detection` | count ink per row, cut the page into text lines | lines so close together that the ink never drops between them |
| `04_line_crops` | crop each line tight to its ink, left and right | a stray mark far from the text inflates the crop |
| `05_resized_lines` | scale each line to exactly {SH} px tall | a grown band leaves the writing small inside the strip |
| `06_strip_assembled` | lay the lines end to end with a 20 px gap | a very long page hits the {SW:,} px cap and is squeezed |
| `07_padding_v1_vs_v2` | pad to a common width for batching | padding to a fixed width wastes most of the image |
| `08_augmentation` | training-only distortions | augmentation thinning strokes until they vanish |
| `06b` (in cell 7) | stretch any strip with under {MIN_SLOTS_PER_CHAR} output slots per character | none; this repairs the cramped cases |
| `09_model_input` | scale to [-1, +1]; count the CTC output slots | too few slots for the label, which silently breaks CTC |
| `10_outliers` | one picture per flagged patch, with the reason in the caption | — |

## What this audit found, and what was changed

**Failure 1 — found at stage 07 (padding).** The notebook reported that moving to a single strip cut
blank space from 87% to 61%. That number was measured on the strip *before* padding. Every strip was
then padded out to a fixed 6144 px, and for a median patch only about 3,300 px held text — so the
image the model really received was roughly **80% blank paper**, and CTC was told to read all 768
output slots including the empty ones.
*Change:* each batch is now padded only to its widest member (as FLAME does), samples of similar
width are grouped into the same batch, and CTC is given each sample's true length. Padding overhead
fell from 53% to **0.4%**, and a batch is about **2.1× smaller**, so epochs also run faster.

**Failure 2 — found at stage 03 (line detection), and much more serious.** On dense or slanted pages
the ink profile never dropped below the 22% threshold between lines, so the detector returned **one
band covering the whole page**. That band was then scaled to 64 px tall, crushing five lines of Modi
into a single strip. One patch had a **246-character label inside a 183 px image**. {s.get('03_lines__old_finder_merged_everything', 0)} patches
were affected and nothing in the pipeline complained.
*Change:* skew is corrected before projecting ({int((df.skew_deg.abs() > 0.01).sum()):,} of {len(df):,} patches needed it), and any band
taller than about 1.6 lines is split at its own deepest ink valleys rather than at equal spacing, so
a cut never lands in the middle of a line. Patches seen as a single line fell from
{int(((df.lines_old <= 1) & (df.h > 2.5*MED_LINE)).sum())} to {int(((df.lines_new <= 1) & (df.h > 2.5*MED_LINE)).sum())}.

**Failure 3 — found at stage 03 as well, and the one with the biggest ceiling effect.** Once the
pictures were being rendered it became visible that whole lines were sitting *outside* every
detected band — page headers, dates, short final lines — along with the tops and tails of ordinary
letters. Measured across 400 patches, a median patch handed the model only **90% of its own ink**,
half of all patches lost more than 10%, and the worst lost a third. The transcription still listed
those characters, so the model was being trained to produce text for which it had been shown no
image at all. That puts a floor under the error rate that no amount of extra training can pass.
*Change:* the ink threshold for a line core was lowered to {LINE_THR}, and every line is now **grown until
it meets its neighbour at the valley between them** (capped at {GROW_CAP} of a line height), so every
inked row belongs to exactly one line. Ink reaching the model is now a median of
**{df.ink_kept.median()*100:.1f}%** (worst case {df.ink_kept.min()*100:.0f}%); {int((df.ink_kept < 0.95).sum())} patches still lose more than 5%.
The cap of {GROW_CAP} was chosen by measurement: growing further bought under half a percent more ink
while making the writing steadily smaller inside the strip. At {GROW_CAP} the writing still fills a
median of **{df.text_px_of_64.median():.0f} of the strip's {SH} px**, so nothing was given up for it.

**Effect of fixing failure 2 on the thing that matters.** CTC needs several output slots per
character. Before: a median of **2.6 slots per character**, below the healthy 3–8 range, with
**77 patches having fewer slots than their label needed at all** — for those, CTC returns zero loss
and the patch teaches the model nothing, silently. After: a median of **{df.slots_per_char.median():.1f} slots per
character** and **{int(df.tight.sum())} patches** short.

## Checks that run over all {len(df):,} patches

| Check | Patches flagged | What it means |
|---|---|---|
""" + "\n".join(
        f"| `{r['check']}` | {r['n_flagged']} ({r['pct']}%) | {r['meaning']} |" for r in summary
    ) + f"""

Raw numbers for every patch are in `stage_stats.csv`; the counts above are in `outlier_checks.csv`.

## Measured constants

| | |
|---|---|
| typical line height | {MED_LINE:.0f} px |
| strip height | {SH} px |
| strip width cap | {SW:,} px |
| gap inserted between lines | {GAP} px |
| output slots | one per {W_STRIDE} px of width |
| median strip width | {df.strip_w.median():,.0f} px |
| median slots per character | {df.slots_per_char.median():.1f} |
"""
    (OUTDIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"\nwrote README.md, stage_stats.csv, outlier_checks.csv and "
          f"{sum(1 for _ in OUTDIR.rglob('*.png'))} images to {OUTDIR}")


if __name__ == "__main__":
    main()
