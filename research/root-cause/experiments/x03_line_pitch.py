"""X0.3 - estimate each patch's own line pitch, and segment lines with it (hypotheses H2, H21).

03 used ONE line height (56 px) for every patch; real lines run from ~30 to ~150 px tall (X0.1, X0.5). Here:
  1. pitch = the dominant period of the row-ink profile (autocorrelation peak, lag 24-260 px);
     for one-line patches (no period) fall back to the height of the main ink band.
  2. lines = the profile smoothed at a scale tied to the pitch, cut at its minima, with cuts at least 0.65 x pitch apart.
Checked against MY OWN line counts for the 24 patches audited in X0.1/X0.5 (counted by eye from the renders).
Output: printed accuracy table, figures/x03_segmentation_{group}.png, pitch distribution over all 2,043 patches.
"""
import io, sys, time
from pathlib import Path
import numpy as np, pandas as pd, cv2
from PIL import Image
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "FYP-modi-script-code"
OUT = CODE / "research" / "root-cause" / "figures"

# visible text lines, counted by eye from x01_*.png / x05_*.png (headers, dates and signatures count as lines)
MANUAL = {"715.jpg": 2, "1028.jpg": 5, "1934.jpg": 2, "324.jpg": 3, "1562.jpg": 2, "132.jpg": 4, "1493.jpg": 2,
          "1073.jpg": 4, "353.jpg": 4, "653.jpg": 4, "838.jpg": 2, "1136.jpg": 3,
          "869.jpg": 2, "1367.jpg": 1, "868.jpg": 2, "1375.jpg": 1, "858.jpg": 2, "701.jpg": 3,
          "1116.jpg": 5, "1090.jpg": 7, "1193.jpg": 5, "1087.jpg": 5, "20.jpg": 5, "1142.jpg": 5}


def row_profile(g):
    return (cv2.GaussianBlur(g, (5, 5), 0) < 128).sum(1).astype(np.float64)


def deskew(g, rng=6.0, step=0.5):
    h, w = g.shape; best, ba = -1.0, 0.0
    for a in np.arange(-rng, rng + 1e-9, step):
        r = g if a == 0 else cv2.warpAffine(g, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0), (w, h),
                                            flags=cv2.INTER_NEAREST, borderValue=255)
        s = float(np.sum(np.diff(row_profile(r)) ** 2))
        if s > best: best, ba = s, a
    if ba == 0: return g, 0.0
    return cv2.warpAffine(g, cv2.getRotationMatrix2D((w / 2, h / 2), ba, 1.0), (w, h),
                          flags=cv2.INTER_LINEAR, borderValue=255), ba


def estimate_pitch(p, lo=24, hi=260):
    """Dominant period of the row profile. Returns (pitch, confidence)."""
    q = p - p.mean()
    ac = np.correlate(q, q, mode="full")[len(q) - 1:]
    if ac[0] <= 0: return None, 0.0
    ac = ac / ac[0]
    hi = min(hi, len(ac) - 1)
    peaks, props = find_peaks(ac[lo:hi], height=0.05)
    if len(peaks) == 0: return None, 0.0
    k = int(np.argmax(props["peak_heights"]))
    return float(peaks[k] + lo), float(props["peak_heights"][k])


def main_band_height(p):
    """Fallback for one-line patches: height of the largest contiguous block of inked rows."""
    on = p > 0.15 * p.max()
    best = cur = 0
    for v in on:
        cur = cur + 1 if v else 0; best = max(best, cur)
    return float(best)


def segment(g):
    g2, ang = deskew(g)
    p = row_profile(g2)
    pitch, conf = estimate_pitch(p)
    if pitch is None or conf < 0.12:                    # no clear rhythm: treat as a single line
        pitch = max(30.0, main_band_height(p)); conf = 0.0
    sm = cv2.GaussianBlur(p.reshape(-1, 1), (1, 0), sigmaX=0, sigmaY=max(2.0, pitch / 8)).ravel()
    inked = np.where(p > 0.02 * p.max())[0]
    top, bot = (int(inked[0]), int(inked[-1]) + 1) if len(inked) else (0, len(p))
    mins, _ = find_peaks(-sm[top:bot], distance=max(4, int(0.65 * pitch)), prominence=0.04 * sm.max())
    cuts = [top] + [int(m + top) for m in mins] + [bot]
    lines = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if (b - a) > 0.35 * pitch and p[a:b].sum() > 0.01 * p.sum()]
    return g2, lines, pitch, conf, ang


def main():
    t0 = time.time()
    pq = pd.read_parquet(ROOT / "FYP-modi-script-data/MoDeTrans-dataset/train-00000-of-00001.parquet")
    row = {r.filename: r for r in pq.itertuples()}
    res = []
    for n, want in MANUAL.items():
        g = np.array(Image.open(io.BytesIO(row[n].image["bytes"])).convert("L"))
        g2, lines, pitch, conf, ang = segment(g)
        res.append(dict(name=n, manual=want, detected=len(lines), pitch=round(pitch), conf=round(conf, 2), g2=g2, lines=lines))
    d = pd.DataFrame(res)
    exact = int((d.detected == d.manual).sum()); within1 = int(((d.detected - d.manual).abs() <= 1).sum())
    print(f"line count vs my manual count on 24 patches: exact {exact}/24, within one {within1}/24")
    # 03's band counts on the same patches, as printed by x05_line_finder_audit.py
    B03 = {"715.jpg": 8, "1028.jpg": 7, "1934.jpg": 6, "324.jpg": 3, "1562.jpg": 5, "132.jpg": 5, "1493.jpg": 4,
           "1073.jpg": 4, "353.jpg": 5, "653.jpg": 6, "838.jpg": 6, "1136.jpg": 4, "869.jpg": 6, "1367.jpg": 2,
           "868.jpg": 6, "1375.jpg": 4, "858.jpg": 6, "701.jpg": 6, "1116.jpg": 5, "1090.jpg": 7, "1193.jpg": 4,
           "1087.jpg": 6, "20.jpg": 5, "1142.jpg": 5}
    e03 = sum(B03[n] == MANUAL[n] for n in MANUAL); w03 = sum(abs(B03[n] - MANUAL[n]) <= 1 for n in MANUAL)
    print(f"  03's line finder on the same patches: exact {e03}/24, within one {w03}/24")
    for _, r in d.iterrows():
        flag = "" if r.detected == r.manual else "   <-- off"
        print(f"  {r['name']:>9}  manual {r.manual}  detected {r.detected}  pitch {r.pitch:>4}  conf {r.conf:4.2f}{flag}")

    for gname, names in [("a", list(MANUAL)[:12]), ("b", list(MANUAL)[12:])]:
        sub = d[d.name.isin(names)]
        fig, axes = plt.subplots(len(sub), 1, figsize=(11, 2.4 * len(sub)))
        for ax, (_, r) in zip(axes, sub.iterrows()):
            ax.imshow(r.g2, cmap="gray", vmin=0, vmax=255); ax.axis("off")
            for k, (a, b) in enumerate(r.lines):
                col = "lime" if k % 2 == 0 else "magenta"
                ax.axhspan(a, b, color=col, alpha=0.15); ax.axhline(a, color=col, lw=1.2)
            ax.set_title(f"{r['name']}: pitch {r.pitch} px (conf {r.conf}) -> {r.detected} lines (manual {r.manual})", fontsize=8, loc="left")
        fig.tight_layout(); fig.savefig(OUT / f"x03_segmentation_{gname}.png", dpi=68); plt.close(fig)

    # pitch over the whole dataset (H2, quantified)
    pitches = []
    for r in pq.itertuples():
        g = np.array(Image.open(io.BytesIO(r.image["bytes"])).convert("L"))
        p = row_profile(g)
        pitch, conf = estimate_pitch(p)
        if pitch is None or conf < 0.12: pitch = max(30.0, main_band_height(p))
        pitches.append(pitch)
    pitches = np.array(pitches)
    q = np.percentile(pitches, [5, 25, 50, 75, 95])
    print(f"\nline pitch over all 2,043 patches: 5% {q[0]:.0f}  25% {q[1]:.0f}  median {q[2]:.0f}  75% {q[3]:.0f}  95% {q[4]:.0f} px"
          f"  -> {q[4]/q[0]:.1f}x range between 5th and 95th percentile")
    np.save(OUT.parent / "experiments" / "x03_pitch.npy", pitches)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
