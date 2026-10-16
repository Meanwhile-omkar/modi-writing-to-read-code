"""X2 step 1 - turn every patch into line images with pseudo line labels (hypotheses H8, H9).

MoDeTrans has no line-level labels (E6), so each patch's label is split across its detected lines in proportion to
each line's ink width, cutting only at word boundaries (spaces). Lines come from the scale-adaptive segmenter of
X0.3 (exact line count 20/24 vs 03's 6/24). Each line is cropped tight to its ink and scaled to height 40 px, which
also removes the 2.6x scale variation between patches (H2).
Output: experiments/cache/lines_h40.pkl, figures/x2_lines_examples.png, printed statistics.
"""
import io, json, pickle, sys, time, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "FYP-modi-script-code"
OUT_FIG = CODE / "research" / "root-cause" / "figures"
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
H = 40
MAXW = 1024
plt.rcParams["font.family"] = ["Nirmala UI", "DejaVu Sans"]

spec = importlib.util.spec_from_file_location("x03", HERE / "x03_line_pitch.py")
x03 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x03)


def split_label(label, weights):
    """Split `label` into len(weights) pieces, proportional to weights, cutting only at spaces."""
    k = len(weights)
    if k == 1:
        return [label]
    spaces = [i for i, ch in enumerate(label) if ch == " "]
    total = len(label); cum = np.cumsum(weights) / np.sum(weights)
    cuts, prev = [], -1
    for j in range(k - 1):
        target = cum[j] * total
        cand = [s for s in spaces if s > prev]
        if not cand:
            break
        s = min(cand, key=lambda x: abs(x - target))
        cuts.append(s); prev = s
    pieces, start = [], 0
    for c in cuts:
        pieces.append(label[start:c].strip()); start = c + 1
    pieces.append(label[start:].strip())
    while len(pieces) < k:                       # fewer word boundaries than lines: leave the extra lines empty
        pieces.append("")
    return pieces


def crop_line(g2, a, b):
    band = g2[a:b]
    ink = band < 128
    rows = np.where(ink.any(1))[0]; cols = np.where(ink.any(0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None, 0
    r0, r1 = max(0, rows[0] - 2), min(band.shape[0], rows[-1] + 3)
    c0, c1 = max(0, cols[0] - 2), min(band.shape[1], cols[-1] + 3)
    crop = band[r0:r1, c0:c1]
    w = max(8, int(round(crop.shape[1] * H / crop.shape[0])))
    w = min(w, MAXW)
    return cv2.resize(crop, (w, H), interpolation=cv2.INTER_AREA), c1 - c0


def main():
    t0 = time.time()
    pq = pd.read_parquet(ROOT / "FYP-modi-script-data/MoDeTrans-dataset/train-00000-of-00001.parquet")
    text = pd.read_csv(CODE / "assets/metadata/modetrans_text.csv", keep_default_na=False).set_index("image_id")["text_train"].to_dict()
    split = json.load(open(CODE / "assets/metadata/splits.json", encoding="utf-8"))["modetrans"]["assignment"]
    lines, per_patch = [], []
    for r in pq.itertuples():
        g = np.array(Image.open(io.BytesIO(r.image["bytes"])).convert("L"))
        g2, bands, pitch, conf, ang = x03.segment(g)
        crops = [crop_line(g2, a, b) for a, b in bands]
        crops = [(c, w) for c, w in crops if c is not None]
        label = text[r.filename]
        pieces = split_label(label, [w for _, w in crops]) if crops else []
        for i, ((img, w), lab) in enumerate(zip(crops, pieces)):
            lines.append(dict(patch=r.filename, split=split[r.filename], idx=i, n_lines=len(crops), img=img, label=lab))
        per_patch.append(dict(patch=r.filename, split=split[r.filename], n_lines=len(crops), pitch=pitch, label_len=len(label)))
    pickle.dump(dict(lines=lines, patches=per_patch, H=H), open(CACHE / "lines_h40.pkl", "wb"))

    d = pd.DataFrame([{k: v for k, v in L.items() if k != "img"} for L in lines])
    d["w"] = [L["img"].shape[1] for L in lines]; d["chars"] = d.label.str.len()
    d["frames_per_char"] = (d.w / 4) / d.chars.clip(lower=1)
    pp = pd.DataFrame(per_patch)
    print(f"{len(lines):,} line images from {len(pp):,} patches in {time.time()-t0:.0f}s")
    print("lines per patch:", pp.n_lines.value_counts().sort_index().to_dict())
    print("by split:", d.split.value_counts().to_dict())
    print(f"line width at {H}px: median {d.w.median():.0f}, 95% {d.w.quantile(.95):.0f}, max {d.w.max()}")
    print(f"characters per line: median {d.chars.median():.0f}, 5% {d.chars.quantile(.05):.0f}, 95% {d.chars.quantile(.95):.0f}; empty lines {int((d.chars==0).sum())}")
    print(f"CTC frames per character at width stride 4: median {d.frames_per_char.median():.1f}; "
          f"below 1.5 (likely infeasible): {int((d.frames_per_char < 1.5).sum())} lines")

    ex = d[d.split == "train"].sample(12, random_state=0).index
    fig, axes = plt.subplots(12, 1, figsize=(16, 13))
    for ax, i in zip(axes, ex):
        L = lines[i]
        ax.imshow(L["img"], cmap="gray", vmin=0, vmax=255, aspect="equal"); ax.axis("off")
        ax.set_title(f"{L['patch']} line {L['idx']+1}/{L['n_lines']}  ->  {L['label']}", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(OUT_FIG / "x2_lines_examples.png", dpi=70); plt.close(fig)
    print("wrote cache/lines_h40.pkl and figures/x2_lines_examples.png")


if __name__ == "__main__":
    main()
