"""X4 step 1 - render synthetic Modi text lines with exactly-known labels (hypothesis: synthetic pretraining
gets the reader past the letter-frequency plateau; UniLipi pretrains every model on ~2.5 M synthetic lines, J14
trains on font-rendered Modi).

Text: TRAINING-split labels only (no val/test text can leak), cut into line-sized chunks at word boundaries.
Script: Devanagari -> Modi by Unicode character NAME (DEVANAGARI LETTER KA -> MODI LETTER KA); the 7 rare marks
with no Modi equivalent (candrabindu, nukta, avagraha, candra e/o, RRA) are mapped to their closest base or dropped,
and the LABEL is kept in the same form so image and label stay consistent.
Font: Noto Sans Modi (OFL), shaped with Pillow + raqm. Augmented to look less like print: random size, slant,
rotation, stroke thickness, blur, speckle, elastic wobble.
Output: cache/synth_lines_h40.pkl (+ figures/x4_synth_examples.png).
usage: python x4_render_synthetic.py [--n 12000] [--seed 0]
"""
import argparse, json, pickle, random, re, time, unicodedata
from pathlib import Path
import numpy as np, pandas as pd, cv2
from PIL import Image, ImageDraw, ImageFont, features
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CODE = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=12000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-gaps", action="store_true", help="render words run together (real Modi has no word gaps); label keeps its spaces")
ap.add_argument("--headline", action="store_true", help="draw one continuous headline stroke across the line, as Modi scribes did")
ap.add_argument("--out", default="synth_lines_h40.pkl")
args = ap.parse_args()
rng = np.random.default_rng(args.seed); random.seed(args.seed)
assert features.check("raqm"), "Pillow needs raqm for complex-script shaping"
FONT = HERE / "cache" / "fonts" / "NotoSansModi-Regular.ttf"
H = 40
plt.rcParams["font.family"] = ["Nirmala UI", "DejaVu Sans"]

# ---- Devanagari -> Modi by Unicode name; marks with no Modi form are simplified in BOTH image and label
SIMPLIFY = {"ँ": "ं",   # candrabindu -> anusvara
            "ऑ": "ओ",   # candra O -> O
            "ऱ": "र",   # RRA -> RA
            "़": "",         # nukta -> dropped
            "ऽ": "",         # avagraha -> dropped
            "ॅ": "े",   # vowel sign candra E -> E
            "ॉ": "ो"}   # vowel sign candra O -> O


def to_modi(ch):
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return ch
    if name.startswith("DEVANAGARI "):
        try:
            return unicodedata.lookup(name.replace("DEVANAGARI", "MODI", 1))
        except KeyError:
            return None
    return ch


def normalise(s):
    return "".join(SIMPLIFY.get(c, c) for c in s)


cs = json.load(open(CODE / "assets/metadata/charset.json", encoding="utf-8"))
split = json.load(open(CODE / "assets/metadata/splits.json", encoding="utf-8"))["modetrans"]["assignment"]
text = pd.read_csv(CODE / "assets/metadata/modetrans_text.csv", keep_default_na=False).set_index("image_id")["text_train"].to_dict()
train_texts = [normalise(text[n]).replace("<FRAC>", "") for n in text if split[n] == "train"]

# line-sized chunks (18-70 characters, cut at spaces), like the real lines (median 37)
chunks = []
for t in train_texts:
    words = t.split()
    i = 0
    while i < len(words):
        target = int(rng.integers(18, 71)); cur = []
        while i < len(words) and len(" ".join(cur + [words[i]])) <= target:
            cur.append(words[i]); i += 1
        if not cur:
            cur = [words[i]]; i += 1
        chunks.append(" ".join(cur))
chunks = [c for c in chunks if c.strip()]
print(f"{len(train_texts)} training labels -> {len(chunks)} line-sized chunks", flush=True)


def render(label):
    shown = label.replace(" ", "") if args.no_gaps else label      # image may hide the spaces; the label keeps them
    modi = "".join(m for m in (to_modi(c) for c in shown) if m is not None)
    size = int(rng.integers(34, 60))
    font = ImageFont.truetype(str(FONT), size, layout_engine=ImageFont.Layout.RAQM)
    l, t, r, b = font.getbbox(modi)
    W, Hh = r - l + 20, b - t + 20
    img = Image.new("L", (max(8, W), max(8, Hh)), 255)
    ImageDraw.Draw(img).text((10 - l, 10 - t), modi, font=font, fill=0)
    g = np.array(img)
    if args.headline:
        ink = (g < 128).sum(1)
        top = int(np.argmax(ink[: max(1, int(0.55 * g.shape[0]))]))          # the glyphs' own top bar
        thick = max(2, size // 14)
        cols = np.where((g < 128).any(0))[0]
        if len(cols):
            g = g.copy(); g[max(0, top - thick // 2): top + (thick + 1) // 2, max(0, cols[0] - 4): cols[-1] + 5] = 0
    # look less like print
    h, w = g.shape
    shear = rng.uniform(-0.25, 0.25)
    M = np.float32([[1, shear, -shear * h / 2], [0, 1, 0]])
    g = cv2.warpAffine(g, M, (w, h), borderValue=255)
    if rng.random() < 0.7:
        k = np.ones((int(rng.integers(1, 4)), int(rng.integers(1, 4))), np.uint8)
        g = cv2.erode(g, k) if rng.random() < 0.7 else cv2.dilate(g, k)      # erode = thicker dark strokes
    if rng.random() < 0.5:
        a = rng.uniform(1.5, 4.0); s = rng.uniform(6, 12)
        dx = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), s) * a * s
        dy = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), s) * a * s
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        g = cv2.remap(g, gx + dx, gy + dy, cv2.INTER_LINEAR, borderValue=255)
    if rng.random() < 0.5:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-2.5, 2.5), 1.0)
        g = cv2.warpAffine(g, M, (w, h), borderValue=255)
    if rng.random() < 0.4:
        g = cv2.GaussianBlur(g, (3, 3), 0)
    if rng.random() < 0.35:
        noise = rng.random(g.shape) < rng.uniform(0.002, 0.02)
        g = g.copy(); g[noise] = 0
    g = np.where(g < 140, 0, 255).astype(np.uint8)        # binarised, like most real patches
    ink = g < 128
    rows = np.where(ink.any(1))[0]; cols = np.where(ink.any(0))[0]
    if len(rows) == 0:
        return None
    g = g[max(0, rows[0] - 2):rows[-1] + 3, max(0, cols[0] - 2):cols[-1] + 3]
    nw = max(8, int(round(g.shape[1] * H / g.shape[0])))
    return cv2.resize(g, (min(nw, 1024), H), interpolation=cv2.INTER_AREA)


t0 = time.time(); out = []
for k in range(args.n):
    lab = chunks[int(rng.integers(len(chunks)))]
    img = render(lab)
    if img is not None:
        out.append(dict(img=img, label=lab))
pickle.dump(dict(lines=out, H=H, no_gaps=args.no_gaps, headline=args.headline), open(HERE / "cache" / args.out, "wb"))
w = np.array([L["img"].shape[1] for L in out]); c = np.array([len(L["label"]) for L in out])
print(f"rendered {len(out):,} synthetic lines in {time.time()-t0:.0f}s | width median {np.median(w):.0f} | "
      f"chars median {np.median(c):.0f} | frames/char at stride 2: median {np.median(w/2/c):.1f}", flush=True)
fig, axes = plt.subplots(10, 1, figsize=(15, 10))
for ax, L in zip(axes, out[:10]):
    ax.imshow(L["img"], cmap="gray", vmin=0, vmax=255); ax.axis("off"); ax.set_title(L["label"], fontsize=10, loc="left")
figname = "x4_synth_examples" + ("_nogaps" if args.no_gaps else "") + ("_headline" if args.headline else "") + ".png"
fig.tight_layout(); fig.savefig(CODE / "research/root-cause/figures" / figname, dpi=72); plt.close(fig)
print(f"wrote cache/{args.out} and figures/{figname}")
