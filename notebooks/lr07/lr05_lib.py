"""lr05_lib.py - the line-level Modi reader and every round-2 hypothesis, in one tested file.

Provenance - every piece below was first proven on CPU (research/root-cause/LAB_NOTEBOOK.md):
  segment / split_label / crop_line   X0.3 + X2: each patch's own line pitch; exact line count on 20/24
                                      hand-counted patches (notebook 03's finder: 6/24)
  render_line                         X4: Noto Sans Modi (OFL), Devanagari -> Modi by Unicode name,
                                      print-breaking augmentation. Only change: the elastic wobble is
                                      drawn from a coarse grid (same strength, ~10x cheaper)
  CRNN                                X2/X4b: conv 32-64-128-128-128 + BiLSTM 2x128 + CTC, width stride 2
  viterbi_ctc / realign_labels        X3: CTC forced alignment of the full patch label across its lines
  CharLM / beam_decode                X6: character n-gram LM from TRAINING labels only, fused into a
                                      CTC prefix beam search
New here: akshara targets (N14), no-space targets (N5), LM-sampled synthetic text (N12b), a shared
on-disk line store, the GPU training loop, error anatomy (N1/N2), paired bootstrap comparisons.
"""
import os, re, io, json, math, time, random, collections, unicodedata, threading, queue
import numpy as np
import cv2

# ======================================================================= alphabet and targets
TOK = re.compile(r"<FRAC>|.", re.S)
_C = "क-हक़-य़ॸ-ॿ"                    # consonants
_V = "ऺऻा-ौॎॏॕ-ॗॢॣ"  # dependent vowel signs
_M = "ऀ-ः"                                               # candrabindu, anusvara, visarga
_I = "ऄ-औॠॡॲ-ॷ"                      # independent vowels
# one akshara = (consonant [nukta] virama)* consonant [nukta] (virama | vowel signs*) modifiers*
AKSHARA = re.compile(rf"<FRAC>|(?:[{_C}]़?्)*[{_C}]़?(?:्|[{_V}]*)[{_M}]*|[{_I}][{_M}]*|.", re.S)
COMBINING = re.compile(rf"[{_V}{_M}़्]")


class Vocab:
    """CTC output alphabet.
    kind='char'    : 0 = blank, the 86 dataset symbols, <FRAC>, <unk>. Identical to notebooks 02-04 and the
                     CPU experiments, so numbers stay comparable.
    kind='akshara' : the same, plus every multi-character akshara (syllable cluster) seen >= min_count times
                     in the TRAINING labels. Rarer aksharas are spelled character by character, so any text
                     can still be written and every metric is still computed on characters.
    no_space=True  : spaces are removed from the targets (Modi is written without word gaps; N5)."""

    def __init__(self, symbols, kind="char", train_texts=(), min_count=3, no_space=False):
        self.kind, self.no_space, self.min_count = kind, bool(no_space), min_count
        self.itos = ["<blank>"] + list(symbols) + ["<FRAC>", "<unk>"]
        if kind == "akshara":
            cnt = collections.Counter(a for t in train_texts for a in AKSHARA.findall(t))
            self.itos += sorted(a for a, c in cnt.items() if c >= min_count and len(TOK.findall(a)) > 1)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.unk = self.stoi["<unk>"]

    def __len__(self):
        return len(self.itos)

    def encode(self, s):
        if self.no_space:
            s = s.replace(" ", "")
        if self.kind == "char":
            return [self.stoi.get(t, self.unk) for t in TOK.findall(s)]
        out = []
        for a in AKSHARA.findall(s):
            i = self.stoi.get(a)
            if i is not None:
                out.append(i)
            else:
                out += [self.stoi.get(t, self.unk) for t in TOK.findall(a)]
        return out

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i > 0)

    def describe(self):
        return dict(kind=self.kind, no_space=self.no_space, size=len(self.itos))


def make_vocab(symbols, train_texts, kind="char", no_space=False):
    return Vocab(symbols, kind, train_texts, 3, no_space)


# ======================================================================= line segmentation (X0.3, X2)
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
    from scipy.signal import find_peaks
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
    from scipy.signal import find_peaks
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


def crop_line(g2, a, b, heights=(40,), maxw=None):
    """Tight ink crop of rows a:b, scaled to each height in `heights` (one resample each from the crop).
    Returns ({H: image}, ink width in source pixels) or (None, 0) for an empty band."""
    band = g2[a:b]
    ink = band < 128
    rows = np.where(ink.any(1))[0]; cols = np.where(ink.any(0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None, 0
    r0, r1 = max(0, rows[0] - 2), min(band.shape[0], rows[-1] + 3)
    c0, c1 = max(0, cols[0] - 2), min(band.shape[1], cols[-1] + 3)
    crop = band[r0:r1, c0:c1]
    out = {}
    for H in heights:
        w = max(8, int(round(crop.shape[1] * H / crop.shape[0])))
        w = min(w, (maxw or {}).get(H, int(1024 * H / 40)))
        out[H] = cv2.resize(crop, (w, H), interpolation=cv2.INTER_AREA)
    return out, c1 - c0


def patch_to_lines(g, label, heights=(40, 64)):
    """One patch -> its line images (per height), proportional pseudo labels and patch statistics."""
    g2, bands, pitch, conf, ang = segment(g)
    crops = [crop_line(g2, a, b, heights) for a, b in bands]
    crops = [(c, w) for c, w in crops if c is not None]
    pieces = split_label(label, [w for _, w in crops]) if crops else []
    lines = [dict(imgs=c, label=lab, idx=i, n_lines=len(crops)) for i, ((c, w), lab) in enumerate(zip(crops, pieces))]
    ink = float((g2 < 128).mean())
    return lines, dict(n_lines=len(crops), pitch=float(pitch), pitch_conf=float(conf), angle=float(ang), ink=ink)


# ======================================================================= shared line store
class LineWriter:
    """Appends line images of one height to <base>.u8 and keeps the index in memory. publish() writes the
    index to <base>.json atomically: a reader that sees the .json may use every line it lists, even while
    the writer keeps appending (milestones)."""

    def __init__(self, base, H):
        self.base, self.H = base, H
        self.f = open(base + ".u8", "wb"); self.off = 0
        self.widths, self.offs, self.labels, self.info = [], [], [], []

    def add(self, img, label, info=None):
        assert img.ndim == 2 and img.shape[0] == self.H and img.dtype == np.uint8, (img.shape, img.dtype)
        self.f.write(np.ascontiguousarray(img).tobytes())
        self.offs.append(self.off); self.widths.append(int(img.shape[1])); self.off += int(img.size)
        self.labels.append(label); self.info.append(info)

    def n_lines_ok(self, limit):
        return limit is None or len(self.widths) < limit

    def publish(self, final=False, extra=None):
        self.f.flush(); os.fsync(self.f.fileno())
        meta = dict(H=self.H, n=len(self.widths), final=final, widths=self.widths, offs=self.offs,
                    labels=self.labels, info=self.info, **(extra or {}))
        with open(self.base + ".json.tmp", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        os.replace(self.base + ".json.tmp", self.base + ".json")

    def close(self, extra=None):
        self.publish(final=True, extra=extra)
        self.f.close()


class LineStore:
    """Read side of LineWriter: memory-mapped, so every process shares one copy through the page cache."""

    def __init__(self, base, limit=None):
        meta = json.load(open(base + ".json", encoding="utf-8"))
        n = meta["n"] if limit is None else min(meta["n"], int(limit))
        self.base, self.H, self.n = base, meta["H"], n
        self.widths = np.asarray(meta["widths"][:n], np.int64)
        self.offs = np.asarray(meta["offs"][:n], np.int64)
        self.labels = list(meta["labels"][:n])
        self.info = list(meta["info"][:n])
        self.meta = {k: v for k, v in meta.items() if k not in ("widths", "offs", "labels", "info")}
        self.pix = np.memmap(base + ".u8", dtype=np.uint8, mode="r")

    def __len__(self):
        return self.n

    def img(self, i):
        w = int(self.widths[i]); o = int(self.offs[i])
        return np.asarray(self.pix[o:o + self.H * w]).reshape(self.H, w)


def store_ready(base, need=None):
    try:
        meta_n = json.load(open(base + ".json", encoding="utf-8"))["n"]
    except Exception:
        return False
    return need is None or meta_n >= need


# ======================================================================= synthetic Modi lines (X4)
SIMPLIFY = {"ँ": "ं",   # candrabindu -> anusvara
            "ऑ": "ओ",   # candra O -> O
            "ऱ": "र",   # RRA -> RA
            "़": "",          # nukta -> dropped
            "ऽ": "",          # avagraha -> dropped
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


def label_chunks(texts, rng, lo=18, hi=70):
    """Cut texts into line-sized chunks (lo-hi characters) at spaces, like the real lines (median 37)."""
    chunks = []
    for t in texts:
        words = t.split()
        i = 0
        while i < len(words):
            target = int(rng.integers(lo, hi + 1)); cur = []
            while i < len(words) and len(" ".join(cur + [words[i]])) <= target:
                cur.append(words[i]); i += 1
            if not cur:
                cur = [words[i]]; i += 1
            chunks.append(" ".join(cur))
    return [c for c in chunks if c.strip()]


class TextSampler:
    """N12b - new text from a character n-gram model of the TRAINING labels (offline stand-in for an
    external Marathi corpus): sampled from the longest seen context, so words recombine and new
    word forms appear (25% of validation words never occur in training)."""

    def __init__(self, texts, order=4):
        self.order = order
        cnt = collections.defaultdict(collections.Counter)
        for t in texts:
            s = "^" * (order - 1) + t + "$"
            for i in range(order - 1, len(s)):
                for k in range(order):                      # contexts of length 0..order-1
                    cnt[s[i - k:i]][s[i]] += 1
        self.table = {}
        for ctx, c in cnt.items():
            chars = list(c.keys()); p = np.array([c[x] for x in chars], np.float64)
            self.table[ctx] = (chars, np.cumsum(p) / p.sum())
        self.bad = re.compile(rf"(^|\s)[{_V}{_M}़्]|््|[{_V}][{_V}]{{2,}}")

    def _next(self, hist, rng):
        for k in range(self.order - 1, -1, -1):
            ctx = hist[len(hist) - k:] if k else ""
            e = self.table.get(ctx)
            if e is not None:
                chars, cdf = e
                return chars[int(np.searchsorted(cdf, rng.random(), side="right").clip(0, len(chars) - 1))]
        return " "

    def sample(self, rng, lo=18, hi=70, tries=20):
        for _ in range(tries):
            target = int(rng.integers(lo, hi + 1))
            hist = "^" * (self.order - 1); out = []
            while len(out) < target + 25:
                ch = self._next(hist, rng)
                if ch in "$^":
                    hist = "^" * (self.order - 1); ch = " "
                out.append(ch); hist = (hist + ch)[-(self.order - 1):]
            s = "".join(out)
            words = s.split()[1:-1] if len(s.split()) > 2 else s.split()   # drop the clipped first/last word
            text, cur = [], 0
            for w in words:
                if cur + len(w) + (1 if text else 0) > target and text:
                    break
                text.append(w); cur += len(w) + (1 if len(text) > 1 else 0)
            t = " ".join(text).strip()
            if t and not self.bad.search(t):
                return t
        return None


_FONTS = {}


def _font(path, size):
    from PIL import ImageFont
    k = (path, size)
    if k not in _FONTS:
        try:
            _FONTS[k] = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
        except Exception:
            _FONTS[k] = ImageFont.truetype(path, size)        # basic layout if raqm is missing (logged by the caller)
    return _FONTS[k]


def _coarse_field(h, w, spacing, amp, rng):
    gh, gw = max(2, int(h / spacing) + 2), max(2, int(w / spacing) + 2)
    f = rng.uniform(-1, 1, (gh, gw)).astype(np.float32)
    return cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC) * amp


def render_line(label, rng, font_path, heights=(40,), no_gaps=False, headline=False, heavy=False):
    """Render one label in Modi glyphs; return {H: image} (one resample per height from the crop) or None.
    The label keeps its spaces even when no_gaps hides them in the image. heavy=True (notebook 07) widens every
    distortion (shear, stroke thickness, elastic wobble, rotation) while drawing the same random numbers."""
    from PIL import Image, ImageDraw
    shown = label.replace(" ", "") if no_gaps else label
    modi = "".join(m for m in (to_modi(c) for c in shown) if m is not None)
    if not modi.strip():
        return None
    size = int(rng.integers(34, 60))
    font = _font(font_path, size)
    l, t, r, b = font.getbbox(modi)
    W, Hh = r - l + 20, b - t + 20
    img = Image.new("L", (max(8, W), max(8, Hh)), 255)
    ImageDraw.Draw(img).text((10 - l, 10 - t), modi, font=font, fill=0)
    g = np.array(img)
    if headline:                                       # one continuous top stroke, as Modi scribes wrote
        ink = (g < 128).sum(1)
        top = int(np.argmax(ink[: max(1, int(0.55 * g.shape[0]))]))
        thick = max(2, size // 14)
        cols = np.where((g < 128).any(0))[0]
        if len(cols):
            g = g.copy(); g[max(0, top - thick // 2): top + (thick + 1) // 2, max(0, cols[0] - 4): cols[-1] + 5] = 0
    h, w = g.shape
    shear = rng.uniform(-0.4, 0.4) if heavy else rng.uniform(-0.25, 0.25)
    M = np.float32([[1, shear, -shear * h / 2], [0, 1, 0]])
    g = cv2.warpAffine(g, M, (w, h), borderValue=255)
    if rng.random() < (0.9 if heavy else 0.7):
        k = np.ones((int(rng.integers(1, 5 if heavy else 4)), int(rng.integers(1, 5 if heavy else 4))), np.uint8)
        g = cv2.erode(g, k) if rng.random() < 0.7 else cv2.dilate(g, k)      # erode = thicker dark strokes
    if rng.random() < (1.0 if heavy else 0.5):        # elastic wobble (X4: blurred noise, std ~0.16*a px)
        a = rng.uniform(3.0, 7.0) if heavy else rng.uniform(1.5, 4.0); s = rng.uniform(5, 10) if heavy else rng.uniform(6, 12)
        dx = _coarse_field(h, w, s, 0.35 * a, rng); dy = _coarse_field(h, w, s, 0.35 * a, rng)
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        g = cv2.remap(g, gx + dx, gy + dy, cv2.INTER_LINEAR, borderValue=255)
    if rng.random() < (0.8 if heavy else 0.5):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-4.5, 4.5) if heavy else rng.uniform(-2.5, 2.5), 1.0)
        g = cv2.warpAffine(g, M, (w, h), borderValue=255)
    if rng.random() < 0.4:
        g = cv2.GaussianBlur(g, (3, 3), 0)
    if rng.random() < 0.35:
        noise = rng.random(g.shape) < rng.uniform(0.002, 0.02)
        g = g.copy(); g[noise] = 0
    g = np.where(g < 140, 0, 255).astype(np.uint8)    # binarised, like most real patches
    ink = g < 128
    rows = np.where(ink.any(1))[0]; cols = np.where(ink.any(0))[0]
    if len(rows) == 0:
        return None
    g = g[max(0, rows[0] - 2):rows[-1] + 3, max(0, cols[0] - 2):cols[-1] + 3]
    out = {}
    for H in heights:
        nw = max(8, int(round(g.shape[1] * H / g.shape[0])))
        out[H] = cv2.resize(g, (min(nw, int(1024 * H / 40)), H), interpolation=cv2.INTER_AREA)
    return out


# ======================================================================= model
class CRNN:
    pass


def build_crnn(n_classes, size="small"):
    """small = the CPU-proven reader (1.07 M parameters with 89 classes); big = twice as wide (N11).
    Height-agnostic: the conv stack halves the height three times and the rest is averaged."""
    import torch.nn as nn

    class _CRNN(nn.Module):
        def __init__(self, n, c, hid):
            super().__init__()
            L, cin = [], 1
            for i, co in enumerate(c):
                L += [nn.Conv2d(cin, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True)]
                if i == 0: L.append(nn.MaxPool2d(2))              # H -> H/2, width / 2
                if i in (1, 2): L.append(nn.MaxPool2d((2, 1)))    # H/2 -> H/4 -> H/8
                cin = co
            self.cnn = nn.Sequential(*L)
            self.rnn = nn.LSTM(cin, hid, 2, bidirectional=True, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(2 * hid, n)

        def forward(self, x):
            f = self.cnn(x).mean(2).permute(0, 2, 1)                 # (B, W/2, C)
            return self.fc(self.rnn(f)[0])                            # (B, T, classes)

    cfg = dict(small=((32, 64, 128, 128, 128), 128), big=((64, 128, 256, 256, 256), 256))[size]
    return _CRNN(n_classes, *cfg)


# ======================================================================= batches
STRIDE = 2


def line_ok(n_targets, width):
    """CTC feasibility rule used since X2: room for every target plus repeats."""
    return 0 < n_targets and n_targets * 1.2 + 2 <= width / STRIDE


def augment_line(img, rng, mode="light"):
    """X2's light augmentation for real lines (N6): small rotation/scale/shear, stroke thickness.
    Notebook 07 adds stronger modes aimed at what failed on unseen pages (Folder 18):
      stretch  first squeeze or stretch the line horizontally, 0.75-1.4x (character density: big vs small writing)
      tilt     first rotate the line by up to +-5 degrees on a taller canvas and re-crop it (tilted pages)
      strong   stretch + tilt, and an occasional blur"""
    h, w = img.shape
    if mode in ("tilt", "strong") and rng.random() < 0.6:
        a = rng.uniform(-5, 5)
        pad = int(math.ceil(abs(math.sin(math.radians(a))) * w / 2)) + 2
        big = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=255)
        big = cv2.warpAffine(big, cv2.getRotationMatrix2D((w / 2, big.shape[0] / 2), a, 1.0), (w, big.shape[0]),
                             borderValue=255)
        ink = big < 128
        rows, cols = np.where(ink.any(1))[0], np.where(ink.any(0))[0]
        if len(rows) and len(cols):
            crop = big[max(0, rows[0] - 2):rows[-1] + 3, max(0, cols[0] - 2):cols[-1] + 3]
            img = cv2.resize(crop, (max(8, min(int(1024 * h / 40), int(round(crop.shape[1] * h / crop.shape[0])))), h),
                             interpolation=cv2.INTER_AREA)
            h, w = img.shape
    if mode in ("stretch", "strong") and rng.random() < 0.5:
        s = rng.uniform(0.75, 1.4)
        img = cv2.resize(img, (max(8, min(int(1024 * h / 40), int(round(w * s)))), h),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        h, w = img.shape
    if rng.random() < 0.5:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-2, 2), rng.uniform(0.92, 1.05))
        M[0, 1] += rng.uniform(-0.15, 0.15)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    if rng.random() < 0.4:
        k = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, k) if rng.random() < 0.5 else cv2.dilate(img, k)
    if mode == "strong" and rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def to_batch(imgs, H, round_to=32):
    W = max(im.shape[1] for im in imgs)
    W = int(math.ceil(W / round_to) * round_to)
    x = np.ones((len(imgs), 1, H, W), np.float32)
    for i, im in enumerate(imgs):
        x[i, 0, :, :im.shape[1]] = im.astype(np.float32) / 127.5 - 1.0      # (x/255 - 0.5)/0.5, white = 1
    lens = np.array([int(math.ceil(im.shape[1] / STRIDE)) for im in imgs], np.int64)
    return x, lens


def batch_plan(widths, idxs, batch, rng, mega=50):
    """Shuffle, then sort by width inside mega-chunks (fast, little padding), then shuffle batch order."""
    idxs = np.array(idxs, np.int64)
    idxs = idxs[rng.permutation(len(idxs))]
    out = []
    for s in range(0, len(idxs), batch * mega):
        ch = idxs[s:s + batch * mega]
        ch = ch[np.argsort(widths[ch], kind="stable")]
        out += [ch[i:i + batch] for i in range(0, len(ch), batch)]
    order = rng.permutation(len(out))
    return [out[i] for i in order]


class Prefetch:
    """Builds batches in a background thread (numpy/cv2 release the GIL) so the GPU is not kept waiting."""

    def __init__(self, gen, depth=6):
        self.q = queue.Queue(depth); self.err = None
        self.t = threading.Thread(target=self._run, args=(gen,), daemon=True); self.t.start()

    def _run(self, gen):
        try:
            for item in gen:
                self.q.put(item)
        except Exception as e:                      # surface worker errors in the main thread
            self.err = e
        self.q.put(None)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                if self.err is not None:
                    raise self.err
                return
            yield item


# ======================================================================= decoding and metrics
def greedy_ids(lp_row):
    s = lp_row.argmax(-1)
    keep = np.ones_like(s, bool); keep[1:] = s[1:] != s[:-1]; keep &= s != 0
    return s[keep]


def cer_parts(ref, hyp):
    import editdistance
    u = lambda s: s.replace("<FRAC>", "")
    return editdistance.eval(u(ref), u(hyp)), max(1, len(u(ref)))


def metrics(refs, hyps):
    import editdistance, sacrebleu
    u = lambda s: s.replace("<FRAC>", "")
    ns = lambda s: u(s).replace(" ", "")
    e = np.array([editdistance.eval(u(r), u(h)) for r, h in zip(refs, hyps)], float)
    L = np.array([max(1, len(u(r))) for r in refs], float)
    e2 = sum(editdistance.eval(ns(r), ns(h)) for r, h in zip(refs, hyps)); n2 = sum(len(ns(r)) for r in refs)
    we = sum(editdistance.eval(r.split(), h.split()) for r, h in zip(refs, hyps)); wl = sum(max(1, len(r.split())) for r in refs)
    per = e / L
    return dict(CER=100 * e.sum() / L.sum(), CER_nospace=100 * e2 / max(1, n2), WER=100 * we / max(1, wl),
                BLEU=sacrebleu.corpus_bleu(hyps, [refs]).score,
                chrF_pp=sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score,
                median_patch_CER=100 * float(np.median(per)), patches_below_50=int((per < 0.5).sum()),
                best_patch_CER=100 * float(per.min()), len_ratio=float(np.mean([len(h) for h in hyps]) / max(1e-9, np.mean([len(r) for r in refs]))),
                n=len(refs))


def edit_anatomy(refs, hyps, focus=(" ", "ा", "्", "र", "ं", "ि", "ी", "ु", "ू", "े", "ो")):
    """Deletions / insertions / substitutions, the most-missed characters and the top confusions."""
    from rapidfuzz.distance import Levenshtein
    D = I = S = 0; miss = collections.Counter(); ins = collections.Counter(); sub = collections.Counter()
    ref_cnt = collections.Counter()
    for r, h in zip(refs, hyps):
        r = r.replace("<FRAC>", "\x01"); h = h.replace("<FRAC>", "\x01")
        ref_cnt.update(r)
        for op in Levenshtein.editops(r, h):
            if op.tag == "delete": D += 1; miss[r[op.src_pos]] += 1
            elif op.tag == "insert": I += 1; ins[h[op.dest_pos]] += 1
            else: S += 1; sub[(r[op.src_pos], h[op.dest_pos])] += 1
    tot = max(1, D + I + S)
    recall = {c: round(100 * (1 - (miss[c] + sum(v for (a, b), v in sub.items() if a == c)) / max(1, ref_cnt[c])), 1) for c in focus}
    return dict(errors=D + I + S, deletions_pct=round(100 * D / tot, 1), insertions_pct=round(100 * I / tot, 1),
                substitutions_pct=round(100 * S / tot, 1),
                most_missed=[(c, n) for c, n in miss.most_common(12)],
                most_inserted=[(c, n) for c, n in ins.most_common(8)],
                top_confusions=[(a + "->" + b, n) for (a, b), n in sub.most_common(15)],
                focus_char_recall=recall)


def paired_bootstrap(e_a, e_b, L, n=10000, seed=0):
    """95% CI of CER(a) - CER(b) over patches resampled together (same patches for both systems)."""
    e_a, e_b, L = (np.asarray(v, float) for v in (e_a, e_b, L))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(L), (n, len(L)))
    d = 100 * (e_a[idx].sum(1) - e_b[idx].sum(1)) / L[idx].sum(1)
    return float(100 * (e_a.sum() - e_b.sum()) / L.sum()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def spearman(a, b):
    import warnings
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(a[ok], b[ok]).correlation)


# ======================================================================= CTC forced alignment (X3)
def viterbi_ctc(logp, labels):
    """Best CTC path of `labels` through log-probs (T, C). Returns, for every frame, the index of the label
    token it is assigned to (blanks belong to the token before them) and the path score."""
    T = logp.shape[0]; L = len(labels)
    S = 2 * L + 1
    ext = np.zeros(S, dtype=int); ext[1::2] = labels
    NEG = -1e30
    dp = np.full((T, S), NEG); bp = np.zeros((T, S), dtype=np.int8)
    dp[0, 0] = logp[0, 0]
    if S > 1: dp[0, 1] = logp[0, ext[1]]
    allow2 = np.zeros(S, bool)
    allow2[3::2] = ext[3::2] != ext[1:-2:2]
    for t in range(1, T):
        prev = dp[t - 1]
        c0 = prev
        c1 = np.concatenate([[NEG], prev[:-1]])
        c2 = np.where(allow2, np.concatenate([[NEG, NEG], prev[:-2]]), NEG)
        stack = np.stack([c0, c1, c2])
        k = stack.argmax(0)
        dp[t] = stack[k, np.arange(S)] + logp[t, ext]
        bp[t] = k
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = np.empty(T, dtype=int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])          # plain int: under NumPy 2 (NEP 50) `int - int8` stays int8 and overflows past 127
    char_of_frame = np.where(path % 2 == 1, path // 2, -1)
    last = -1
    for t in range(T):
        if char_of_frame[t] >= 0: last = char_of_frame[t]
        else: char_of_frame[t] = last
    return char_of_frame, float(dp[T - 1, path[-1]])


def realign_patch(line_logps, label, vocab):
    """X3 for one patch: force-align the FULL label across its lines (laid end to end) and cut it where the
    alignment crosses into the next line, snapped to the nearest space. Returns the new line labels or None."""
    toks = AKSHARA.findall(label) if vocab.kind == "akshara" else TOK.findall(label)
    ids = []
    for a in toks:                                   # token ids of the label, keeping a map back to the strings
        ids.append(vocab.encode(a))
    flat = [i for x in ids for i in x]
    owner_tok = [k for k, x in enumerate(ids) for _ in x]
    logp = np.concatenate(line_logps); owner = np.concatenate([[i] * lp.shape[0] for i, lp in enumerate(line_logps)])
    if not flat or len(flat) > logp.shape[0]:
        return None
    cof, _ = viterbi_ctc(logp, np.array(flat))
    starts = [0]
    for i in range(1, len(line_logps)):
        ci = cof[owner == i]; ci = ci[ci >= 0]
        starts.append(owner_tok[int(ci.min())] if len(ci) else starts[-1])
    spaces = [j for j, t in enumerate(toks) if t == " "]
    cuts = []
    for st in starts[1:]:
        cand = [s for s in spaces if (not cuts or s > cuts[-1])]
        cuts.append(min(cand, key=lambda s: abs(s + 1 - st)) if cand else (cuts[-1] if cuts else 0))
    pieces, b = [], 0
    for c in cuts:
        pieces.append("".join(toks[b:c]).strip()); b = c + 1
    pieces.append("".join(toks[b:]).strip())
    return pieces


# ======================================================================= character LM and beam search (X6)
class CharLM:
    """Character n-gram LM with interpolated absolute discounting, built from TRAINING labels only."""

    def __init__(self, texts, n, d=0.75):
        self.n, self.d = n, d
        self.c = [collections.Counter() for _ in range(n + 1)]
        self.ctx = [collections.Counter() for _ in range(n + 1)]
        self.types = [collections.defaultdict(set) for _ in range(n + 1)]
        V = set()
        for t in texts:
            s = ["^"] * (n - 1) + TOK.findall(t) + ["$"]
            V.update(s)
            for k in range(1, n + 1):
                for i in range(k - 1, len(s)):
                    h = tuple(s[i - k + 1:i]); w = s[i]
                    self.c[k][h + (w,)] += 1; self.ctx[k][h] += 1; self.types[k][h].add(w)
        self.V = len(V)
        self.cache = {}

    def logp(self, hist, w):
        key = (tuple(hist[-(self.n - 1):]) if self.n > 1 else (), w)
        v = self.cache.get(key)
        if v is not None:
            return v
        hist = list(key[0])
        p = 1.0 / self.V
        for k in range(1, self.n + 1):
            h = tuple(hist[len(hist) - (k - 1):]) if k > 1 else ()
            if len(h) < k - 1: continue
            cn = self.ctx[k][h]
            if cn == 0: continue
            lam = self.d * len(self.types[k][h]) / cn
            p = max(self.c[k][h + (w,)] - self.d, 0) / cn + lam * p
        v = math.log(p)
        self.cache[key] = v
        return v


def topk_compress(lp, k=8):
    """Keep the k most likely tokens per frame (all the beam search ever looks at)."""
    idx = np.argsort(lp, axis=1)[:, -k:][:, ::-1]
    return idx.astype(np.int32), np.take_along_axis(lp, idx, 1).astype(np.float32)


def beam_decode(top_idx, top_lp, itos, lm, alpha, beta, beam=10):
    """CTC prefix beam search with shallow LM fusion over TOKEN STRINGS (a token may be one character or a
    whole akshara; its LM score is the sum over its characters). Identical to X6 for the char alphabet."""
    order = lm.n
    lse = lambda a, b: max(a, b) + math.log1p(math.exp(-abs(a - b))) if min(a, b) > -1e29 else max(a, b)
    toks_of = {}
    tail = {(): ("^",) * (order - 1)}
    beams = {(): (0.0, -1e30)}
    lmc = {}

    def lmscore(prefix, c):
        key = (tail[prefix], c)
        v = lmc.get(key)
        if v is None:
            ts = toks_of.get(c)
            if ts is None:
                ts = toks_of[c] = TOK.findall(itos[c])
            h = list(tail[prefix]); v = 0.0
            for t in ts:
                v += lm.logp(h, t); h.append(t)
            lmc[key] = v
        return v

    for t in range(top_idx.shape[0]):
        nb = collections.defaultdict(lambda: [-1e30, -1e30])
        for prefix, (pb, pnb) in beams.items():
            tot = lse(pb, pnb)
            for c, p in zip(top_idx[t], top_lp[t]):
                c = int(c); p = float(p)
                if c == 0:
                    e = nb[prefix]; e[0] = lse(e[0], tot + p)
                    continue
                last = prefix[-1] if prefix else None
                ext = prefix + (c,)
                if ext not in tail:
                    ts = toks_of.get(c)
                    if ts is None:
                        ts = toks_of[c] = TOK.findall(itos[c])
                    tail[ext] = tuple((list(tail[prefix]) + ts)[-(order - 1):]) if order > 1 else ()
                bonus = alpha * lmscore(prefix, c) + beta
                if c == last:
                    e = nb[ext]; e[1] = lse(e[1], pb + p + bonus)
                    e2 = nb[prefix]; e2[1] = lse(e2[1], pnb + p)
                else:
                    e = nb[ext]; e[1] = lse(e[1], tot + p + bonus)
        beams = dict(sorted(((k, (v[0], v[1])) for k, v in nb.items()), key=lambda kv: -lse(*kv[1]))[:beam])
    best = max(beams.items(), key=lambda kv: lse(*kv[1]))[0]
    return "".join(itos[i] for i in best if i > 0)


def restore_spaces(s, lm, gamma, beam=16):
    """N5: put word spaces back into a space-free reading with the character LM (spaces never go before a
    combining mark). gamma = bonus per inserted space, tuned on validation."""
    toks = TOK.findall(s)
    if not toks:
        return s
    order = lm.n
    beams = [(0.0, ("^",) * (order - 1), [])]
    for i, t in enumerate(toks):
        nb = []
        for sc, tl, out in beams:
            opts = [False]
            if i > 0 and out and out[-1] != " " and not COMBINING.match(t):
                opts.append(True)
            for sp in opts:
                h = list(tl); sc2 = sc; o2 = list(out)
                if sp:
                    sc2 += lm.logp(h, " ") + gamma; h.append(" "); o2.append(" ")
                sc2 += lm.logp(h, t); h.append(t); o2.append(t)
                nb.append((sc2, tuple(h[-(order - 1):]), o2))
        nb.sort(key=lambda x: -x[0]); beams = nb[:beam]
    return "".join(beams[0][2])


# ======================================================================= small helpers
class Tee:
    """print() to the notebook AND to a log file, flushed line by line."""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n"); self.f.flush()


def fmt_s(sec):
    sec = int(max(0, sec)); return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def gpu_status():
    import subprocess
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        return " | ".join(f"GPU{r.split(',')[0].strip()} {r.split(',')[1].strip()}% {r.split(',')[2].strip()}/{r.split(',')[3].strip()}MB {r.split(',')[4].strip()}C"
                          for r in q.splitlines() if r.strip())
    except Exception:
        return "nvidia-smi unavailable"
