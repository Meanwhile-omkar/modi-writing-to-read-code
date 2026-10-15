"""Factory Step 3: draw a text as a fake handwritten Modi patch.

Copies the layout of real MoDeTrans patches (all 1268 x 463, 1-4+ lines of ~25-35 characters):
words in Noto Sans Modi with little or no gap between them, a long horizontal stroke along the top of
each line that starts left of the text, each line slightly tilted and indented. Then the whole patch is
slanted, warped, its strokes thickened or thinned, roughened and thresholded back to black and white,
with a few specks. Finally it is resized the way the Reader resizes real patches.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

from modi.synthetic.mapping import dev_to_modi
from modi.utils.paths import CODE_ROOT

FONT = CODE_ROOT / 'assets/fonts/NotoSansModi-Regular.ttf'
W, H = 1268, 463          # size of every real MoDeTrans patch
OUT_H, OUT_W = 512, 1408  # Reader input: height 512 with aspect kept (1402 wide), padded to a multiple of 8
CONSONANTS = dev_to_modi('कगपमनलवसतद')  # letters that hang from the top stroke
# Tuned in Step 4 against real training patches (see calibrate.py):
TALL = 1.5             # letters are stretched vertically by TALL .. TALL + 0.6
LINE_CHARS = (30, 50)  # characters per line; real patches mostly have 2-3 lines


def wrap(text, n):
    """Split text at spaces into at most n lines of about equal length (each word goes to the line its
    middle falls in). ' '.join(wrap(text, n)) == text."""
    lines = [[] for _ in range(n)]
    pos = 0
    for word in text.split(' '):
        lines[min(n - 1, int(n * (pos + len(word) / 2) / len(text)))].append(word)
        pos += len(word) + 1
    return [' '.join(line) for line in lines if line]


def draw_line(line, font, rng):
    """One text line: words with small gaps, hanging from a long top stroke. White background, cropped."""
    size = font.size
    head = int(np.median([font.getbbox(c, anchor='ls')[1] for c in CONSONANTS]))  # top stroke, above baseline
    canvas = Image.new('L', (size * (len(line) + 8), size * 4), 255)
    draw = ImageDraw.Draw(canvas)
    base = size * 2
    x = lead = rng.uniform(0.3, 2.5) * size  # the top stroke starts before the first letter
    gap = rng.choice([0, rng.uniform(0, 0.3)]) * size  # half the writers leave no space between words
    for word in line.split(' '):
        modi = dev_to_modi(word)
        draw.text((x, base + rng.uniform(-0.02, 0.02) * size), modi, font=font, fill=0, anchor='ls')
        x += font.getlength(modi) + gap
    width = max(2, round(size * rng.uniform(0.05, 0.1)))
    y = base + head + width / 2  # the stroke's upper edge on the letters' tops, so they hang from it
    draw.line([(lead - rng.uniform(0.3, 2.5) * size, y), (x + rng.uniform(0, 0.4) * size, y)], fill=0, width=width)
    return canvas.crop(Image.eval(canvas, lambda v: 255 - v).getbbox())


def render(text, rng):
    """Devanagari label text -> numpy uint8 (512, 1408), dark ink 0 on white 255. All randomness from rng."""
    if not features.check('raqm'):
        raise RuntimeError('Pillow cannot join Modi letters (raqm is off); on macOS run: source .venv/bin/activate')
    nr = np.random.default_rng(rng.getrandbits(64))
    # LINE_CHARS characters per line, at most 4-6 lines:
    # long texts get longer, denser lines, as real ones do
    n = min(max(1, round(len(text) / rng.uniform(*LINE_CHARS))), rng.randint(4, 6))
    lines = wrap(text, n)
    pitch = H / (len(lines) + rng.uniform(0.2, 0.8))  # vertical space per line
    font = ImageFont.truetype(str(FONT), round(rng.uniform(70, 110)), layout_engine=ImageFont.Layout.RAQM)
    images = [draw_line(line, font, rng) for line in lines]
    # One scale for the whole patch: the longest line spans 75-97 % of the width, as in real patches.
    # Handwritten Modi letters are taller and narrower than the font's, so the height is stretched more,
    # but not so much that lines become far taller than their share of the patch.
    sx = rng.uniform(0.75, 0.97) * W / max(im.width for im in images)
    sy = min(sx * rng.uniform(TALL, TALL + 0.6), 1.8 * pitch / max(im.height for im in images))
    indent = rng.uniform(0, 0.08) * W

    page = np.full((H, W), 255, np.uint8)
    top = rng.uniform(0.1, 0.6) * pitch
    for i, im in enumerate(images):
        im = im.resize((round(im.width * sx * rng.uniform(0.95, 1.05)), round(im.height * sy)), Image.LANCZOS)
        im = np.asarray(im.rotate(rng.uniform(-2, 2), expand=True, fillcolor=255, resample=Image.BICUBIC))
        x = round(indent + rng.uniform(-0.03, 0.05) * W)
        y = round(top + (i + 0.5) * pitch - im.shape[0] / 2)
        ys, xs = max(0, y), max(0, x)
        ye, xe = min(H, y + im.shape[0]), min(W, x + im.shape[1])
        if ye > ys and xe > xs:
            page[ys:ye, xs:xe] = np.minimum(page[ys:ye, xs:xe], im[ys - y:ye - y, xs - x:xe - x])

    a = page.astype(np.float32)
    k = rng.uniform(-0.2, 0.2)  # slant
    a = cv2.warpAffine(a, np.float32([[1, k, -k * H / 2], [0, 1, 0]]), (W, H), borderValue=255)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    shift, smooth = rng.uniform(2, 6), rng.uniform(20, 40)  # elastic warp: pixels moved, field smoothness
    # smooth random fields, made at 1/8 size and enlarged (same result as a wide blur, much faster)
    dx, dy = (cv2.resize(cv2.GaussianBlur(nr.standard_normal((H // 8, W // 8)).astype(np.float32), (0, 0), smooth / 8),
                         (W, H), interpolation=cv2.INTER_CUBIC) for _ in range(2))
    a = cv2.remap(a, xx + dx * shift / dx.std(), yy + dy * shift / dy.std(), cv2.INTER_LINEAR, borderValue=255)
    t = rng.choice([-1, 0, 1, 1, 2])  # stroke width: negative thins, positive thickens (ink is dark)
    if t:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(t) + 1, 2 * abs(t) + 1))
        a = cv2.erode(a, kernel) if t > 0 else cv2.dilate(a, kernel)
    a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.6, 1.5)) + nr.normal(0, rng.uniform(5, 30), a.shape)
    a = np.where(a < rng.uniform(100, 160), 0, 255).astype(np.uint8)  # back to black and white, rough edges
    for _ in range(nr.poisson(rng.uniform(0, 30))):  # specks
        cv2.circle(a, (int(nr.integers(W)), int(nr.integers(H))), int(nr.integers(1, 4)), 0, -1)

    a = cv2.resize(a, (round(W * OUT_H / H), OUT_H), interpolation=cv2.INTER_AREA)
    out = np.full((OUT_H, OUT_W), 255, np.uint8)
    out[:, :a.shape[1]] = a
    return out
