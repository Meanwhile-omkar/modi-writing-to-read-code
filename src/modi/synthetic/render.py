"""Factory Step 3: draw a text as a fake handwritten Modi patch (render) or a single line (render_line).

Copies the layout of real MoDeTrans patches (all 1268 x 463, 1-4+ lines of ~25-35 characters):
words in Noto Sans Modi with little or no gap between them, a long horizontal stroke along the top of
each line that starts left of the text, each line slightly tilted and indented. Then the whole patch is
slanted, warped, its strokes thickened or thinned, roughened and thresholded back to black and white,
with a few specks. Finally it is resized the way the Reader resizes real patches.

Dotted circles: when a vowel sign or virama follows something it cannot attach to (e.g. 'अै' typed for 'ऐ',
'रू्प' in the transcriptions), text shaping inserts a dotted circle (U+25CC) that is not in the label. It does
so only if the font has that glyph, so the font is loaded without it: the mark is drawn on its own instead.
"""
import functools
import io

import cv2
import numpy as np
from fontTools.ttLib import TTFont
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
# A line must suit the line Reader (notebooks/lr05): at most 1024 px wide at height 40, and wide enough for CTC,
# width / STRIDE >= 1.2 * characters + 2 (lr05 line_ok). Both are checked at height 40.
MAX_WIDTH_40 = 1024
STRIDE = 2


@functools.lru_cache(maxsize=None)
def _font_bytes(dotted_circle):
    font = TTFont(FONT)
    if not dotted_circle:
        for table in font['cmap'].tables:
            table.cmap.pop(0x25CC, None)
    buf = io.BytesIO()
    font.save(buf)
    return buf.getvalue()


@functools.lru_cache(maxsize=None)
def _font(size, dotted_circle=False):
    return ImageFont.truetype(io.BytesIO(_font_bytes(dotted_circle)), size, layout_engine=ImageFont.Layout.RAQM)


def has_dotted_circle(text):
    """True if the plain font would draw a dotted circle in this Devanagari text (the renderers never do)."""
    modi = dev_to_modi(text)
    def draw(font):
        im = Image.new('L', (60 * (len(modi) + 4), 200), 255)
        ImageDraw.Draw(im).text((20, 150), modi, font=font, fill=0, anchor='ls')
        return im.tobytes()
    return draw(_font(60, dotted_circle=True)) != draw(_font(60))


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
        raise RuntimeError('Pillow cannot join Modi letters (raqm is off): see Setup in src/modi/synthetic/README.md')
    nr = np.random.default_rng(rng.getrandbits(64))
    # LINE_CHARS characters per line, at most 4-6 lines:
    # long texts get longer, denser lines, as real ones do
    n = min(max(1, round(len(text) / rng.uniform(*LINE_CHARS))), rng.randint(4, 6))
    lines = wrap(text, n)
    pitch = H / (len(lines) + rng.uniform(0.2, 0.8))  # vertical space per line
    font = _font(round(rng.uniform(70, 110)))
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

    a = _roughen(page.astype(np.float32), rng, nr)
    _specks(a, rng, nr)

    a = cv2.resize(a, (round(W * OUT_H / H), OUT_H), interpolation=cv2.INTER_AREA)
    out = np.full((OUT_H, OUT_W), 255, np.uint8)
    out[:, :a.shape[1]] = a
    return out


def _roughen(a, rng, nr):
    """Slant, elastic warp, thicker or thinner strokes, blur + noise, threshold: float image -> uint8 0 / 255."""
    h, w = a.shape
    k = rng.uniform(-0.2, 0.2)  # slant
    a = cv2.warpAffine(a, np.float32([[1, k, -k * h / 2], [0, 1, 0]]), (w, h), borderValue=255)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shift, smooth = rng.uniform(2, 6), rng.uniform(20, 40)  # elastic warp: pixels moved, field smoothness
    # smooth random fields, made at 1/8 size and enlarged (same result as a wide blur, much faster)
    dx, dy = (cv2.resize(cv2.GaussianBlur(nr.standard_normal((max(2, h // 8), max(2, w // 8))).astype(np.float32),
                                          (0, 0), smooth / 8), (w, h), interpolation=cv2.INTER_CUBIC) for _ in range(2))
    a = cv2.remap(a, xx + dx * shift / dx.std(), yy + dy * shift / dy.std(), cv2.INTER_LINEAR, borderValue=255)
    t = rng.choice([-1, 0, 1, 1, 2])  # stroke width: negative thins, positive thickens (ink is dark)
    if t:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(t) + 1, 2 * abs(t) + 1))
        a = cv2.erode(a, kernel) if t > 0 else cv2.dilate(a, kernel)
    a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.6, 1.5)) + nr.normal(0, rng.uniform(5, 30), a.shape)
    return np.where(a < rng.uniform(100, 160), 0, 255).astype(np.uint8)  # back to black and white, rough edges


def _specks(a, rng, nr):
    h, w = a.shape
    for _ in range(nr.poisson(rng.uniform(0, 30) * h * w / (H * W))):  # as many per area as on a patch
        cv2.circle(a, (int(nr.integers(w)), int(nr.integers(h))), int(nr.integers(1, 4)), 0, -1)


def render_line(text, rng, height):
    """One text line -> numpy uint8 (height, width), dark ink 0 on white 255, or None when it does not suit the
    line Reader (wider than MAX_WIDTH_40, or too narrow for its characters at height 40)."""
    if not features.check('raqm'):
        raise RuntimeError('Pillow cannot join Modi letters (raqm is off): see Setup in src/modi/synthetic/README.md')
    nr = np.random.default_rng(rng.getrandbits(64))
    im = draw_line(text, _font(round(rng.uniform(70, 110))), rng)
    sx = rng.uniform(0.7, 1.1)  # about the size of a line in a real 1268 x 463 patch
    im = im.resize((round(im.width * sx), round(im.height * sx * rng.uniform(TALL, TALL + 0.6))), Image.LANCZOS)
    a = np.asarray(im.rotate(rng.uniform(-2, 2), expand=True, fillcolor=255, resample=Image.BICUBIC), np.float32)
    m = a.shape[0] // 2  # room on every side for the slant and the warp
    a = _roughen(cv2.copyMakeBorder(a, m, m, m, m, cv2.BORDER_CONSTANT, value=255), rng, nr)
    # The text's extent from its strokes; single pixels left by the noise are not text and are ignored.
    _, _, stats, _ = cv2.connectedComponentsWithStats((a < 128).astype(np.uint8), connectivity=8)
    strokes = stats[1:][stats[1:, cv2.CC_STAT_AREA] >= 20]
    if not len(strokes):
        return None
    x0, y0 = strokes[:, 0].min(), strokes[:, 1].min()
    x1, y1 = (strokes[:, 0] + strokes[:, 2]).max(), (strokes[:, 1] + strokes[:, 3]).max()
    if x0 == 0 or y0 == 0 or x1 == a.shape[1] or y1 == a.shape[0]:  # fit-to-page validation
        raise RuntimeError(f'line ink reaches the canvas edge, letters would be cut off: {text!r}')
    a = a[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2].copy()
    _specks(a, rng, nr)  # after the crop, so specks never move it
    width40 = a.shape[1] * 40 / a.shape[0]
    if width40 > MAX_WIDTH_40 or width40 / STRIDE < 1.2 * len(text.replace('<FRAC>', '_')) + 2:
        return None
    return cv2.resize(a, (max(8, round(a.shape[1] * height / a.shape[0])), height), interpolation=cv2.INTER_AREA)
