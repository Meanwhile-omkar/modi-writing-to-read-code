"""How much of the out-of-domain gap on the Folder-18 book is page layout? (task of 16 Oct, experiment 2)

Same pipeline as s11_folder18_ood.py (half-size pages, text block from the ink, the Stage-1 line finder, bands under
30 % of the block width skipped, the frozen Stage-1 Readers, the team metrics). One thing changes per variant:
  base          s11 as it is: deskew searched within ±6° (lr05_lib default)
  tilt12        page deskew searched within ±12°
  tilt12_cols   ±12°, and a page with a vertical gutter is cut into its two columns: the left column's lines are read
                top to bottom, then the right column's

The answer key follows the same order: the printed pages of a two-column letter are split at their gutter and each
half is OCR'd with Tesseract (`mar`) separately. Every variant is scored against the same references, per page
(handwritten page p+1 against printed page p) and per letter. References: the proofread files of
modi.evaluation.folder18_gt (proofread or Tesseract draft) and, for two-column pages, the half-page OCR.

  python s12_folder18_layout.py lines     cut lines for every variant (fast)
  python s12_folder18_layout.py read      read them with the frozen models (s11 `read`, per variant; slow)
  python s12_folder18_layout.py score     -> outputs/reports/folder18_layout.md (numbers only)
Everything else goes to assets/folder18_gt/ood_layout/ (git-ignored: page images and book text stay out of git).
"""
import contextlib
import functools
import json
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s11_folder18_ood as S  # noqa: E402  (also puts notebooks/lr05 and src on the path)
from s11_folder18_ood import L  # noqa: E402
from modi.evaluation.folder18_gt import OUT as GT, TESSDATA, body_lines, join, to_train  # noqa: E402
from modi.evaluation.metrics import cer  # noqa: E402
from modi.utils.paths import CODE_ROOT, P  # noqa: E402

OUT = GT / 'ood_layout'
VARIANTS = {'base': (6.0, False), 'tilt12': (12.0, False), 'tilt12_cols': (12.0, True)}
MIN_GAP, AGREE = 15, 0.55  # gutter: a gap >= 15 px (at half size) shared by >= 55 % of the inked strips


def find_gutter(block, lo=0.3, hi=0.7, strips=40):
    """x of a gutter between two text columns, or None. The block is cut into horizontal strips; each strip marks its
    empty gaps at least MIN_GAP wide (Modi has almost no word gaps, so wide gaps are rare inside a column). A gutter is
    a position in the middle 40 % that lies in such a gap in most strips. On the 14 strict pages: two-column pages
    0.59-0.71, single-column pages <= 0.50."""
    ink = block < 128
    H, W = ink.shape
    votes, used = np.zeros(W), 0
    for k in range(strips):
        s = ink[k * H // strips:(k + 1) * H // strips]
        if s.mean() < 0.01:
            continue
        cols = s.sum(0)
        empty = np.concatenate([[False], cols <= 0.05 * cols.max(), [False]])
        starts, ends = np.where(np.diff(empty.astype(int)) == 1)[0], np.where(np.diff(empty.astype(int)) == -1)[0]
        for a, b in zip(starts, ends):
            if b - a >= MIN_GAP:
                votes[a:b] += 1
        used += 1
    if used < 5:
        return None
    share = votes / used
    a, b = int(lo * W), int(hi * W)
    x = a + int(np.argmax(share[a:b]))
    return x if share[x] >= AGREE else None


@contextlib.contextmanager
def deskew_range(rng):
    """lr05_lib.segment calls lr05_lib.deskew(g) with its default ±6°; widen it for the duration."""
    original = L.deskew
    L.deskew = functools.partial(original, rng=rng)
    try:
        yield
    finally:
        L.deskew = original


def page_block(p):
    g = np.array(Image.open(P.hhdoc_root / f'18/page{p}.jpg').convert('L'))
    g = cv2.resize(g, None, fx=S.SCALE, fy=S.SCALE, interpolation=cv2.INTER_AREA)
    y0, y1, x0, x1 = S.text_box(g)
    return g[y0:y1, x0:x1]


def cut(block):
    """s11's line cutting of one block: segment, crop to 40 px, skip bands under 30 % of the block width."""
    g2, bands, pitch, conf, ang = L.segment(block)
    lines = []
    for a, b in bands:
        crop, w = L.crop_line(g2, a, b, (40,))
        if crop is not None and w >= S.MIN_WIDTH * g2.shape[1]:
            lines.append(crop[40])
    return lines, ang


def cut_lines():
    for name, (rng, columns) in VARIANTS.items():
        (OUT / name).mkdir(parents=True, exist_ok=True)
        res, stats = {}, {}
        with deskew_range(rng):
            for p in [p for ps in S.FACSIMILE.values() for p in ps]:
                block = page_block(p)
                x = find_gutter(block) if columns else None  # on the page as scanned: rotating first blurs the votes
                if x is None:
                    lines, ang = cut(block)
                else:  # each column is cut (and deskewed by the line finder) on its own
                    (left, ang), (right, _) = cut(block[:, :x]), cut(block[:, x:])
                    lines = left + right
                res[p] = lines
                stats[p] = dict(lines=len(lines), angle=float(ang), gutter=None if x is None else int(x))
        pickle.dump(res, open(OUT / name / 'lines.pkl', 'wb'))
        json.dump(stats, open(OUT / name / 'line_stats.json', 'w'), indent=1)
        print(name, {p: (s['lines'], round(s['angle'], 1), s['gutter']) for p, s in stats.items()}, flush=True)


def read():
    for name in VARIANTS:
        S.OUT = OUT / name  # s11's read_lines reads <OUT>/lines.pkl and writes <OUT>/predictions.json
        S.read_lines()
        print('read', name, flush=True)


def half_page_ocr(p):
    """Text of a two-column printed page, left half then right half, each OCR'd on its own (cached)."""
    cache = OUT / f'printed{p}_halves.txt'
    if not cache.exists():
        g = np.array(Image.open(GT / f'ocr/page{p}.png').convert('L'))
        y0, y1, x0, x1 = S.text_box(g)
        x = x0 + find_gutter(g[y0:y1, x0:x1])
        # A full-width heading (the English title and summary on a letter's first page) crosses the gutter; start both
        # halves below the last row in the upper half whose ink crosses it, so no half gets heading fragments.
        crossing = np.where((g[:y0 + (y1 - y0) // 2, x - MIN_GAP // 2:x + MIN_GAP // 2] < 128).any(1))[0]
        top = int(crossing[-1]) + 1 if len(crossing) else 0
        g = g[top:]
        lines = []
        for half in (g[:, :x], g[:, x:]):
            png = OUT / 'half.png'
            Image.fromarray(half).save(png)
            txt = subprocess.run(['tesseract', str(png), '-', '-l', 'mar', '--tessdata-dir', str(TESSDATA)],
                                 capture_output=True, text=True, check=True).stdout
            lines += body_lines(txt.splitlines())
        cache.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return cache.read_text(encoding='utf-8').splitlines()


def references():
    """{printed page: reference text}: the proofread file's section for that page, or the half-page OCR for
    printed pages with a gutter (letter 11)."""
    clean, refs = to_train(), {}
    for n, facsimile in S.FACSIMILE.items():
        sections, cur = {}, None
        for line in (GT / f'proofread/letter{n:02d}.txt').read_text(encoding='utf-8').splitlines():
            if line.startswith('## page'):
                cur = int(line.split()[2]); sections[cur] = []
            elif cur is not None:
                sections[cur].append(line)
        for f in facsimile:
            p = f - 1
            g = np.array(Image.open(GT / f'ocr/page{p}.png').convert('L'))
            y0, y1, x0, x1 = S.text_box(g)
            two_columns = find_gutter(g[y0:y1, x0:x1]) is not None and n == 11
            refs[p] = clean(join(half_page_ocr(p) if two_columns else sections.get(p, [])))
    return refs


def score():
    refs = references()
    preds = {name: json.load(open(OUT / name / 'predictions.json', encoding='utf-8'))['best'] for name in VARIANTS}
    stats = {name: json.load(open(OUT / name / 'line_stats.json')) for name in VARIANTS}
    hyp = lambda name, f: ' '.join(l for l in preds[name][str(f)] if l)
    lines = ['# Folder-18 layout: tilt and two columns', '',
             'Out-of-domain CER on the 14 handwritten pages of the strict letters, read by the Stage-1 ensemble '
             '(4 models + char 7-gram LM, as in `s11_folder18_ood.py`). One change at a time: `base` = s11 '
             '(deskew within ±6°), `tilt12` = deskew within ±12°, `cols` = also cut pages with a gutter into two '
             'columns (left column read first). Same references for every variant: handwritten page p+1 against '
             'printed page p; letter 11 against half-page OCR of its printed pages (left column, then right). '
             'References are proofread for letter 3 and Tesseract drafts otherwise, so absolute numbers are rough; '
             'the differences between columns are the result. Made by `research/root-cause/offline/s12_folder18_layout.py`.', '',
             '| letter | page | base CER | tilt12 CER | tilt12 + cols CER | angle base -> tilt12 | gutter |', '|---|---|---|---|---|---|---|']
    for n, facsimile in S.FACSIMILE.items():
        for f in facsimile:
            c = [cer([refs[f - 1]], [hyp(name, f)]) for name in VARIANTS]
            a0, a1 = stats['base'][str(f)]['angle'], stats['tilt12'][str(f)]['angle']
            g = stats['tilt12_cols'][str(f)]['gutter']
            lines.append(f'| {n} | {f} | {c[0]:.1f} | {c[1]:.1f} | {c[2]:.1f} | {a0:+.1f}° -> {a1:+.1f}° | {"yes" if g else ""} |')
    lines += ['', '| letter | base CER | tilt12 CER | tilt12 + cols CER |', '|---|---|---|---|']
    for n, facsimile in list(S.FACSIMILE.items()) + [('all', [f for fs in S.FACSIMILE.values() for f in fs])]:
        r = [refs[f - 1] for f in facsimile]
        c = [cer([' '.join(r)], [' '.join(hyp(name, f) for f in facsimile)]) for name in VARIANTS]
        lines.append(f'| {n} | {c[0]:.1f} | {c[1]:.1f} | {c[2]:.1f} |')
    report = CODE_ROOT / 'outputs/reports/folder18_layout.md'
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    {'lines': cut_lines, 'read': read, 'score': score}[sys.argv[1]]()
