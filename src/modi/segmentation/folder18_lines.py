"""Referee Step 6 (revised in the lead review): cut the Folder-18 handwritten pages into text lines.

Uses the Stage-1 line finder itself (notebooks/lr05/lr05_lib.py: deskew, line pitch from the row-ink profile,
cuts at its minima, crop_line), so the line Reader sees the book pages exactly the way it saw MoDeTrans.
The 600-dpi scans are halved first: there the line pitch (112-146 px) is close to MoDeTrans's (median 153).
Ink touching the page edge (dark scan borders, the book's edge) is removed first; it is never handwriting.
Known limits (see the overlays, and the hand-count comparison in the report): round seals and printed page
numbers are not removed (on some pages handwriting loops have a seal's shape, and deleting handwriting is
worse), so they can form a band of their own with no text in it; and a line with very little ink (a lone
"श्री" at the top) is dropped by the finder's rule that a band needs 1 % of the page's ink.

Run from the repository root:  PYTHONPATH=src python -m modi.segmentation.folder18_lines
  outputs/figures/folder18_lines/pageNN.jpg       every page with its detected lines in colour
  outputs/reports/folder18_lines.md, .csv         lines per page; agreement with the hand counts
  <data>/processed/folder18_lines/                line images at 40 and 64 px + manifest.csv (strict and partial
                                                  letters only; pages that are MoDeTrans sources are skipped)
Hand counts: assets/folder18_gt/line_counts_by_hand.csv (column lines_by_hand), counted before looking at these.
"""
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from modi.utils.paths import CODE_ROOT, P

sys.path.insert(0, str(CODE_ROOT / 'notebooks/lr05'))
import lr05_lib as L  # noqa: E402

SCALE = 0.5
HEIGHTS = (40, 64)
COLORS = [(214, 120, 42), (52, 104, 235), (122, 175, 27)]  # BGR: blue, orange, aqua bands, alternating
HAND = CODE_ROOT / 'assets/folder18_gt/line_counts_by_hand.csv'


def remove_edge_ink(g):
    """Whiten every ink component that touches the page edge (scan borders and the book's edge)."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats((g < 128).astype(np.uint8), connectivity=8)
    x, y, w, h = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3]
    edge = (x == 0) | (y == 0) | (x + w >= g.shape[1]) | (y + h >= g.shape[0])
    edge[0] = False  # label 0 is the background
    g = g.copy()
    g[edge[labels]] = 255
    return g


def page_lines(path):
    """Book page -> (deskewed half-size page, [(top, bottom)] line bands, {height: line image} per line, info)."""
    g = np.array(Image.open(path).convert('L'))
    g = cv2.resize(g, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    g = remove_edge_ink(g)
    g2, bands, pitch, conf, angle = L.segment(g)
    crops = [L.crop_line(g2, a, b, HEIGHTS)[0] for a, b in bands]
    keep = [i for i, c in enumerate(crops) if c is not None]
    return g2, [bands[i] for i in keep], [crops[i] for i in keep], dict(pitch=pitch, pitch_conf=conf, angle=angle)


def overlay(g2, bands, path):
    img = cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)
    tint = img.copy()
    for i, (a, b) in enumerate(bands):
        cv2.rectangle(tint, (0, a), (img.shape[1] - 1, b - 1), COLORS[i % len(COLORS)], -1)
        cv2.putText(img, str(i + 1), (8, a + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    img = np.where(g2[..., None] < 128, img, cv2.addWeighted(img, 0.65, tint, 0.35, 0))  # ink stays black
    cv2.imwrite(str(path), cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])


def main():
    labels = pd.read_csv(CODE_ROOT / 'assets/metadata/folder18_labels.csv', keep_default_na=False)
    pages = labels[labels.page_type == 'modi_facsimile']
    figs = P.figures / 'folder18_lines'
    figs.mkdir(parents=True, exist_ok=True)
    out = P.processed / 'folder18_lines'
    out.mkdir(parents=True, exist_ok=True)
    rows, manifest = [], []
    for r in pages.itertuples():
        g2, bands, crops, info = page_lines(P.hhdoc_root / '18' / r.filename)
        overlay(g2, bands, figs / f'page{r.page}.jpg')
        rows.append(dict(page=r.page, letter_no=int(float(r.doc_no)), tier=r.ood_tier, lines_detected=len(bands), **info))
        if r.ood_tier in ('strict', 'partial'):
            for i, c in enumerate(crops):
                for h in HEIGHTS:
                    cv2.imwrite(str(out / f'page{r.page}_line{i + 1:02d}_h{h}.png'), c[h])
                manifest.append(dict(page=r.page, line_no=i + 1, letter_no=int(float(r.doc_no)), tier=r.ood_tier,
                                     **{f'image_h{h}': f'page{r.page}_line{i + 1:02d}_h{h}.png' for h in HEIGHTS}))
        print(f'page {r.page}: {len(bands)} lines', flush=True)
    pd.DataFrame(manifest).to_csv(out / 'manifest.csv', index=False)
    df = pd.DataFrame(rows)
    hand = pd.read_csv(HAND, keep_default_na=False) if HAND.exists() else pd.DataFrame(columns=['page', 'lines_by_hand'])
    hand = hand[hand.lines_by_hand.astype(str).str.strip() != '']
    df = df.merge(hand[['page', 'lines_by_hand']].astype({'lines_by_hand': int}), on='page', how='left')
    df.to_csv(P.reports / 'folder18_lines.csv', index=False)

    lines = ['# Folder-18 line extraction', '',
             f'{len(df)} handwritten pages, cut with the Stage-1 line finder (`notebooks/lr05/lr05_lib.segment`) at half '
             f'resolution. Overlays: `outputs/figures/folder18_lines/pageNN.jpg`. Line images for the {len(manifest)} lines '
             'of the strict and partial letters: `<data>/processed/folder18_lines/`.', '']
    counted = df.dropna(subset=['lines_by_hand'])
    if len(counted):
        diff = counted.lines_detected - counted.lines_by_hand
        lines += [f'**Agreement with the hand counts** ({len(counted)} of {len(df)} pages counted): exactly equal on '
                  f'{(diff == 0).sum()} ({(diff == 0).mean():.0%}), within ±1 on {(diff.abs() <= 1).sum()} '
                  f'({(diff.abs() <= 1).mean():.0%}).', '']
        worst = counted.assign(diff=diff).sort_values('diff', key=abs, ascending=False).head(5)
        lines += ['Hardest pages (largest difference):', '', '| page | letter | hand | detected | pitch confidence |',
                  '|---|---|---|---|---|']
        lines += [f'| {w.page} | {w.letter_no} | {int(w.lines_by_hand)} | {w.lines_detected} | {w.pitch_conf:.2f} |'
                  for w in worst.itertuples()]
    else:
        lines += ['**Hand counts missing**: fill `assets/folder18_gt/line_counts_by_hand.csv` and run again.', '']
    lines += ['', '| page | letter | tier | lines detected | lines by hand | pitch (px) | pitch confidence | angle |',
              '|---|---|---|---|---|---|---|---|']
    lines += [f'| {r.page} | {r.letter_no} | {r.tier} | {r.lines_detected} | '
              f'{"" if pd.isna(r.lines_by_hand) else int(r.lines_by_hand)} | {r.pitch:.0f} | {r.pitch_conf:.2f} | {r.angle:+.1f} |'
              for r in df.itertuples()]
    (P.reports / 'folder18_lines.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:8]))


if __name__ == '__main__':
    main()
