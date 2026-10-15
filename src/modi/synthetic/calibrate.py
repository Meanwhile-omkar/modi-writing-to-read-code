"""Factory Step 4: how close are fake patches to real ones? Four statistics, real vs fake, with KS distances.

Real = the 1,430 training patches, resized like the Reader resizes them; fake = make_patch(0 .. 1429).
Run: PYTHONPATH=src python -m modi.synthetic.calibrate
  -> outputs/reports/synthetic_calibration.md, outputs/figures/synthetic_calibration.png
"""
import io
import json

import cv2
import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from scipy.signal import find_peaks
from scipy.stats import ks_2samp
from skimage.morphology import skeletonize

from modi.synthetic import make_patch
from modi.synthetic.render import OUT_H, H, W
from modi.utils.paths import CODE_ROOT, P

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

READER_W = round(W * OUT_H / H)  # 1402: real and fake are compared at the Reader's input size, without padding


def stats(img):
    """ink share, stroke width (ink area / skeleton length, in pixels), number of text lines (row-profile peaks)."""
    ink = img[:, :READER_W] < 128
    skeleton = skeletonize(ink).sum()
    profile = cv2.GaussianBlur(ink.sum(1).astype(np.float32).reshape(-1, 1), (1, 0), 8).ravel()
    lines, _ = find_peaks(profile, distance=40, prominence=0.25 * profile.max())
    return ink.mean(), ink.sum() / max(skeleton, 1), len(lines)


def main():
    meta = CODE_ROOT / 'assets/metadata'
    split = json.load(open(meta / 'splits.json', encoding='utf-8'))['modetrans']['assignment']
    text = pd.read_csv(meta / 'modetrans_text.csv', keep_default_na=False).set_index('image_id')['text_train']
    table = pq.read_table(P.modetrans_parquet, columns=['filename', 'image'])
    rows = []
    for i, name in enumerate(table.column('filename').to_pylist()):
        if split[name] != 'train':
            continue
        img = np.asarray(Image.open(io.BytesIO(table.column('image')[i].as_py()['bytes'])).convert('L'))
        img = cv2.resize(img, (READER_W, OUT_H), interpolation=cv2.INTER_AREA)
        rows.append(('real', *stats(img), len(text[name])))
    for seed in range(len(rows)):
        img, t = make_patch(seed)
        rows.append(('fake', *stats(img), len(t)))
    df = pd.DataFrame(rows, columns=['kind', 'ink share', 'stroke width (px)', 'text lines', 'text length'])

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    report = ['# Synthetic patch calibration (Factory Step 4)', '',
              f'{(df.kind == "real").sum()} real training patches vs {(df.kind == "fake").sum()} fake patches '
              '(`make_patch(0..)`), both at the Reader input size (512 x 1402).', '',
              '| statistic | real median | fake median | KS distance (0 = same, 1 = disjoint) |', '|---|---|---|---|']
    for ax, col in zip(axes, df.columns[1:]):
        real, fake = df[df.kind == 'real'][col], df[df.kind == 'fake'][col]
        bins = np.histogram_bin_edges(df[col], bins=30 if col != 'text lines' else np.arange(-0.5, df[col].max() + 1.5))
        ax.hist(real, bins, alpha=0.6, label='real', density=True)
        ax.hist(fake, bins, alpha=0.6, label='fake', density=True)
        ks = ks_2samp(real, fake).statistic
        ax.set_title(f'{col}  (KS {ks:.2f})')
        report.append(f'| {col} | {real.median():.3g} | {fake.median():.3g} | {ks:.2f} |')
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(P.figures / 'synthetic_calibration.png', dpi=110)
    (P.reports / 'synthetic_calibration.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report))


if __name__ == '__main__':
    main()
