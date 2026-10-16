"""Save a fixed set of synthetic patches as PNG files + labels.csv, zipped, for sharing.

Patch i is make_patch(BASE_SEED + i), the same pool the README's SyntheticPatches uses, so the files and
on-the-fly generation give the same data (up to a few pixels if another machine has other library versions).

Run: PYTHONPATH=src python -m modi.synthetic.export 4290 ../FYP-modi-script-data/synthetic/armB_synthetic_4290
  -> <out>/syn_000000.png ..., <out>/labels.csv (image_id, seed, text), and <out>.zip
"""
import csv
import multiprocessing
import sys
import zipfile
from pathlib import Path

from PIL import Image

from modi.synthetic import make_patch

BASE_SEED = 1_000_000


def _save(job):
    i, out = job
    img, text = make_patch(BASE_SEED + i)
    Image.fromarray(img).save(out / f'syn_{i:06d}.png', optimize=True)
    return f'syn_{i:06d}.png', BASE_SEED + i, text


def main(n, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with multiprocessing.Pool() as pool:
        rows = pool.map(_save, [(i, out) for i in range(n)], chunksize=20)
    with open(out / 'labels.csv', 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows([('image_id', 'seed', 'text'), *rows])
    with zipfile.ZipFile(out.with_suffix('.zip'), 'w', zipfile.ZIP_STORED) as z:  # PNGs are already compressed
        for path in sorted(out.iterdir()):
            z.write(path, f'{out.name}/{path.name}')
    print(f'{n} patches -> {out.with_suffix(".zip")}')


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2])
