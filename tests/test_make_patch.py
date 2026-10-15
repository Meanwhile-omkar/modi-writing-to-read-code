import hashlib
import json
import time
from pathlib import Path

import numpy as np

from modi.synthetic import make_patch
from modi.synthetic.mapping import FRAC, unmappable
from modi.synthetic.render import OUT_H, OUT_W, W, H

CHARSET = set(json.load(open(Path(__file__).resolve().parents[1] / 'assets/metadata/charset.json', encoding='utf-8'))['symbols'])


def digest(seed):
    img, text = make_patch(seed)
    return hashlib.sha256(img.tobytes() + text.encode()).hexdigest()


def test_same_seed_same_bytes_different_seed_different():
    assert digest(123) == digest(123)
    assert digest(123) != digest(124)


def test_agreed_format():
    for seed in range(20):
        img, text = make_patch(seed)
        assert img.shape == (OUT_H, OUT_W) and img.dtype == np.uint8
        assert (img[:, round(W * OUT_H / H):] == 255).all()  # padding is white, like the Reader pads real patches
        assert 0.01 < (img < 128).mean() < 0.4  # some ink, mostly paper; dark ink on white
        assert text and not unmappable(text) and set(text.replace(FRAC, '')) <= CHARSET


def test_speed():
    t = time.perf_counter()
    for seed in range(1000, 1020):
        make_patch(seed)
    rate = 20 / (time.perf_counter() - t)
    print(f'\nmake_patch: {rate:.1f} patches per second on one core')
    assert rate > 2
