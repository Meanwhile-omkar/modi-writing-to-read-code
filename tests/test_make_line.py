import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fontTools.ttLib import TTFont

from modi.synthetic import _chunks, make_line
from modi.synthetic.mapping import FRAC, unmappable
from modi.synthetic.render import MAX_WIDTH_40, STRIDE, _font_bytes, has_dotted_circle
from modi.synthetic.text_source import training_sentences

CHARSET = set(json.load(open(Path(__file__).resolve().parents[1] / 'assets/metadata/charset.json', encoding='utf-8'))['symbols'])


@pytest.mark.parametrize('height', [40, 64])
def test_lines_fit_the_line_reader(height):
    for seed in range(150):  # also the fit-to-page check: render_line raises if letters would be cut off
        img, text = make_line(seed, height)
        assert img.dtype == np.uint8 and img.shape[0] == height
        width40 = img.shape[1] * 40 / height
        assert width40 <= MAX_WIDTH_40 + 1                                     # not wider than the Reader takes
        assert width40 / STRIDE >= 1.2 * len(text.replace(FRAC, '_')) + 2 - 1  # room for CTC (lr05 line_ok)
        assert 0.01 < (img < 128).mean() < 0.6                                 # ink on white
        assert text and not unmappable(text) and set(text.replace(FRAC, '')) <= CHARSET


def test_chunk_mode_uses_real_training_text():
    train = training_sentences()
    for seed in range(30):
        _, text = make_line(seed)
        assert any(text in s for s in train), text
    lengths = [len(c) for c in _chunks()]
    assert sorted(lengths)[len(lengths) // 2] in range(25, 45)  # line-sized, like the real lines (median ~37)


def test_same_seed_same_line():
    digest = lambda *a: hashlib.sha256(b''.join(x.tobytes() if hasattr(x, 'tobytes') else x.encode() for x in make_line(*a))).hexdigest()
    assert digest(7, 40) == digest(7, 40) and digest(7, 40) != digest(8, 40)
    assert digest(7, 40, 'ngram') == digest(7, 40, 'ngram') != digest(7, 40)
    with pytest.raises(ValueError):
        make_line(7, 40, 'other')


def test_dotted_circles_are_never_drawn():
    # these transcription spellings make the plain font insert a dotted circle (U+25CC) that is not in the label
    assert has_dotted_circle('अैवज') and has_dotted_circle('सुखरू्प')
    assert not has_dotted_circle('ऐवज') and not has_dotted_circle('राजश्री')
    # the renderers' font has no dotted-circle glyph, so shaping cannot insert one
    assert 0x25CC not in TTFont(io.BytesIO(_font_bytes(False))).getBestCmap()
    assert 0x25CC in TTFont(io.BytesIO(_font_bytes(True))).getBestCmap()
