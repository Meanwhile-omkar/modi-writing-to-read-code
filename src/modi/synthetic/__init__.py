"""Font-based synthetic patches (Arm B): Devanagari-to-Modi mapping, text source, renderer, make_patch(seed)."""
import functools
import random

from modi.synthetic.render import render as _render  # "as": keep modi.synthetic.render the module
from modi.synthetic.text_source import TextSource, training_sentences


@functools.lru_cache(maxsize=1)
def _source():
    return TextSource(training_sentences())


def make_patch(seed: int):
    """-> (image, text). image: numpy uint8 (512, 1408), dark ink 0 on white 255. text: its Devanagari label,
    only charset.json symbols (and <FRAC>). The same seed always gives the same output."""
    rng = random.Random(seed)
    text = _source().sample(rng)
    return _render(text, rng), text
