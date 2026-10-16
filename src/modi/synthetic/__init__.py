"""Font-based synthetic data (Arm B): make_patch(seed) for 3-4-line patches, make_line(seed, height) for single lines."""
import functools
import random

from modi.synthetic.render import render as _render  # "as": keep modi.synthetic.render the module
from modi.synthetic.render import render_line as _render_line
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


@functools.lru_cache(maxsize=1)
def _chunks():
    """Training transcriptions cut at spaces into line-sized pieces of 18-70 characters, like the real lines
    (notebooks/lr05 label_chunks). A fixed cut, so the pool is the same everywhere."""
    rng, chunks = random.Random(0), []
    for sentence in training_sentences():
        words, i = sentence.split(' '), 0
        while i < len(words):
            target, line = rng.randint(18, 70), [words[i]]
            i += 1
            while i < len(words) and len(' '.join(line + [words[i]])) <= target:
                line.append(words[i])
                i += 1
            chunks.append(' '.join(line))
    return chunks


def make_line(seed: int, height: int = 40, text_mode: str = 'chunk'):
    """-> (image, text): one synthetic Modi text line. image: numpy uint8 (height, width), dark ink 0 on white 255,
    at most 1024 * height / 40 wide and wide enough for its characters (the lr05 line_ok rule). text: its
    Devanagari label. text_mode 'chunk' (default): a real 18-70-character piece of a training transcription
    (invented text trained worse in Stage 1, results/README.md); 'ngram': new text from the text source.
    The same (seed, height, text_mode) always gives the same output."""
    if text_mode not in ('chunk', 'ngram'):
        raise ValueError(f"text_mode must be 'chunk' or 'ngram', not {text_mode!r}")
    rng = random.Random(seed)
    for _ in range(50):  # a text that does not fit the line Reader is replaced by another one
        text = rng.choice(_chunks()) if text_mode == 'chunk' else _source().sample(rng, rng.randint(18, 70))
        img = _render_line(text, rng, height)
        if img is not None:
            return img, text
    raise RuntimeError(f'make_line({seed}, {height}): no text fitted in 50 tries')
