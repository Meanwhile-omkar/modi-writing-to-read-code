"""Factory Step 2: new archaic-Marathi texts for the fake patches.

A character n-gram model learned from the TRAINING sentences only (validation and test are never read
here): it counts which character follows each run of 5 characters, then writes new text one character
at a time by drawing the next character with those odds. Text lengths are drawn from the training lengths.

Run: PYTHONPATH=src python -m modi.synthetic.text_source   (prints 10 examples)
"""
import json
import random
from collections import Counter, defaultdict
from itertools import accumulate

import pandas as pd

from modi.synthetic.mapping import FRAC, unmappable
from modi.utils.paths import CODE_ROOT

ORDER = 6          # next character given the previous ORDER - 1 = 5 characters
BACKOFF = 0.05     # chance of looking at one character less, so the text is not only copied spans
FRAC_CHAR = ''  # <FRAC> as a single symbol while modelling, so it is never cut apart


def training_sentences():
    """text_train of the training split, minus the sentences with a symbol that has no Modi form."""
    meta = CODE_ROOT / 'assets/metadata'
    split = json.load(open(meta / 'splits.json', encoding='utf-8'))['modetrans']['assignment']
    text = pd.read_csv(meta / 'modetrans_text.csv', keep_default_na=False)
    return [s for i, s in zip(text.image_id, text.text_train) if split[i] == 'train' and not unmappable(s)]


class TextSource:
    def __init__(self, sentences, order=ORDER):
        self.order = order
        texts = [s.replace(FRAC, FRAC_CHAR) for s in sentences]
        self.lengths = [len(t) for t in texts]
        stream = ' '.join(texts)  # one long stream, so every context has a next character
        counts = defaultdict(Counter)
        for i, ch in enumerate(stream):
            for k in range(min(order, i + 1)):  # contexts of 0 .. order-1 preceding characters
                counts[stream[i - k:i]][ch] += 1
        self.next = {ctx: (list(c), list(accumulate(c.values()))) for ctx, c in counts.items()}

    def sample(self, rng):
        """One new text, as long as a randomly chosen training sentence, ending at a word boundary."""
        target = rng.choice(self.lengths)
        out = ' '  # start as if after a space, i.e. at the beginning of a word
        while True:
            k = self.order - 1 - (rng.random() < BACKOFF)
            ctx = out[-k:]
            while ctx not in self.next:  # only happens after a back-off step
                ctx = ctx[1:]
            chars, cum = self.next[ctx]
            ch = rng.choices(chars, cum_weights=cum)[0]
            if ch == ' ' and len(out) > target:
                return out[1:].replace(FRAC_CHAR, FRAC)
            out += ch


def main():
    sentences = training_sentences()
    source = TextSource(sentences)
    vocab = {w for s in sentences for w in s.split()}
    texts = [source.sample(random.Random(seed)) for seed in range(1000)]
    words = [w for t in texts for w in t.split()]
    print(f'{len(sentences)} training sentences; 10 examples:')
    for t in texts[:10]:
        print(' -', t)
    print(f'{100 * sum(w in vocab for w in words) / len(words):.1f} % of generated words are real training words')


if __name__ == '__main__':
    main()
