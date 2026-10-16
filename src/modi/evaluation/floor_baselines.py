"""Referee Step 3: what does "terrible" look like? Six silly predictors scored on validation and test.

Run: PYTHONPATH=src python -m modi.evaluation.floor_baselines  ->  outputs/reports/floor_baselines.md
"""
import json
import random

import numpy as np
import pandas as pd

from modi.evaluation.metrics import FRAC, _FRAC_CHAR, cer, chrf_pp, normalize, wer
from modi.utils.paths import CODE_ROOT, P

META = CODE_ROOT / 'assets/metadata'
ALPHABET = [s for s in json.load(open(META / 'charset.json', encoding='utf-8'))['symbols'] if s != ' ']


def damage(text, share, rng):
    """Damage `share` of the characters: each one is deleted, replaced, or followed by an extra letter (a random
    choice, so about a third each). Each damaged character is one edit, except that a deletion next to an insertion
    can merge into one substitution, so the CER of the result is close to (slightly below) 100 * share."""
    chars = list(normalize(text))
    for i in sorted(rng.sample(range(len(chars)), round(share * len(chars))), reverse=True):
        op = rng.choice(('delete', 'replace', 'insert'))
        if op == 'delete':
            del chars[i]
        elif op == 'replace':
            chars[i] = rng.choice([c for c in ALPHABET if c != chars[i]])
        else:
            chars.insert(i + 1, rng.choice(ALPHABET))
    return ''.join(chars).replace(_FRAC_CHAR, FRAC)


def predictors(refs, train, rng):
    """The six floor 'models': name -> one prediction per reference."""
    def random_same_length(ref):
        text = ''
        while len(text) < len(ref):
            text += rng.choice(train) + ' '
        return text[:len(ref)]

    def cut_or_padded(ref):  # the handout's row (c): one sentence, cut, or padded with spaces
        s = rng.choice(train)[:len(ref)]
        return s + ' ' * (len(ref) - len(s))

    def shuffled(ref):  # shuffle symbols, so <FRAC> moves as one symbol instead of six letters
        symbols = normalize(ref)
        return ''.join(rng.sample(symbols, len(symbols))).replace(_FRAC_CHAR, FRAC)

    return {
        'empty text': ['' for _ in refs],
        'random training sentence': [rng.choice(train) for _ in refs],
        'random training sentence, cut or padded with spaces': [cut_or_padded(r) for r in refs],
        'random training text, same length': [random_same_length(r) for r in refs],
        'correct letters, shuffled': [shuffled(r) for r in refs],
        'correct text, 10 % damaged': [damage(r, 0.10, rng) for r in refs],
        'correct text, 30 % damaged': [damage(r, 0.30, rng) for r in refs],
    }


SEEDS = range(42, 142)  # every random predictor is repeated with 100 seeds: mean and spread


def main():
    split = json.load(open(META / 'splits.json', encoding='utf-8'))['modetrans']['assignment']
    text = pd.read_csv(META / 'modetrans_text.csv', keep_default_na=False)
    by_split = {s: text[text.image_id.map(split) == s].text_train.tolist() for s in ('train', 'val', 'test')}
    runs = {}  # predictor -> one [val CER, val WER, val chrF++, test CER, test WER, test chrF++] per seed
    for seed in SEEDS:
        row = {}
        for s in ('val', 'test'):
            refs = by_split[s]
            for name, hyps in predictors(refs, by_split['train'], random.Random(seed)).items():
                row.setdefault(name, []).extend([cer(refs, hyps), wer(refs, hyps), chrf_pp(refs, hyps)])
        for name, values in row.items():
            runs.setdefault(name, []).append(values)
    stats = {name: (np.mean(v, axis=0), np.std(v, axis=0, ddof=1)) for name, v in runs.items()}
    lines = ['# Floor baselines (Referee Step 3)', '',
             f'Silly predictors on the MoDeTrans validation ({len(by_split["val"])}) and test ({len(by_split["test"])}) '
             f'patches, `text_train`. Each random predictor is run with {len(SEEDS)} seeds ({SEEDS[0]}-{SEEDS[-1]}): '
             'mean ± standard deviation. CER and WER in %, lower is better; chrF++ higher is better.', '',
             '| predictor | val CER | val WER | val chrF++ | test CER | test WER | test chrF++ |', '|---|---|---|---|---|---|---|']
    lines += [f'| {name} | ' + ' | '.join(f'{m:.1f} ± {sd:.1f}' for m, sd in zip(*stats[name])) + ' |' for name in stats]
    (P.reports / 'floor_baselines.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    mean = {name: m for name, (m, _) in stats.items()}
    assert mean['empty text'][0] == mean['empty text'][3] == 100  # the handout's checks
    assert abs(mean['correct text, 10 % damaged'][3] - 10) < 1.5 and abs(mean['correct text, 30 % damaged'][3] - 30) < 3


if __name__ == '__main__':
    main()
