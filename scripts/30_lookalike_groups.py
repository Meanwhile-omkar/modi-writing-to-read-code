"""30 - derive the look-alike letter pairs of the error taxonomy from VALIDATION confusions.

The handout's groups ({भ म} {क फ} {ट ठ ढ} {न ण}) were a guess. This counts which letters the Stage-1 Reader
actually swaps, on validation patches only (test predictions are never used to define how test is scored):
the four models of the final ensemble (AUG, BIG, B60, MIX), notebook 05, VAL-SEL + VAL-REP = all 307
validation patches, greedy decoding.

A pair of letters (consonants or independent vowels) is look-alike if, on these predictions, it is swapped
in at least 2 % of the two letters' occurrences (rate, with at least 20 swaps as evidence) or at least 150
times (count). Pairs are kept as pairs:
chaining them (ज-न-ल-म-क-ळ) would also count swaps nobody makes, such as ज/ळ.

Run: python scripts/30_lookalike_groups.py  ->  outputs/reports/lookalike_confusions.md
Exits with 1 if src/modi/evaluation/taxonomy.py LOOKALIKE differs from what it derives.
"""
import csv
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from modi.evaluation.metrics import normalize  # noqa: E402
from modi.evaluation.taxonomy import LOOKALIKE, align  # noqa: E402
from modi.utils import console  # noqa: F401,E402
from modi.utils.paths import CODE_ROOT, P  # noqa: E402

RUNS = CODE_ROOT / 'notebooks/outputs-of-notebook/05-notebook-kaggle/extracted/runs'
MODELS = ['AUG', 'BIG', 'B60', 'MIX']
MIN_RATE, MIN_EVIDENCE, MIN_COUNT = 0.02, 20, 150
OLD = [set('भम'), set('कफ'), set('टठढ'), set('नण')]


def is_letter(c):
    return unicodedata.category(c) == 'Lo' and unicodedata.name(c, '').startswith('DEVANAGARI')


def confusions():
    pairs, freq = Counter(), Counter()
    for model in MODELS:
        for part in ('val_sel', 'val_rep'):
            for row in csv.DictReader(open(RUNS / model / f'{part}_predictions.csv', encoding='utf-8')):
                ref, hyp = normalize(row['ref']), normalize(row['hyp'])
                freq.update(c for c in ref if is_letter(c))
                pairs.update(frozenset((a, b)) for op, a, b in align(ref, hyp)
                             if op == 'sub' and is_letter(a) and is_letter(b))
    return pairs, freq


def main():
    pairs, freq = confusions()
    rate = {p: n / sum(freq[c] for c in p) for p, n in pairs.items()}
    by_rate = lambda p: rate[p] >= MIN_RATE and pairs[p] >= MIN_EVIDENCE
    chosen = sorted((p for p in pairs if by_rate(p) or pairs[p] >= MIN_COUNT), key=lambda p: -pairs[p])
    name = lambda p: '/'.join(sorted(p))
    lines = ['# Look-alike letter pairs, from validation confusions', '',
             f'Letter-for-letter substitutions of the four final-ensemble models ({", ".join(MODELS)}) on all 307 validation '
             f'patches (notebook 05 VAL-SEL + VAL-REP, greedy): {sum(pairs.values())} substitutions between letters. '
             'Rate = swaps / occurrences of the two letters in the references. Made by `scripts/30_lookalike_groups.py`.', '',
             f'## Chosen: rate >= {MIN_RATE:.0%} (with >= {MIN_EVIDENCE} swaps) or count >= {MIN_COUNT} ({len(chosen)} pairs, used by `taxonomy.LOOKALIKE`)', '',
             '| pair | swaps | rate | chosen because |', '|---|---|---|---|']
    lines += [f'| {name(p)} | {pairs[p]} | {rate[p]:.1%} | '
              + ' and '.join(x for x, ok in (('rate', by_rate(p)), ('count', pairs[p] >= MIN_COUNT)) if ok) + ' |'
              for p in chosen]
    lines += ['', '## The handout\'s groups on the same predictions', '', '| pair | swaps | rate |', '|---|---|---|']
    for group in OLD:
        for p in sorted({frozenset((a, b)) for a in group for b in group if a < b}, key=name):
            lines.append(f'| {name(p)} | {pairs[p]} | {rate.get(p, 0):.1%} |')
    lines += ['', '## Next 15 pairs by rate (not chosen)', '', '| pair | swaps | rate |', '|---|---|---|']
    rest = sorted((p for p in pairs if p not in chosen), key=lambda p: -rate[p])[:15]
    lines += [f'| {name(p)} | {pairs[p]} | {rate[p]:.1%} |' for p in rest]
    (P.reports / 'lookalike_confusions.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    derived = sorted(''.join(sorted(p)) for p in chosen)
    current = sorted(''.join(sorted(g)) for g in LOOKALIKE)
    if derived != current:
        print(f'\ntaxonomy.LOOKALIKE differs from the derived pairs; set it to: {derived}')
        sys.exit(1)


if __name__ == '__main__':
    main()
