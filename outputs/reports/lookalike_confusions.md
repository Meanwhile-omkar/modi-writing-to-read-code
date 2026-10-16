# Look-alike letter pairs, from validation confusions

Letter-for-letter substitutions of the four final-ensemble models (AUG, BIG, B60, MIX) on all 307 validation patches (notebook 05 VAL-SEL + VAL-REP, greedy): 13396 substitutions between letters. Rate = swaps / occurrences of the two letters in the references. Made by `scripts/30_lookalike_groups.py`.

## Chosen: rate >= 2% (with >= 20 swaps) or count >= 150 (16 pairs, used by `taxonomy.LOOKALIKE`)

| pair | swaps | rate | chosen because |
|---|---|---|---|
| ज/न | 494 | 5.6% | rate and count |
| अ/आ | 386 | 14.9% | rate and count |
| ब/ह | 273 | 4.9% | rate and count |
| क/म | 273 | 2.8% | rate and count |
| क/ल | 261 | 2.6% | rate and count |
| न/ल | 230 | 2.3% | rate and count |
| न/म | 193 | 2.0% | rate and count |
| द/ह | 185 | 2.7% | rate and count |
| ल/ळ | 183 | 3.6% | rate and count |
| प/य | 170 | 2.0% | count |
| त/र | 155 | 0.9% | count |
| द/प | 151 | 2.4% | rate and count |
| ट/ठ | 101 | 11.2% | rate |
| इ/ई | 65 | 13.3% | rate |
| घ/ध | 27 | 2.4% | rate |
| उ/ऊ | 21 | 2.6% | rate |

## The handout's groups on the same predictions

| pair | swaps | rate |
|---|---|---|
| भ/म | 39 | 0.8% |
| क/फ | 14 | 0.2% |
| ट/ठ | 101 | 11.2% |
| ट/ढ | 3 | 0.5% |
| ठ/ढ | 1 | 0.2% |
| ण/न | 26 | 0.3% |

## Next 15 pairs by rate (not chosen)

| pair | swaps | rate |
|---|---|---|
| ऐ/औ | 6 | 21.4% |
| ख/द | 72 | 2.0% |
| द/ब | 89 | 2.0% |
| द/ध | 73 | 2.0% |
| ख/ध | 28 | 1.8% |
| ध/ह | 78 | 1.6% |
| ज/म | 114 | 1.4% |
| थ/ध | 19 | 1.4% |
| म/ल | 120 | 1.3% |
| ढ/ळ | 6 | 1.3% |
| य/व | 141 | 1.3% |
| ज/भ | 53 | 1.3% |
| ग/ण | 51 | 1.2% |
| श/स | 92 | 1.2% |
| ऊ/ड | 18 | 1.2% |
