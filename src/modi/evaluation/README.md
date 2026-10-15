# Evaluation (the Referee)

Scores for every model, the floor baselines, error types, and the tooling for the Folder-18 answer key.
Definitions follow `docs/BASELINE_DESIGN.md`, Sections 7.2 and 7.3.

## Scoring a run

A run's predictions file is `results/<run_name>/test_predictions.csv` with columns
`image_id,reference,prediction` (UTF-8, one row per test image).

```python
import sys; sys.path.insert(0, 'src')
import pandas as pd
from modi.evaluation.metrics import cer, cer_nospace, wer, chrf_pp, median_cer, catastrophic_rate, bootstrap_ci

p = pd.read_csv('results/B1_seed42/test_predictions.csv', keep_default_na=False)  # keeps empty predictions as ''
refs, hyps = p.reference.tolist(), p.prediction.tolist()
score, low, high = bootstrap_ci(refs, hyps)  # CER with its 95 % interval (10,000 resamples)
print(f'CER {score:.2f} [{low:.2f}, {high:.2f}]  WER {wer(refs, hyps):.2f}  chrF++ {chrf_pp(refs, hyps):.2f}')
```

`keep_default_na=False` matters: without it pandas reads an empty prediction as NaN instead of an empty string.

## The rules every metric follows

- Text is Unicode NFC-normalised first.
- `<FRAC>` counts as one character.
- An empty reference raises `ValueError`.
- CER and WER are micro-averaged: all edits in the set divided by all reference characters (words).
  A 10-character patch weighs less than a 400-character one. Values are percentages and can exceed 100.

## Functions (`metrics.py`)

| Function | Returns |
|---|---|
| `cer(refs, hyps)` | character error rate, % (the headline number) |
| `cer_nospace(refs, hyps)` | CER ignoring spaces (word boundaries in Modi are the editor's choice) |
| `wer(refs, hyps)` | word error rate, % |
| `chrf_pp(refs, hyps)` | chrF++, 0-100, higher is better |
| `bleu(refs, hyps)` | character BLEU, only to compare with the MoDeTrans paper |
| `median_cer`, `catastrophic_rate` | median per-image CER; % of images with CER above 50 |
| `bootstrap_ci(refs, hyps)` | `(cer, low, high)`, 95 % interval |
| `paired_bootstrap(refs, hyps_a, hyps_b)` | `(cer_a - cer_b, low, high)` on the same resamples; a difference is real only if the interval excludes 0 |

`taxonomy.error_types(refs, hyps)` counts mistakes by kind: look-alike letters ({भ म} {क फ} {ट ठ ढ} {न ण}),
virama (broken conjuncts), anusvara dropped or added, vowel signs, other.

## Reference points (`floor_baselines.py`)

What deliberately bad predictors score on the test split (`outputs/reports/floor_baselines.md`):

| Predictor | Test CER |
|---|---|
| empty text | 100.0 |
| random training sentence | 95.0 |
| random training text of the right length | 83.2 |
| correct letters, shuffled | 84.5 |
| correct text, 10 % of characters damaged | 9.9 |
| correct text, 30 % of characters damaged | 28.8 |

A Reader at CER 80 or above is no better than guessing.

## Folder-18 answer key (`folder18_gt.py`)

The printed transcriptions of the book's strict letters (3, 5, 8, 11, 12, 13, 15), OCR'd twice and
proofread by hand. The answer key itself stays out of git (`assets/folder18_gt/` is ignored because of the
book's terms) and is shared separately. Rules are in `assets/folder18_gt/RULES.md`.

```bash
PYTHONPATH=src python -m modi.evaluation.folder18_gt ocr     # OCR + proofreading files (needs PaddleOCR in ../ocr-venv)
PYTHONPATH=src python -m modi.evaluation.folder18_gt build   # after proofreading: folder18_gt.csv + audit + OOV report
```

The book is scored per letter: all band predictions of a letter's handwritten pages joined, against the
letter's full text.

## Tests

```bash
python -m pytest -q tests/test_metrics.py tests/test_taxonomy.py tests/test_folder18_gt.py
```
