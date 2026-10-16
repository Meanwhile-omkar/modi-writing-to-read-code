# Evaluation (the Referee)

Scores for every model, the floor baselines, error types, and the tooling for the Folder-18 answer key.
Definitions follow `docs/BASELINE_DESIGN.md`, Sections 7.2 and 7.3.

## Scoring a run

A run's predictions file is `results/<run_name>/test_predictions.csv` with columns
`image_id,reference,prediction` (UTF-8, one row per test image). The 12 Stage-1 Reader runs are already in
`results/s1_*/` (see `results/README.md`); the example scores the best one.

```python
import sys; sys.path.insert(0, 'src')
import pandas as pd
from modi.evaluation.metrics import cer, cer_nospace, wer, chrf_pp, median_cer, catastrophic_rate, bootstrap_ci

p = pd.read_csv('results/s1_offline_ens4_lm7/test_predictions.csv', keep_default_na=False)  # keeps empty predictions as ''
refs, hyps = p.reference.tolist(), p.prediction.tolist()
score, low, high = bootstrap_ci(refs, hyps)  # CER with its 95 % interval (10,000 resamples)
print(f'CER {score:.2f} [{low:.2f}, {high:.2f}]  WER {wer(refs, hyps):.2f}  chrF++ {chrf_pp(refs, hyps):.2f}')
```

`keep_default_na=False` matters: without it pandas reads an empty prediction as NaN instead of an empty string.
Results files do not share a row order: before comparing two runs, merge them on `image_id` (and pass
`refs_b=` to `paired_bootstrap`, which then refuses runs that are not aligned).

## The rules every metric follows

- Text is Unicode NFC-normalised first.
- `<FRAC>` counts as one character.
- An empty reference raises `ValueError` (for WER also a reference without any word).
- CER and WER are micro-averaged: all edits in the set divided by all reference characters (words).
  A 10-character patch weighs less than a 400-character one. Values are percentages and can exceed 100.

## Functions (`metrics.py`)

| Function | Returns |
|---|---|
| `cer(refs, hyps)` | character error rate, % (the headline number) |
| `cer_nospace(refs, hyps)` | CER ignoring spaces (word boundaries in Modi are the editor's choice) |
| `wer(refs, hyps)` | word error rate, % |
| `chrf_pp(refs, hyps)` | chrF++, 0-100, higher is better |
| `bleu(refs, hyps)` | character BLEU, only to compare with the MoDeTrans paper; sacreBLEU's `char` tokenizer ignores spaces (`keep_spaces=True` counts them) |
| `median_cer`, `catastrophic_rate` | median per-image CER; % of images with CER above 50 (exactly 50 is not catastrophic) |
| `bootstrap_ci(refs, hyps)` | `(cer, low, high)`, 95 % interval |
| `paired_bootstrap(refs, hyps_a, hyps_b, refs_b=None)` | `(cer_a - cer_b, low, high)` on the same resamples; a difference is real only if the interval excludes 0 |

`taxonomy.error_types(refs, hyps)` counts mistakes by kind: look-alike letters (16 pairs the Reader really swaps on
validation, e.g. ज/न, ब/ह, ल/ळ, ट/ठ; derived by `scripts/30_lookalike_groups.py`, see `outputs/reports/lookalike_confusions.md`),
virama (broken conjuncts), anusvara dropped or added, space (word boundary missed or invented), vowel signs, other.

## Reference points (`floor_baselines.py`)

What deliberately bad predictors score on the test split (`outputs/reports/floor_baselines.md`; regenerate with
`PYTHONPATH=src python -m modi.evaluation.floor_baselines`, in PowerShell `$env:PYTHONPATH='src'; python -m modi.evaluation.floor_baselines`):

| Predictor | Test CER, mean ± std over 100 seeds |
|---|---|
| empty text | 100.0 |
| random training sentence | 98.1 ± 1.7 |
| random training sentence, cut or padded with spaces to the right length | 84.3 ± 0.1 |
| random training text of the right length | 83.1 ± 0.1 |
| correct letters, shuffled | 84.6 ± 0.1 |
| correct text, 10 % of characters damaged | 9.9 ± 0.0 |
| correct text, 30 % of characters damaged | 28.9 ± 0.1 |

A Reader at CER 80 or above is no better than guessing.

## Folder-18 answer key (`folder18_gt.py`)

The answer key is made from the printed transcriptions of the book's strict letters (3, 5, 8, 11, 12, 13, 15):
each page is OCR'd twice and proofread by hand with the tool below. The answer key itself stays out of git
(`assets/folder18_gt/` is ignored because of the book's terms) and is shared on the team Drive, together with
its rules file `RULES.md`, the audit file `folder18_gt_audit.csv` and the out-of-alphabet list `oov_report.csv`.

```bash
PYTHONPATH=src python -m modi.evaluation.folder18_gt ocr     # OCR + proofreading files
PYTHONPATH=src python -m modi.evaluation.folder18_gt build   # after proofreading: folder18_gt.csv + audit + OOV report
```

In PowerShell set the path first: `$env:PYTHONPATH='src'`. `ocr` needs:
- the Folder-18 page images (`18/page<N>.jpg` under `hhdoc_root` in `configs/paths.yaml`);
- Tesseract on the PATH with the Marathi model `mar.traineddata` in `<data_root>/tessdata`;
- PaddleOCR in its own environment, by default `../ocr-venv` (`bin/python`, on Windows `Scripts\python.exe`),
  or any interpreter named in the `PADDLE_PYTHON` environment variable.

The book is scored per letter: the handwritten pages are cut into lines by `modi.segmentation.folder18_lines`
(the Stage-1 line finder; images and `manifest.csv` in `<data>/processed/folder18_lines/`), the Reader's line
predictions of a letter are joined with spaces in page and line order, and the result is scored against the
letter's full text.

## Tests

```bash
python -m pytest -q tests/test_metrics.py tests/test_taxonomy.py tests/test_folder18_gt.py
```
