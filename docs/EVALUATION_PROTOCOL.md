# Evaluation protocol

How every number in the results is computed. Code: `src/modi/evaluation/` and `scripts/30_aggregate_results.py`.
Anyone with the repository, the MoDeTrans parquet and a run's predictions file can reproduce a reported number
with one command (below).

## 1. Test sets

| ID | What | Size | Role |
|---|---|---|---|
| **T1** | MoDeTrans `test` split (`assets/metadata/splits.json`, seed 42, split by source document) | 306 patches | Headline number |
| **T2-strict** | Folder 18 (*Illustrative Modi Documents*), letters 3, 5, 8, 11, 12, 13, 15 | 14 handwritten pages | Out-of-domain: no page is in MoDeTrans |
| T2-partial | Folder 18, letters 2, 4, 6, 7, 14, 17 | 10 handwritten pages | Reported separately, never pooled with strict |

- **Validation** (`val` split, 307 patches) is the only data used to choose anything: checkpoints, decoding
  settings, language-model weights, and the look-alike letter pairs of the error taxonomy.
- **T1 is scored once per final configuration.** Nothing is tuned on it.
- **Never evaluated:** Folder-18 pages that are MoDeTrans sources (letters 1, 9, 10, 16 and the other
  `excluded` pages in `folder18_labels.csv`), because the Reader trained on them; MODI-HHDoc pages and
  MODI-HChar characters, which have no transcriptions.
- Robustness check reported with T1: 8 test patches share at least 50 characters of text with a training patch;
  scores are also given without them (`results/README.md`).

## 2. References and predictions

- T1 references: `text_train` in `assets/metadata/modetrans_text.csv` (the expert text after
  `scripts/05` and `scripts/15`: NFC, σ written as `<FRAC>`, editor's `?` dropped, `[x]` joined, newlines as spaces).
- T2 references: `folder18_gt.csv`, made by OCR with two tools, proofread line by line, then the same
  `scripts/05` + `scripts/15` cleaning (`assets/folder18_gt/RULES.md`, shared on the team Drive, not in git).
- A run hands in `results/<run>/test_predictions.csv` (`image_id,reference,prediction`, UTF-8, one row per
  test image) next to its `config.json`. Read with `keep_default_na=False` so an empty prediction stays empty.
- T2 is scored per letter: the handwritten pages are cut into lines by `modi.segmentation.folder18_lines`
  (the Stage-1 line finder), the line predictions of a letter are joined with spaces in page and line order,
  and scored against the letter's full text.

## 3. Metrics (`metrics.py`)

Before any comparison both texts are Unicode **NFC**-normalised and `<FRAC>` counts as **one** character.
An empty reference is an error; an empty prediction scores CER 100.

| Metric | Definition |
|---|---|
| **CER** (primary) | Levenshtein edits over code points, **summed over all images and divided by all reference characters** (micro-average, so a long patch weighs more than a short one). In %, can exceed 100. |
| CER without spaces | the same with all spaces removed (Modi word boundaries are the editor's, not the writer's) |
| WER | the same over whitespace-separated words |
| chrF++ | sacreBLEU `corpus_chrf`, character 6-grams + word bigrams (`word_order=2`), 0-100 |
| character BLEU | sacreBLEU `corpus_bleu`, `tokenize='char'` (spaces not counted); only to compare with the MoDeTrans paper |
| median CER | median of the per-image CERs |
| catastrophic rate | % of images with per-image CER above 50 |

## 4. Statistics

- **Confidence interval** of a CER: 10,000 bootstrap resamples of the test images (with replacement, seed 0),
  CER recomputed on each; the 2.5th and 97.5th percentiles.
- **Comparing two systems**: paired bootstrap, the same 10,000 resamples for both, rows matched by `image_id`
  (and checked to have identical references). A difference is reported as real only if its 95 % interval
  excludes 0.
- **Seeds**: every reported configuration is trained with seeds 42, 43, 44; reported as mean ± standard deviation
  (n - 1). Runs named `<config>_seed<NN>` are grouped automatically.

## 5. Error types (`taxonomy.py`)

Each prediction is aligned with its reference (minimum edit operations; ties pair letters with letters and
signs with signs), and every edit goes to the first category that fits: **look-alike** (a swap between two of
the 16 letter pairs the Stage-1 Reader really confuses on validation, `outputs/reports/lookalike_confusions.md`),
**virama**, **anusvara** (inserted or dropped), **space**, **vowel sign**, **other**.

## 6. Reference points (`floor_baselines.py`)

Deliberately bad predictors on T1, mean ± std over 100 seeds: empty output 100.0; a random training sentence
98.1 ± 1.7; random training text of the right length 83.1 ± 0.1; the correct letters shuffled 84.6 ± 0.1; the
correct text with 10 % / 30 % of characters damaged 9.9 / 28.9. A Reader near 80 or above is not reading.

## 7. Producing the results table

```bash
python scripts/30_aggregate_results.py --compare <run A> <run B>
```

reads every `results/*/test_predictions.csv` and writes `outputs/reports/results.md` and `results.json`
(all metrics above, CIs, seed means, paired comparisons, error types) plus two figures. On the 12 Stage-1 runs
it reproduces every CER and CI of `results/README.md` to two decimals. The machine itself is tested on fake
runs with known answers (`tests/test_results_machine.py`).
