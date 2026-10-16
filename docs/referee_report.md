# Referee report (Stage 1)

Udit Sevak · branch `feat/eval` · how numbers are computed: `docs/EVALUATION_PROTOCOL.md`

## What was built

| Step | Result | Where |
|---|---|---|
| 2 Metrics | CER (micro), CER without spaces, WER, chrF++, character BLEU, median CER, catastrophic rate, bootstrap CI, paired bootstrap. Hand-computed tests; agrees with `jiwer` on 1,000 random pairs; deliberately broken versions fail the tests. | `src/modi/evaluation/metrics.py` |
| 3 Floor baselines | Empty 100.0; random sentence 98.1 ± 1.7; random text of the right length 83.1; shuffled letters 84.6; 10 % / 30 % damage 9.9 / 28.9 (test CER, 100 seeds). | `outputs/reports/floor_baselines.md` |
| 4 Error types | Look-alike, virama, anusvara, space, vowel sign, other. Look-alike pairs derived from validation confusions of the Stage-1 Reader (16 pairs, ज/न the most frequent). | `taxonomy.py`, `outputs/reports/lookalike_confusions.md` |
| 5 Answer key | Tooling done: OCR with Tesseract and PaddleOCR, review pages with the two OCR texts diffed, `build` applying the rules, audit and out-of-alphabet files. Proofreading in progress. | `folder18_gt.py`, `assets/folder18_gt/` (Drive) |
| 6 Lines of the book | Line extractor using the Stage-1 line finder on the 38 handwritten pages; 351 line images for the strict and partial letters. Agreement with hand counts pending. | `src/modi/segmentation/folder18_lines.py` |
| 7 Results machine | One command from `results/*/test_predictions.csv` to the results table, seed means, paired comparisons, error types, figures. Reproduces every CER and CI of the 12 Stage-1 runs to two decimals. | `scripts/30_aggregate_results.py` |
| 8 Protocol | This report's companion. | `docs/EVALUATION_PROTOCOL.md` |

## What worked

- Testing every scorer on inputs with known answers first: hand-computed cases, damaged text that should give
  CER 10 and 30, fake runs whose differences are known. Every real number afterwards could be trusted.
- Reporting median CER and the catastrophic rate next to the mean: two fake systems with the same CER (9.85 and
  10.11) had medians of 9.9 and 0 and confidence intervals 70 times apart.
- Deriving the look-alike pairs from real validation errors instead of guessing: the handout's groups covered
  about 100 errors of the final pipeline, the derived pairs 467.

## What did not, and what was fixed

- The first alignment explained भंग -> मग as "भ deleted, ं read as म"; ties now pair letters with letters.
- The proofreading status check accepted unchecked files, because the hint text contained the words it looked for.
- A random-damage fake meant to give CER 10 gave 6.6 (overlapping edits); it now damages until the real edit
  count is reached.
- The first floor table used one seed; seed 42 alone gave 95.0 for the random-sentence row, the 100-seed mean is 98.1.
- The lead review found more: comparisons could misalign rows of differently ordered files, WER divided by zero on a
  reference without words, space errors were filed as "other", and OCR failed on Windows. All fixed with tests
  (`tests/test_lead_review.py`).

## Open

1. **Answer key**: 7 strict letters to proofread completely (letter 3 started). T2 cannot be scored before.
2. **Hand line counts** for the 38 pages, then the agreement figure and the five hardest pages.
3. **Book pages**: seals and printed page numbers can form a band without text, and a lone "श्री" line is
   dropped by the line finder's 1 % ink rule. Measure their effect once T2 is scored; decide whether to filter.
4. `<FRAC>` in the answer key and the synthetic data: check with a Modi reader how fractions are written.
