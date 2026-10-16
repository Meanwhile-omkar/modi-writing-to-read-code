# Merge notes: three changes and one failure per merge

## PR #1 · Factory steps 0-5 (merged 15 Oct)
- Font check: Noto Sans Modi, GO, with renders.
- Devanagari -> Modi mapping by Unicode names; the 7 symbols without a Modi form drop their sentence instead of a letter.
- `make_patch(seed)`: text from an order-6 character model of the training sentences, patches calibrated
  against the real ones (KS 0.11-0.16).
- **Failure:** the handout's zero-copy leakage test cannot pass for an honest generator (letter formulas); replaced
  by a comparison with a generator that read val/test.

## PR #2 · Referee steps 2-5 (merged 15 Oct)
- Metrics library with hand-computed tests and a jiwer cross-check.
- Floor baselines and the error-type counter.
- Answer-key tooling: two OCR tools, review pages, `build`.
- **Failure:** the proofreading status check treated unchecked files as checked; caught before any letter was built.

## PR #3 · Synthetic-set export
- `export.py` and the release `armB-synthetic-v1` (4,290 patches + labels).
- README: how to download and load the set.
- Merged with the lead review; the README conflict kept both additions.
- **Failure:** the first check of the set ran in a way macOS multiprocessing cannot start; rerun from a script file.

## PR #4 · Results machine
- `scripts/30_aggregate_results.py`: table, seed means, paired comparisons, error types, figures.
- Fake runs with known answers as its tests (even and concentrated errors).
- Reproduces the 12 Stage-1 CERs and CIs of the results card.
- **Failure:** the concentrated fake gave CER 6.6 instead of 10 (overlapping edits); now damages until the target is met.

## Lead-review round (16 Oct, `feat/synth` + `feat/eval`)
- `make_line(seed, height)` for the line Reader, real-text chunks by default, fit-to-page check, no dotted circles.
- Look-alike pairs from validation confusions; floor rows over 100 seeds.
- Line extractor for the book pages with the Stage-1 line finder; protocol, reports, this record.
- **Failure:** on book pages, seals and printed page numbers form bands without text, and the lone "श्री" line is
  missed. Left open until T2 is scored and the effect can be measured.
