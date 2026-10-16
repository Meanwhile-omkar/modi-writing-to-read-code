# Work log: Udit Sevak (Referee + Factory)

Times are from the commits. Failed attempts are kept.

## 12 Oct
- Read the three handouts; set up the environment.
- Data: MoDeTrans parquet (154 MB, SHA-256 matches `BASELINE_DESIGN.md`). Folder 18 taken out of the 7.2 GB
  HHDoc archive by byte range (7 MB; the archive is RAR5, not solid, and its quick-open index lists every file).
  Folder-18 pages turned out to be 600-dpi bi-level TIFFs named `.jpg`.
- Pillow could not join Modi letters (raqm off): fribidi from Homebrew was not found at run time. Fixed in the
  virtual environment's activate script.
- Factory step 0, font check: Noto Sans Modi draws all 79 Modi code points and shapes conjuncts. GO.

## 13 Oct
- 00:01 Step 0 committed. 01:24 step 1, mapping: 70 of 77 symbols map; round trip exact on all usable training sentences.
- Step 2, text source. **Failure:** the handout's "zero 20-character copies from val/test" test failed for an
  honest generator (87 of 5,000 texts): letter formulas recombine into variants that occur only in val/test. A
  generator that also read val/test: 504. Test changed to "far below a leaky control"; decision asked of the lead.
- Steps 3-5, renderer and `make_patch`. **Failures caught by the step-4 calibration:** writing too small (lines
  covered a third of the width); short texts drew every word on its own line (up to 11 lines, from a wrong wrap
  width), found when the line-count statistic would not match. Tuned `TALL` and `LINE_CHARS`; KS 0.16 / 0.11 / 0.13 / 0.03.
- 17:45 Referee steps 2-4. **Failure:** the alignment explained भंग -> मग the wrong way round; tie-break fixed.
- 22:16 Step 5 tooling; both OCR tools run on the 14 printed pages. **Failures:** OCR'd English headings passed the
  line filter (now cut at the "क्र." line); unchecked proofreading files counted as checked (status line fixed).
- PR #1 (Factory) opened after the README; Colab check of `make_patch`: runs, same text as on the Mac.

## 14 Oct
- PR #2 (Referee) opened. Started proofreading letter 3.

## 15 Oct
- Lead review merged #1 and #2 and fixed small defects (row order in paired comparisons, WER on empty references,
  a "space" error type, OCR on Windows).

## 16 Oct
- 16:00 Synthetic set of 4,290 patches exported and published as release `armB-synthetic-v1`.
- 16:10 Results machine. **Failure:** a fake run meant to give CER 10 gave 6.6; fixed. On the 12 Stage-1 runs the
  machine reproduces every CER and CI of the results card.
- Lead-review items: font-gate evidence committed; `make_line` (**failure:** the fit-to-page check fired at once,
  because noise pixels touched the canvas edge; text extent now taken from strokes only); dotted circles removed
  (29 of the 4,290 release images had one); floor rows over 100 seeds; look-alike pairs from validation confusions;
  line extractor for the book pages (**open:** seals and page numbers form text-free bands, a lone "श्री" line is
  dropped); evaluation protocol and referee report.
