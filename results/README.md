# Stage 1 final results — start here

**Stage 1 = the baseline Reader** (Modi handwriting → Devanagari text), frozen on 15 October 2026 under the git tag
**`stage1-final-baseline`** (`git checkout stage1-final-baseline` returns to exactly this state).

**Best result: test CER 24.55 % (95 % CI 23.09–26.15)** on the 306 MoDeTrans test patches, i.e. about 75 % of
characters correct. Trained on free Kaggle T4 GPUs (one 5-hour run) plus CPU.

## The final pipeline (the "Arm B" Reader recipe)
1. **Lines, not paragraphs.** Each 1268×463 strip is deskewed (±6°); its own line pitch is estimated from the
   autocorrelation of the row-ink profile; lines are cut at profile minima ≥ 0.65 × pitch apart, cropped to their ink
   and scaled to 40 px high (64 px variant). 5,901 lines from 2,043 strips.
2. **Line labels:** each strip's transcription is split across its lines in proportion to line ink width, cutting only at
   spaces. All scores are computed on whole strips, so this approximation never enters a reported number.
3. **Synthetic pretraining:** training transcriptions only, cut into 18–70-character chunks, mapped Devanagari → Modi by
   Unicode name, rendered with Noto Sans Modi (Pillow + raqm), then sheared, thickened/thinned, wobbled, rotated, blurred,
   speckled and binarised. 60k rendered lines (each a random degradation of one of 4,536 distinct training-text
   chunks), ~430k presentations.
4. **Model:** CRNN — conv 32-64-128-128-128 (or 2× wider "BIG", 4.2 M parameters) + 2-layer BiLSTM + CTC.
5. **Fine-tuning on real lines:** Adam 1e-3, warm-up + cosine, batch 32, augmentation (small rotation/scale/shear,
   erode/dilate); best epoch chosen on the first 150 validation patches (VAL-SEL).
6. **Decoding:** CTC prefix beam search (beam 10, top-8) + character 7-gram LM from training transcriptions
   (α 0.3, β 1.25).
7. **Ensemble:** average of the per-frame probabilities of four 40-px models — AUG, BIG, MIX, B60.

Code: `notebooks/lr05/` (source of notebook 05; `build_nb05.py` builds the notebook, `rehearse_nb05.py` rehearses it on
CPU) and `research/root-cause/offline/` (ensemble + LM search that produced 24.55).

## All Stage-1 pipelines on the same test set (scored once each)
Each folder below holds `test_predictions.csv` (`image_id,reference,prediction`, all runs in the same row order),
`config.json` and `metrics.json` (team results format). Re-create and re-check them with
`python scripts/22_export_stage1_predictions.py`. CI and character BLEU come from the team metric library
(`src/modi/evaluation/metrics.py`; 10,000 image-level bootstrap draws).

| run folder | pipeline | test CER (95 % CI) | CER without spaces | char BLEU | patches < 50 % CER |
|---|---|---|---|---|---|
| `s1_offline_ens4_lm7` | **final: 4-model ensemble + char 7-gram LM** | **24.55 (23.09–26.15)** | 24.26 | 64.7 | 277 / 306 |
| `s1_nb05_final_ens3_lm4` | notebook 05 final: 3-model ensemble + char 4-gram LM | 25.91 (24.41–27.53) | 25.54 | 62.2 | 273 |
| `s1_nb05_final_ens3_greedy` | same ensemble, no LM | 32.46 (30.82–34.29) | 31.25 | 50.7 | 243 |
| `s1_nb05_anchor_lm4` | single model (anchor) + char 4-gram LM | 32.20 (30.59–33.96) | 31.64 | 55.0 | 245 |
| `s1_nb05_anchor_greedy` | single model (anchor), no LM | 36.88 (35.29–38.61) | 35.04 | 47.4 | 229 |
| `s1_cpu_x7_lm4` | first CPU line baseline + LM | 36.33 (34.85–37.87) | 35.90 | 48.9 | 236 |
| `s1_cpu_x7_greedy` | first CPU line baseline, no LM | 44.88 (43.43–46.46) | 43.02 | 33.7 | 180 |
| `s1_armA_nopre_lm4` | **Arm A**: same CRNN, real lines only (no synthetic pretraining) + char 4-gram LM | 73.18 (72.71–73.68) | 78.24 | 4.8 | 0 |
| `s1_armA_nopre_greedy` | **Arm A**: real lines only, no LM | 76.69 (76.10–77.29) | 81.08 | 1.6 | 0 |
| `s1_span_v2_iam` | paragraph model (SPAN), IAM-pretrained encoder | 69.90 (69.37–70.42) | 74.12 | 9.6 | 1 |
| `s1_span_v2_scratch` | paragraph model (SPAN) from scratch | 72.56 (72.25–72.88) | 78.61 | 4.1 | 0 |
| `s1_nb03_strip_crnn` | notebook 03: lines joined into one strip | 73.09 (72.81–73.37) | 80.02 | 1.9 | 0 |
| `s1_span_v1` | first SPAN-style paragraph model | 74.36 (74.07–74.67) | 81.68 | 1.8 | 0 |
| `s1_nb02_band_crnn` | notebook 02: stacked bands | 81.32 (80.58–82.12) | 87.90 | 0.3 | 0 |

Reference points: an empty output scores 100; a random sentence of the right length ~84; text with 30 % of its
characters damaged ~29. Anything near 75–85 is not reading.

**Arm A vs Arm B (the study's first comparison).** Arm A — the same CRNN trained on real lines only (notebook 05's
NOPRE control, 2× the fine-tuning epochs, scored on test once on 16 Oct) — does not read: 76.69 CER, and its output
matches its own patch's text no better than another patch's text of the same length (own better on 155/306, a coin
flip). Arm B — synthetic-line pretraining then the same fine-tuning (the anchor) — reads: 36.88, own better on 304/306.
Paired difference Arm B − Arm A: **−39.81 CER (95 % CI −41.55 to −37.96)**; with the LM −40.98 (−42.49 to −39.33).
Code: `research/root-cause/offline/s10_arm_a_test.py`.

**Out-of-domain test: unseen documents (Folder 18, preliminary, 16 Oct).** The book *Illustrative Modi Documents*
(folder 18 of MODI-HHDoc) prints each handwritten Modi letter next to the editors' typed Devanagari copy, so the
answer key is in the book. For the 7 "strict" letters (14 handwritten pages) the key was made with Tesseract
(`mar`) through `modi.evaluation.folder18_gt` and proofread line by line against the printed page images for 6 of
the 7 letters (Tesseract needed 3–4 % of characters corrected); letter 11, printed in two columns, is still OCR-only.
The handwritten pages were cut into lines automatically by the Stage-1 line finder (no hand layout) and read by the
frozen pipelines; each letter is scored as one text. A text-overlap check found that **22–31 % of the text of letters
3, 5, 8 and 15 already occurs in MoDeTrans training transcriptions** — those letters are not truly unseen even
though their pages are. Letters 11, 12, 13 (≤ 2 % overlap: shared greeting formulas only) count as unseen documents.

| letters | final pipeline CER | single model (anchor), no LM |
|---|---|---|
| **unseen, single column: 12, 13** | **41.8** (per letter 44.8 / 31.9) | 53.7 |
| unseen, two columns: 11 (OCR key; line order across columns differs from the key) | 63.2 | 68.6 |
| all unseen: 11, 12, 13 | 52.7 | 61.3 |
| partly in MoDeTrans: 3, 5, 8, 15 | 15.1–22.9 (inflated by the overlap) | 28.9–38.9 |

Every letter's output matches its own text far better than other letters' texts, so the Reader does read unseen
pages; on new documents it loses about 17 CER points against MoDeTrans (24.55 → 41.8 on single-column letters).
Main losses: page layout (two columns; tilt beyond the ±6° deskew range on letter 12). The key and the page images
stay outside git (HHDoc terms).
Code: `research/root-cause/offline/s11_folder18_ood.py` (outputs in the git-ignored `assets/folder18_gt/ood/`).

**Robustness check.** 8 of the 306 test patches share at least 50 characters of text with a training patch
(the two pairs inspected are the same handwriting scanned twice). Without them the best pipeline scores
**24.80 (23.29–26.43)** on the other 298 patches. The split stays frozen so that every number remains comparable.

## What worked, what did not (details: `research/root-cause/NB05_ANALYSIS.md`, `LAB_NOTEBOOK.md`)
- **Real data alone never learns to read** (it stays at the letter-frequency level, ~3.3 nats/char): every real-only
  run, including a GPU-scale one (notebook 05 NOPRE, val +35 CER vs the anchor), fails.
- **Cutting strips into lines with a per-strip pitch** and **pretraining on font-rendered synthetic lines** is what
  unlocks reading (CPU: 90.2 → 52.2 val CER at equal time).
- **Augmenting real lines during fine-tuning** is the largest single gain (−4.0 CER); a 2× wider model −1.9; 64-px lines
  −1.3; a character LM −4.7 to −8.6; LM order 6–7 beats 4 by ~0.7–1.0; ensembles help mainly through the LM.
- **No gain:** more synthetic data of the same kind (60k vs 120k lines), 3× longer pretraining, "realistic" font
  rendering, keeping synthetic data in fine-tuning, 3× longer fine-tuning. **Worse:** LM-invented synthetic text,
  syllable (akshara) targets, too little synthetic diversity (12k lines).
- **Synthetic data has a ceiling:** a Reader trained only on font-rendered lines reads real handwriting at ~77–84 CER
  whether it saw 0.1 M or 1.5 M lines. Only more realistic synthetic handwriting can lower it — the motivation for the
  project's learned Writer. Zero-shot real CER of a synthetic-only Reader is a cheap measure of synthetic realism.
- Remaining errors of the final pipeline: 43 % deletions, 17 % insertions, 40 % substitutions. By kind (team error
  taxonomy): vowel signs 28 %, spaces 18 %, anusvara 6 %, virama 5 %, look-alike letter pairs 0.2 %, other 43 %.
  Most-missed symbols are the space, anusvara ं, ा and virama ्; error is U-shaped in character density (big,
  squeezed writing and very sparse writing are hardest).

## Reproducing
| what | how |
|---|---|
| re-score every run from its saved predictions | `python scripts/22_export_stage1_predictions.py` (seconds) |
| the 24.55 pipeline from the checkpoints | `research/root-cause/offline/README.md` (CPU: ~25 minutes) |
| retrain everything | build notebook 05 with `notebooks/lr05/build_nb05.py`; Kaggle, 2× T4, ~5 hours |

**Checkpoints (27 files, in git):** `notebooks/outputs-of-notebook/05-notebook-kaggle/extracted/runs/<model>/best.pt`
(the four final models: `AUG`, `BIG`, `MIX`, `B60`; the anchor seeds `FT_A0_s0/1/2`; 64-px `H64`; every other notebook-05
experiment) and `research/root-cause/experiments/cache/x*_model.pt` (the six CPU-study models).

## Metric notes
`metrics.json` in each run folder comes from the Stage-1 code, `notebooks/lr05/lr05_lib.py:metrics`: micro-averaged CER
over patches; the `<FRAC>` token is removed before scoring (no test reference contains one); "CER without spaces"
ignores spaces; **its "BLEU" is sacreBLEU's default word BLEU** (12.99 for the final pipeline), not character BLEU.
The team metric library (`src/modi/evaluation/metrics.py`) is the reference for everything after Stage 1. Checked on
these 12 runs: it reproduces every CER to within 0.003 (the only differences are a few predicted `<FRAC>` tokens, which
it counts as one character), and the 95 % CI and the paired test exactly. The character BLEU above comes from it
(sacreBLEU `tokenize='char'`, which ignores spaces). The MoDeTrans paper reports BLEU 27.3 for its CRNN; its BLEU
settings and split are not verified, so the numbers are not compared here.

## Not in git, and why
- MoDeTrans images: download from Hugging Face (`historyHulk/MoDeTrans`).
- Cached line images and log-probabilities (`*.pkl`, hundreds of MB): re-created by the code.
- `05_results.zip` (the raw notebook-05 bundle, already extracted here) and the paragraph models' checkpoint zips:
  too large for git or of no further use.
- MODI-HHDoc page images and the typed transcriptions of the Folder-18 book: their terms do not allow redistribution.

## Data and licences
MoDeTrans (MIT licence; arXiv 2503.13060). Noto Sans Modi (SIL Open Font License). All figures in this repository are
made from MoDeTrans images or from synthetic renders.
