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
   speckled and binarised. 60k distinct lines, ~430k presentations.
4. **Model:** CRNN — conv 32-64-128-128-128 (or 2× wider "BIG", 4.2 M parameters) + 2-layer BiLSTM + CTC.
5. **Fine-tuning on real lines:** Adam 1e-3, warm-up + cosine, batch 32, augmentation (small rotation/scale/shear,
   erode/dilate); best epoch chosen on the first 150 validation patches (VAL-SEL).
6. **Decoding:** CTC prefix beam search (beam 10, top-8) + character 7-gram LM from training transcriptions
   (α 0.3, β 1.25).
7. **Ensemble:** average of the per-frame probabilities of four 40-px models — AUG, BIG, MIX, B60.

Code: `notebooks/lr05/` (source of notebook 05; `build_nb05.py` builds the notebook, `rehearse_nb05.py` rehearses it on
CPU) and `research/root-cause/offline/` (ensemble + LM search that produced 24.55).

## All Stage-1 pipelines on the same test set (scored once each)
Each folder below holds `test_predictions.csv` (`image_id,reference,prediction`), `config.json` and `metrics.json`
(team results format). Re-create and re-check them with `python scripts/22_export_stage1_predictions.py`.

| run folder | pipeline | test CER | CER without spaces | patches < 50 % CER |
|---|---|---|---|---|
| `s1_offline_ens4_lm7` | **final: 4-model ensemble + char 7-gram LM** | **24.55** | 24.26 | 277 / 306 |
| `s1_nb05_final_ens3_lm4` | notebook 05 final: 3-model ensemble + char 4-gram LM | 25.91 | 25.54 | 273 |
| `s1_nb05_final_ens3_greedy` | same ensemble, no LM | 32.46 | 31.25 | 243 |
| `s1_nb05_anchor_lm4` | single model (anchor) + char 4-gram LM | 32.20 | 31.64 | 245 |
| `s1_nb05_anchor_greedy` | single model (anchor), no LM | 36.88 | 35.04 | 229 |
| `s1_cpu_x7_lm4` | first CPU line baseline + LM | 36.33 | 35.90 | 236 |
| `s1_cpu_x7_greedy` | first CPU line baseline, no LM | 44.88 | 43.02 | 180 |
| `s1_span_v2_iam` | paragraph model (SPAN), IAM-pretrained encoder | 69.90 | 74.12 | 1 |
| `s1_span_v2_scratch` | paragraph model (SPAN) from scratch | 72.56 | 78.61 | 0 |
| `s1_nb03_strip_crnn` | notebook 03: lines joined into one strip | 73.09 | 80.02 | 0 |
| `s1_span_v1` | first SPAN-style paragraph model | 74.36 | 81.68 | 0 |
| `s1_nb02_band_crnn` | notebook 02: stacked bands | 81.32 | 87.90 | 0 |

Reference points: an empty output scores 100; a random sentence of the right length ~84; text with 30 % of its
characters damaged ~29. Anything near 75–85 is not reading.

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
- Remaining errors: 45 % deletions; most-missed symbols are the space, anusvara ं, ा, virama ्; error is U-shaped in
  character density (big, squeezed writing and very sparse writing are hardest).

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
Scores here are computed with `notebooks/lr05/lr05_lib.py:metrics`: micro-averaged CER over patches; the `<FRAC>` token is
removed before scoring (no test patch contains one); "CER without spaces" ignores spaces; **its BLEU is sacreBLEU's
default word BLEU**, so it is not the character-level BLEU reported by the MoDeTrans paper. The team's metric library
(`src/modi/evaluation/metrics.py`) is the reference for all later comparisons.

## Not in git, and why
- MoDeTrans images: download from Hugging Face (`historyHulk/MoDeTrans`).
- Cached line images and log-probabilities (`*.pkl`, hundreds of MB): re-created by the code.
- `05_results.zip` (the raw notebook-05 bundle, already extracted here) and the paragraph models' checkpoint zips:
  too large for git or of no further use.
- MODI-HHDoc page images and the typed transcriptions of the Folder-18 book: their terms do not allow redistribution.

## Data and licences
MoDeTrans (MIT licence; arXiv 2503.13060). Noto Sans Modi (SIL Open Font License). All figures in this repository are
made from MoDeTrans images or from synthetic renders.
