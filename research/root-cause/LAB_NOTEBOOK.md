# Lab notebook — why the Modi reader does not learn

**If you are resuming after a context reset, read this file top to bottom first.** It is the single source of truth
for this investigation. The approved plan is `C:\Users\Omkar\.claude\plans\stateful-sprouting-melody.md`.

## Goal
Find the root cause of the failed baselines and get a working Modi handwriting-recognition baseline
(val CER clearly below the ~84 "no-reading" baseline, target ≤ 50, predictions visibly reading), or a
well-supported explanation of why that is not possible with this data and compute.

## Ground rules
- Never touch the Kaggle job that is running (04 SPAN v2).
- Every experiment gets a row in `results.csv` (config, seed, result) and a dated entry below.
- Any claim without data behind it is marked *not verified*.
- CPU only, **≤ 15 min per experiment, ~3 h total** (agreed with Omkar). Downscaled images, small models.
- One script per experiment in `experiments/`, figures in `figures/`.

## Key files
| what | where |
|---|---|
| data (images + labels) | `FYP-modi-script-data/MoDeTrans-dataset/train-00000-of-00001.parquet` (2,043 patches, all 1268×463) |
| cleaned labels / splits / charset | `FYP-modi-script-code/assets/metadata/{modetrans_text.csv, splits.json, charset.json}` (`text_train` column) |
| 03 run (strip + BiLSTM, finished) | `notebooks/outputs-of-notebook/kaggle-output-v2-03/{logs.txt, best.zip (= best.pt), test_predictions.txt}` |
| 03 notebook (model + strip code) | `notebooks/03_baseline_reader_KAGGLE_v2.ipynb` |
| SPAN v1 (unfaithful, still running) | `notebooks/outputs-of-notebook/kaggle-output-span-version1/{logs.txt, modi-script-span-model.ipynb}` |
| SPAN v2 (faithful, running) | `notebooks/04_baseline_reader_SPAN.ipynb`, code `notebooks/span/{span_lib.py, run_span.py}` |
| line-finding code (deskew, bands, valley split) | `scripts/19_image_pipeline_audit.py` |
| papers | `FYP-modi-script-work/Research Papers/` (C01 anchor MoScNet/MoDeTrans, C02 UniLipi, J12, J14) |

## Evidence established before experiments (2026-10-14, read-only)
| ID | Finding |
|---|---|
| E1 | 03 Stage B: **training** loss 3.45 → 3.01 over 171 epochs; letter-frequency prior is 3.32 → it never fitted even the training set |
| E2 | SPAN v1: training loss 7.0 → 2.57, still falling; **val CER flat 74–77 from epoch ~20 to 123**; 228 s/epoch on a T4 |
| E3 | 20-patch memorisation test passes in every version (03 at epoch 243, SPAN v1 at epoch 106) |
| E4 | Our labels equal the dataset's own labels for 2043/2043 filenames → no mapping bug |
| E5 | Label length vs ink: Spearman 0.36 (shuffled 0.04); scale-corrected vs total line length: 0.29; characters per detected line 9–65 (5th–95th pct); ink per character varies 6.5× |
| E6 | Only 25/2043 labels contain an internal line break → no line-level labels exist |
| E7 | Split is effectively per patch (2008 units / 2043) → val shares pages and hands with train |
| E8 | 15.6% of label tokens are spaces; i/ī matras 6.3%, u/ū 2.4%; the anchor paper says Modi is written without word gaps (*not verified* how visible gaps are in these images) |
| E9 | 130/2043 images are RGB, the rest greyscale; all resized to 1268×463 |
| E10 | 03's final line finder gives 4–9 bands per patch (median 4–5); the dataset paper says 3–4 lines → likely over-segmentation (*not verified*) |
| E11 | 03 test: per-patch CER 0.67–0.89 for all 306 patches (std 0.03), none below 0.5; predictions 82 characters on average vs 143 true → output nearly independent of the input |

## What the papers did
- **C01 anchor (MoScNet, IIT Roorkee):** reports BLEU only. CRNN baseline BLEU 27.3 (no details). Best = SigLIP + LLaMA decoder, 128 H100s. Images 128×256.
- **C02 UniLipi:** Modi CER 18.6 at **line level**; every model pretrained on ~2.5 M synthetic lines per script, then fine-tuned on 500–3000 real lines; 4×H100.
- **SPAN:** from scratch on RIMES (1,400 paragraphs) works after ~2 days on a V100; headline numbers use a line-pretrained encoder.

## Hypothesis register
Status: **confirmed / refuted / inconclusive / open**. Update the status and the evidence column as experiments land.

| ID | Hypothesis | Test | Status | Evidence |
|---|---|---|---|---|
| H1 | Labels do not consistently cover the visible text (extra / missing / boundary lines) | X0.1, X1 | **not supported** (subtle boundary errors still possible) | X0.1: 24/24 plausible |
| H2 | Character scale varies too much (fixed 1268×463 resize) | X0.3, X5 | **supported** (effect on learning not yet measured) | X0.3: line pitch 89–231 px (5–95%), 2.6× |
| H3 | Spaces and vowel length are non-visual → error floor + harder alignment | X0.2, metric variants | **supported** | E8; no visible word gaps in line crops (X2); J14's top errors are i/ī, u/ū, anusvara |
| H4 | Residual label noise / odd tokens | X0.2 | **low impact** | doubled ा = consistent abbreviation encoding in source |
| H5 | Image-quality subgroups (RGB vs L, speckle) behave differently | X0.4, X1 | **refuted (RGB)** | RGB images are greyscale + binarised like the rest |
| H8 | Paragraph-level CTC is the wrong unit at this data size; line level learns much faster | **X2** | **supported** (SPAN v1: paragraph model never reads, content gap 0.6; line reader gap 32.7) | line reader + synthetic pretraining reads (X4b 52.2) where every paragraph run stalled; not yet an equal-footing paragraph vs line comparison |
| H9 | Proportional pseudo line labels are good enough to start; EM re-alignment improves them | X2, X3 | **first half supported; second half inconclusive** | X4b learns from them; X3a vs X3ctl −0.23 CER (paired CI −0.80..+0.31), train loss 1.80 vs 2.04 |
| H10 | Akshara / romanised targets are easier than code points | X6 | → notebook 05 (AKS) | |
| H11 | Simply not enough epochs | SPAN v2 log, X1 | **paragraphs: REFUTED** (SPAN v1 10.75 h, val flat 74-76); lines: still improving (X3ctl) → notebook 05 FT_LONG | SPAN v1 final |
| H12 | IAM pretraining helps | SPAN v2 | **SUPPORTED (weakly)** — test 69.9 vs 72.6 scratch (CIs disjoint); iam_init starts to read (same-length gap 5.0, 253/306; 46 correct long words) — scratch does not (0.4, 148/306) | SPAN v2 test |
| H13 | Model too big → memorises | X2 sweep | **refuted** — nothing memorises (line reader train ≥ val; SPAN v1 train 70.4 vs val 74.2) | |
| H16 | Metric artefact (CER with spaces vs BLEU) | report both | **supported** — 03 no-space CER 80.0 vs 73.1; all runs report CER, no-space CER, BLEU | |
| H18 | Label mapping bug | E4 | **refuted** | E4 |
| H21 | 03's line finder over-segments → corrupted strips | X0.5 | **CONFIRMED** — explains 03's failure | X0.5: ~80% of strips unreadable |
| H23 | A character LM in the decoder recovers non-visual content (spaces, vowel length) | X6 | **CONFIRMED** | held-out val half: greedy 48.4 → LM beam 40.8 CER (no-space 47.1 → 41.0) |
| H24 | Synthetic pretraining is what gets the reader off the letter-frequency plateau | X4, X2b | **CONFIRMED** | real-only 7 epochs/26 min: val 90.2, loss 3.24 (plateau); synthetic 1 + real 3 epochs/23 min: val 52.2, loss 2.57 |
| H22 | Full-set failure is underfitting, not memorisation | **X1** | **supported** | 03 train loss at prior (E1); X2a train CER 94.4 ≈ val 93.4; X4b train 65.5 > val 52.2 |

## Experiment log
(newest at the bottom; one entry per experiment, matching a row in results.csv)

### X0.1 — label/image audit (2026-10-14) · `experiments/x01_label_image_audit.py` · figures `x01_{high,low,random}.png`
Rendered the 6 patches with the most characters per ink pixel, the 6 with the fewest, and 12 random ones beside their labels.
- **No gross mismatch in 24/24.** Line counts and label lengths are plausible everywhere (e.g. 1028: 5 dense lines / 243 chars; 1562: 2 lines / 67 chars).
- **Lowest ratio = giant writing:** 1–2 lines of very large, thick text fill the whole 1268×463 frame (1375: one line, 13 chars; 869: two lines, 28 chars).
- **Highest ratio = tiny thin-pen writing:** 4–6 small lines, often later-era letters with dates and signatures, much white space.
- So the weak label-vs-ink correlation (E5) is explained by **~10× variation in character size** inside one fixed frame, not by wrong labels.
- Side notes, *not verified as problems*: text running off the left/right edge (715, 1934, 132); a two-column table (838).
- **H1 → not supported (keep open for subtle boundary-line errors). H2 → strongly supported.**

### X0.5 — did 03's line finder cut lines apart? (2026-10-14) · `experiments/x05_line_finder_audit.py` · figures `x05_bands_{random,low,high}.png`
Ran the exact line finder 03 shipped with on the same 24 patches and showed the strip it fed the model.
- **Readable strips: random 2–3 of 12; large writing 0 of 6; small thin writing ~5 of 6.** So roughly **80% of 03's training
  inputs were unreadable.** Band boundaries run through the writing; one line of giant text becomes 4 horizontal slices,
  each stretched to 64 px tall, which look like barcodes.
- **Mechanism:** one global line height (56 px, measured on the first 300 patches) drives smoothing and valley splitting,
  but real line height runs from ~30 to ~150 px across patches (H2). Large writing has internal valleys (between the
  headline stroke and the letter bodies, and below the letters) that the splitter treats as line gaps.
- My earlier audit (scripts/19) measured ink kept and band counts but never checked whether a band cut through writing — that check would have caught this.
- **H21 → CONFIRMED. This explains 03 (training loss stuck at the prior, output independent of the input).**
- Consequence: **X1 on 03's checkpoint is deprioritised** — the model was trained on corrupted input, so its train-vs-val
  CER says nothing more about the method. SPAN (no line finder) failed for a different reason, still open.
- Any line-level approach (X2) needs a **scale-adaptive** segmenter: estimate each patch's own line pitch first.

## Next steps (updated 2026-10-14, after X4b)
1. X2b control result (real-only, 26 min total) — does more time alone escape the plateau?
2. X6 LM decoding on X4b (no training, ~5 min).
3. X3 re-align line labels with X4b, then fine-tune (13 min) vs plain continued fine-tune (13 min, control).
4. If budget allows: X4c — more / more realistic synthetic data (no visible word gaps, cursive font).
5. Write the mentor findings and, if the result holds, one Kaggle notebook 05 (lines + synthetic pretraining + fine-tune), rehearsed on CPU first.

### X0.3 — each patch's own line pitch, and a scale-adaptive segmenter (2026-10-14) · `experiments/x03_line_pitch.py` · figures `x03_segmentation_{a,b}.png`
- Pitch = dominant period of the row-ink profile (autocorrelation), fallback = height of the main ink band for one-line patches.
- **Line pitch over all 2,043 patches: median 154 px, 5th–95th percentile 89–231 px (2.6× range).** This corrects my earlier
  "10×": that figure was ink per character, which also includes stroke thickness. 03 assumed 56 px for every patch.
- Segmenting at profile minima with cuts ≥ 0.65 × pitch apart: **exact line count 20/24, within one 23/24** against my hand
  counts (03's finder: 6/24 and 12/24). Visual check: one line per band for giant text, dense small text and the table.
- Pitch saved to `experiments/x03_pitch.npy` (one value per parquet row) for X2/X5.

### Papers J12 and J14 (on disk), read 2026-10-14
- **J12 (IJIMAI 2024):** isolated-character classification only (46 classes, 4,140 images) — not text recognition. Not comparable.
- **J14 (Kundaikar, IIETA 2024):** LSTM (OCRopus) trained on a **synthetic, font-rendered** Modi line dataset, 94.67% character
  accuracy on synthetic data. Its dominant error types are **anusvara, i/ī (Eekar), u/ū (Ookar)** — the same non-visual
  distinctions measured in our labels (E8). States Modi has no word or sentence separators. **Supports H3.**
- Neither paper trains on real handwritten paragraphs.

### X2 step 1 — line images with pseudo line labels (2026-10-14) · `experiments/x2_build_lines.py` · `cache/lines_h40.pkl` · figure `x2_lines_examples.png`
- 5,901 line images from 2,043 patches (64 s). Lines per patch: 1:86, 2:858, 3:526, 4:368, 5:187, 6+:18.
  Split: train 3,975 / val 946 / test 980 lines.
- Each line cropped tight to its ink and scaled to 40 px tall → removes the 2.6× scale variation (H2) by construction.
- Labels split across a patch's lines in proportion to each line's ink width, cutting only at spaces.
- Line width at 40 px: median 354, 95% 587. Characters per line: median 37 (5%: 18, 95%: 78).
- **At width stride 4 the median line has only 2.1 CTC frames per character and 1,242 lines (21%) are below 1.5** → stride 2 used for training.
- Visual check: clean single lines; label lengths plausible; **no visible gaps between words** inside a line (supports H3).

### X2a — line reader on real pseudo-labelled lines, 13 CPU min (2026-10-14) · `experiments/x2_train_lines.py --tag x2a`
- CRNN 1.07 M params (conv 32-64-128-128-128 + BiLSTM 2×128), lines 40 px, width stride 2, Adam 1e-3, batch 8, no augmentation, seed 0.
- 4 epochs (~2,000 updates) in 13.1 min (3.8 min/epoch, slower than the benchmark because stride 2 doubles the sequence length).
- Train loss 3.72 → 3.45 → 3.37 → 3.34: **still at the letter-frequency plateau**. Output: `क क स`.
- **Val 150 patches: CER 93.4, no-space 97.3, BLEU 0.0, 0/150 patches below 50%. Train 150 patches: CER 94.4** → no memorisation; both stuck.
- **Verdict: inconclusive for H8** — the CPU budget ended before the plateau was escaped (every run so far has spent its first
  thousands of updates there). Not evidence for or against the line formulation.

### X4 step 1 — synthetic Modi lines (2026-10-14) · `experiments/x4_render_synthetic.py` · `cache/synth_lines_h40.pkl` · figure `x4_synth_examples.png`
- 12,000 lines rendered in 168 s from **training-split labels only** (1,430 labels → 4,570 line-sized chunks), Noto Sans Modi
  (OFL) shaped by Pillow+raqm; Devanagari→Modi by Unicode character name (70 of 86 symbols map; 7 rare marks simplified in
  image AND label). Augmented: size, shear, rotation, stroke thickness, elastic wobble, blur, speckle, binarised.
- Width median 391 px at 40 px; 33 chars median; 6.0 CTC frames per char at stride 2.
- Visual check: correct Modi glyph shaping. **Known gaps vs real data:** printed glyphs (not cursive handwriting) and
  **visible word spaces** (real Modi has none).

### X4a — pretrain the same line reader on synthetic lines only, 10 CPU min (2026-10-14) · `x2_train_lines.py --tag x4a --train-data cache/synth_lines_h40.pkl`
- Same CRNN/settings as X2a, trained on the 12,000 synthetic lines; evaluated **zero-shot on real** validation patches.
- 1 epoch (1,500 updates) in 10.1 min. **Synthetic training loss 2.82 — below the 3.32 plateau after one epoch**, which no run on real data ever did quickly.
- **Zero-shot real val 150 patches: CER 84.9, no-space 83.9** (X2a after 13 min on real lines: 93.4 / 97.3). Train patches 83.8.
- Output now depends on the image (36 chars per patch vs 8 for X2a); fragments align with the truth (`…चे` for `इनामतीचे`).
- **Verdict:** synthetic data gets the model off the plateau fast; zero-shot transfer to real handwriting is weak but non-zero. The real test is X4b (fine-tune on real).

### Light checks run while X4b trained (2026-10-14)
- **H4 (odd tokens):** most frequent unusual sequence is a doubled ा (`पाा`, `राा`, `सुा`): 311× in 244 labels; present in the
  SOURCE transcriptions (not our cleaning) and used consistently — looks like the transcribers' encoding of the Modi
  abbreviation sign. 376/2043 labels contain some such pair. **Low impact; consistent convention, not noise.**
- **H5 (RGB vs L):** the 130 "RGB" images are effectively greyscale (saturation 0.0) and binarised like the rest (mid-grey
  pixels 3.9% vs 4.7%). **Refuted** — no visual difference. (They are over-represented in val/test: 47/38 vs 45 train.)
- **H23 (new) — text predictability:** character n-gram cross-entropy of val text under train text: 1-gram **3.31** nats/char
  (= exactly the 3.32 plateau every failed run sat at → the plateau literally means "knows letter frequencies only"),
  2-gram 2.59, 3-gram 2.19, **4-gram 2.10** (perplexity 8.2). 25.2% of val word tokens never occur in training → a
  character LM (not a word LM) is the useful decoding prior. **Supported in principle; decoding gain not yet measured.**
- **03 no-space CER = 80.0** (vs 73.1 with spaces): part of 03's "accuracy" was spaces guessed inside a babble.

### X4b — synthetic-pretrained reader fine-tuned on real lines, 13 CPU min (2026-10-14) · `x2_train_lines.py --tag x4b --init cache/x4a_model.pt`
- Same model and settings as X2a, initialised from X4a. 3 epochs in 13.1 min (epoch 1 slowed to 6.8 min by the light checks above).
- Epoch val(60 patches): **76.7 → 61.2 →** final. Train loss **3.09 → 2.57** — off the plateau within the first epoch.
- **FINAL val 150 patches: CER 52.2, no-space 51.7, BLEU 0.20, 56/150 patches below 50% CER, best patch 33.3.**
  Train 150 patches: CER 65.5 (worse than val → not memorising; still underfitting after 3 epochs).
- Predictions are visibly reading: `…पंसर देणो` for `…पेस्तर देणे`, `आहे` recognised.
- **This is the best result of the project so far, in 23 CPU-minutes** (03: test 73.1; SPAN v1: val ~75 after 8–11 GPU-hours).
- Pending control: X2b (real-only, +13 min = 26 min total) to separate "synthetic pretraining" from "more minutes".

### X2b — CONTROL: real-only, continue X2a for 13 more CPU min (2026-10-14) · `x2_train_lines.py --tag x2b --init cache/x2a_model.pt`
- 3 more epochs (7 real epochs in total, 26 min). Train loss 3.29 → 3.21 → 3.24: **still on the plateau.** Output `स प स`.
- **Val 150 patches: CER 90.2, no-space 95.7; train 91.6.** 0/150 patches below 50%.
- **Equal-footing comparison:** real-only, 7 real epochs / 26 min → val CER **90.2**; synthetic 1 epoch + real 3 epochs / 23 min →
  val CER **52.2**. More real-data training than X4b got, and it still has not started reading.
- **H24 CONFIRMED: synthetic pretraining — not extra time — is what gets the reader off the letter-frequency plateau.**
  This is the root cause of "memorises 20 patches, never learns 1,430": from real data alone the model never leaves the
  prior within any budget we have tried (03: 171 GPU epochs; X2: 7 CPU epochs); a few minutes of synthetic text unlocks it.

### X6 — character-LM beam decoding on the X4b model, no training (2026-10-14) · `experiments/x6_lm_decode.py`
- CTC prefix beam search (beam 10, top-8 chars per frame) fused with a character 4-gram LM (interpolated absolute
  discounting) built from TRAINING labels only. alpha/beta tuned on val[:75], reported on val[75:150] (never used for tuning).
- Grid: best on the tuning half alpha 0.3, beta 1.0 (CER 60.1 → 51.6). Larger alpha hurts (the LM over-rides the image);
  beta = length bonus matters (the reader under-predicts length). Best beta was at the edge of the grid → may improve further (*not verified*).
- **Report half: greedy CER 48.4 → LM beam 40.8 (−7.6); no-space 47.1 → 41.0.** 104 s on CPU.
- **H23 CONFIRMED.** Best result so far: **≈ 41 CER** (X4b + LM), from 23 CPU-minutes of training.

### X3 step 1 — re-split training line labels by CTC forced alignment with X4b (2026-10-14) · `experiments/x3_realign_labels.py` · figure `x3_realign_examples.png`
- Viterbi CTC alignment of each multi-line training patch's FULL label against its lines laid end to end; cut where the
  alignment crosses into the next line, snapped to the nearest space. Unit-tested on 3 known cases (all correct).
- **2,279 of 3,894 multi-line training line labels changed (59%)**; mean length change +0.0, median |change| 6 chars (≈ one word moved).
- Visual spot-check: where the first glyph is legible, the aligned label usually fits better (1542: image starts `वं…`, new
  label `वनिती…`, old `प्रार्थना…`; 1097: new label starts `म्हणजे…`). *Not verified* beyond that.
- Running: X3a = fine-tune X4b 13 min on aligned labels; X3ctl = same 13 min on the original proportional labels.

### Error analysis of X4b on the 150 val patches (2026-10-14, no training)
- 17,969 character errors: **deletions 11,465 (64%)**, insertions 3,542, substitutions 2,962 → the reader mainly
  **leaves characters out** (under-generation), typical of an early CTC model; matches the LM length bonus (beta) helping in X6.
- Most missed: space 1,725 · ा 1,170 · virama ् 844 · र 703 · anusvara ं 516 (small marks and the invisible space).
- Most frequent substitutions: ि→ी 93, अ→आ 54, ो→े 49, ी→ि 44, म→क 29, ज→न 28 — the same confusions the anchor paper reports
  (म/क, ि/ी). i/ī or u/ū swaps are only 0.9% of errors; spaces are involved in 15%.
- Implication: more training and a length-aware decoder should cut errors most; vowel-length ambiguity is a small share at this stage.

## ROUND 1 SUMMARY (2026-10-14, reported to Omkar)
19 hypotheses, ~16 experiments. **Confirmed/supported 9:** H2, H3, H8 (partly), H9, H21, H22, H23, H24, H16.
**Refuted/low impact 4:** H1, H4, H5, H18. **Pending 6:** X3 re-alignment, H10, H11, H12 (Kaggle SPAN v2), H13, X4c.
Root causes: (1) 03's line finder destroyed ~80% of inputs; (2) no real-only model leaves the letter-frequency plateau;
synthetic pretraining unlocks it (controlled: 90.2 vs 52.2). Best so far ≈ 41 CER (X4b + char-LM, held-out val half).
CPU used ≈ 95 of ~180 min.

## ROUND 2 — new hypotheses (proposed 2026-10-14)
| ID | Direction | Hypothesis | Test | Cost | Status |
|---|---|---|---|---|---|
| N1 | errors | Deletions (64%) come from too few CTC frames on dense lines | correlate frames/char with deletion rate per line | analysis | **REFUTED** (Spearman +0.14, wrong sign) |
| N2 | errors | Failures cluster by writing size / stroke style | CER by line pitch and ink density | analysis | **SUPPORTED** (large writing worse: pitch +0.29) |
| N3 | decoding | Better LM decoding (wider beta grid, order 5–6) cuts more | x6 with larger grid | no training | open |
| N4 | synthetic | Realistic rendering (no word gaps, continuous headline) transfers better | X4c: render + pretrain 10 + finetune 13 vs X4b | 26 min | open |
| N5 | targets | Training without spaces in targets cleans visual learning; LM restores spaces | finetune X4b 13 min on space-free targets | 13 min | open |
| N6 | regularise | Augmenting real lines during fine-tuning helps | finetune X4b 13 min with --aug 1 vs control | 13 min | open |
| N7 | resolution | 64 px lines recover small marks (virama, anusvara) | rebuild at 64 px, pretrain + finetune | ~26 min | open |
| N8 | labels | A second EM re-alignment round keeps improving | realign with X3a model, finetune | ~15 min | open |
| N9 | mixing | Mixing synthetic into fine-tuning beats sequential | GPU | GPU | open |
| N10 | scale | 10–50× more synthetic lines + longer pretraining | GPU | GPU | open |
| N11 | capacity | Larger model helps now that learning works | GPU | GPU | open |
| N12 | text | External Marathi text for rendering covers the 25% unseen words | GPU | GPU | open |
| N13 | semi-supervised | Self-training on unlabelled HHDoc Modi pages | GPU | GPU | open |
| N14 | units | Syllable (akshara) targets | GPU | GPU | open |
Order: N1–N3 (cheap) once the X3 control frees the CPU, then N4–N6; N7/N8 if budget remains; N9–N14 go into Kaggle notebook 05.

### X3a / X3ctl — re-aligned vs original line labels, 13 CPU min each from X4b (2026-10-14)
- Both: init X4b, 3 epochs, same settings; X3a on `lines_h40_realigned.pkl`, X3ctl on `lines_h40.pkl`.
- **Val 150: X3a CER 42.4 (no-space 40.7, 102/150 below 50%) vs X3ctl 42.7 (no-space 41.2, 94/150).**
  Paired bootstrap over the same patches: **−0.23 CER, 95% CI −0.80 to +0.31** → re-alignment is **inconclusive** on val
  (better on 80 patches, worse on 61). Per epoch val(60): 54.6/50.3/50.6 vs 59.3/51.0/51.7 (re-aligned ahead early).
- Re-aligned labels are clearly **easier to fit**: train loss 1.80 vs 2.04 at epoch 3 → labels agree better with the images.
- **Bigger finding: 3 more real epochs alone took X4b 52.2 → 42.7** → on the line formulation, more training still pays
  (unlike every paragraph run). Train-patch CER 54.8 > val 42.7 → still underfitting.
- **H9 second half → inconclusive at this budget** (tested again at GPU scale in notebook 05: FT_EM1/FT_EM2).

### Decision after Omkar's request for one GPU notebook (2026-10-14)
Omkar asked for ONE run-and-forget Kaggle notebook (~2 h, very detailed logs, variations included) covering the GPU
hypotheses. N4–N8 are tested better at GPU scale, so they move into it with N9–N12, N14 and seeds; the local CPU is
used only for code tests and the notebook rehearsal. **N13 is excluded:** project rule — HHDoc must not be used for
training until the overlap check with MoDeTrans is done (test leakage risk). **N12 is run as N12b:** synthetic text
sampled from a character n-gram model of the TRAINING labels (offline proxy; no external corpus).
Notebook: `notebooks/05_line_reader_round2_GPU.ipynb`, code `notebooks/lr05/`.

### N1 / N2 — error anatomy of X3a on the 150 val patches (2026-10-14, analysis only)
- Deletions are **63%** of errors (X4b: 64%) — the reader still leaves characters out.
- **N1 REFUTED:** deletions do not come from too few CTC frames. Spearman(frames per character, deletion rate) = **+0.14**
  (wrong sign); deletion rate by frames/char quartile 28.4 / 26.1 / 24.9 / **36.2** — the most deletions where frames are
  most plentiful (frames/char 2.4–8.8, 5–95%). Resolution/stride is not the bottleneck for deletions.
- **N2 SUPPORTED:** errors cluster in **large-writing** patches: line pitch Spearman **+0.29** with CER (largest-pitch quartile
  CER 54.3 vs 42–44 for the rest); fewer lines −0.39, shorter labels −0.41, thinner relative stroke −0.26 (all the same
  big-letter patches: 1–2 lines, few characters). Possible reasons (*not verified*): rarer in training, more segmentation or
  label-split errors on 1–2-line patches, more names/dates/headers. The GPU notebook repeats this analysis for every job.

### SPAN v1 — the finished Kaggle run, analysed (2026-10-14) · `notebooks/outputs-of-notebook/kaggle-output-span-version1/`
- 167 epochs / 10.75 GPU-h (budget stop). Best val CER **74.21 at epoch 105**; **test CER 74.36** (95% CI 74.06–74.66),
  WER 108, chrF++ 6.3, BLEU 0.015; **0/306 test patches below 50% CER**. Train loss 7.0 → 2.35 and still falling; val CER flat
  74–76 from epoch ~20 to 167; blank 92%; predicted length ~0.54× the truth. Its 20-patch smoke test passed (CER 4.98).
- **Is the output about the image at all? (new test)** Score each prediction against its own transcription and against the
  patch next to it in transcription LENGTH. SPAN v1: **75.0 vs 75.6 — a 0.6-point gap, own better on 174/306 (coin flip)**.
  03: 73.7 vs 74.1, 153/306. Positive control, the CPU line reader X3a: **45.9 vs 78.6, own better on 149/150**.
- Prediction length tracks transcription length (Spearman **+0.94**, as for 03 and X3a), so the paragraph models learned
  **how much text a page holds** and the letter frequencies — and nothing about **which** letters are written. **0 correct words
  of ≥ 4 characters in 306 test patches** (03: 0; X3a: 204 in 150 val patches). Top outputs are single letters/syllables
  (क, कर, य, त, का), never the page's frequent words (आहे, करून, राजश्री).
- **Memorised the training set? No.** Its best checkpoint (epoch 105), re-scored here exactly as v1 scored (greedy, height 320):
  **train patches CER 70.4** (2/100 below 50%, best patch 22.4) vs **val 74.2** (0/100 below 50%; matches v1's own logged 74.21,
  so the re-scoring is faithful). A 4-point gap: it under-fitted its OWN training set almost as badly — the falling training
  loss (2.35) was length/frequency priors, not reading. (`scratchpad/span_v1_train_vs_val.py`)
- **Verdict:** the discarded run confirms the diagnosis rather than adding a new one: 10.75 more GPU-hours of paragraph-level
  training from scratch did not start reading (H11 for paragraphs: **refuted** — more hours alone do not help). Nothing in it
  needs to be carried forward except this test, which is now the cheapest way to tell "reads" from "babbles".

### Notebook 05 — what was verified before handing it over (2026-10-14)
| check | result |
|---|---|
| line finder vs the CPU line set | **bit-exact**: 5,901 lines, 0 of 2,043 patches differ |
| character alphabet / encoder | identical to every earlier notebook on all 2,043 labels |
| akshara alphabet | 704 tokens, round-trips all 2,043 labels, 0.64x as many CTC targets |
| beam decoder vs X6 | identical output on 54/54 real val lines (25% faster) |
| renderer | 217 lines/s/core (X4: 71); elastic strength 0.175 vs 0.181 px; LM-sampled text 100% valid, 37% new words |
| `job.py` evaluation vs CPU logs | **52.22/42.55 at first → cause: batch composition (BiLSTM sees different padding) → fixed by reading one patch per batch → 52.18 / 42.44 exact** |
| `job.py` training loop learns | from X4b, 2 epochs (batch 32, cosine): val 52.18 → **47.7 → 46.4** (train-150 58.5) |
| rehearsal 1 (whole notebook, CPU, tiny) | all cells ran: renderer, bench, planner (x0.55), 17 jobs, combination, LM grid, ensemble, test, report, zip |
| priority-3 job types (64-px, long fine-tune, long pretraining) | all ran OK through the scheduler (also tests the `known`-state fix) |
| inside-the-network figure on real models | correct: e.g. large cursive line 57.jpg emits 26 chars, 81% blank (N2 visible) |
| rehearsal 3 (final build, 40-min local budget) | running |

**Not verified (cannot be on this PC):** T4 throughput (the notebook measures it and sizes every job); that Kaggle's Pillow has
raqm (cell 10 checks, tries apt/pip, warns loudly); the CUDA AMP path (standard GradScaler code); `os.nice` for the renderer
(Linux only); by how much GPU-scale training beats the CPU numbers.

## STATUS CHECKLIST (2026-10-14 evening, given to Omkar)
**Round 1 (19 hypotheses):** all resolved except **H12** (IAM pretraining — needs the SPAN v2 / notebook-04 logs). H9b, H10,
H11-lines are re-tested at GPU scale in notebook 05. Confirmed: H21, H23, H24. Supported: H2, H3, H8, H9a, H16, H22.
Refuted/low impact: H1, H4, H5, H13, H18, H11-paragraphs.
**Round 2 (15):** N1 refuted, N2 supported, N3 supported (X3a + LM: 31.3 CER held-out val) — done on CPU. N4–N12, N14, N15 →
notebook 05. N13 excluded (HHDoc overlap rule).
**SPAN v1:** fully analysed (test 74.4; content gap 0.6 = does not read; train 70.4 vs val 74.2 = no memorisation). Done.
**Notebook 05:** built and verified piece by piece; rehearsal 4 of the final build running; content-gap column being
re-tested; then rebuild + hand-over. **X7** (one-time TEST score of the best CPU pipeline X3a + LM α0.3 β1.5) running.

### X7 — one-time TEST score of the best CPU pipeline (2026-10-14) · `experiments/x7_test_eval.py --model cache/x3a_model.pt --alpha 0.3 --beta 1.5`
- Pipeline fixed beforehand on validation only: X3a line reader (X4a synthetic 10 min → X4b real 13 min → X3a re-aligned
  real 13 min, all CPU) + character 4-gram LM beam search (alpha 0.3, beta 1.5 chosen on val[:75] in X6b).
- **TEST (306 patches, never used before): greedy CER 44.9 (95% CI 43.4–46.5); with the LM CER 36.3 (95% CI 34.9–37.9)**,
  no-space 35.9, BLEU 3.79, 236/306 patches below 50% CER (greedy: 180/306).
- Same test set: notebook 03 **73.1**, SPAN v1 **74.4**. **This is the working baseline: error halved, from ~40 CPU-minutes of training.**
- (First attempt crashed before scoring — x7 exec'd x6's header without `__file__`; fixed; the test set was scored once.)
- Not comparable to the anchor paper's BLEU 27.3 (different model, unknown evaluation details — *not verified*).

### Notebook 05 redesigned for quality, ~5 h (2026-10-14, at Omkar's request: "don't reduce quality to save GPU time")
- Budget 120 → **300 min** (extends itself by up to 60 min rather than shrink experiments if the T4 is slow); planner now
  scales budgets **up to 2x** when the GPU is fast (was: only down). All 19 experiments incl. priority 3 are planned.
- Base budgets: pretraining 240k → **300k** lines, fine-tuning 30 → **40** epochs (NOPRE 80, FT_LONG 120, PRE_LONG 900k).
- Pools: A **200k** (milestones 12k/60k/120k; 64-px copy of the first 60k), B and C **120k** (milestone 60k). Experiments use
  equal-size subsets exactly as before; falls back to the old sizes if < 15 GB scratch disk.
- Final phase of training (~100 min, both GPUs): **COMBO** (winners, normal length — do they add up?) and **FINAL_s0/s1**
  (winning recipe, WHOLE pool, lengthened by `fill_to` to the time left: fine-tuning up to 2.5x, rest to pretraining; two
  seeds, ensembled in the final phase). COMBO_LONG removed. Verified: logic test (no winners / B60+BIG+EM2 winners;
  fill_to 32 → 80 min, never shortens). Full rehearsal 5 running.

### SPAN v2 — the faithful SPAN run (notebook 04), analysed (2026-10-15) · `notebooks/outputs-of-notebook/kaggle-output-span-version2/`
Only checkpoints, test metrics and test predictions were provided (no per-epoch logs), ~10 h on Kaggle T4 x2.
| | test CER (95% CI) | best epoch / val CER | blank | same-length test: own vs neighbour (own better) | correct words >=4 chars |
|---|---|---|---|---|---|
| scratch | 72.56 (72.25–72.88) | 52 / 72.47 | 21.9% (over-generates, WER 116) | 73.2 vs 73.6, gap 0.4 (148/306 = coin flip) | 1 |
| **iam_init** | **69.90 (69.37–70.42)** | 147 / 68.57 | 90.7% | **71.1 vs 76.1, gap 5.0 (253/306)** | **46** (राजश्री ×9, आहेत, केले, विशेष) |
| SPAN v1 (ref) | 74.36 | 105 / 74.21 | 92.9% | 75.0 vs 75.3, gap 0.3 (157/306) | 0 |
| CPU line baseline + LM (X7, ref) | **36.3** | — | — | **39.9 vs 79.1, gap 39.2 (304/306)** | **1,083** |
- **H12 SUPPORTED (weakly):** the IAM-pretrained encoder is the first paragraph model that starts to read (e.g. GT `राजश्री नारो
  बाबाजी व …` → `राजश्री जारेबाबाजीव …`); from scratch it does not. Pretraining is the lever, as for the line reader — but English
  handwriting features give a small fraction of what Modi synthetic pretraining at line level gives (gap 5 vs 39).
- **Memorised? (checkpoints re-scored exactly as v2 scores: 320x880, train mean 215.6 / std 85.3, greedy grid decoding;
  val reproduces the logged values, so the re-scoring is faithful)** — `scratchpad/span_v2_train_vs_val.py`
  | | train (first 100) | val (first 100) | train patches < 50% |
  |---|---|---|---|
  | scratch | 71.5 | 72.5 | 0/100 — under-fits, never reads (like v1) |
  | **iam_init** | **43.6** (median 39.2, best 5.6) | **68.2** | **73/100** (val 0/100) — **memorises** |
  The IAM-initialised SPAN learned to read its TRAINING pages (some almost perfectly) but transfers little to unseen pages: a
  24.6-point train/val gap — classic over-fitting of 1,430 paragraphs. A different failure from v1/scratch (under-fitting),
  and the same cure: much more training data — i.e. synthetic data, the project's thesis.
- Best epoch 147 for iam_init: val may still have been creeping down at the end, but with training already memorised more
  hours would mostly deepen the memorisation (*not verified* — no per-epoch logs were provided). Not worth GPU time
  against the line route (test 69.9 vs 36.3).
- **Carry forward:** (1) pretrained features matter (H12); (2) possible follow-up, *not tested*: the authors' IAM encoder is
  a LINE-level model (Pool-Line-R) — it could serve as the backbone of our line reader, combined with Modi synthetic
  pretraining (N16, for a later notebook; notebook 05 stays as verified).
- Same-length numbers for 03/SPAN v1 differ slightly from the earlier table (75.6 → 75.3) because ties in length are now
  ordered stably; conclusions unchanged.

## NOTEBOOK 05 RAN ON KAGGLE (2026-10-15) — full analysis in `NB05_ANALYSIS.md`
- Exact verified build, 299 min, 2 x T4, scale 1.43 (57 fine-tune epochs, 429k synthetic lines per job).
- **TEST: final pipeline (ensemble AUG+BIG+B60 + char 4-gram LM) CER 25.91 (24.41–27.53)**; anchor + LM 32.20; anchor greedy 36.88.
- Round-2 verdicts (VAL-REP, paired bootstrap, seed sd 0.43): **AUG −4.03, BIG −1.86, H64 −1.30 BETTER**; A12 +2.32, C60 +1.57,
  AKS +1.11 WORSE; A120, PRE_LONG, B60, NS, MIX, FT_LONG no difference; **NOPRE (real only, GPU scale) does not read** (+35.3,
  same-length test a coin flip). EM1/EM2 crashed (NumPy-2 int8 overflow in Viterbi — fixed); COMBO/FINAL time-capped after
  1 real epoch (benchmark lacked big@64px; estimate fell back to small@40 speed — fixed).
- Zero-shot synthetic→real CER flat at 77–84 for 0.1–1.5 M synthetic lines and every style: the font-rendered data has a
  fixed ceiling — the thesis lever is synthetic realism.
- Error is U-shaped in character density (frames/char at 40 px): <3: 44.7, 3–4: 26.6, >8: 42.0; 2-line patches 34.5 vs
  5-line 19.7 on TEST. Test-time per-line rescaling: VAL-REP 36.50 → 35.89.
- **Offline (CPU, from the run's checkpoints): ensemble AUG+BIG+MIX+B60 + char 7-gram LM (0.3, 1.25) → TEST 24.55
  (23.09–26.15), −1.35 vs the notebook pipeline (paired CI −1.68..−1.04). BEST RESULT SO FAR.** LM order 4 → 6/7 alone is
  worth ~0.7–1.0 CER; ensemble gains flatten after 3–4 members.
- Folder-18 OOD test: page map exists, transcription text NOT digitised yet (needs OCR + proofreading).
- Next: notebook 06 (strong diverse members: BIG+AUG, 64-px BIG+AUG, AUG variants incl. horizontal stretch, fixed EM,
  early stopping, 6/7-gram LM) and the synthetic-realism track (zero-shot CER as the metric).

## STAGE 1 CLOSED (2026-10-15)
Omkar's Stage 1 (baseline Reader + experimental variants) is complete: best test CER 24.55 (offline recombination of
notebook 05), on free Kaggle T4s. The project-level summary, all observations and the whole-project future plan are in
`Final_year_project/PROJECT_HANDOVER.md` — the entry point for the next session. Next: pull the teammate's GitHub work,
commit the Stage-1 code, then Phase 0–1 of the handover plan (contracts → line level, Arm A vs B table, folder-18 OOD test).

## 16 OCT 2026 — Arm A on test, and the first out-of-domain test (Folder 18)
- **Arm A (real only) on TEST, once** (`offline/s10_arm_a_test.py`, NOPRE checkpoint of notebook 05): greedy 76.69
  (76.10–77.29), + char 4-gram LM (0.3, 1.0 chosen on VAL-SEL) 73.18. Same-length test: own text better on 155/306 —
  does not read. Arm B anchor − Arm A = −39.81 (−41.55..−37.96), with LM −40.98. Exported as `results/s1_armA_*`.
- **Folder 18 = the answer key is in the book**: even pages print the editors' Devanagari copy of the handwritten letter on
  the next page, and the printed lines follow the handwritten lines. Key made with Tesseract `mar` (tessdata_best,
  downloaded into `FYP-modi-script-data/tessdata`) through `modi.evaluation.folder18_gt`; PaddleOCR 3.0.2 lacks the
  PP-OCRv5 Devanagari model, so the second engine is PP-OCRv3 Devanagari (much weaker). Letters 3, 5, 8 proofread by
  Claude against the page images: Tesseract needed 3.1–4.2 % of characters corrected.
- Handwritten pages: half size, text block from the ink, `lr05_lib.segment` → 218 lines on 14 pages; bands with ink
  over < 30 % of the block width skipped (stamps). Two-column (letter 11) and tilted (up to 6°) pages kept as they are.
- Scores (final pipeline, letters as one text each): 3: 15.6, 5: 17.7, 8: 15.1, 11: 63.2, 12: 46.1, 13: 33.7, 15: 27.6.
  **Better than in-domain on 3/5/8 → leakage check (lead-guide rule): 22–31 % of their 20-character pieces occur in
  MoDeTrans TRAIN transcriptions (shared runs of 73–105 characters with val/test patches 66, 75, 23)** — the page-level
  image matching of Oct 3 did not catch that MoDeTrans contains other scans of these letters. Text-overlap ≤ 1 %:
  letters 11, 12, 13 → unseen-text CER **53.2** (anchor greedy 61.6). Inside the lead guide's expected 45–80.
- Every letter reads (own text vs other letters' text, letters only: e.g. 34 vs 82 for letter 13).
- Open: human transcription of the key (Omkar, in progress); per-column line cutting for two-column pages; hand line
  counts (Udit). The Folder-18 strict tier should be redefined by TEXT overlap, not page matching.
