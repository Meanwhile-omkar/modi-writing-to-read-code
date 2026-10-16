# Why the baseline reader did not learn — and what does work

*MODI-SYNTH, baseline Reader investigation, 14 October 2026. CPU phase complete; notebook 05 (GPU) is the next step.
Every number below comes from a logged experiment (`results.csv`) or a log file; the full record is `LAB_NOTEBOOK.md`.*

## In one paragraph
Three GPU runs (about 10–11 GPU-hours each) produced no reader: the best scored **CER 73** on the test set, and its
output did not depend on the letters in the image. The investigation found two causes. **First**, the line-cutting front end of notebook 03 was
destroying about **80%** of its own training images. **Second, and more fundamental**, no model we trained on the
1,430 real training patches ever got past the **letter-frequency plateau**: the point where a model has learned how often
each letter occurs but nothing about what the letters look like. Given a few minutes of **synthetic Modi text** first,
the same small model leaves that plateau at once and **reads real Modi**. With a character language model in the decoder,
the resulting baseline scores **CER 36.3 on the untouched test set** (95% CI 34.9–37.9) after about 40 CPU-minutes of
training — against 73.1 and 74.4 for the earlier GPU runs on the same test set.

## 1. What failed, measured
| run | what it was | result |
|---|---|---|
| 03 | lines cut out and joined into one strip, CNN + BiLSTM + CTC, 10.7 GPU-h | **test CER 73.1**, BLEU 0.01; training loss 3.45 → 3.01 over 171 epochs; every one of 306 test patches scored between 67% and 89% CER |
| 04 v2 | faithful SPAN, ~10 GPU-h, two models side by side | **from scratch: test CER 72.6**, never reads; **IAM-pretrained encoder: 69.9** — reads its training pages (train CER 43.6) but not new ones (val 68.2): it memorises |
| 04 v1 | whole-paragraph model (SPAN-style), 10.75 GPU-h, 167 epochs | **test CER 74.4**, BLEU 0.02; val CER flat at ~75 from epoch 20 to 167 while training loss kept falling; every one of 306 test patches above 50% CER |

**Did either model look at the letters at all?** Score each prediction against its own transcription and against the
patch whose transcription has nearly the same length. A model that reads must do clearly better on its own patch.
| | own transcription | same-length other patch | own patch better on |
|---|---|---|---|
| 03 | 73.7 | 74.1 | 153 / 306 (a coin flip) |
| 04 v1 (SPAN-style) | 75.0 | 75.6 | 174 / 306 |
| 04 v2, from scratch | 73.2 | 73.6 | 148 / 306 |
| 04 v2, IAM-pretrained encoder | 71.1 | 76.1 | 253 / 306 — the first paragraph model to read a little |
| CPU line reader (X3a, for comparison) | 45.9 | 78.6 | **149 / 150** |

Both failed models learned *how much* text a page holds (their output length follows the true length closely) and which
letters are common — and nothing about *which* letters are written: not one correct word of four or more characters in
306 test patches (the line reader: 204 in 150 validation patches). Nor did the paragraph model memorise its training
data: re-scored on its own training patches it reaches only **70.4** CER (validation 74.2). Ten more GPU-hours of the
same recipe moved nothing — more hours alone are not the answer.

**The plateau, explained.** A model that ignores the image and only knows how often each character appears scores
**3.31 nats per character** on our validation text. Every failed run sat at **3.3**. That is what "stuck" meant: the models
had learned letter frequencies, nothing more. Their predictions were babble of the commonest syllables (`परा पा करा सरा`).

## 2. Cause one — notebook 03 was training on destroyed images
The line finder assumed one line height (56 px) for every patch. Measured over all 2,043 patches, the real line pitch runs
from **89 to 231 px** (5th–95th percentile, median 154). Large writing was cut into horizontal slices, each slice stretched
to full height, producing stripes with no recognisable letter. On a random sample only **2–3 of 12** strips were readable;
on large writing **0 of 6**. (`figures/x05_bands_*.png`)

A replacement that first measures each patch's own line pitch matches a hand count of lines on **20 of 24** patches
(the old one: 6 of 24). (`figures/x03_segmentation_*.png`)

## 3. Cause two — real data alone never gets the reader off the plateau
A controlled comparison on this PC's CPU, same small line reader (1.07 M parameters), same lines, same evaluation:

| | training | real-data passes | val CER (150 patches) |
|---|---|---|---|
| real lines only | 26 min | 7 | **90.2** — still on the plateau |
| synthetic Modi first, then real lines | 10 + 13 min | 3 | **52.2** — reading |
| the same, decoded with a character language model | (no extra training) | — | **≈ 41** (held-out half) |

With more than twice the real-data training, the real-only model still had not started. Ten minutes of synthetic text,
rendered from the **training** transcriptions in a Modi font, changed that. This is the root cause behind "it memorises 20
patches but never learns 1,430": from scratch, the reader has to discover what each letter looks like from noisy,
whole-patch labels, and with this little data it does not get there. Clean synthetic text, where every letter's label is
exact, teaches it first.

## 4. Properties of the data that any reader has to live with
- **No line-level labels.** Only 25 of 2,043 transcriptions contain a line break. Lines were cut automatically and each
  patch's text split across its lines (then re-aligned with the reading model).
- **Writing size varies 2.6×** between patches, all stored at 1268×463. Cutting lines and scaling each to one height removes this.
- **About a sixth of the label cannot be seen.** 15.6% of label characters are spaces, but Modi is written without gaps
  between words; another ~9% are i/ī and u/ū vowel signs that Modi often writes alike. A character language model
  recovers part of this — on the test set it takes the reader from 44.9 to 36.3 CER.
- **Labels are faithful.** Checked against the images on 24 patches including the most extreme ones: no gross mismatches.

## 5. Why published systems did better
| work | what they used that we did not |
|---|---|
| MoScNet (C01, the dataset paper) | SigLIP vision encoder + LLaMA decoders pretrained on huge corpora, **128 H100 GPUs**; reports BLEU only (their CRNN: BLEU 27.3) |
| UniLipi (C02), Modi CER 18.6 | **line-level** training, ~**2.5 million synthetic lines per script** before fine-tuning on real lines, 4 H100s |
| SPAN | ~2 days on a V100; headline results use a line-pretrained encoder |

None trains a reader from scratch on ~1,400 real paragraphs. Our result agrees with theirs: **synthetic pretraining at the
line level is the missing ingredient.**

## 6. What this means for the project
The project's thesis — *writing to read*: generate handwriting to train a reader — is exactly the lever this investigation
found. Even plain font-rendered Modi moves CER from ~90 to ~52 in minutes; better synthetic handwriting is the natural
next step and the paper's contribution.

Concrete plan:
1. **Baseline at GPU scale (notebook 05, ~5 h):** the same line-level recipe with up to 200,000 rendered lines instead of
   12,000, far longer training, every round-2 question answered on the way, and a final model trained as long as the
   budget allows. Expected to improve on the CPU test result of 36.3; *not verified* by how much.
2. **Then the thesis experiment:** replace font rendering with style-conditioned synthetic handwriting and measure the gain
   over this baseline.
3. **On the 5,000-image idea:** synthesise *lines*, not paragraphs, and in far larger numbers. The same text costs about
   **7× fewer pixels** as lines (a 40-px line is ~14,200 px; a patch holds ~2.9 lines ≈ 41,000 px, versus 281,600 px for
   a whole patch at 320×880), and line models leave the plateau far sooner.

## 7. The working baseline, on the test set
The best CPU pipeline — line reader (synthetic 10 min → real 26 min of CPU training) plus a character language model in the
decoder, every setting chosen on validation only — scored once on the 306 untouched test patches:
**CER 36.3** (95% CI 34.9–37.9; 44.9 without the language model), against **73.1** for notebook 03 and **74.4** for the
SPAN-style run on the same test set. 236 of 306 test patches are now read with under 50% error (notebook 03: none).

## 8. What the last CPU checks showed (round 2)
- **More training on lines keeps paying.** Three more CPU epochs took the 52.2 model to **42.7** (X3ctl) — unlike every
  paragraph run, the line reader is still improving when the budget ends. Its training-patch CER (54.8) is *worse* than
  validation, so it is still under-fitting: more GPU time should help (*not verified* by how much — notebook 05 measures it).
- **Re-aligning the line labels with the model (X3)** makes them easier to fit (training loss 1.80 vs 2.04) but the
  validation gain is within noise: **42.4 vs 42.7, difference −0.23, 95% CI −0.80 to +0.31**. Re-tested at GPU scale in notebook 05.
- **Where the errors are (N1/N2).** Two thirds of the errors are characters left out. It is *not* a shortage of output
  positions (more frames per character does not reduce deletions — the opposite). Errors concentrate in **large writing**:
  patches with 1–2 big lines score ~54 CER against ~43 for the rest.

## 9. Notebook 05 — result (ran on Kaggle, 5 h; details in `NB05_ANALYSIS.md`)
**Test CER 25.9** for the notebook's final pipeline (three models + language model), and **24.6** after re-combining its
checkpoints offline (four models + a higher-order language model) — against 36.3 for the CPU baseline and 73.1 for
notebook 03. What helped: augmenting the real lines (−4.0), a larger model (−1.9), 64-px lines (−1.3), ensembles and the
language model. What did not: more or longer synthetic pretraining, 'realistic' rendering, invented synthetic text,
syllable targets. Real data alone still does not read, even with five GPU-hours. Synthetic-only models stay at ~80 CER
however much synthetic text they see — the font-rendered data has a ceiling, which is exactly what the project's
handwriting generator is meant to lift.

### The run's design (as planned)
One Kaggle notebook runs the line reader at GPU scale and, around one fixed anchor, every round-2 hypothesis as a job
that changes one thing: synthetic volume (12k / 60k / 120k lines, longer pretraining), realistic rendering (no word gaps,
continuous headline), synthetic text with new word forms, a 4× larger model, syllable (akshara) targets, no-space
targets, 64-px lines, synthetic data kept in fine-tuning, augmentation, one and two label re-alignments, 3× longer
fine-tuning, three seeds of the anchor (to measure noise), and a real-data-only control at 2× epochs. It then combines
whatever beat the anchor, tunes a character-LM decoder and an ensemble on one half of validation, reports on the other
half, and scores the test set once. Everything was rehearsed end to end on CPU first; GPU speed is measured at the start
and every job is sized to fit (the budget may grow by up to an hour rather than shorten any experiment). Last, the
winning recipe is trained twice for as long as the time allows (about 100 minutes) on the largest synthetic pool, and
the two models are ensembled. Self-training on HHDoc pages is deliberately left out until the HHDoc/MoDeTrans overlap
check is done.
