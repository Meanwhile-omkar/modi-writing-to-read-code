# Notebook 05 on Kaggle — deep analysis of the round-2 run

*15 October 2026. Source: `notebooks/outputs-of-notebook/05-notebook-kaggle/` (05_results.zip, extracted to `extracted/`;
the notebook as run). Every number below is computed from those files or re-computed from the run's checkpoints on CPU
(re-scoring reproduces the run's own validation CER to within 0.03). Nothing here is tuned on the test set except where
it says "TEST, once".*

## 0. What actually ran
- The **exact verified build** (code identical to `notebooks/lr05/`, apart from one cosmetic report sentence changed later),
  5-hour budget, 2 × Tesla T4, Python 3.13, torch 2.11 / CUDA 12.8, 4 CPUs, 31 GB RAM.
- The T4s were faster than planned (small model **705 lines/s**, big 443, 64-px 571), so every experiment ran at **1.43×** its
  base budget: 429k synthetic lines of pretraining and **57** real fine-tuning epochs per job.
- Modi shaping (raqm) worked; the line set matched the CPU set exactly (5,901 lines); the renderer produced pools A 200k,
  B 120k, C 120k in 89 min in the background (~90 lines/s at low priority).
- 23 jobs: 17 experiments finished, **2 crashed (EM1, EM2)**, **3 were cut by their time cap (COMBO, FINAL_s0, FINAL_s1)** — §5.

## 1. Headline (TEST, 306 patches, scored once for the two pre-registered pipelines)
| pipeline | decoding | CER (95% CI) | no-space CER | chrF++ | BLEU | patches < 50% CER |
|---|---|---|---|---|---|---|
| anchor (60k plain synthetic → real) | greedy | 36.88 (35.29–38.61) | 35.04 | 33.6 | 1.71 | 229 |
| anchor | char LM (0.45, 1.0) | 32.20 (30.59–33.96) | 31.64 | 41.9 | 6.57 | 245 |
| **ensemble AUG+BIG+B60** | greedy | 32.46 (30.82–34.29) | 31.25 | 38.2 | 3.36 | 243 |
| **ensemble AUG+BIG+B60** | **char LM (0.3, 1.0)** | **25.91 (24.41–27.53)** | **25.54** | **49.3** | **10.77** | **273** |

Same test set: notebook 03 **73.1**, SPAN v1 **74.4**, SPAN v2 (IAM encoder) **69.9**, CPU baseline (X3a + LM) **36.3**.
Against the CPU baseline the final pipeline is better on **289 of 306** patches (mean +10.0 points) and reads **1,781**
correct words of ≥ 4 characters (CPU baseline 1,083; anchor + LM 1,344).

## 2. What each hypothesis showed (VAL-REP, paired bootstrap; noise = anchor seed sd 0.43)
| change | Δ CER vs anchor (95% CI) | verdict | reading |
|---|---|---|---|
| **AUG** — augment real lines in fine-tuning | **−4.03 (−4.80 to −3.26)** | **BETTER** | biggest single gain; also the least over-fitting (§3) |
| **BIG** — 2× wider CRNN (4.2 M params) | **−1.86 (−2.71 to −1.03)** | **BETTER** | capacity helps, but it memorises (train-150 21.7 vs val 38.7) |
| **H64** — 64-px lines | **−1.30 (−2.31 to −0.37)** | **BETTER** | finer marks help; frame rate differs, so it cannot join 40-px ensembles |
| A120 — 120k vs 60k distinct synthetic lines | +0.51 (−0.31 to +1.34) | no difference | more of the same synthetic data does not help … |
| PRE_LONG — 3× longer pretraining | +0.35 (−0.41 to +1.11) | no difference | … nor does longer pretraining |
| A12 — only 12k distinct lines | +2.32 (+1.55 to +3.11) | WORSE | … but too little diversity hurts |
| B60 — realistic rendering (no gaps + headline) | +0.76 (+0.01 to +1.53) | no difference | best zero-shot transfer (76.9 vs 79.7), gone after fine-tuning |
| C60 — LM-sampled synthetic text | +1.57 (+0.75 to +2.40) | WORSE | invented word forms hurt; real training text is better |
| AKS — akshara targets | +1.11 (+0.24 to +1.98) | WORSE | but its errors are the most different (good ensemble partner) |
| NS — no spaces in targets (no-space CER) | +0.44 (−0.37 to +1.23) | no difference | spaces in the targets do not hurt reading the letters |
| MIX — keep synthetic in fine-tuning | +0.85 (+0.06 to +1.66) | no difference | |
| FT_LONG — 3× fine-tuning epochs | +0.02 (−0.59 to +0.61) | no difference | fine-tuning is saturated by epoch ~20 |
| **NOPRE** — real lines only, 2× epochs | **+35.34** | **WORSE** | **does not read even at GPU scale**: own-patch vs same-length 75.9 vs 75.7 (a coin flip) |
| EM1 / EM2 — label re-alignment | — | not tested | crashed at the first re-alignment (§5); on track with the anchor until then |
| COMBO (BIG+H64+AUG) / FINAL | — | not tested | starved of fine-tuning by a planning bug (§5) |

## 3. Training dynamics
- Every synthetic-pretrained model falls from ~70 to ~35 val CER within 10 epochs and plateaus; best epochs **15–34 of 57**.
  After the best epoch val CER rises 1–2 points while training loss keeps falling (anchor 1.27 → 0.41): over-fitting.
- **AUG** is the exception: best at epoch 34, only +0.4 after it, training loss stays at ~1.0 — augmentation regularises.
- **BIG** memorises (training loss 0.065, train-150 CER 21.7 vs val 38.7) — the strongest case for BIG + AUG together.
- **FT_LONG** (171 epochs) peaks at epoch 20; the extra 150 epochs only memorise (loss 0.038).
- **Zero-shot transfer is flat at 77–84 CER** after 0.1 M or 1.5 M synthetic lines, for every style: the font-rendered
  synthetic data has a fixed gap to real handwriting that more of it cannot close. Fine-tuning does all the work.
  **This is the thesis in one figure: the next lever is synthetic data that looks like the real hands.**
- Inside the network (`runs/*/inside_alignment.png`): AUG fires sharp, confident spikes (confidence 0.83–0.85); NOPRE's are
  weak and diffuse (0.23–0.28, 78–91 % blank) — it never commits to a character.

## 4. Where the remaining errors are (final pipeline, TEST)
- 11,369 errors: **deletions 45 %**, insertions 15 %, substitutions 40 %; 18 % involve a space; i/ī + u/ū swaps only 3.1 %.
- Most missed: space 546, **anusvara ं 419**, ा 373, **virama ् 337**, र 298, े 244 — small marks and the invisible space.
- Top confusions: ि→ी 140, **ि→space 114**, ी→ि 112, अ→आ 66, space→ा 53, ू→ु 53.
- The LM removes most deletions (greedy ensemble: 63 % deletions → 45 %), at the price of a few insertions.
- Patch CER: median 27.4, 90th percentile 50.0. The patches every model fails (17 of 157 in VAL-REP at ≥ 50 % for all
  15 models) are **layout** and **label-coverage** problems, not reading problems: 665 (stamp + overlapping two-column
  lines → the line finder breaks, 4 characters output), 724 (scattered date words with underlines), 690 (lines cut off
  at the patch edges; the label covers 35 characters, the model reads 65 from what is visible), 82 (a very cursive hand).
- **Errors are spread out, not concentrated:** the worst 10 % of test patches hold 13.7 % of the errors (without them CER
  23.7 instead of 25.9). Layout fixes are therefore a small lever.
- **The big pattern is patch size.** Final pipeline on TEST, by number of detected lines: **2 lines 34.5** CER (88 patches),
  3 lines 28.6 (109), 4 lines 23.8 (68), **5 lines 19.7** (38). Strongest correlate: label length (Spearman −0.40), then
  number of lines (−0.32) and line pitch (+0.20). Short, big-letter patches lose most — a likely cause is **partial lines at
  the patch edges** (text cut at the top/bottom of the strip weighs more when the whole label is short; *not verified*).
- The line finder is rarely the problem: only **11 of 2,043** patches get an implausible line pitch (< 60 px; e.g. test
  382: 8 "lines" at 26 px → the worst test patch, 83 % CER). A guard against harmonics would fix these few.
- Models make highly correlated errors (per-patch Spearman 0.85–0.91); AKS and H64 are the most different.
  Per-patch oracle (best model per patch, an upper bound for picking one model): AUG alone 36.5 → AUG+BIG 35.3 →
  AUG+BIG+H64 34.6 → all 15 models 33.6 (VAL-REP greedy). Picking is a small lever; averaging posteriors + LM is the big one.

### 4b. Character density — a cause of the size effect (new)
- At a fixed 40-px line height, real characters span **~2 to 13 CTC frames each** (1-line, big-writing patches median 2.2;
  2-line 3.6; 4–5-line 6.5–7.0; synthetic pool A 6.1). Error is **U-shaped** in it — AUG model, all val:
  < 3 frames/char **44.7** · 3–4 **26.6** · 4–5 28.0 · 5–6 32.3 · 6–8 37.7 · > 8 **42.0**. Big writing is squeezed (CTC
  has barely a frame per character, so it drops them); very sparse writing is the other extreme.
- Edge lines are not the cause: force-aligning each patch label to its lines gives top 35.7 / middle 32.1 / bottom 36.7, but
  2-line patches are bad on both lines (41.7 / 44.8) and 5-line patches good on all (23.6–30.2).
- **Test-time check (no retraining):** stretching every line by one factor always hurts (×0.7: 34.2, ×1.3: 31.2, ×1.6:
  34.8 vs 29.7 VAL-SEL); rescaling **each line** toward 3.5 frames per character (from a first-pass length estimate) helps:
  VAL-SEL 29.69 → 28.99, **VAL-REP 36.50 → 35.89**. Real but small — the model never saw that density in training.
  The stronger version (training with horizontal-stretch augmentation or on density-normalised lines) is untested.
- This also explains part of why 64-px lines helped (1.6× more frames per character).

## 5. The two failures — root causes and fixes
1. **EM1/EM2 crashed** with `OverflowError: Python integer 222 out of bounds for int8` in the Viterbi back-trace. Kaggle
   runs NumPy 2, where `int - int8` stays int8 (NEP 50); this PC runs NumPy 1.26, which widened it, so every local test
   passed. It fires only for labels longer than 63 characters. **Fix: one line** (`int(...)` around the back-pointer).
2. **COMBO and FINAL were starved.** The winners were BIG + H64 + AUG. The benchmark measured big@40 px and small@64 px
   but never **big@64 px**, and the time estimate silently fell back to small@40 (705 lines/s) while big@64 runs at **~190**.
   FINAL was given 6.1 M synthetic lines (≈ 8.5 h of work); it hit its time cap after 1.5 M lines and got **one** real
   epoch (val ~80). **Fixes:** benchmark every (model, height) any job can use; never fall back silently; cap FINAL's
   pretraining (zero-shot is flat beyond ~0.2 M lines anyway, §3) and give the time to fine-tuning and more models.

## 6. Better combinations from the checkpoints, offline (no GPU)
All 12 40-px character models were re-scored on this PC's CPU (every model within ±0.04 CER of the run's own numbers; the
notebook's ensemble reproduces exactly: VAL-SEL 24.05, VAL-REP 29.15). Everything below was chosen on VAL-SEL only.
- **Ensemble membership, chosen by LM-decoded CER** (the notebook took the top-3 by greedy CER): AUG 25.88 → +BIG 24.77 →
  +MIX 24.03 → +B60 23.75 → +C60 23.71 (stopped). Gains flatten after 3–4 models; many alternatives are within ±0.3 —
  ensembling *these* models is close to saturated.
- **LM order — the free gain the notebook missed** (its order-5/6 step was skipped for time):
  | pipeline (VAL-SEL) | order 4 | 5 | 6 | 7 | 8 |
  |---|---|---|---|---|---|
  | AUG alone (α 0.45, β 1.0) | 25.66 | 24.98 | **24.85** | 24.85 | 24.92 |
  | notebook ensemble AUG+BIG+B60 (0.3, 1.0) | 24.05 | 23.55 | **23.34** | 23.46 | 23.57 |
  | new ensemble AUG+BIG+MIX+B60 (0.3, 1.25) | 23.63 | 22.98 | 22.68 | **22.67** | 22.71 |
  The LM weights were inside the grid after all (α 0.3–0.45, β 1.0–1.25; lower β did not help). Order saturates at 6–7.
- **Greedy vs LM:** the 4-model average reads *worse* greedily (VAL-REP 37.86 vs AUG 36.50) but *better* with the LM —
  averaging flattens each model's spikes, and the LM turns the softer evidence into better choices.
- **New pipeline: AUG+BIG+MIX+B60, char 7-gram LM (α 0.3, β 1.25).** VAL-REP greedy 37.86 → **LM 27.63** (notebook 29.15).
  **TEST (scored once): CER 24.55 (95% CI 23.09–26.15)**, no-space 24.26, BLEU 12.99, chrF++ 52.1, 277/306 patches < 50 %.
  Against the notebook's pipeline on the same patches: **−1.35 CER (paired 95% CI −1.68 to −1.04)**, better on 188, worse on 79.
- Test-time density rescaling (§4b) adds a further −0.6 for a single greedy model (not yet combined with the ensemble/LM).

## 7. The next big step
Ranked by expected value for the project; "expected" numbers are *not verified*.
1. **Adopt the new pipeline now** (no GPU): test CER **24.55** is the current best result of the project.
2. **Notebook 06 — strong, diverse members** (GPU, ~5 h). Round 2 says *where* gains are: AUG (−4.0), BIG (−1.9),
   64 px (−1.3), ensembles + LM (large), LM order 6–7 (−0.7 to −1.0). Plan:
   - fix the two bugs (done in source: Viterbi int8; benchmark covers big@64 px, no silent fallback) and redesign FINAL:
     no giant pretraining (zero-shot is flat after ~0.2 M lines); spend the time on **more strong models** instead;
   - members: **BIG + AUG** (the untested pairing: BIG memorises, AUG regularises) × 3 seeds, small AUG × 2, and a 64-px
     family **BIG + H64 + AUG** × 2 (64-px models ensemble among themselves; frame rates differ from 40 px);
   - **AUG variants**, since augmentation is the biggest lever: horizontal **stretch** 0.6–1.6× (targets the density
     U-curve, §4b) and **stronger** augmentation (elastic, blur, noise, stroke dropout);
   - **EM re-alignment** on top of AUG (now fixed; H9 still open);
   - early stopping (best epochs were 15–34 of 57 → ~40 % of fine-tuning time was wasted);
   - decode every ensemble with the char 6/7-gram LM.
3. **The thesis track — synthetic realism.** Zero-shot transfer from font-rendered lines is stuck at ~80 CER for any
   amount of it; that ceiling is what better synthetic data must move, and **zero-shot real CER is a cheap, direct metric
   for synthetic-data quality**. Steps: (a) a quick realism test — compose new lines from **real glyph images cut out of
   training lines** with the reader's CTC alignment; (b) then the style-conditioned generator itself, judged by the same
   zero-shot and fine-tuned CERs against the font baseline.
4. **Out-of-distribution test (folder 18):** the strict tier (7 letters, 14 facsimile pages) needs its transcriptions
   digitised from the printed pages (OCR, then a team member proofreads every line). That is the only real answer to
   "does it work beyond MoDeTrans" — worth doing before the paper.
5. Small fixes: a guard against line-pitch harmonics (11 patches); a word-aware LM bonus to stop mid-word spaces.
