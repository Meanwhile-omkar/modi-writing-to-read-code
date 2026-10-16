# Notebook 05 - round-2 hypotheses at GPU scale: results

*Generated automatically at the end of the run. GPUs: GPU 0: Tesla T4 (UUID: GPU-34a5e0d2-a1f5-008d-8b51-afb011c829f9); GPU 1: Tesla T4 (UUID: GPU-8546828b-7a64-1db0-e53d-9d54bbb69a26); budget scale 1.43; raqm shaping on; total time 299 min*

Rules fixed before the run: VAL-SEL (first 150 validation patches) chose every epoch, LM weight and the final pipeline; VAL-REP (the rest of validation) is only reported; TEST was scored once, for the anchor and the pipeline VAL-SEL chose.

## Headline: TEST

| pipeline | models | decoding | CER (95% CI) | CER no-space | WER | BLEU | chrF++ | patches <50% CER |
|---|---|---|---|---|---|---|---|---|
| anchor | FT_A0_s0 | greedy | 36.88 (35.29-38.61) | 35.04 | 96.00 | 1.71 | 33.6 | 229/306 |
| anchor | FT_A0_s0 | lm_beam (0.45, 1.0, 4) | 32.20 (30.59-33.96) | 31.64 | 89.17 | 6.57 | 41.9 | 245/306 |
| ensemble (FINAL) | AUG+BIG+B60 | greedy | 32.46 (30.82-34.29) | 31.25 | 84.86 | 3.36 | 38.2 | 243/306 |
| ensemble (FINAL) | AUG+BIG+B60 | lm_beam (0.3, 1.0, 4) | 25.91 (24.41-27.53) | 25.54 | 76.37 | 10.77 | 49.3 | 273/306 |

For reference: notebook 03 test CER 73.1 (BLEU 0.01); SPAN v1 val CER ~75; CPU line reader (X3a) val CER 42.4 greedy, ~41 with the LM (different validation subset).

## Hypothesis verdicts (VAL-REP)

A change counts as BETTER/WORSE only if the paired-bootstrap 95% CI excludes zero AND the difference is larger than twice the anchor's seed-to-seed standard deviation.

| id | question | variant vs control | metric | variant | control | difference (95% CI) | verdict |
|---|---|---|---|---|---|---|---|
| H24-GPU | Can GPU-scale training on real lines alone (no synthetic pretraining) learn to read? | NOPRE vs FT_A0_s0 | CER | 75.9 | 40.57 | +35.34 (+32.97 to +37.56) | **WORSE** |
| N10a | Fewer distinct synthetic lines (12k instead of 60k, same number of steps) - worse? | A12 vs FT_A0_s0 | CER | 42.89 | 40.57 | +2.32 (+1.55 to +3.11) | **WORSE** |
| N10b | More distinct synthetic lines (120k instead of 60k, same steps) - better? | A120 vs FT_A0_s0 | CER | 41.08 | 40.57 | +0.51 (-0.31 to +1.34) | **INCONCLUSIVE (within noise)** |
| N10c | Three times longer pretraining on 120k lines - better than the normal length? | PRE_LONG vs A120 | CER | 41.43 | 41.08 | +0.35 (-0.41 to +1.11) | **INCONCLUSIVE (within noise)** |
| N4 | Realistic rendering (no word gaps + continuous headline) instead of plain print - better? | B60 vs FT_A0_s0 | CER | 41.32 | 40.57 | +0.76 (+0.01 to +1.53) | **INCONCLUSIVE (within noise)** |
| N12b | Synthetic text sampled from a character LM (new word forms) instead of training-label text - better? | C60 vs FT_A0_s0 | CER | 42.14 | 40.57 | +1.57 (+0.75 to +2.40) | **WORSE** |
| N11 | A 2x wider model (4x the parameters) - better now that learning works? | BIG vs FT_A0_s0 | CER | 38.7 | 40.57 | -1.86 (-2.71 to -1.03) | **BETTER** |
| N14 | Akshara (syllable) targets instead of Unicode characters - better? | AKS vs FT_A0_s0 | CER | 41.68 | 40.57 | +1.11 (+0.24 to +1.98) | **WORSE** |
| N5 | No spaces in the targets (Modi has no word gaps) - better at the letters themselves? | NS vs FT_A0_s0 | CER_nospace | 39.97 | 39.53 | +0.44 (-0.37 to +1.23) | **INCONCLUSIVE (within noise)** |
| N7 | 64-px lines instead of 40-px (small marks: virama, anusvara) - better? | H64 vs FT_A0_s0 | CER | 39.27 | 40.57 | -1.30 (-2.31 to -0.37) | **BETTER** |
| N9 | Keep synthetic lines in the mix during fine-tuning - better than real lines only? | MIX vs FT_A0_s0 | CER | 41.42 | 40.57 | +0.85 (+0.06 to +1.66) | **INCONCLUSIVE (within noise)** |
| N6 | Augment real lines during fine-tuning - better? | AUG vs FT_A0_s0 | CER | 36.53 | 40.57 | -4.03 (-4.80 to -3.26) | **BETTER** |
| H9/N8a | One CTC re-alignment of the line labels during fine-tuning - better? | EM1 vs FT_A0_s0 | CER | | | | NOT TESTED: not run (EM1: failed, FT_A0_s0: ok) |
| N8b | A second re-alignment round - better than one? | EM2 vs EM1 | CER | | | | NOT TESTED: not run (EM2: failed, EM1: failed) |
| H11-line | Three times more fine-tuning epochs - still improving? | FT_LONG vs FT_A0_s0 | CER | 40.59 | 40.57 | +0.02 (-0.59 to +0.61) | **INCONCLUSIVE (within noise)** |
| COMBO | All winning changes together (normal length) - better than the anchor? | COMBO vs FT_A0_s0 | CER | 81.77 | 40.57 | +41.20 (+39.49 to +42.84) | **WORSE** |
| FINAL | The final recipe trained as long as the budget allows - how much better than at normal length? | FINAL_s0 vs COMBO | CER | 81.67 | 81.77 | -0.10 (-0.70 to +0.51) | **INCONCLUSIVE (within noise)** |

Run-to-run noise: anchor VAL-REP CER over fine-tune seeds = [40.565649438466515, 41.212441935673546, 41.37120009407891] (sd 0.42670375087428924).

## Decoding (N3) and ensembles (N15), VAL-REP

| pipeline | models | LM (alpha, beta, order) chosen on VAL-SEL | greedy CER | LM CER | LM CER no-space | LM BLEU |
|---|---|---|---|---|---|---|
| anchor | FT_A0_s0 | (0.45, 1.0, 4) | 40.55 | 35.53 | 35.88 | 5.98 |
| best_single | AUG | (0.45, 1.0, 4) | 36.5 | 31.6 | 31.87 | 9.22 |
| combo | COMBO | (0.45, 1.0, 4) | 81.76 | 79.76 | 79.36 | 0.0 |
| final_s0 | FINAL_s0 | (0.45, 1.0, 4) | 81.66 | 79.97 | 80.2 | 0.0 |
| final_s1 | FINAL_s1 | (0.45, 1.0, 4) | 83.32 | 81.1 | 82.04 | 0.0 |
| ensemble | AUG+BIG+B60 | (0.3, 1.0, 4) | 36.73 | 29.15 | 29.6 | 10.22 |

**N5 space restoration:** no-space model + LM spaces (gamma -1.0): VAL-REP CER 40.87 (no-space 39.97).

## Every job

*Reads?* = CER against each patch's own transcription vs against a patch with the same amount of text (VAL-REP). A reader is far better on its own patch (CPU line reader: own patch better on 149/150); a model that only learned how much text a page holds is a coin flip (notebook 03: 153/306; SPAN v1: 174/306).

| job | status | what changed | val-sel CER | val-rep CER | val-rep no-space | train-150 CER | reads? own vs same-length (own better) | epochs (best) | minutes |
|---|---|---|---|---|---|---|---|---|---|
| AUG | ok | anchor fine-tune with augmented real lines | 29.69 | 36.53 | 35.65 | 32.39 | 38.3 vs 80.7 (99.4%) | 57 (34) | 5.46 |
| BIG | ok | 2x wider CRNN (conv 64-128-256-256-256, BiLSTM 2x256) | 31.19 | 38.70 | 37.84 | 21.70 | 40.3 vs 80.8 (98.7%) | 57 (20) | 31.91 |
| H64 | ok | 64-px line images (pretrain + fine-tune) | 31.52 | 39.27 | 38.11 | 33.51 | 41.1 vs 80.6 (98.1%) | 57 (28) | 24.94 |
| FT_A0_s0 | ok | ANCHOR: PT_A60 -> real lines, proportional line labels, seed 0 | 33.58 | 40.57 | 39.53 | 35.82 | 42.5 vs 80.4 (97.5%) | 57 (17) | 5.32 |
| FT_LONG | ok | anchor fine-tune for 3x the epochs | 33.87 | 40.59 | 39.78 | 34.62 | 42.0 vs 80.2 (99.4%) | 171 (20) | 13.59 |
| A120 | ok | pretrain on 120,000 distinct synthetic lines (same steps) | 33.63 | 41.08 | 40.56 | 36.65 | 43.0 vs 80.1 (98.1%) | 57 (15) | 13.41 |
| FT_A0_s1 | ok | anchor, fine-tune seed 1 (measures run-to-run noise) | 34.41 | 41.21 | 40.35 | 32.09 | 43.0 vs 81.1 (98.7%) | 57 (23) | 5.34 |
| B60 | ok | pretrain on 60,000 REALISTIC synthetic lines (no word gaps + continuous headline) | 33.21 | 41.32 | 41.00 | 35.42 | 43.4 vs 79.9 (99.4%) | 57 (17) | 12.82 |
| FT_A0_s2 | ok | anchor, fine-tune seed 2 | 34.25 | 41.37 | 40.92 | 34.40 | 43.0 vs 80.6 (98.7%) | 57 (19) | 5.26 |
| MIX | ok | anchor fine-tune with one synthetic batch after every real batch | 33.47 | 41.42 | 40.44 | 32.19 | 43.3 vs 81.3 (99.4%) | 57 (24) | 9.46 |
| PRE_LONG | ok | 3x longer pretraining on 120,000 lines | 34.27 | 41.43 | 40.63 | 34.97 | 43.2 vs 80.9 (98.7%) | 57 (19) | 26.23 |
| AKS | ok | akshara (syllable) targets, pretrain + fine-tune | 34.43 | 41.68 | 41.04 | 29.83 | 43.4 vs 80.8 (99.4%) | 57 (29) | 13.92 |
| C60 | ok | pretrain on 60,000 lines whose TEXT is sampled from a char 4-gram of the training labels | 33.48 | 42.14 | 41.66 | 33.48 | 43.9 vs 80.6 (98.7%) | 57 (21) | 14.27 |
| A12 | ok | pretrain on only 12,000 distinct synthetic lines (same steps as the anchor) | 34.88 | 42.89 | 42.35 | 35.40 | 44.6 vs 80.9 (99.4%) | 57 (21) | 13.24 |
| NS | ok | no spaces in any target, pretrain + fine-tune (spaces put back by the LM afterwards) | 41.21 | 48.32 | 39.97 | 47.20 | 49.7 vs 83.9 (100.0%) | 57 (15) | 13.26 |
| NOPRE | ok | CONTROL: real lines only, no synthetic pretraining, 2x the epochs | 76.19 | 75.90 | 80.62 | 77.87 | 75.9 vs 75.7 (50.3%) | 114 (44) | 10.25 |
| FINAL_s0 | ok (time-capped) | FINAL model, seed 0: the winning recipe on the whole A pool (60,000 lines), trained as long as the time allows | 79.50 | 81.67 | 82.77 | 85.62 | 83.4 vs 87.6 (79.0%) | 1 (1) | 149.38 |
| PT_A60 | ok | shared pretraining: 60,000 plain synthetic lines (X4 style) | 79.67 | 81.75 | 82.82 | 85.01 | 83.1 vs 87.6 (80.9%) | - (-) | 8.22 |
| COMBO | ok (time-capped) | COMBINATION of big=True, h64=True, aug=True - normal length | 79.91 | 81.77 | 82.38 | 83.64 | 83.3 vs 87.6 (77.7%) | 1 (1) | 25.93 |
| FINAL_s1 | ok (time-capped) | FINAL model, seed 1: the winning recipe on the whole A pool (60,000 lines), trained as long as the time allows | 80.97 | 83.31 | 85.70 | 87.16 | 85.2 vs 89.3 (75.8%) | 1 (1) | 123.39 |
| EM1 | failed | anchor + one CTC re-alignment of the line labels after 1/3 of fine-tuning | nan | nan | nan | nan | - | - (-) | 1.79 |
| EM2 | failed | anchor + re-alignment after 1/3 and 2/3 of fine-tuning | nan | nan | nan | nan | - | - (-) | 1.79 |

## Error anatomy of the anchor (N1, N2)

Errors on all validation: deletions 40.2%, insertions 14.4%, substitutions 45.5%. Most missed: [[' ', 1077], ['ा', 681], ['्', 448], ['ं', 444], ['र', 331], ['े', 274], ['त', 204], ['ी', 197], ['म', 168], ['न', 158], ['य', 157], ['ल', 154]]. Top confusions: [['ि->ी', 207], ['ी->ि', 149], ['ा-> ', 98], ['ि-> ', 92], ['अ->आ', 82], ['ज->न', 79], ['ा->ो', 73], ['्-> ', 63]].

| patch feature | Spearman with CER | Spearman with deletion rate | CER by quartile (low -> high) |
|---|---|---|---|
| frames_per_char | +0.09 | +0.12 | {'Q1 low': 42.2, 'Q2': 36.4, 'Q3': 37.1, 'Q4 high': 45.3} |
| pitch | +0.20 | +0.13 | {'Q1 low': 35.8, 'Q2': 40.8, 'Q3': 40.2, 'Q4 high': 44.3} |
| n_lines | -0.31 | -0.18 | {'Q1 low': 45.5, 'Q2': 38.6, 'Q3': 42.3, 'Q4 high': 34.6} |
| label_len | -0.32 | -0.22 | {'Q1 low': 46.1, 'Q2': 40.4, 'Q3': 41.4, 'Q4 high': 33.2} |
| ink | -0.11 | -0.25 | {'Q1 low': 43.7, 'Q2': 38.2, 'Q3': 39.7, 'Q4 high': 39.5} |
| stroke | -0.11 | -0.02 | {'Q1 low': 46.5, 'Q2': 37.2, 'Q3': 34.6, 'Q4 high': 42.7} |

## Files
`summary.csv` (every job), `runs/<job>/` (log.txt, history.csv, result.json, predictions, best.pt), `lm_grid_val_sel.csv`, `pipelines_val.csv`, `test_predictions_*.csv`, `figures/`, `logs.txt` (everything printed).
