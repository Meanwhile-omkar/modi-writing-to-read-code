# Image-processing audit — what the model is actually handed

Every image in `FYP-modi-script-data/MoDeTrans-dataset` is edited several times before the network
sees it. This folder renders each of those edits so they can be checked by eye rather than trusted.
It was produced by `scripts/19_image_pipeline_audit.py` and covers all **2,043 patches**;
the galleries show a fixed random sample of 8 (listed in `00_sample.txt`) plus the
worst outlier of every automatic check.

Start with **`00_pipeline_overview.png`** — one patch through all six stages on a single sheet.

## The stages

| Folder | What is done to the image | What can go wrong |
|---|---|---|
| `01_original` | nothing; the patch as downloaded | — |
| `02_deskew` | rotate to the angle where the text lines are most horizontal (±8°, 0.5° steps) | a page slanted by more than 8° stays slanted |
| `03_line_detection` | count ink per row, cut the page into text lines | lines so close together that the ink never drops between them |
| `04_line_crops` | crop each line tight to its ink, left and right | a stray mark far from the text inflates the crop |
| `05_resized_lines` | scale each line to exactly 64 px tall | a grown band leaves the writing small inside the strip |
| `06_strip_assembled` | lay the lines end to end with a 20 px gap | a very long page hits the 8,192 px cap and is squeezed |
| `07_padding_v1_vs_v2` | pad to a common width for batching | padding to a fixed width wastes most of the image |
| `08_augmentation` | training-only distortions | augmentation thinning strokes until they vanish |
| `06b` (in cell 7) | stretch any strip with under 3.2 output slots per character | none; this repairs the cramped cases |
| `09_model_input` | scale to [-1, +1]; count the CTC output slots | too few slots for the label, which silently breaks CTC |
| `10_outliers` | one picture per flagged patch, with the reason in the caption | — |

## What this audit found, and what was changed

**Failure 1 — found at stage 07 (padding).** The notebook reported that moving to a single strip cut
blank space from 87% to 61%. That number was measured on the strip *before* padding. Every strip was
then padded out to a fixed 6144 px, and for a median patch only about 3,300 px held text — so the
image the model really received was roughly **80% blank paper**, and CTC was told to read all 768
output slots including the empty ones.
*Change:* each batch is now padded only to its widest member (as FLAME does), samples of similar
width are grouped into the same batch, and CTC is given each sample's true length. Padding overhead
fell from 53% to **0.4%**, and a batch is about **2.1× smaller**, so epochs also run faster.

**Failure 2 — found at stage 03 (line detection), and much more serious.** On dense or slanted pages
the ink profile never dropped below the 22% threshold between lines, so the detector returned **one
band covering the whole page**. That band was then scaled to 64 px tall, crushing five lines of Modi
into a single strip. One patch had a **246-character label inside a 183 px image**. 150 patches
were affected and nothing in the pipeline complained.
*Change:* skew is corrected before projecting (1,349 of 2,043 patches needed it), and any band
taller than about 1.6 lines is split at its own deepest ink valleys rather than at equal spacing, so
a cut never lands in the middle of a line. Patches seen as a single line fell from
150 to 6.

**Failure 3 — found at stage 03 as well, and the one with the biggest ceiling effect.** Once the
pictures were being rendered it became visible that whole lines were sitting *outside* every
detected band — page headers, dates, short final lines — along with the tops and tails of ordinary
letters. Measured across 400 patches, a median patch handed the model only **90% of its own ink**,
half of all patches lost more than 10%, and the worst lost a third. The transcription still listed
those characters, so the model was being trained to produce text for which it had been shown no
image at all. That puts a floor under the error rate that no amount of extra training can pass.
*Change:* the ink threshold for a line core was lowered to 0.08, and every line is now **grown until
it meets its neighbour at the valley between them** (capped at 0.35 of a line height), so every
inked row belongs to exactly one line. Ink reaching the model is now a median of
**99.1%** (worst case 86%); 68 patches still lose more than 5%.
The cap of 0.35 was chosen by measurement: growing further bought under half a percent more ink
while making the writing steadily smaller inside the strip. At 0.35 the writing still fills a
median of **64 of the strip's 64 px**, so nothing was given up for it.

**Effect of fixing failure 2 on the thing that matters.** CTC needs several output slots per
character. Before: a median of **2.6 slots per character**, below the healthy 3–8 range, with
**77 patches having fewer slots than their label needed at all** — for those, CTC returns zero loss
and the patch teaches the model nothing, silently. After: a median of **9.6 slots per
character** and **0 patches** short.

## Checks that run over all 2,043 patches

| Check | Patches flagged | What it means |
|---|---|---|
| `02_deskew__angle_at_search_limit` | 4 (0.2%) | skew pinned at the +-8 deg search limit, so the page may still be slanted |
| `03_lines__old_finder_merged_everything` | 150 (7.34%) | v1 returned one band for a multi-line page |
| `03_lines__still_single_band` | 6 (0.29%) | v2 STILL returns one band - needs attention |
| `03b_lines__ink_left_outside_every_line` | 68 (3.33%) | ink the model never sees, although its characters are in the label |
| `04_crops__near_blank_line` | 0 (0.0%) | a detected line is almost pure paper |
| `04_crops__extreme_aspect` | 0 (0.0%) | a crop is far wider than tall; probably two lines joined |
| `05_resize__writing_too_small_in_strip` | 0 (0.0%) | after growing the band, the writing fills under half the 64 px strip |
| `06_strip__hit_the_width_cap` | 190 (9.3%) | strip longer than the 8192 px cap, so it was squeezed horizontally |
| `07_ctc__too_few_slots` | 0 (0.0%) | fewer output slots than the label needs; CTC would return zero loss |
| `07_ctc__slots_per_char_low` | 0 (0.0%) | still below 3 slots per character after widening |
| `06b_widened_to_fit_the_label` | 142 (6.95%) | strip stretched horizontally so CTC has room; not a failure, a repair |

Raw numbers for every patch are in `stage_stats.csv`; the counts above are in `outlier_checks.csv`.

## Measured constants

| | |
|---|---|
| typical line height | 56 px |
| strip height | 64 px |
| strip width cap | 8,192 px |
| gap inserted between lines | 20 px |
| output slots | one per 4 px of width |
| median strip width | 3,888 px |
| median slots per character | 9.6 |
