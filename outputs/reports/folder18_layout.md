# Folder-18 layout: tilt and two columns

Out-of-domain CER on the 14 handwritten pages of the strict letters, read by the Stage-1 ensemble (4 models + char 7-gram LM, as in `s11_folder18_ood.py`). One change at a time: `base` = s11 (deskew within ±6°), `tilt12` = deskew within ±12°, `cols` = also cut pages with a gutter into two columns (left column read first). Same references for every variant: handwritten page p+1 against printed page p; letter 11 against half-page OCR of its printed pages (left column, then right). References are proofread for letter 3 and Tesseract drafts otherwise, so absolute numbers are rough; the differences between columns are the result. Made by `research/root-cause/offline/s12_folder18_layout.py`.

| letter | page | base CER | tilt12 CER | tilt12 + cols CER | angle base -> tilt12 | gutter |
|---|---|---|---|---|---|---|
| 3 | 15 | 16.3 | 16.3 | 16.3 | -0.5° -> -0.5° |  |
| 5 | 21 | 19.3 | 19.3 | 19.3 | -1.0° -> -1.0° |  |
| 8 | 41 | 15.8 | 15.8 | 15.8 | -0.5° -> -0.5° |  |
| 11 | 51 | 69.8 | 69.8 | 37.7 | +0.0° -> +0.0° | yes |
| 11 | 53 | 69.5 | 69.5 | 48.4 | -1.5° -> -1.5° | yes |
| 11 | 55 | 70.7 | 70.7 | 42.1 | -2.0° -> -2.0° | yes |
| 11 | 57 | 69.4 | 69.4 | 50.1 | +0.0° -> +0.0° | yes |
| 12 | 59 | 40.6 | 40.6 | 40.6 | -3.0° -> -3.0° |  |
| 12 | 61 | 43.7 | 40.3 | 40.3 | -6.0° -> -7.5° |  |
| 12 | 63 | 45.3 | 45.3 | 45.3 | -5.0° -> -5.0° |  |
| 12 | 65 | 53.2 | 56.3 | 56.3 | -6.0° -> -8.5° |  |
| 13 | 67 | 34.7 | 34.7 | 34.7 | +1.0° -> +1.0° |  |
| 13 | 69 | 29.8 | 29.8 | 29.8 | -3.0° -> -3.0° |  |
| 15 | 77 | 28.1 | 28.1 | 28.1 | -1.5° -> -1.5° |  |

| letter | base CER | tilt12 CER | tilt12 + cols CER |
|---|---|---|---|
| 3 | 16.3 | 16.3 | 16.3 |
| 5 | 19.3 | 19.3 | 19.3 |
| 8 | 15.8 | 15.8 | 15.8 |
| 11 | 68.6 | 68.6 | 44.9 |
| 12 | 45.9 | 45.8 | 45.8 |
| 13 | 32.0 | 32.0 | 32.0 |
| 15 | 28.1 | 28.1 | 28.1 |
| all | 44.2 | 44.2 | 36.0 |

## Findings (16 Oct)

- **Columns explain most of letter 11's gap.** Reading the left column, then the right, cuts letter 11 from 68.6 to
  44.9 CER (−23.7); every two-column page improves (−19 to −32). The gutter is found on exactly the four two-column
  pages (51, 53, 55, 57) and on no other page: a gap at least 15 px wide shared by 59-71 % of the text strips there,
  at most 50 % on single-column pages. That margin is narrow and was set on these 14 pages.
- **A wider deskew finds the larger tilts but does not help.** Pages 61 and 65 straighten to −7.5° and −8.5° instead of
  stopping at −6.0°; page 61 improves (43.7 → 40.3), page 65 gets worse (53.2 → 56.3), letter 12 is unchanged
  (45.9 → 45.8). Its lines probably curve rather than tilt as a whole, which one rotation per page cannot fix.
- **Single-column letters are untouched**: 3, 5, 8, 13, 15 score the same in all three variants.
- Over all 7 letters: 44.2 → 36.0. The absolute numbers differ from s11's because the references here are mostly
  Tesseract drafts (letter 11: half-page OCR); the comparison between the columns of this table is the result.
