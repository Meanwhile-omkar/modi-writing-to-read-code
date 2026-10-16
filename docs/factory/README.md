# Factory evidence (Arm B, font-based synthetic data)

Udit Sevak · Factory steps 0 and 4 · code in `src/modi/synthetic/`

## Step 0 · font go / no-go: GO with Noto Sans Modi

`font_gate/NOTE.md` has the decision and its limits. The renders:

| file | shows |
|---|---|
| `font_gate/01_charset.png` | all 79 Modi code points drawn by the font, labelled with code point and Devanagari twin |
| `font_gate/02_words.png` | 12 archaic words: Devanagari, Modi without shaping, Modi with shaping (11 of 12 change) |
| `font_gate/03_real_vs_font.png` | a real training patch (MoDeTrans 10.jpg) above the same text in the font |

Rebuild: `PYTHONPATH=src python -m modi.synthetic.font_gate`.

## Step 4 · calibration against real patches

`calibration/synthetic_calibration.png`: histograms of four statistics for the 1,430 real training patches and
1,430 synthetic patches (`make_patch(0..1429)`), both at the Reader input size. Numbers in
`outputs/reports/synthetic_calibration.md`. Rebuild: `PYTHONPATH=src python -m modi.synthetic.calibrate`.

| statistic | real median | synthetic median | KS distance |
|---|---|---|---|
| ink share | 0.147 | 0.149 | 0.16 |
| stroke width (px) | 9.11 | 8.49 | 0.11 |
| text lines | 2 | 2 | 0.13 |
| text length | 91.5 | 91.5 | 0.03 |

`calibration/real_vs_fake.png`: four real training patches (left) next to four synthetic ones (right).

The settings `TALL` (letter height stretch) and `LINE_CHARS` (characters per line) in `render.py` were chosen
by trying a few values on 400 patches and keeping the one with the lowest KS distances. Two things the
calibration caught on the way: words of short texts were each put on their own line (up to 11 lines instead
of at most 6; fixed with `render.wrap`), and the first width rule made letters too small.

## What still differs

The four statistics match, but they describe ink amount and layout, not letter shapes. What is still different,
from looking at the pictures:

1. **Letter shapes.** The glyphs come from a printed font: upright, even and separate. Real writers join
   neighbouring letters, slant them, and draw tall loops and long strokes above and below the line
   (`real_vs_fake.png`, left column). Warping and slanting the whole patch does not create these.
2. **Pen.** Real strokes change width inside one stroke and taper at the ends; ours have one width per patch.
3. **Paper.** Many real scans have speckle, bleed-through, grey backgrounds (45 of the 1,430 training patches
   are colour scans) and cut-off letters at the edges. Ours are clean white with a few specks.
4. **Lines.** Real lines curve and drift; ours are straight with a small tilt.
5. **Two known gaps in the text.** The font draws u and ū the same after a consonant, and `<FRAC>` is drawn as
   the quarter sign, which is still an assumption.

How large this gap is, measured by the team: a Reader trained only on font-rendered lines reads real
handwriting at about 77-84 CER, whether it saw 0.1 M or 1.5 M lines (`results/README.md`). More font data does
not close the gap; only more realistic handwriting can, which is what the learned Writer (Arm C) is for.
The realism classifier (Factory step 6) will put one number on it.
