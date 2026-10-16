# Font-based synthetic patches (Arm B)

This package makes fake handwritten Modi patches from a font, with their text, so the Reader can be
trained on real + synthetic data (Arm B). It is the cheap baseline that the learned Writer (Arm C) has to beat.

Nothing is stored or downloaded: each patch is drawn when it is asked for, on the CPU.

## The function

After the Setup below (the package is imported from `src/`):

```python
from modi.synthetic import make_patch

img, text = make_patch(seed)
```

| | |
|---|---|
| `img` | `numpy.uint8`, shape `(512, 1408)`, dark ink = 0 on white = 255. Same geometry as a real MoDeTrans patch after the Reader's resize: height 512, width 1402, padded with white to 1408. Almost black and white: the final resize leaves grey edge pixels (about 4 % of all pixels; the real scans are greyer). |
| `text` | The Devanagari label, in the same format as `text_train` in `assets/metadata/modetrans_text.csv`: only symbols from `charset.json`, fractions written as `<FRAC>`, words separated by single spaces. |
| `seed` | Any int. The same seed always gives byte-identical output; different seeds give different patches. |
| Speed | About 16 patches per second on one CPU core (Apple M1); 5-10 on an Intel Windows laptop. |
| Needs | Only files already in the repository (`assets/metadata/`, `assets/fonts/`). Not the MoDeTrans images. |

## Setup (Colab or local)

```python
!git clone https://github.com/Meanwhile-omkar/modi-writing-to-read-code.git
%cd modi-writing-to-read-code
!pip install -q -r requirements.txt
import sys; sys.path.insert(0, 'src')

from PIL import features
print(features.check('raqm'))   # must print True
```

Modi letters only join correctly when Pillow's text-shaping library (Raqm) is active. If it is not,
`make_patch` raises a `RuntimeError` instead of quietly drawing broken letters.

- Colab / Linux: if it prints `False`, run `!apt-get -qq install libfribidi0`, restart the runtime, check again.
- macOS: `brew install fribidi`, and add `export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` to `.venv/bin/activate`.
- Windows: the standard Pillow wheel already has Raqm. In PowerShell write `$env:PYTHONPATH='src'; python -m ...`
  where this page says `PYTHONPATH=src python -m ...`.

## Using it in Reader training (Reader handout, Step 9)

1. Training data per epoch: all 1,430 real training patches **plus** `n` synthetic ones. Start with
   `n = 1430` (1:1), then try `n = 4290` (1:3).
2. Use a **fixed pool of seeds**, `make_patch(base_seed + i)` for `i` in `range(n)`, the same for every
   epoch and every Reader seed. Runs stay comparable, and Arm C can later use the same number of patches.
3. Validation and test stay **100 % real**. Never validate or test on synthetic patches.
4. Treat a synthetic patch exactly like a real one after loading: same normalisation (`(x / 255 - 0.5) / 0.5`),
   same training augmentation, same tokeniser for the text (one token per code point, `<FRAC>` = one token).
   Do not resize it again; it is already 512 x 1408.
5. Use `num_workers >= 2` in the DataLoader so drawing never slows training. Each worker builds the text
   model once (about one second).

A map-style dataset that works with `torch.utils.data.DataLoader` as it is. `transform` is whatever turns
a real `(image, text)` pair into the model's input, so both kinds go through the same code:

```python
from modi.synthetic import make_patch

class SyntheticPatches:
    def __init__(self, n, transform, base_seed=1_000_000):
        self.n, self.transform, self.base_seed = n, transform, base_seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img, text = make_patch(self.base_seed + i)
        return self.transform(img, text)

# train_set = torch.utils.data.ConcatDataset([real_train_set, SyntheticPatches(1430, transform)])
```

On Windows (and macOS) DataLoader workers start fresh processes: with `num_workers >= 2`, define the dataset
class and `transform` at module level (no lambda) and start training under `if __name__ == '__main__':`.
In a notebook there, use `num_workers=0`.

## Single lines: `make_line`

```python
from modi.synthetic import make_line

img, text = make_line(seed, height=40)              # or height=64
img, text = make_line(seed, height=40, text_mode='ngram')
```

| | |
|---|---|
| `img` | `numpy.uint8`, shape `(height, width)`, dark ink = 0 on white = 255, cropped to the ink like the real lines of `notebooks/lr05` (`crop_line`). |
| width | Always suits the line Reader, checked at height 40: at most 1,024 px, and at least `2 * (1.2 * characters + 2)` px so CTC has room (lr05 `line_ok`). A text that does not fit is replaced by another one; nothing is squeezed. |
| `text` | `'chunk'` (default): a real 18-70-character piece of a training transcription, cut at spaces (4,422 pieces, median 33 characters). Invented text trained worse in Stage 1 (`results/README.md`). `'ngram'`: new text from the text source. |
| seed | The same `(seed, height, text_mode)` always gives the same output. |
| Speed | About 27 lines per second on one CPU core (Apple M1). |

A line is drawn like one line of a patch (words hanging from a long top stroke, a lead-in stroke on the left,
little or no word gap), stretched vertically like the patches (`TALL`), then slanted, warped, its strokes
thickened or thinned and thresholded, and cropped. **Fit-to-page check:** the line is drawn on a canvas with
room on every side, and `render_line` raises an error if any stroke still reaches the edge, so letters are
never cut off silently.

**Dotted circles.** Some transcriptions contain a vowel sign or virama that cannot attach to the letter before
it (`अै` typed for `ऐ`, `सुखरू्प`). Text shaping then draws a dotted circle (◌) that is not in the label. Both
`make_line` and `make_patch` load the font without its dotted-circle glyph, so the mark is drawn on its own.
`render.has_dotted_circle(text)` tells whether a text would have shown one. The release `armB-synthetic-v1`
was made before this fix: 29 of its 4,290 images (0.7 %) contain a dotted circle; the other 4,261 are
byte-identical to what `make_patch` draws now.

## Pre-generated set (download instead of generating)

The same pool, saved once as files: patch `i` is `make_patch(1_000_000 + i)`, 4,290 patches (226 MB zip),
attached to the GitHub release `armB-synthetic-v1`. Rows 0-1,429 are the 1:1 set, all 4,290 the 1:3 set.
Using the files gives byte-identical training data on every machine (generating on another machine can
differ by a few pixels if its library versions differ).

```python
!wget -q https://github.com/Meanwhile-omkar/modi-writing-to-read-code/releases/download/armB-synthetic-v1/armB_synthetic_4290.zip
!unzip -q armB_synthetic_4290.zip            # -> armB_synthetic_4290/syn_000000.png ... + labels.csv

import numpy as np, pandas as pd
from PIL import Image
labels = pd.read_csv('armB_synthetic_4290/labels.csv', keep_default_na=False)   # image_id, seed, text
pool = labels.iloc[:1430]                                                      # 1:1 set
img = np.asarray(Image.open('armB_synthetic_4290/' + pool.image_id[0]))        # uint8 (512, 1408)
text = pool.text[0]
```

To rebuild the set: `PYTHONPATH=src python -m modi.synthetic.export 4290 <folder>` (`export.py`).

## How a patch is made

1. **Text** (`text_source.py`). A character model (order 6: the next character given the previous five)
   learned from the training sentences only: the `train` split of `modetrans_text.csv`, minus 32
   sentences that contain a character with no Modi form. It writes new text whose length is drawn from
   the real training lengths. 95 % of the words it writes are real words from the training text, but
   98 % of its 20-character stretches are new combinations. Validation and test text are never read.
2. **Letters** (`mapping.py`). Devanagari is converted to Modi code points through the Unicode names
   (DEVANAGARI LETTER KA -> MODI LETTER KA). 70 of the 77 Devanagari symbols in `charset.json` have a
   Modi form; sentences with the other 7 (ऱ ़ ॉ ऑ ऽ ँ ॅ) are not used. Space, punctuation and the
   zero-width non-joiner stay as they are. `<FRAC>` is drawn as the North Indic quarter sign (U+A830).
3. **Drawing** (`render.py`). Noto Sans Modi (`assets/fonts/`, SIL Open Font License). 1-6 lines,
   words hanging from a long top stroke that starts left of the text, little or no gap between words,
   a slight tilt and indent per line. Then the whole patch is slanted, warped, its strokes thickened or
   thinned, roughened and thresholded to black and white, with a few specks.

`TALL` and `LINE_CHARS` in `render.py` were tuned so the fakes match the real training patches
(`calibrate.py`, below).

## How close to real

1,430 real training patches against 1,430 synthetic ones, both at 512 x 1402
(`outputs/reports/synthetic_calibration.md`). KS distance: 0 = same distribution, 1 = no overlap.

| statistic | real median | synthetic median | KS distance |
|---|---|---|---|
| ink share | 0.147 | 0.149 | 0.16 |
| stroke width (px) | 9.11 | 8.49 | 0.11 |
| text lines | 2 | 2 | 0.13 |
| text length | 91.5 | 91.5 | 0.03 |

## Known limits

- The letter shapes come from a printed font: upright, even, no cursive joins and no tall flourishes.
  This is the gap a learned Writer should close.
- After a consonant the font draws u and ū the same way (पु = पू), so synthetic patches cannot teach
  that difference.
- Drawing `<FRAC>` as the quarter sign is an assumption, still to be checked with someone who reads Modi.
  It affects 11 training sentences.
- Leakage: the handout asks for zero 20-character pieces copied from validation / test text. Because
  letters reuse fixed phrases (salutations, closings), a generator trained on training text alone still
  rebuilds some variants that occur only in validation / test: 87 of 5,000 texts. A generator that also
  read validation / test reaches 504 of 5,000. `tests/test_text_source.py` therefore checks that we stay
  far below that leaky generator. This change from "zero" needs the team's agreement.

## Files

| File | What it does |
|---|---|
| `__init__.py` | `make_patch(seed)`, `make_line(seed, height)` |
| `mapping.py` | Devanagari <-> Modi characters |
| `text_source.py` | new training-style text |
| `render.py` | draws a text as a patch or a line |
| `font_gate.py` | Step 0 check that the font draws Modi correctly |
| `calibrate.py` | Step 4 comparison of synthetic and real patches |
| `export.py` | saves a fixed set of patches as PNG + labels.csv (the release zip) |

Checks, from the repository root:

```bash
python -m pytest -q tests/test_mapping.py tests/test_text_source.py tests/test_make_patch.py tests/test_make_line.py
PYTHONPATH=src python -m modi.synthetic.calibrate     # 3-7 minutes; needs the MoDeTrans parquet at raw.modetrans_parquet in configs/paths.yaml
```
