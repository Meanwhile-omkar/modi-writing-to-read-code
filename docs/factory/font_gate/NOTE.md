# Font go / no-go (Factory Step 0)

12 October 2026 · Udit Sevak · draft written with an AI assistant (Claude); to be reviewed and reworded by me

**Decision: GO with Noto Sans Modi Regular** (Google Noto, SIL Open Font License 1.1, in `assets/fonts/` with `OFL.txt`).

Reproduce from the repository root: `source .venv/bin/activate && PYTHONPATH=src python -m modi.synthetic.font_gate`
(writes the three figures to outputs/figures/font_gate/, copied here, and prints the numbers below). Only training patches are read.

## What works
- **Every Modi character is drawn.** All 79 assigned code points (U+11600–U+1165F) are in the font and none renders blank (`01_charset.png`). The shapes match the Unicode 18.0 Modi code chart, checked by eye cell by cell.
- **Joining works.** With the Raqm layout engine the font forms conjuncts (श्र, क्ष्म, न्य), repha (र्व, र्ज) and Modi's consonant + vowel ligatures (मा, कु, हा). 11 of 12 sample words change when shaping is switched on; the one that does not, अलफ, has nothing to join (`02_words.png`). All 12 words occur in the training text.
- **It covers our alphabet.** Of the 77 Devanagari symbols in `charset.json`, 70 map to Modi. 1,398 of the 1,430 training patches (97.8 %) can be drawn completely.

## Limits
1. **7 symbols have no Modi code point:** ऱ (in 16 training patches), nukta ़ (8), ॉ (6), ऑ (3), ँ (1), ॅ (1), ऽ (0) — 32 training patches in all. Step 1 drops sentences that contain them and lists them; nothing is deleted silently.
2. **Joining depends on the set-up.** Pillow only shapes Modi when Raqm is active. On this Mac that needs Homebrew's `fribidi` (enabled in `.venv/bin/activate`); the script refuses to run without it instead of drawing unjoined letters. Must be checked on Colab before the Reader imports `make_patch`.
3. **u and ū look the same after a consonant** (पु = पू, कु = कू). Font patches cannot teach this difference, while the labels still contain it. To check with someone who reads Modi whether real handwriting distinguishes them.
4. **It looks printed, not handwritten** (`03_real_vs_font.png`): upright, even stroke width, gaps between words, no top stroke. The real patch has no gaps between words, a long horizontal stroke across each line, slant and joined letters. Step 3 should therefore draw lines without word gaps (the label keeps its spaces, exactly as in the real data), add the horizontal stroke, and tilt each line.

## Open
- Other free Unicode Modi fonts not yet surveyed. More fonts would give Arm B more style variety. (The design document also names "Marathi CursiveT", which is a Devanagari font, so it would need checking.)
- The non-Devanagari symbols in `charset.json` (space, `( ) , - . / :`, zero-width non-joiner) and `<FRAC>` each need a decision in Step 1.
