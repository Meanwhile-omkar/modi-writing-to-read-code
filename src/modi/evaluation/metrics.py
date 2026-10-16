"""Referee Step 2: the scores every model is judged by (docs/BASELINE_DESIGN.md, Section 7.2-7.3).

Rules, applied to every function:
- Text is Unicode NFC-normalised first, so the same letter typed two ways counts as the same letter.
- <FRAC> counts as ONE character (it is one symbol in the alphabet, not six letters).
- An empty reference raises ValueError: there is nothing to score against.
- CER and WER are micro-averaged: all edits across the set divided by all reference characters (words),
  so a 10-character patch weighs less than a 400-character one. Values are percentages and can exceed 100.
"""
import statistics
import unicodedata

import editdistance
import numpy as np
import sacrebleu

FRAC = '<FRAC>'
_FRAC_CHAR = ''  # one private-use code point stands in for <FRAC> while counting


def normalize(s):
    return unicodedata.normalize('NFC', s).replace(FRAC, _FRAC_CHAR)


def _pairs(refs, hyps):
    if len(refs) != len(hyps):
        raise ValueError(f'{len(refs)} references but {len(hyps)} predictions')
    if not refs:
        raise ValueError('nothing to score')
    pairs = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps)]
    if any(not r for r, _ in pairs):
        raise ValueError('empty reference')
    return pairs


def char_counts(refs, hyps):
    """Per image: (character edits, reference length) as two numpy arrays. Basis of CER and its bootstrap."""
    pairs = _pairs(refs, hyps)
    return (np.array([editdistance.eval(r, h) for r, h in pairs]), np.array([len(r) for r, _ in pairs]))


def cer(refs, hyps):
    edits, lengths = char_counts(refs, hyps)
    return 100 * edits.sum() / lengths.sum()


def cer_nospace(refs, hyps):
    """CER ignoring spaces: in Modi the word boundaries are chosen by the editor, not written on the page."""
    return cer([r.replace(' ', '') for r in refs], [h.replace(' ', '') for h in hyps])


def wer(refs, hyps):
    pairs = _pairs(refs, hyps)
    if any(not r.split() for r, _ in pairs):
        raise ValueError('reference without words')
    edits = sum(editdistance.eval(r.split(), h.split()) for r, h in pairs)
    return 100 * edits / sum(len(r.split()) for r, _ in pairs)


def per_image_cer(refs, hyps):
    edits, lengths = char_counts(refs, hyps)
    return 100 * edits / lengths


def median_cer(refs, hyps):
    return float(statistics.median(per_image_cer(refs, hyps)))


def catastrophic_rate(refs, hyps, threshold=50):
    """Share of images (%) read with CER above 50 %: shows collapse that the average hides."""
    return float(100 * (per_image_cer(refs, hyps) > threshold).mean())


def chrf_pp(refs, hyps):
    """chrF++ (character n-grams plus word bigrams), 0-100, higher is better."""
    pairs = _pairs(refs, hyps)
    return sacrebleu.corpus_chrf([h for _, h in pairs], [[r for r, _ in pairs]], word_order=2).score


def bleu(refs, hyps, keep_spaces=False):
    """Character-level BLEU, 0-100. Only for comparison with the MoDeTrans paper (C01).
    sacreBLEU's 'char' tokenizer drops spaces, so a missing or extra space costs nothing; keep_spaces=True makes
    each space a token of its own."""
    pairs = _pairs(refs, hyps)
    if keep_spaces:
        pairs = [(r.replace(' ', '▁'), h.replace(' ', '▁')) for r, h in pairs]
    return sacrebleu.corpus_bleu([h for _, h in pairs], [[r for r, _ in pairs]], tokenize='char').score


def bootstrap_ci(refs, hyps, n=10_000, seed=0, level=95):
    """CER and its confidence interval: re-draw the test images with replacement n times, recompute CER."""
    edits, lengths = char_counts(refs, hyps)
    idx = np.random.default_rng(seed).integers(0, len(edits), (n, len(edits)))
    draws = 100 * edits[idx].sum(1) / lengths[idx].sum(1)
    tail = (100 - level) / 2
    return cer(refs, hyps), float(np.percentile(draws, tail)), float(np.percentile(draws, 100 - tail))


def paired_bootstrap(refs, hyps_a, hyps_b, n=10_000, seed=0, level=95, refs_b=None):
    """CER(a) - CER(b) and its confidence interval, with the SAME re-drawn images for both systems.
    The difference is real only if the interval excludes 0. Pass refs_b (run b's own reference column) to check
    that both runs list the same images in the same order; results files are not guaranteed to share a row order."""
    if refs_b is not None and [normalize(r) for r in refs_b] != [normalize(r) for r in refs]:
        raise ValueError('the two runs are not aligned: sort or merge both on image_id first')
    edits_a, lengths = char_counts(refs, hyps_a)
    edits_b, _ = char_counts(refs, hyps_b)
    idx = np.random.default_rng(seed).integers(0, len(lengths), (n, len(lengths)))
    total = lengths[idx].sum(1)
    draws = 100 * (edits_a[idx].sum(1) - edits_b[idx].sum(1)) / total
    tail = (100 - level) / 2
    diff = 100 * (edits_a.sum() - edits_b.sum()) / lengths.sum()
    return float(diff), float(np.percentile(draws, tail)), float(np.percentile(draws, 100 - tail))
