"""Referee Step 4: what kind of mistakes does a model make?

Aligns each prediction with its reference (Levenshtein, keeping the operations) and puts every edit in the
first category that fits, in this order:
  look-alike  a substitution inside one of the groups {भ म} {क फ} {ट ठ ढ} {न ण}
  virama      the edit touches the virama ् (the conjunct marker), so a conjunct is broken or invented
  anusvara    ं inserted or dropped
  vowel sign  the edit touches a vowel sign (U+093E-U+094C)
  other       everything else
When several alignments need the same number of edits, the one that pairs letters with letters and signs with
signs wins (भंग -> मग is "भ read as म, ं dropped", not "भ dropped, ं read as म"); after that substitutions
are preferred, then deletions, then insertions.
"""
import unicodedata
from collections import Counter

from modi.evaluation.metrics import _pairs

LOOKALIKE = [set('भम'), set('कफ'), set('टठढ'), set('नण')]
VIRAMA, ANUSVARA = '्', 'ं'
CATEGORIES = ['look-alike', 'virama', 'anusvara', 'vowel sign', 'other']


EDIT = 1000  # cost of one edit; a substitution between a letter and a sign costs one more, only to break ties


def _sub_cost(a, b):
    if a == b:
        return 0
    return EDIT + (unicodedata.category(a).startswith('M') != unicodedata.category(b).startswith('M'))


def align(ref, hyp):
    """Edit operations turning ref into hyp: a list of (op, ref_char, hyp_char), op in sub / del / ins.
    Always the minimum number of edits (texts are far shorter than EDIT characters)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i * EDIT
    for j in range(m + 1):
        d[0][j] = j * EDIT
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + _sub_cost(ref[i - 1], hyp[j - 1]), d[i - 1][j] + EDIT, d[i][j - 1] + EDIT)
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + _sub_cost(ref[i - 1], hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                ops.append(('sub', ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + EDIT:
            ops.append(('del', ref[i - 1], ''))
            i -= 1
        else:
            ops.append(('ins', '', hyp[j - 1]))
            j -= 1
    return ops[::-1]


def category(op, r, h):
    if op == 'sub' and any({r, h} <= group for group in LOOKALIKE):
        return 'look-alike'
    if VIRAMA in (r, h):
        return 'virama'
    if op != 'sub' and ANUSVARA in (r, h):
        return 'anusvara'
    if any('ा' <= c <= 'ौ' for c in (r, h) if c):
        return 'vowel sign'
    return 'other'


def error_types(refs, hyps):
    """Counter of edit categories over a whole set (every category present, possibly 0)."""
    counts = Counter({c: 0 for c in CATEGORIES})
    for r, h in _pairs(refs, hyps):
        counts.update(category(*op) for op in align(r, h))
    return counts
