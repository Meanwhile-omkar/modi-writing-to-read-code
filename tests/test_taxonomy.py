"""Error-type tests: 12 hand-made (reference, prediction) pairs whose answer is known in advance."""
import random

import editdistance
import pytest

from modi.evaluation.metrics import normalize
from modi.evaluation.taxonomy import align, error_types

CASES = [
    ('जर', 'नर', {'look-alike': 1}),                 # ज read as न (the most frequent swap on validation)
    ('बाग', 'हाग', {'look-alike': 1}),               # ब read as ह
    ('टप', 'ठप', {'look-alike': 1}),                 # ट read as ठ
    ('कमल', 'कमळ', {'look-alike': 1}),               # ल read as ळ
    ('की', 'कि', {'vowel sign': 1}),                 # long ी read as short ि
    ('काम', 'कम', {'vowel sign': 1}),                # ा dropped
    ('संत', 'सत', {'anusvara': 1}),                  # ं dropped
    ('सत', 'संत', {'anusvara': 1}),                  # ं inserted
    ('श्री', 'शरी', {'virama': 1}),                  # conjunct broken: श ् र ी -> श र ी
    ('बंग', 'हग', {'look-alike': 1, 'anusvara': 1}),  # two mistakes in one word
    ('भाव', 'माव', {'other': 1}),                    # भ/म: 0.8 % on validation, not a look-alike pair
    ('राजश्री', 'राजश्री', {}),                       # perfect reading: no errors
]


@pytest.mark.parametrize('ref, hyp, expected', CASES)
def test_hand_made_cases(ref, hyp, expected):
    counts = error_types([ref], [hyp])
    assert {k: v for k, v in counts.items() if v} == expected


def test_alignment_cost_equals_edit_distance():
    rng = random.Random(0)
    letters = 'कखगभमनणटठढ' + 'ािीुे' + '्ं '
    for _ in range(300):
        ref = ''.join(rng.choice(letters) for _ in range(rng.randint(1, 30)))
        hyp = ''.join(rng.choice(letters) for _ in range(rng.randint(0, 30)))
        assert len(align(normalize(ref), normalize(hyp))) == editdistance.eval(normalize(ref), normalize(hyp))
