"""Metric tests. Every expected number here is worked out by hand in the comment next to it."""
import random

import jiwer
import pytest

from modi.evaluation.metrics import (bleu, bootstrap_ci, catastrophic_rate, cer, cer_nospace, chrf_pp, median_cer,
                                     paired_bootstrap, per_image_cer, wer)


def test_one_wrong_letter_of_three():
    assert cer(['abc'], ['abd']) == pytest.approx(100 / 3)  # 1 substitution / 3 characters


def test_devanagari_vowel_signs_and_conjuncts():
    assert cer(['कमल'], ['कमळ']) == pytest.approx(100 / 3)  # क म ल: 1 substitution / 3
    assert cer(['की'], ['कि']) == pytest.approx(50)  # क ी -> क ि: 1 substitution / 2
    assert cer(['श्री'], ['शी']) == pytest.approx(50)  # श ् र ी -> श ी: 2 deletions / 4


def test_nfc_makes_two_spellings_of_one_letter_equal():
    # U+0958 (क़ as one code point) and क + nukta are the same letter; NFC turns both into क + ़
    assert cer(['क़'], ['क़']) == 0


def test_frac_is_one_character():
    assert cer(['१<FRAC>'], ['१']) == pytest.approx(50)  # 1 deletion / 2 characters, not 6 / 7
    assert cer(['१<FRAC>'], ['१<FRAC>']) == 0


def test_longer_prediction_can_exceed_100():
    assert cer(['ab'], ['abcdef']) == pytest.approx(200)  # 4 insertions / 2


def test_empty_prediction_is_100_and_empty_reference_is_an_error():
    assert cer(['abc'], ['']) == 100  # 3 deletions / 3
    with pytest.raises(ValueError):
        cer([''], ['abc'])
    with pytest.raises(ValueError):
        cer(['a', 'b'], ['a'])


def test_micro_average_weighs_long_images_more():
    refs = ['a' * 10, 'b' * 390]
    hyps = ['x' * 10, 'b' * 390]
    assert cer(refs, hyps) == pytest.approx(2.5)  # 10 edits / 400 characters (averaging per image would say 50)


def test_cer_nospace_and_wer():
    assert cer_nospace(['राज श्री'], ['राजश्री']) == 0  # only the editor's space differs
    assert cer(['राज श्री'], ['राजश्री']) == pytest.approx(100 / 8)  # 1 deletion / 8 code points
    assert wer(['a b c'], ['a x c']) == pytest.approx(100 / 3)  # 1 substituted word / 3


def test_median_and_catastrophic_rate():
    refs = ['aaaaa'] * 4
    hyps = ['aaaaa', 'aaaab', 'abbbb', '']  # per image: 0, 20, 80, 100
    assert list(per_image_cer(refs, hyps)) == [0, 20, 80, 100]
    assert median_cer(refs, hyps) == 50  # middle of 20 and 80
    assert catastrophic_rate(refs, hyps) == 50  # 2 of 4 images above 50


def test_chrf_and_bleu_are_100_for_a_perfect_reading():
    refs = ['राजश्री स्वामी', 'मौजे वाठार']
    assert chrf_pp(refs, refs) == pytest.approx(100)
    assert bleu(refs, refs) == pytest.approx(100)
    assert chrf_pp(refs, ['', '']) < 1


def test_agrees_with_jiwer_on_1000_random_pairs():
    rng = random.Random(0)
    letters = 'कखगघचजटडणतदनपबमयरलवशसह' + 'ािीुूेैोौं्' + ' '
    def text():
        return 'क' + ''.join(rng.choice(letters) for _ in range(rng.randint(0, 60))) + 'म'  # no edge spaces
    refs, hyps = [text() for _ in range(1000)], [text() for _ in range(1000)]
    assert cer(refs, hyps) == pytest.approx(100 * jiwer.cer(refs, hyps))


def test_bootstrap_interval_contains_the_score_and_is_reproducible():
    refs = ['abcdefghij'] * 50
    hyps = ['abcdefghij'] * 40 + ['abcdefghxx'] * 10  # 20 edits / 500 = 4 %
    score, low, high = bootstrap_ci(refs, hyps)
    assert score == pytest.approx(4) and low < 4 < high
    assert bootstrap_ci(refs, hyps) == (score, low, high)


def test_paired_bootstrap_tells_real_differences_from_none():
    refs = ['abcdefghij'] * 100
    good = ['abcdefghij'] * 90 + ['abcdefghix'] * 10  # 1 %
    bad = ['abcdefghij'] * 60 + ['abcdefxxxx'] * 40  # 16 %
    diff, low, high = paired_bootstrap(refs, bad, good)
    assert diff == pytest.approx(15) and low > 0  # clearly worse: interval excludes 0
    assert paired_bootstrap(refs, good, good) == (0, 0, 0)  # a system against itself: no difference at all
