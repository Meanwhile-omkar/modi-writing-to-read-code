import json
import unicodedata
from pathlib import Path

import pandas as pd
import pytest

from modi.synthetic.mapping import FRAC, dev_to_modi, modi_to_dev, unmappable

META = Path(__file__).resolve().parents[1] / 'assets/metadata'
# The 7 Devanagari symbols in charset.json with no Modi code point (Factory handout, Step 1).
NO_MODI_FORM = set('ऱ़ॉऑऽँॅ')


def training_sentences():
    split = json.load(open(META / 'splits.json', encoding='utf-8'))['modetrans']['assignment']
    text = pd.read_csv(META / 'modetrans_text.csv', keep_default_na=False)
    return text[text.image_id.map(split) == 'train'].text_train.tolist()


def test_hand_checked_examples():
    # श्री = SHA + VIRAMA + RA + VOWEL SIGN II (Unicode chart U11600)
    assert dev_to_modi('श्री') == '\U0001162B\U0001163F\U00011628\U00011632'
    assert dev_to_modi('१२।') == '\U00011651\U00011652\U00011641'
    assert dev_to_modi('राज श्री') == dev_to_modi('राज') + ' ' + dev_to_modi('श्री')


def test_frac_is_a_fraction_sign_and_comes_back():
    assert dev_to_modi('१<FRAC><FRAC>') == '\U00011651꠰꠰'
    assert modi_to_dev('\U00011651꠱') == '१' + FRAC


def test_exactly_the_seven_known_symbols_are_unmappable():
    symbols = json.load(open(META / 'charset.json', encoding='utf-8'))['symbols']
    assert set().union(*(unmappable(s) for s in symbols)) == NO_MODI_FORM
    with pytest.raises(ValueError):
        dev_to_modi('ऱ')


def test_output_is_only_modi_or_kept_symbols():
    out = dev_to_modi('अखंडित लक्ष्मी (१/२), उपरी.')
    assert all(unicodedata.name(c).startswith('MODI ') or c in ' (),-./:' for c in out)


def test_round_trip_on_every_training_sentence():
    sentences = training_sentences()
    dropped = [s for s in sentences if unmappable(s)]
    for s in sentences:
        if not unmappable(s):
            assert modi_to_dev(dev_to_modi(s)) == s, s
    # Every dropped sentence is dropped only for one of the 7 known symbols, and they are rare (~2 %).
    assert set().union(*(unmappable(s) for s in dropped)) <= NO_MODI_FORM
    assert len(dropped) / len(sentences) < 0.03
