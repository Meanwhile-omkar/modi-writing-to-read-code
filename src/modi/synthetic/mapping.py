"""Devanagari <-> Modi character mapping (Factory Step 1).

Unicode names mirror each other across the two blocks: 'DEVANAGARI LETTER KA' <-> 'MODI LETTER KA',
'DEVANAGARI VOWEL SIGN II' <-> 'MODI VOWEL SIGN II', and so on, so the name does the mapping.
This covers vowels, consonants, vowel signs, anusvara, visarga, virama, digits and danda:
70 of the 77 Devanagari symbols in charset.json.

Decisions for everything else in charset.json:
- The 7 Devanagari symbols with no Modi code point (ऱ, nukta ़, ॉ, ऑ, ऽ, ँ, ॅ) raise ValueError.
  Sentences containing them are dropped by the text source, never repaired by deleting characters;
  `python -m modi.synthetic.mapping` lists the training sentences this affects.
- Space: kept. The label needs it. Word gaps are the renderer's business (real Modi has none).
- ( ) , - . / : kept as they are. They are not script-specific, the Modi font draws them,
  and each occurs in at most 5 training patches.
- Zero-width non-joiner (U+200C): kept. Modi shaping honours it like Devanagari does. It does not
  occur in the training text.
- <FRAC> (a fraction whose value the transcribers did not record) -> U+A830 NORTH INDIC FRACTION ONE
  QUARTER, which the Modi font draws. Any North Indic fraction sign (U+A830-U+A835) maps back to <FRAC>.
  Assumption: the fractions in the documents are written with these signs; to confirm with
  someone who reads Modi.
"""
import unicodedata

FRAC = '<FRAC>'
MODI_FRAC = '꠰'
FRACTION_SIGNS = {chr(c) for c in range(0xA830, 0xA836)}
KEEP = set(' (),-./:‌')


def _swap(ch, src, dst):
    if ch in KEEP:
        return ch
    name = unicodedata.name(ch, '')
    if name.startswith(src + ' '):
        try:
            return unicodedata.lookup(dst + name[len(src):])
        except KeyError:
            pass
    raise ValueError(f'no {dst.title()} form for {ch!r} ({name or "unnamed"})')


def dev_to_modi(s):
    """Devanagari training text -> Modi code points. Raises ValueError on a symbol with no Modi form."""
    return MODI_FRAC.join(''.join(_swap(c, 'DEVANAGARI', 'MODI') for c in part) for part in s.split(FRAC))


def modi_to_dev(s):
    """Modi code points -> Devanagari training text (the inverse of dev_to_modi)."""
    return ''.join(FRAC if c in FRACTION_SIGNS else _swap(c, 'MODI', 'DEVANAGARI') for c in s)


def unmappable(s):
    """The symbols of s that dev_to_modi cannot convert. Empty set = safe to render."""
    bad = set()
    for c in set(s.replace(FRAC, '')):
        try:
            _swap(c, 'DEVANAGARI', 'MODI')
        except ValueError:
            bad.add(c)
    return bad


def main():
    """Write outputs/reports/mapping_exceptions.csv: the training sentences the text source must drop."""
    import json

    import pandas as pd

    from modi.utils.paths import CODE_ROOT, P

    meta = CODE_ROOT / 'assets/metadata'
    split = json.load(open(meta / 'splits.json', encoding='utf-8'))['modetrans']['assignment']
    text = pd.read_csv(meta / 'modetrans_text.csv', keep_default_na=False)
    train = text[text.image_id.map(split) == 'train']
    rows = [(i, ' '.join(sorted(unmappable(s))), s) for i, s in zip(train.image_id, train.text_train) if unmappable(s)]
    out = P.reports / 'mapping_exceptions.csv'
    pd.DataFrame(rows, columns=['image_id', 'symbols', 'text_train']).to_csv(out, index=False, encoding='utf-8')
    print(f'{len(rows)} of {len(train)} training sentences contain a symbol with no Modi form -> {out}')


if __name__ == '__main__':
    main()
