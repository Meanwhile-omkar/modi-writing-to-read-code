"""Devanagari <-> Modi character mapping.

Unicode names mirror each other across the two blocks: 'DEVANAGARI LETTER KA' <-> 'MODI LETTER KA',
'DEVANAGARI VOWEL SIGN II' <-> 'MODI VOWEL SIGN II', and so on, so the name does the mapping.
"""
import unicodedata


def _swap(ch, src, dst):
    name = unicodedata.name(ch, '')
    if name.startswith(src + ' '):
        try:
            return unicodedata.lookup(dst + name[len(src):])
        except KeyError:
            pass
    # ponytail: unmapped characters pass through unchanged; Step 1 decides how each one is handled
    return ch


def dev_to_modi(s):
    return ''.join(_swap(c, 'DEVANAGARI', 'MODI') for c in s)


def modi_to_dev(s):
    return ''.join(_swap(c, 'MODI', 'DEVANAGARI') for c in s)
