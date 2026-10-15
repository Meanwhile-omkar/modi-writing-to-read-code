"""Factory Step 0: font go / no-go.

Renders (1) every Modi character, (2) sample words without and with shaping, (3) one real MoDeTrans
patch next to its text drawn in the font, and prints the numbers for the go / no-go note.
Only training patches are read, so no validation or test text reaches the Factory.

Run from the repository root:  PYTHONPATH=src python -m modi.synthetic.font_gate
Writes outputs/figures/font_gate/*.png
"""
import io
import json
import textwrap
import unicodedata

import pandas as pd
import pyarrow.parquet as pq
from fontTools.ttLib import TTFont
from PIL import Image, ImageChops, ImageDraw, ImageFont, features

from modi.synthetic.mapping import dev_to_modi, modi_to_dev, unmappable
from modi.utils import console  # noqa: F401
from modi.utils.paths import CODE_ROOT, P

MODI_FONT = CODE_ROOT / 'assets/fonts/NotoSansModi-Regular.ttf'
LABEL_FONT = CODE_ROOT / 'assets/fonts/NotoSansDevanagari.ttf'
META = CODE_ROOT / 'assets/metadata'

# Archaic administrative words, chosen to cover conjuncts, repha, anusvara and every vowel sign.
WORDS = ['श्री', 'राजमान्य', 'लक्ष्मी', 'मौजे', 'पातशाहा', 'अलफ', 'विनंती', 'कुशल', 'पूर्वी', 'मोकदम',
         'सबैन', 'आलंकृत']
SENTENCE_FILE = '10.jpg'  # a clean training patch, shown next to its font rendering


def render(text, font, pad=10):
    """Black text on white, cropped to the ink plus a margin."""
    l, t, r, b = ImageDraw.Draw(Image.new('L', (1, 1))).multiline_textbbox((0, 0), text, font=font, spacing=30)
    im = Image.new('L', (r - l + 2 * pad, b - t + 2 * pad), 255)
    ImageDraw.Draw(im).multiline_text((pad - l, pad - t), text, font=font, fill=0, spacing=30)
    return im


def stack(images, gap=10):
    w = max(im.width for im in images)
    out = Image.new('L', (w, sum(im.height for im in images) + gap * (len(images) - 1)), 255)
    y = 0
    for im in images:
        out.paste(im, (0, y))
        y += im.height + gap
    return out


def charset_sheet(modi, label, cmap):
    """Grid of every assigned Modi code point; returns the sheet and the code points drawn without ink."""
    chars = [c for c in range(0x11600, 0x11660) if unicodedata.name(chr(c), '')]
    cols, cw, ch = 10, 140, 170
    sheet = Image.new('L', (cols * cw, -(-len(chars) // cols) * ch), 255)
    empty = []
    for i, c in enumerate(chars):
        glyph = render(chr(c), modi)
        if c not in cmap or glyph.getextrema()[0] == 255:
            empty.append(c)
        x, y = (i % cols) * cw, (i // cols) * ch
        sheet.paste(glyph, (x + (cw - glyph.width) // 2, y + max(0, 100 - glyph.height) // 2))
        try:
            dev = modi_to_dev(chr(c))
        except ValueError:  # Modi-only signs such as ardhacandra have no Devanagari twin
            dev = '-'
        sheet.paste(render(f'{c:X} {dev}', label), (x + 8, y + 108))
    return sheet, chars, empty


def words_sheet(modi_basic, modi_raqm, label):
    """Each word as Devanagari | Modi without shaping | Modi with shaping."""
    header = render('Devanagari  |  Modi, no shaping (BASIC)  |  Modi, shaped (RAQM)', label)
    rows, changed = [header], 0
    for w in WORDS:
        m = dev_to_modi(w)
        basic, shaped = render(m, modi_basic), render(m, modi_raqm)
        changed += basic.size != shaped.size or ImageChops.difference(basic, shaped).getbbox() is not None
        cells = [render(w, label), basic, shaped]
        row = Image.new('L', (900, max(c.height for c in cells)), 255)
        for x, c in zip((0, 300, 600), cells):
            row.paste(c, (x, 0))
        rows.append(row)
    return stack(rows), changed


def main():
    if not features.check('raqm'):
        raise SystemExit('Pillow cannot shape Modi (raqm is off): see Setup in src/modi/synthetic/README.md')
    out = P.figures / 'font_gate'
    out.mkdir(exist_ok=True)
    modi = ImageFont.truetype(str(MODI_FONT), 64, layout_engine=ImageFont.Layout.RAQM)
    modi_basic = ImageFont.truetype(str(MODI_FONT), 64, layout_engine=ImageFont.Layout.BASIC)
    label = ImageFont.truetype(str(LABEL_FONT), 22, layout_engine=ImageFont.Layout.RAQM)
    big_label = ImageFont.truetype(str(LABEL_FONT), 44, layout_engine=ImageFont.Layout.RAQM)
    cmap = TTFont(MODI_FONT).getBestCmap()

    split = json.load(open(META / 'splits.json', encoding='utf-8'))['modetrans']['assignment']
    text = pd.read_csv(META / 'modetrans_text.csv', keep_default_na=False).set_index('image_id')['text_train']
    train = [t for i, t in text.items() if split[i] == 'train']
    symbols = json.load(open(META / 'charset.json', encoding='utf-8'))['symbols']

    sheet, chars, empty = charset_sheet(modi, label, cmap)
    sheet.save(out / '01_charset.png')

    words, changed = words_sheet(modi_basic, modi, big_label)
    words.save(out / '02_words.png')

    table = pq.read_table(P.modetrans_parquet, columns=['filename', 'image'])
    row = table.column('filename').to_pylist().index(SENTENCE_FILE)
    real = Image.open(io.BytesIO(table.column('image')[row].as_py()['bytes'])).convert('L')
    fake = render('\n'.join(textwrap.wrap(dev_to_modi(text[SENTENCE_FILE].strip()), 32)), modi)
    fake = fake.resize((real.width, round(fake.height * real.width / fake.width)))
    stack([render(f'Real handwriting (MoDeTrans {SENTENCE_FILE})', big_label), real,
           render('Same text, Noto Sans Modi', big_label), fake]).save(out / '03_real_vs_font.png')

    dev = [s for s in symbols if unicodedata.name(s, '').startswith('DEVANAGARI')]
    unmapped = [s for s in dev if unmappable(s) or ord(dev_to_modi(s)) not in cmap]
    print(f'Font: {MODI_FONT.name}')
    print(f'  Modi code points in font: {sum(c in cmap for c in chars)} of {len(chars)} assigned')
    print(f'  drawn without ink (empty): {[f"{c:X}" for c in empty] or "none"}')
    print(f'  sample words changed by shaping: {changed} of {len(WORDS)}')
    print(f'  sample words found in training text: {sum(any(w in t for t in train) for w in WORDS)} of {len(WORDS)}')
    print(f'charset.json: {len(symbols)} symbols, {len(dev)} Devanagari, {len(dev) - len(unmapped)} drawable after mapping')
    print(f'  other symbols (Step 1 decides): {[s for s in symbols if s not in dev]}')
    print(f'  training patches with no undrawable symbol: '
          f'{sum(not any(c in t for c in unmapped) for t in train)} of {len(train)}')
    print('  undrawable (symbol, name, training patches containing it):')
    for s in sorted(unmapped, key=lambda s: -sum(s in t for t in train)):
        print(f'    {s}  {unicodedata.name(s)}  {sum(s in t for t in train)}')


if __name__ == '__main__':
    main()
