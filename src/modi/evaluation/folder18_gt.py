"""Referee Step 5: the answer key for the Folder-18 book (the secret exam).

Two commands, run from the repository root:

  PYTHONPATH=src python -m modi.evaluation.folder18_gt ocr
      OCR every printed page of the strict letters with two different tools (Tesseract `mar`, PaddleOCR
      Devanagari), then write for each letter
        proofread/letterNN.txt   Tesseract text with the obvious non-body lines removed: correct it by hand
        review/letterNN.html     the page next to both OCR texts, disagreements highlighted
  PYTHONPATH=src python -m modi.evaluation.folder18_gt build
      after proofreading: apply RULES.md and write folder18_gt.csv, folder18_gt_audit.csv, oov_report.csv

Everything is written to assets/folder18_gt/, which git ignores (the book's terms): share it outside git.
PaddleOCR lives in its own environment (default ../ocr-venv, override with PADDLE_PYTHON).
"""
import base64
import difflib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys

import editdistance
import pandas as pd
from PIL import Image

from modi.evaluation.metrics import FRAC
from modi.utils.paths import CODE_ROOT, P

META = CODE_ROOT / 'assets/metadata'
OUT = CODE_ROOT / 'assets/folder18_gt'
TESSDATA = P.data_root / 'tessdata'
PADDLE_PYTHON = os.environ.get('PADDLE_PYTHON', str(CODE_ROOT.parent / 'ocr-venv/bin/python'))
NOT_DONE = '# status: NOT PROOFREAD  <- change to "# status: proofread" when every line matches the page'

PADDLE_SCRIPT = """
import json, os, sys
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
from paddleocr import PaddleOCR
ocr = PaddleOCR(text_detection_model_name='PP-OCRv5_mobile_det', text_recognition_model_name='devanagari_PP-OCRv5_mobile_rec',
                use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
for path in sys.argv[1:]:
    print(json.dumps({'path': path, 'lines': list(ocr.predict(path)[0]['rec_texts'])}, ensure_ascii=False), flush=True)
"""


def letters(tier='strict'):
    """{letter number: (facsimile pages, transcription pages)} for the letters of one test tier."""
    labels = pd.read_csv(META / 'folder18_labels.csv', keep_default_na=False)
    docs = sorted({int(float(d)) for d in labels[labels.ood_tier == tier].doc_no})
    pages = lambda d, kind: [int(p) for p in labels[(labels.doc_no != '') & (labels.page_type == kind)
                                                   & (labels.doc_no.map(lambda x: x and int(float(x)) == d))].page]
    return {d: (pages(d, 'modi_facsimile'), pages(d, 'devanagari_transcription')) for d in docs}


def is_body(line):
    """False for lines the rules drop: English (also when OCR turned it into Latin digits and symbols), the
    'क्र. N' / date line, page numbers, footnotes, star rows."""
    s = line.strip()
    return bool(s) and not (re.search('[A-Za-z0-9&$%@|]', s) or s.startswith(('क्र', '†', '*'))
                            or re.fullmatch(r'[०-९\s.,\[\]()-]+', s))


def body_lines(lines):
    """The letter body of one page: on a letter's first page the English header comes before the 'क्र. N'
    line, so everything up to that line goes; then the other non-body lines."""
    start = next((i + 1 for i, l in enumerate(lines) if l.strip().startswith('क्र')), 0)
    return [l.strip() for l in lines[start:] if is_body(l)]


def ocr():
    (OUT / 'ocr').mkdir(parents=True, exist_ok=True)
    (OUT / 'proofread').mkdir(exist_ok=True)
    (OUT / 'review').mkdir(exist_ok=True)
    todo = letters()
    pages = [p for _, printed in todo.values() for p in printed]
    for p in pages:  # the scans are 600 dpi; OCR works best near 300, so halve them (the handout's 2x is for small scans)
        png = OUT / f'ocr/page{p}.png'
        if not png.exists():
            im = Image.open(P.hhdoc_root / f'18/page{p}.jpg').convert('L')
            im.resize((im.width // 2, im.height // 2), Image.LANCZOS).save(png)
        if not (OUT / f'ocr/page{p}.tesseract.txt').exists():  # OCR output is kept: re-runs only redo what is missing
            subprocess.run(['tesseract', str(png), str(OUT / f'ocr/page{p}.tesseract'), '-l', 'mar',
                            '--tessdata-dir', str(TESSDATA)], check=True, capture_output=True)
            print(f'tesseract page {p}', flush=True)
    missing = [str(OUT / f'ocr/page{p}.png') for p in pages if not (OUT / f'ocr/page{p}.paddle.txt').exists()]
    if missing:
        run = subprocess.Popen([PADDLE_PYTHON, '-I', '-c', PADDLE_SCRIPT, *missing],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        for line in run.stdout:
            r = json.loads(line)
            (OUT / f'ocr/{os.path.basename(r["path"])[:-4]}.paddle.txt').write_text('\n'.join(r['lines']) + '\n',
                                                                                     encoding='utf-8')
            print(f'paddleocr {os.path.basename(r["path"])}', flush=True)
        if run.wait():
            sys.exit('PaddleOCR failed; is it installed in ' + PADDLE_PYTHON + '?')

    for n, (_, printed) in todo.items():
        proof = OUT / f'proofread/letter{n:02d}.txt'
        if not proof.exists():  # never overwrite proofreading work
            body = [f'## page {p}\n' + '\n'.join(body_lines(read(p, 'tesseract'))) for p in printed]
            proof.write_text('\n'.join([
                f'# Letter {n} · printed page(s) {", ".join(map(str, printed))} · compare with review/letter{n:02d}.html',
                NOT_DONE, '# minutes: 0', '# Rules: RULES.md. Lines starting with # are ignored.',
                '# End a line with + when the print breaks it in the middle of a word (joined without a space).',
                *body]) + '\n', encoding='utf-8')
        html = [f'<meta charset="utf-8"><title>Letter {n}</title><style>body{{font-family:sans-serif}} td{{font-size:18px}}'
                ' .page{display:flex;gap:16px;align-items:flex-start} img{width:48%;border:1px solid #888}</style>',
                f'<h1>Letter {n}</h1><p>Left: the printed page. Right: Tesseract vs PaddleOCR; coloured = they disagree, '
                'so check those words on the page first. Then correct proofread/letter{n:02d}.txt line by line.</p>']
        for p in printed:
            buf = io.BytesIO()
            Image.open(OUT / f'ocr/page{p}.png').convert('L').save(buf, 'JPEG', quality=80)
            table = difflib.HtmlDiff(wrapcolumn=60).make_table(body_lines(read(p, 'tesseract')),
                                                                body_lines(read(p, 'paddle')),
                                                                'Tesseract', 'PaddleOCR')
            html.append(f'<h2>Page {p}</h2><div class="page"><img src="data:image/jpeg;base64,'
                        f'{base64.b64encode(buf.getvalue()).decode()}"><div>{table}</div></div>')
        (OUT / f'review/letter{n:02d}.html').write_text('\n'.join(html), encoding='utf-8')
    print(f'Done. Proofread {OUT / "proofread"} using {OUT / "review"}')


def is_proofread(text):
    """A status line whose first word is 'proofread' (any case) counts; 'NOT PROOFREAD' does not."""
    return re.search(r'^# status:\s*proofread\b', text, re.M | re.I) is not None


def read(page, tool):
    return (OUT / f'ocr/page{page}.{tool}.txt').read_text(encoding='utf-8').splitlines()


def join(lines):
    """Printed lines -> one text: a line ending in + continues the word on the next line, others get a space."""
    text = ''
    for line in (l.strip() for l in lines if l.strip() and not l.lstrip().startswith('#')):
        text += line[:-1] if line.endswith('+') else line + ' '
    return re.sub(r'\s+', ' ', text).strip()


def to_train():
    """The team's text-cleaning rule (scripts/15_text_decisions.py), so the answer key matches the training text."""
    spec = importlib.util.spec_from_file_location('text_decisions', CODE_ROOT / 'scripts/15_text_decisions.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.to_train


def build():
    clean = to_train()
    charset = set(json.load(open(META / 'charset.json', encoding='utf-8'))['symbols'])
    rows, audit, oov, waiting = [], [], [], []
    for n, (facsimile, printed) in letters().items():
        proof = (OUT / f'proofread/letter{n:02d}.txt').read_text(encoding='utf-8')
        if not is_proofread(proof):
            waiting.append(n)
            continue
        text = clean(join(proof.splitlines()))
        minutes = re.search(r'# minutes:\s*(\d+)', proof)
        ocr_text = {tool: clean(join(l for p in printed for l in body_lines(read(p, tool)))) for tool in ('tesseract', 'paddle')}
        rows.append((n, 'strict', ' '.join(map(str, facsimile)), ' '.join(map(str, printed)), text,
                     len(text.replace(FRAC, '_'))))
        audit.append((n, ocr_text['tesseract'], ocr_text['paddle'], text, editdistance.eval(ocr_text['tesseract'], text),
                      int(minutes.group(1)) if minutes else 0))
        for ch in sorted(set(text.replace(FRAC, '')) - charset):
            oov.append((ch, f'U+{ord(ch):04X}', n, next(w for w in text.split() if ch in w), ''))
    pd.DataFrame(rows, columns=['letter_no', 'tier', 'facsimile_pages', 'transcription_pages', 'text', 'n_chars']) \
        .to_csv(OUT / 'folder18_gt.csv', index=False, encoding='utf-8')
    pd.DataFrame(audit, columns=['letter_no', 'ocr1_tesseract', 'ocr2_paddle', 'final', 'edits_vs_ocr1', 'minutes']) \
        .to_csv(OUT / 'folder18_gt_audit.csv', index=False, encoding='utf-8')
    pd.DataFrame(oov, columns=['character', 'code_point', 'letter_no', 'example_word', 'decision']) \
        .to_csv(OUT / 'oov_report.csv', index=False, encoding='utf-8')
    print(f'{len(rows)} letters written; not proofread yet: {waiting or "none"}; characters outside charset.json: {len(oov)}')
    for n, *_, edits, _ in audit:
        if edits == 0:
            print(f'  WARNING letter {n}: final text equals the OCR exactly - was it really checked?')


if __name__ == '__main__':
    {'ocr': ocr, 'build': build}[sys.argv[1]]()
