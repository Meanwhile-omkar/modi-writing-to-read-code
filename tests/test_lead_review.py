"""Tests added in the lead's review of the Referee and Factory work (16 Oct 2026).

Each test in the first group failed on the code as merged (commit cba1145) and passes after the matching fix in
the same commit. The second group passed already but closes gaps found by a mutation test: each one fails on a
plausible bug that the original tests let through.
"""
import os
import random
import subprocess
import sys
from pathlib import Path

import editdistance
import pandas as pd
import pytest
import sacrebleu
from PIL import Image

from modi.evaluation.folder18_gt import body_lines, is_body
from modi.evaluation.metrics import (bleu, bootstrap_ci, catastrophic_rate, cer, chrf_pp, median_cer, normalize,
                                     paired_bootstrap, wer)
from modi.evaluation.taxonomy import align, error_types

CODE_ROOT = Path(__file__).resolve().parents[1]


def nonzero(counts):
    return {k: v for k, v in counts.items() if v}


# ---------- metrics ----------

def test_paired_bootstrap_refuses_runs_in_a_different_row_order():
    refs = ['राजश्री', 'स्वामी', 'मौजे']
    refs_b = ['स्वामी', 'राजश्री', 'मौजे']  # the same images, listed in another order in run b's file
    with pytest.raises(ValueError):
        paired_bootstrap(refs, refs, refs_b, refs_b=refs_b)
    assert paired_bootstrap(refs, refs, refs, refs_b=refs) == (0, 0, 0)


def test_character_bleu_says_how_it_treats_spaces():
    assert cer(['राज श्री'], ['राजश्री']) == pytest.approx(100 / 8)       # CER sees the missing space
    assert bleu(['राज श्री'], ['राजश्री']) == pytest.approx(100)          # sacreBLEU's char tokenizer does not
    assert bleu(['राज श्री'], ['राजश्री'], keep_spaces=True) < 100        # opt-in: a space is a token


def test_wer_with_a_wordless_reference_raises_valueerror():
    with pytest.raises(ValueError):
        wer([' '], ['a'])


# ---------- taxonomy ----------

def test_space_errors_have_their_own_category():
    assert nonzero(error_types(['राज श्री'], ['राजश्री'])) == {'space': 1}


@pytest.mark.parametrize('ref, hyp, expected', [
    ('भिंत', 'मीत', {'look-alike': 1, 'vowel sign': 1, 'anusvara': 1}),  # भ->म, ि->ी, ं dropped
    ('कां', 'कि', {'vowel sign': 1, 'anusvara': 1}),                     # ा->ि, ं dropped
])
def test_dropped_anusvara_next_to_a_vowel_sign_error(ref, hyp, expected):
    assert nonzero(error_types([ref], [hyp])) == expected


def test_alignment_stays_minimal_on_letter_length_texts():
    ref, hyp = 'का' * 600, 'ीग' * 600  # 1,200 letter<->sign pairs in place
    assert len(align(ref, hyp)) == editdistance.eval(ref, hyp)


# ---------- floor baselines ----------

def test_shuffled_floor_row_moves_frac_as_one_symbol():
    from modi.evaluation.floor_baselines import predictors
    shuffled = predictors(['१<FRAC>क'], ['x'], random.Random(0))['correct letters, shuffled'][0]
    assert sorted(normalize(shuffled)) == sorted(normalize('१<FRAC>क'))


# ---------- Folder-18 answer-key tooling ----------

def test_a_body_line_starting_with_a_kra_word_is_kept():
    page = ['श्री', 'राजश्री बाजीराव पंडित प्रधान', 'क्रमाने पाठविले आहे.', 'बहुत काय लिहिणे']
    assert body_lines(page) == page
    assert body_lines(['No. 3]', 'RADHABAI', 'क्र. ३] [७-५-१७३३', 'श्री', '१२']) == ['श्री']  # header still goes


def test_one_ocr_slip_does_not_drop_a_body_line():
    assert is_body('श्री यासह चिरंजीव विजीभव राजमान्य राज |')  # danda read as a pipe
    assert is_body('राजश्री बाजीराव 1 पंडित प्रधान')            # one Latin digit
    assert is_body('मौजे वाठार पा l फलटण')                        # one Latin letter


def _fake_book(tmp_path, monkeypatch, F):
    monkeypatch.setattr(F, 'OUT', tmp_path)
    monkeypatch.setattr(F, 'letters', lambda tier='strict': {3: ([15], [14])})
    (tmp_path / 'ocr').mkdir()
    Image.new('L', (40, 20), 255).save(tmp_path / 'ocr/page14.png')
    (tmp_path / 'ocr/page14.tesseract.txt').write_text('No. 3]\nक्र. ३] [७-५-१७३३\nश्री\nराजश्री बाजीराव\n',
                                                       encoding='utf-8')


def test_review_page_names_the_proofreading_file(tmp_path, monkeypatch):
    from modi.evaluation import folder18_gt as F
    _fake_book(tmp_path, monkeypatch, F)
    (tmp_path / 'ocr/page14.paddle.txt').write_text('श्री\nराजश्री बाजीराव\n', encoding='utf-8')
    F.ocr()
    assert 'proofread/letter03.txt' in (tmp_path / 'review/letter03.html').read_text(encoding='utf-8')


STUB = ("import json, sys\nfor p in sys.argv[1:]:\n"
        "    print(json.dumps({'path': p, 'lines': ['श्री', 'राजश्री बाजीराव']}, ensure_ascii=False), flush=True)\n")


def test_paddle_output_survives_the_pipe(tmp_path, monkeypatch):
    from modi.evaluation import folder18_gt as F
    _fake_book(tmp_path, monkeypatch, F)
    monkeypatch.setattr(F, 'PADDLE_PYTHON', sys.executable)
    monkeypatch.setattr(F, 'PADDLE_SCRIPT', STUB)  # stands in for PaddleOCR: prints Devanagari JSON the same way
    F.ocr()
    assert (tmp_path / 'ocr/page14.paddle.txt').read_text(encoding='utf-8') == 'श्री\nराजश्री बाजीराव\n'


def test_default_paddle_interpreter_fits_the_os():
    from modi.evaluation import folder18_gt as F
    if 'PADDLE_PYTHON' in os.environ:
        pytest.skip('PADDLE_PYTHON is set')
    assert Path(F.PADDLE_PYTHON).parts[-2:] == (('Scripts', 'python.exe') if os.name == 'nt' else ('bin', 'python'))


def test_build_applies_the_step05_normalisation(tmp_path, monkeypatch):
    from modi.evaluation import folder18_gt as F
    _fake_book(tmp_path, monkeypatch, F)
    (tmp_path / 'ocr/page14.paddle.txt').write_text('श्री\n', encoding='utf-8')
    (tmp_path / 'proofread').mkdir()
    (tmp_path / 'proofread/letter03.txt').write_text('# status: proofread\n# minutes: 5\nश्री क़र १σ\n',
                                                     encoding='utf-8')
    F.build()
    gt = pd.read_csv(tmp_path / 'folder18_gt.csv', keep_default_na=False)
    assert gt.text[0] == 'श्री क़र १<FRAC>'  # NFC splits क़ into क + nukta; σ is the fraction sign
    assert gt.n_chars[0] == 11
    assert len(pd.read_csv(tmp_path / 'oov_report.csv')) == 0


def test_command_line_help_works():
    env = dict(os.environ, PYTHONPATH=str(CODE_ROOT / 'src'))
    run = subprocess.run([sys.executable, '-m', 'modi.evaluation.folder18_gt', '--help'], cwd=CODE_ROOT, env=env,
                         capture_output=True, text=True, encoding='utf-8')
    assert run.returncode == 0 and 'build' in run.stdout


# ---------- tests that close mutation-test gaps (passed before; fail on the bug named) ----------

def test_vowel_sign_range_ends_at_au():  # bug: range ending at U+094B
    assert nonzero(error_types(['कौल'], ['कल'])) == {'vowel sign': 1}


def test_median_is_not_the_mean():  # per image 0, 0, 100: median 0, mean 33.3
    assert median_cer(['aaaa'] * 3, ['aaaa', 'aaaa', '']) == 0


def test_catastrophic_means_strictly_above_50():  # 50 % is not catastrophic, 100 % is
    assert catastrophic_rate(['ab', 'ab'], ['a', 'xy']) == 50


def test_wer_normalises_like_cer():  # precomposed क़ vs क + nukta
    assert wer(['क़र'], ['क़र']) == 0


def test_chrf_pp_is_chrf_with_word_bigrams():
    refs, hyps = ['राम आला घरी'], ['राम गेला घरी']
    assert chrf_pp(refs, hyps) == pytest.approx(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score)
    assert chrf_pp(refs, hyps) != pytest.approx(sacrebleu.corpus_chrf(hyps, [refs], word_order=0).score)


def test_bootstrap_level_changes_the_interval():
    refs = ['abcdefghij'] * 50
    hyps = ['abcdefghij'] * 40 + ['abcdefghxx'] * 10
    _, lo90, hi90 = bootstrap_ci(refs, hyps, level=90)
    _, lo95, hi95 = bootstrap_ci(refs, hyps)
    assert lo95 < lo90 < hi90 < hi95
