"""Fake runs for testing the results machine: the test-split references with a chosen share of characters
damaged, written in the agreed format (<results>/<run>/test_predictions.csv + config.json).

  python tests/fakes/make_fake_predictions.py <results_dir> <run_name> <share> [--concentrated] [--seed N]

even (default)  every image has `share` of its characters damaged         -> CER ~ 100 * share, median ~ the same
concentrated    whole images damaged heavily (80 %) until the same total is reached, the rest perfect
                -> same CER, but median 0 and many catastrophic images
"""
import argparse
import json
import random
import sys
from pathlib import Path

import editdistance
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from modi.evaluation.floor_baselines import damage  # noqa: E402
from modi.evaluation.metrics import normalize  # noqa: E402

HEAVY = 0.8


def make(results, run, share, concentrated=False, seed=0):
    meta = ROOT / 'assets/metadata'
    split = json.load(open(meta / 'splits.json', encoding='utf-8'))['modetrans']['assignment']
    text = pd.read_csv(meta / 'modetrans_text.csv', keep_default_na=False)
    test = text[text.image_id.map(split) == 'test']
    rng = random.Random(seed)
    if concentrated:  # damage whole images, in random order, until the real edit count reaches `share`
        refs = test.text_train.tolist()
        preds, edits, budget = list(refs), 0, share * sum(len(normalize(r)) for r in refs)
        order = list(range(len(refs)))
        rng.shuffle(order)
        for i in order:
            if edits >= budget:
                break
            preds[i] = damage(refs[i], HEAVY, rng)
            edits += editdistance.eval(normalize(refs[i]), normalize(preds[i]))
    else:
        preds = [damage(r, share, rng) for r in test.text_train]
    out = Path(results) / run
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'image_id': test.image_id, 'reference': test.text_train, 'prediction': preds}) \
        .to_csv(out / 'test_predictions.csv', index=False, encoding='utf-8')
    (out / 'config.json').write_text(json.dumps({'run': run, 'fake': True, 'damaged_share': share,
                                                 'concentrated': concentrated, 'seed': seed}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('run')
    ap.add_argument('share', type=float)
    ap.add_argument('--concentrated', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    make(a.results, a.run, a.share, a.concentrated, a.seed)
