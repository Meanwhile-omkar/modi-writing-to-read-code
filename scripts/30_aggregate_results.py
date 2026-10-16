"""30 - the results machine: every results/*/test_predictions.csv -> one results table with statistics.

Run from the repository root:
  python scripts/30_aggregate_results.py
  python scripts/30_aggregate_results.py --compare B1_seed42 B1-noLSTM_seed42 --compare ...
  python scripts/30_aggregate_results.py --results <runs dir> --out <report dir>

Writes results.md, results.json, results_cer_per_image.png and results_error_types.png to --out
(default outputs/reports). Runs whose names differ only in a final _seedNN (B1_seed42, B1_seed43, ...) are
one configuration: their scores are averaged as mean +- std. Metric definitions: src/modi/evaluation/README.md.
"""
import argparse
import datetime
import json
import math
import re
import statistics
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from modi.evaluation.metrics import (bootstrap_ci, catastrophic_rate, cer_nospace, chrf_pp, median_cer,  # noqa: E402
                                     paired_bootstrap, per_image_cer, wer)
from modi.evaluation.taxonomy import CATEGORIES, error_types  # noqa: E402
from modi.utils import console  # noqa: F401,E402
from modi.utils.paths import CODE_ROOT, P  # noqa: E402

SERIES = '#2a78d6'
TYPE_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']  # one per error type, fixed order
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e4e3dc'
AVERAGED = ['cer', 'cer_nospace', 'wer', 'chrf_pp', 'median_cer', 'catastrophic_pct']


def load(results):
    runs = {}
    for f in sorted(Path(results).glob('*/test_predictions.csv')):
        p = pd.read_csv(f, keep_default_na=False, dtype=str)  # an empty prediction stays '' (not NaN)
        missing = {'image_id', 'reference', 'prediction'} - set(p.columns)
        if missing:
            sys.exit(f'{f}: missing columns {sorted(missing)}')
        if not (f.parent / 'config.json').exists():
            print(f'warning: {f.parent.name} has no config.json next to its predictions')
        runs[f.parent.name] = p.set_index('image_id')
    if not runs:
        sys.exit(f'no runs found in {results}/*/test_predictions.csv')
    image_sets = {frozenset(p.index) for p in runs.values()}
    if len(image_sets) > 1:
        print('warning: the runs were not all scored on the same images; compare them with care')
    return runs


def score(p):
    refs, hyps = p.reference.tolist(), p.prediction.tolist()
    cer, low, high = bootstrap_ci(refs, hyps)
    return {'images': len(p), 'cer': cer, 'cer_low': low, 'cer_high': high, 'cer_nospace': cer_nospace(refs, hyps),
            'wer': wer(refs, hyps), 'chrf_pp': chrf_pp(refs, hyps), 'median_cer': median_cer(refs, hyps),
            'catastrophic_pct': catastrophic_rate(refs, hyps), 'error_types': dict(error_types(refs, hyps))}


def seed_groups(scores):
    groups = {}
    for run in scores:
        groups.setdefault(re.sub(r'_seed\d+$', '', run), []).append(run)
    return {g: {'runs': rs, **{m: (statistics.mean(scores[r][m] for r in rs), statistics.stdev(scores[r][m] for r in rs))
                               for m in AVERAGED}}
            for g, rs in groups.items() if len(rs) > 1}


def compare(runs, a, b):
    for r in (a, b):
        if r not in runs:
            sys.exit(f'--compare: no run called {r}')
    pa, pb = runs[a], runs[b]
    if set(pa.index) != set(pb.index):
        sys.exit(f'--compare: {a} and {b} were scored on different images')
    pb = pb.loc[pa.index]
    if not (pa.reference == pb.reference).all():
        sys.exit(f'--compare: {a} and {b} have different references for the same images')
    diff, low, high = paired_bootstrap(pa.reference.tolist(), pa.prediction.tolist(), pb.prediction.tolist(),
                                       refs_b=pb.reference.tolist())
    if low > 0:
        verdict = f'{a} has a clearly higher CER than {b} (worse)'
    elif high < 0:
        verdict = f'{a} has a clearly lower CER than {b} (better)'
    else:
        verdict = f'the difference between {a} and {b} is not clearly different from zero'
    return {'a': a, 'b': b, 'cer_difference': diff, 'low': low, 'high': high, 'verdict': verdict}


def plot_per_image(runs, path):
    cols = min(3, len(runs))
    rows = math.ceil(len(runs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.8 * rows), sharex=True, sharey=True, squeeze=False)
    for ax, (name, p) in zip(axes.flat, runs.items()):
        per = per_image_cer(p.reference.tolist(), p.prediction.tolist()).clip(max=150)
        ax.hist(per, bins=range(0, 155, 5), color=SERIES, edgecolor='white', linewidth=1)
        ax.axvline(50, color=MUTED, linewidth=1, linestyle='--')
        ax.set_title(name, fontsize=10, color=INK, loc='left')
        ax.grid(axis='y', color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
    for ax in list(axes.flat)[len(runs):]:
        ax.set_visible(False)
    fig.supxlabel('CER of one test image (%), values above 150 shown at 150; dashed line: catastrophic (> 50)',
                  fontsize=9, color=MUTED)
    fig.supylabel('test images', fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_error_types(scores, path):
    names = list(scores)[::-1]  # first run on top
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 1.4))
    left = [0.0] * len(names)
    for category, color in zip(CATEGORIES, TYPE_COLORS):
        shares = []
        for name in names:
            counts = scores[name]['error_types']
            total = sum(counts.values())
            shares.append(100 * counts[category] / total if total else 0)
        ax.barh(names, shares, left=left, color=color, edgecolor='white', linewidth=2, label=category, height=0.6)
        for y, (x0, s) in enumerate(zip(left, shares)):
            if s >= 8:
                ax.text(x0 + s / 2, y, f'{s:.0f}%', ha='center', va='center', fontsize=8, color=INK)
        left = [x0 + s for x0, s in zip(left, shares)]
    ax.set_xlim(0, 100)
    ax.set_xlabel('share of all character errors (%)', fontsize=9, color=MUTED)
    ax.legend(ncol=len(CATEGORIES), loc='lower left', bbox_to_anchor=(0, 1.02), frameon=False, fontsize=8)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def report(scores, groups, comparisons, results_dir):
    images = {s['images'] for s in scores.values()}
    lines = ['# Results', '',
             f'Generated by `scripts/30_aggregate_results.py` on {datetime.date.today()} from '
             f'`{results_dir}/*/test_predictions.csv` ({len(scores)} runs, {"/".join(map(str, sorted(images)))} test images). '
             'CER, WER: % of the reference, lower is better. chrF++: higher is better. '
             'Catastrophic: % of images with CER above 50. CER interval: 95 % bootstrap, 10,000 resamples.', '',
             '## Runs', '',
             '| run | CER [95 % CI] | CER without spaces | WER | chrF++ | median CER | catastrophic |',
             '|---|---|---|---|---|---|---|']
    lines += [f'| {n} | {s["cer"]:.2f} [{s["cer_low"]:.2f}, {s["cer_high"]:.2f}] | {s["cer_nospace"]:.2f} | {s["wer"]:.2f} '
              f'| {s["chrf_pp"]:.2f} | {s["median_cer"]:.2f} | {s["catastrophic_pct"]:.1f} % |' for n, s in scores.items()]
    if groups:
        lines += ['', '## Over seeds (mean ± std)', '',
                  '| configuration | seeds | CER | CER without spaces | WER | chrF++ | median CER | catastrophic |',
                  '|---|---|---|---|---|---|---|---|']
        lines += [f'| {g} | {len(v["runs"])} | ' + ' | '.join(f'{v[m][0]:.2f} ± {v[m][1]:.2f}' for m in AVERAGED) + ' |'
                  for g, v in groups.items()]
    if comparisons:
        lines += ['', '## Comparisons (paired bootstrap on the same images)', '',
                  '| A | B | CER(A) - CER(B) [95 % CI] | verdict |', '|---|---|---|---|']
        lines += [f'| {c["a"]} | {c["b"]} | {c["cer_difference"]:+.2f} [{c["low"]:+.2f}, {c["high"]:+.2f}] | {c["verdict"]} |'
                  for c in comparisons]
    lines += ['', '## Error types (number of character errors)', '',
              '| run | ' + ' | '.join(CATEGORIES) + ' |', '|---|' + '---|' * len(CATEGORIES)]
    lines += [f'| {n} | ' + ' | '.join(str(s['error_types'][c]) for c in CATEGORIES) + ' |' for n, s in scores.items()]
    lines += ['', '![CER per test image](results_cer_per_image.png)', '', '![Error types](results_error_types.png)', '']
    return '\n'.join(lines)


assert len(TYPE_COLORS) >= len(CATEGORIES), 'add a colour for the new error type'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default=CODE_ROOT / 'results', type=Path)
    ap.add_argument('--out', default=P.reports, type=Path)
    ap.add_argument('--compare', nargs=2, action='append', default=[], metavar=('RUN_A', 'RUN_B'))
    a = ap.parse_args()
    runs = load(a.results)
    scores = {name: score(p) for name, p in runs.items()}
    groups = seed_groups(scores)
    comparisons = [compare(runs, x, y) for x, y in a.compare]
    a.out.mkdir(parents=True, exist_ok=True)
    plot_per_image(runs, a.out / 'results_cer_per_image.png')
    plot_error_types(scores, a.out / 'results_error_types.png')
    md = report(scores, groups, comparisons, a.results)
    (a.out / 'results.md').write_text(md, encoding='utf-8')
    (a.out / 'results.json').write_text(json.dumps({'runs': scores, 'over_seeds': groups, 'comparisons': comparisons},
                                                   ensure_ascii=False, indent=2), encoding='utf-8')
    print(md)


if __name__ == '__main__':
    main()
