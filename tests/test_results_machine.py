"""The results machine, end to end on fake runs whose answers are known in advance (Referee Step 7)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/fakes'))
from make_fake_predictions import make  # noqa: E402


@pytest.fixture(scope='module')
def report(tmp_path_factory):
    results, out = tmp_path_factory.mktemp('results'), tmp_path_factory.mktemp('report')
    make(results, 'fake10_seed42', 0.10, seed=42)
    make(results, 'fake10_seed43', 0.10, seed=43)
    make(results, 'fake20_seed42', 0.20, seed=42)
    make(results, 'fake10-concentrated_seed42', 0.10, concentrated=True, seed=42)
    subprocess.run([sys.executable, str(ROOT / 'scripts/30_aggregate_results.py'), '--results', str(results),
                    '--out', str(out), '--compare', 'fake20_seed42', 'fake10_seed42',
                    '--compare', 'fake10_seed42', 'fake10_seed42'], check=True, capture_output=True)
    return json.loads((out / 'results.json').read_text(encoding='utf-8')), out


def test_damage_comes_out_as_the_cer_we_put_in(report):
    runs = report[0]['runs']
    assert runs['fake10_seed42']['cer'] == pytest.approx(10, abs=1.5)
    assert runs['fake20_seed42']['cer'] == pytest.approx(20, abs=2.5)
    assert runs['fake10_seed42']['cer_low'] < runs['fake10_seed42']['cer'] < runs['fake10_seed42']['cer_high']


def test_seeds_of_one_configuration_are_averaged(report):
    group = report[0]['over_seeds']['fake10']
    assert group['runs'] == ['fake10_seed42', 'fake10_seed43']
    mean, std = group['cer']
    assert mean == pytest.approx(10, abs=1.5) and 0 < std < 1
    assert 'fake20' not in report[0]['over_seeds']  # one seed only: nothing to average


def test_comparisons_tell_a_real_difference_from_none(report):
    worse, same = report[0]['comparisons']
    assert worse['cer_difference'] == pytest.approx(10, abs=3) and worse['low'] > 0 and 'clearly higher' in worse['verdict']
    assert (same['cer_difference'], same['low'], same['high']) == (0, 0, 0) and 'not clearly different' in same['verdict']


def test_concentrated_errors_show_in_median_catastrophic_rate_and_interval(report):
    runs = report[0]['runs']
    even, conc = runs['fake10_seed42'], runs['fake10-concentrated_seed42']
    assert conc['cer'] == pytest.approx(even['cer'], abs=1)  # the same average ...
    assert conc['median_cer'] == 0 < even['median_cer']       # ... but most images are perfect
    assert conc['catastrophic_pct'] > 5 and even['catastrophic_pct'] == 0
    assert conc['cer_high'] - conc['cer_low'] > 2 * (even['cer_high'] - even['cer_low'])  # far less certain


def test_report_and_figures_are_written(report):
    out = report[1]
    md = (out / 'results.md').read_text(encoding='utf-8')
    assert '| fake10 | 2 |' in md and 'clearly higher' in md
    assert (out / 'results_cer_per_image.png').stat().st_size > 0 and (out / 'results_error_types.png').stat().st_size > 0
