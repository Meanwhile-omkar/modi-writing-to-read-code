"""Build notebooks/07_robust_reader_GPU.ipynb (notebook 07, a variant of notebook 05 built from notebooks/lr07/).

The five code files in notebooks/lr05/ are embedded verbatim with %%writefile, so the notebook runs exactly
the code that was tested locally (unit tests + the CPU rehearsal, rehearse_nb07.py). The font is embedded
as base64 so the notebook needs no input beyond colab_pack.zip.
"""
import base64, hashlib, json, pathlib

HERE = pathlib.Path(__file__).resolve().parent
NB = HERE.parent / "07_robust_reader_GPU.ipynb"
FONT = HERE.parents[1] / "assets/fonts/NotoSansModi-Regular.ttf"   # same file (SHA-256 df1b8c67...) as the old cache copy; this one is in git
FILES = ["lr05_lib.py", "render_pools.py", "job.py", "lr05_run.py", "lr05_final.py"]
CELLS = []


def md(src):
    CELLS.append(dict(cell_type="markdown", metadata={}, source=src.strip("\n").splitlines(True)))


def code(src):
    CELLS.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=src.strip("\n").splitlines(True)))


md(r"""
# MODI-SYNTH · Notebook 07 — a Reader for unseen pages, and heavier synthetic lines

**Run with: Accelerator `GPU T4 x2`, Internet `On`, input `colab_pack.zip` attached (the same one notebooks 04/05 use).
Then *Save Version → Save & Run All (Commit)* and come back in about 2.5 hours.** Nothing needs watching.

## Why this notebook exists
Stage 1 ends at test CER 24.55 on MoDeTrans. On unseen documents (the Folder-18 book) the same Reader scores 41.8 on
single-column letters, and worst on **tilted pages** and on **very large or very small writing** (error is U-shaped in
characters per frame). Separately, a Reader trained only on font-rendered lines stays at ~80 CER on real handwriting
however much synthetic data it sees (the "synthetic ceiling"). This notebook asks two questions with the notebook-05 code
(same line finder, model, training loop, LM decoding, report), one change per job:

| job | question | id |
|---|---|---|
| `PT_A60` → `FT_A0_s0`, `FT_A0_s1` | the anchor of notebook 05 (plain synthetic → real lines), two seeds | anchor |
| `AUG`, `AUG_s1` | + light augmentation of the real lines (notebook 05's best change) = **the control here** | N6 |
| `AUG_STRETCH` | light augmentation + horizontal stretch 0.75–1.4× (character density) — better than `AUG`? | R2 |
| `AUG_TILT` | light augmentation + rotation up to ±5°, re-cropped (tilted pages) — better than `AUG`? | R3 |
| `AUG_STRONG` | stretch + tilt + blur — better than `AUG`? | R4 |
| `PT_D60` → `HEAVY_AUG` | pretrain on **heavily distorted** synthetic lines (stronger shear, wobble, stroke and rotation), then the same fine-tuning as `AUG` — better? and does its **zero-shot** CER on real lines (logged during pretraining) break the ~80 ceiling of `PT_A60`? | R1 |
| final phase | character-LM decoding grid, checkpoint ensemble, **test once**, REPORT.md | — |

The plain synthetic style and the light augmentation are byte-identical to notebook 05's, so the controls are the same.
The answer on unseen pages comes afterwards, offline: `research/root-cause/offline/s11_folder18_ood.py` with these
checkpoints.

## Rules fixed before the run
* **VAL-SEL** (first 150 validation patches) picks every epoch, LM weights and the final pipeline. **VAL-REP** (the other
  157) is only reported. **TEST** is scored once, for the anchor and for the pipeline VAL-SEL picks.
* A change is called **better/worse** only if the paired-bootstrap 95% CI excludes zero **and** the difference is larger
  than twice the anchor's seed-to-seed spread. Otherwise: inconclusive.

## What was checked before this was handed over
* Built from `notebooks/lr07/` (copies of the notebook-05 code that ran on Kaggle, with the two post-run bug fixes) and
  rehearsed end to end on a CPU with a few patches (`notebooks/lr07/rehearse_nb07.py`): every job type, the planner, the
  scheduler, LM decoding and the report.
* Unit checks: plain rendering and light augmentation identical to notebook 05; the new modes keep 40-px lines within the
  width cap (on real lines: tilt 0.60–1.00×, stretch 0.81–1.34× of the original width).

## What was NOT checked
* GPU speed (the notebook benchmarks the GPU first and sizes every job to the budget), the CUDA mixed-precision path,
  and whether Kaggle's Pillow can shape Modi text (Cell 10 checks and says so loudly).

## Outputs (all in `/kaggle/working`)
`REPORT.md`, `summary.csv`, `logs.txt`, `results_bundle.zip` (download this), `runs/<job>/` (log, history, predictions,
`best.pt`), `figures/`.
""")

code(r"""
# Cell 1 - environment, GPUs, and the time budget (the whole notebook finishes in about 2.5 hours)
import os, sys, time, json, re, math, glob, shutil, subprocess, zipfile, base64, hashlib, random
T_START = time.time()
LOCAL = os.environ.get('LR05_LOCAL') == '1'      # set ONLY by the CPU rehearsal script; False on Kaggle
BUDGET_MIN = 40 if LOCAL else 150                # start -> REPORT.md (notebook 07: about 2.5 h)
FINAL_RESERVE_MIN = 7 if LOCAL else 20           # kept for LM decoding, ensembles, the test score, the report
COMBO_RESERVE_MIN = 0                            # notebook 07 has no combination / FINAL phase
OUT = '/kaggle/working'
WORK = ('/kaggle/temp' if os.path.isdir('/kaggle/temp') else '/tmp') + '/lr05'      # big scratch files, not saved
CODE, RUNS, FIGS = f'{OUT}/lr05_code', f'{OUT}/runs', f'{OUT}/figures'
for d in (WORK, CODE, RUNS, FIGS):
    os.makedirs(d, exist_ok=True)
DEADLINE_ALL = T_START + BUDGET_MIN * 60
DEADLINE_JOBS = DEADLINE_ALL - FINAL_RESERVE_MIN * 60

_LOGF = open(f'{OUT}/logs.txt', 'a', encoding='utf-8')
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); _LOGF.write(s + '\n'); _LOGF.flush()
def clock():
    return time.strftime('%H:%M:%S', time.gmtime(time.time() - T_START))

smi = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.strip()
GPUS = [int(l.split()[1].rstrip(':')) for l in smi.splitlines() if l.startswith('GPU')]
import torch, platform
log(f'MODI-SYNTH notebook 07 | started {time.strftime("%Y-%m-%d %H:%M:%S")} | budget {BUDGET_MIN} min | rehearsal: {LOCAL}')
log(f'python {platform.python_version()} | torch {torch.__version__} | CUDA {torch.version.cuda} | CPUs {os.cpu_count()}')
log(smi or 'NO GPU FOUND')
try:
    import psutil
    log(f'RAM {psutil.virtual_memory().total / 2**30:.1f} GB | free disk for scratch files: {shutil.disk_usage(WORK).free / 2**30:.0f} GB')
except Exception:
    pass
assert GPUS or LOCAL, 'Turn on the GPU: Settings -> Accelerator -> GPU T4 x2'
if len(GPUS) == 1:
    log('NOTE: only one GPU is visible. Everything still runs, one job at a time; the planner shortens jobs to fit.')
""")

code(r"""
# Cell 2 - libraries Kaggle does not ship
!pip -q install datasets editdistance sacrebleu rapidfuzz 2>&1 | tail -1
log('libraries installed')
""")

code(r"""
# Cell 3 - find colab_pack.zip anywhere under /kaggle/input and unpack the metadata (as in notebook 04)
DATA = f'{WORK}/metadata_unzipped'
os.makedirs(DATA, exist_ok=True)
need = ['splits.json', 'modetrans_text.csv', 'charset.json']

def find_metadata():
    for root, _, files in os.walk('/kaggle/input'):
        if all(n in files for n in need):
            return root
    for root, _, files in os.walk('/kaggle/input'):
        for f in files:
            if f.endswith('.zip'):
                zipfile.ZipFile(os.path.join(root, f)).extractall(DATA)
                for r2, _, f2 in os.walk(DATA):
                    if all(n in f2 for n in need):
                        return r2
    return None

META = find_metadata()
assert META, 'Could not find the metadata. Add colab_pack.zip as an input dataset (Add Input -> Upload).'
log('metadata:', META)
""")

code(r"""
# Cell 4 - MoDeTrans images + our fixed splits and cleaned labels (identical to notebooks 02-04 and the CPU work)
import numpy as np, pandas as pd
from datasets import load_dataset
for attempt in range(4):
    try:
        ds = load_dataset('historyHulk/MoDeTrans', split='train')     # 2,043 rows
        break
    except Exception as e:
        log(f'download problem (attempt {attempt + 1}): {e}'); time.sleep(30)
names = list(ds['filename'])
splits = json.load(open(f'{META}/splits.json', encoding='utf-8'))['modetrans']['assignment']
text = pd.read_csv(f'{META}/modetrans_text.csv', keep_default_na=False).set_index('image_id')['text_train'].to_dict()
cs = json.load(open(f'{META}/charset.json', encoding='utf-8'))
assert len(names) == 2043 and set(names) == set(splits) == set(text), 'dataset / split mismatch'
SPLIT = {s: [n for n in names if splits[n] == s] for s in ('train', 'val', 'test')}
assert not (set(SPLIT['train']) & (set(SPLIT['val']) | set(SPLIT['test']))), 'leakage!'
if LOCAL:      # rehearsal only: a handful of patches so every code path runs in minutes on a CPU
    SPLIT = {'train': SPLIT['train'][:40], 'val': SPLIT['val'][:22], 'test': SPLIT['test'][:10]}
n_sel = 12 if LOCAL else 150
VAL_SEL, VAL_REP = SPLIT['val'][:n_sel], SPLIT['val'][n_sel:]
TEST, TR150 = SPLIT['test'], SPLIT['train'][:10 if LOCAL else 150]
KEEP = set(SPLIT['train']) | set(SPLIT['val']) | set(SPLIT['test'])
log(f"splits: train {len(SPLIT['train'])} | val {len(SPLIT['val'])} = VAL-SEL {len(VAL_SEL)} (picks epochs and settings) "
    f"+ VAL-REP {len(VAL_REP)} (only reported) | test {len(TEST)} (scored once, at the end)")
META_JSON = dict(symbols=cs['symbols'], train_texts=[text[n] for n in SPLIT['train']], text={n: text[n] for n in KEEP},
                 val_sel=VAL_SEL, val_rep=VAL_REP, test=TEST, train150=TR150, patch_info={})
json.dump(META_JSON, open(f'{WORK}/meta.json', 'w', encoding='utf-8'), ensure_ascii=False)
""")

md(r"""
### Cells 5–9 write the code files
They are written to `/kaggle/working/lr05_code/` and are the tested files verbatim: `lr05_lib.py` (line finder,
renderer, model, decoding, metrics), `render_pools.py` (background synthetic-line renderer), `job.py` (one experiment
on one GPU), `lr05_run.py` (experiment matrix, planner, scheduler), `lr05_final.py` (LM decoding, verdicts, report).
""")

for fn in FILES:
    code(f"%%writefile /kaggle/working/lr05_code/{fn}\n" + (HERE / fn).read_text(encoding="utf-8"))

font_bytes = FONT.read_bytes()
code("# Cell 10a - Noto Sans Modi (SIL Open Font License 1.1, notofonts.github.io), embedded so no extra input is needed\n"
     f"FONT_MD5 = '{hashlib.md5(font_bytes).hexdigest()}'\n"
     f"FONT_B64 = '{base64.b64encode(font_bytes).decode()}'\n")

code(r"""
# Cell 10 - the font, a check that Modi text is shaped correctly, and the synthetic-line renderer (background)
sys.path.insert(0, CODE)
import lr05_lib as L, lr05_run as LR, lr05_final as LF
FONT = f'{WORK}/NotoSansModi-Regular.ttf'
open(FONT, 'wb').write(base64.b64decode(FONT_B64))
assert hashlib.md5(open(FONT, 'rb').read()).hexdigest() == FONT_MD5, 'font file corrupted'

def raqm_ok():
    r = subprocess.run([sys.executable, '-c', "from PIL import features; print(features.check('raqm'))"],
                       capture_output=True, text=True)
    return r.stdout.strip() == 'True'

RAQM = raqm_ok()
if not RAQM and not LOCAL:
    log('Pillow cannot shape complex scripts here (raqm/fribidi missing). Trying to install them ...')
    subprocess.run('apt-get -qq update && apt-get -qq install -y libraqm0 libfribidi0 libharfbuzz0b',
                   shell=True, timeout=400, capture_output=True)
    RAQM = raqm_ok()
    if not RAQM:
        subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', '--upgrade', 'pillow'], timeout=400, capture_output=True)
        RAQM = raqm_ok()
log(f'complex-script shaping (raqm) for the renderer: {"OK" if RAQM else "NOT AVAILABLE"}')
if not RAQM:
    log('WARNING: synthetic Modi will be drawn WITHOUT shaping (vowel signs may sit in the wrong place). Every synthetic '
        'result is still produced, but read the synthetic-data comparisons with that in mind.')

# Notebook 07: pool A = plain synthetic lines (as notebook 05), pool D = HEAVILY distorted synthetic lines; 60k each
SIZES = (dict(A=300, A_small=300, A_mid=300, A_big=300, A_h64=0, D=300, D_mid=300) if LOCAL else
         dict(A=60000, A_small=60000, A_mid=60000, A_big=60000, A_h64=0, D=60000, D_mid=60000))
POOL_PLAN = [dict(name='A', n=SIZES['A'], style='plain', text='labels', seed=1, milestones=[SIZES['A_mid']], h64_limit=0),
             dict(name='D', n=SIZES['D'], style='heavy', text='labels', seed=4, milestones=[SIZES['D_mid']])]
json.dump(POOL_PLAN, open(f'{WORK}/pool_plan.json', 'w'))
N_RENDER = max(1, min(3, (os.cpu_count() or 4) - 1))
RENDER_LOG = f'{OUT}/render_log.txt'
RENDERER = subprocess.Popen([sys.executable, '-u', f'{CODE}/render_pools.py', '--work', WORK, '--font', FONT,
                             '--plan', f'{WORK}/pool_plan.json', '--procs', str(N_RENDER)],
                            stdout=open(RENDER_LOG, 'w', encoding='utf-8'), stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONIOENCODING='utf-8'),
                            # lowest CPU priority: the GPU benchmark and the training jobs always get the CPU first
                            preexec_fn=(lambda: os.nice(19)) if hasattr(os, 'nice') else None)
log(f'[{clock()}] renderer started in the background with {N_RENDER} processes: '
    + ', '.join(f"pool {p['name']} = {p['n']:,} {p['style']} lines (text: {p['text']})" for p in POOL_PLAN))

# a look at what the renderer draws (saved for the record)
import matplotlib, warnings, logging
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', message='Glyph .* missing from')                  # Devanagari titles: no log flood
warnings.filterwarnings('ignore', message='Matplotlib currently does not support Devanagari')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
plt.rcParams['font.family'] = ['Nirmala UI', 'Noto Sans Devanagari', 'Lohit Devanagari', 'FreeSans', 'DejaVu Sans']
_rng = np.random.default_rng(5)
_chunks = L.label_chunks([L.normalise(t).replace('<FRAC>', '') for t in META_JSON['train_texts']], _rng)
fig, ax = plt.subplots(8, 1, figsize=(15, 8))
for k in range(8):
    style = 'heavy' if k % 2 else 'plain'
    lab = _chunks[int(_rng.integers(len(_chunks)))]
    o = L.render_line(lab, _rng, FONT, (40,), heavy=style == 'heavy')
    if o is not None:
        ax[k].imshow(o[40], cmap='gray', vmin=0, vmax=255)
    ax[k].set_title(f'{style}: {lab}', fontsize=8, loc='left'); ax[k].axis('off')
fig.tight_layout(); fig.savefig(f'{FIGS}/synthetic_examples_main_process.png', dpi=80); plt.close(fig)
""")

code(r"""
# Cell 11 - real line images: each patch's own line pitch -> lines -> tight crop -> 40 px and 64 px versions,
# patch label split across its lines at word boundaries (re-aligned later by the EM jobs). Same code as the CPU work.
t0 = time.time()
W40, W64 = L.LineWriter(f'{WORK}/real_h40', 40), L.LineWriter(f'{WORK}/real_h64', 64)
PINFO, n_done = {}, 0
for r in ds:
    n = r['filename']
    if n not in KEEP:
        continue
    g = np.array(r['image'].convert('L'))
    lines, st = L.patch_to_lines(g, text[n], (40, 64))
    PINFO[n] = st
    for x in lines:
        info = dict(patch=n, split=splits[n], idx=x['idx'], n_lines=x['n_lines'])
        W40.add(x['imgs'][40], x['label'], info); W64.add(x['imgs'][64], x['label'], info)
    n_done += 1
    if n_done % 250 == 0:
        log(f'  [{clock()}] {n_done}/{len(KEEP)} patches -> {len(W40.widths):,} lines')
W40.close(); W64.close()
META_JSON['patch_info'] = PINFO
json.dump(META_JSON, open(f'{WORK}/meta.json', 'w', encoding='utf-8'), ensure_ascii=False)
REAL = L.LineStore(f'{WORK}/real_h40')
d = pd.DataFrame(REAL.info); d['w'] = REAL.widths; d['chars'] = [len(l) for l in REAL.labels]
N_TRAIN_LINES = int((d.split == 'train').sum())
log(f'[{clock()}] {len(REAL):,} line images from {len(PINFO):,} patches in {time.time() - t0:.0f}s'
    + ('' if LOCAL else f"  (the CPU-verified set has 5,901: {'MATCH' if len(REAL) == 5901 else 'DIFFERENT - check'})"))
log('lines per patch:', pd.Series([v['n_lines'] for v in PINFO.values()]).value_counts().sort_index().to_dict())
log('lines by split:', d.split.value_counts().to_dict())
log(f"width at 40 px: median {d.w.median():.0f}, 95% {d.w.quantile(.95):.0f} | characters per line: median {d.chars.median():.0f} | "
    f"CTC frames per character (stride 2): median {(d.w / 2 / d.chars.clip(lower=1)).median():.1f}")
pit = np.array([v['pitch'] for v in PINFO.values()])
log(f'line pitch per patch: 5% {np.percentile(pit, 5):.0f} | median {np.median(pit):.0f} | 95% {np.percentile(pit, 95):.0f} px')
ex = d[d.split == 'train'].sample(min(12, N_TRAIN_LINES), random_state=0).index
fig, ax = plt.subplots(len(ex), 1, figsize=(16, 1.1 * len(ex)))
for a, i in zip(np.atleast_1d(ax), ex):
    a.imshow(REAL.img(i), cmap='gray', vmin=0, vmax=255); a.axis('off')
    a.set_title(f"{REAL.info[i]['patch']} line {REAL.info[i]['idx'] + 1}/{REAL.info[i]['n_lines']} -> {REAL.labels[i]}", fontsize=8, loc='left')
fig.tight_layout(); fig.savefig(f'{FIGS}/real_line_examples.png', dpi=70); plt.close(fig)
""")

code(r"""
# Cell 12 - measure this GPU, then size every job so the whole plan fits the budget
bench_spec = dict(tag='BENCH', work=WORK, run_dir=f'{RUNS}/_bench', size='small', H=40, vocab='char', no_space=False,
                  deadline=DEADLINE_JOBS,
                  bench=[dict(size='small', H=40), dict(size='big', H=40), dict(size='small', H=64), dict(size='big', H=64)])
if LOCAL:
    bench_spec.update(cpu=True, threads=2, bench=[dict(c, warm=1, steps=3) for c in bench_spec['bench']])
os.makedirs(bench_spec['run_dir'], exist_ok=True)
json.dump(bench_spec, open(f'{RUNS}/_bench/spec.json', 'w'))
rc = subprocess.run([sys.executable, '-u', f'{CODE}/job.py', '--spec', f'{RUNS}/_bench/spec.json']
                    + ([] if LOCAL else ['--gpu', str(GPUS[0])]), capture_output=True, text=True,
                    env=dict(os.environ, PYTHONIOENCODING='utf-8'))
for line in (rc.stdout + rc.stderr).splitlines():
    log('[bench]', line)
BENCH = json.load(open(f'{RUNS}/_bench/bench.json'))

BASE = (dict(pre_samples=320, ft_epochs=2, nopre_epochs=3, long_ft_epochs=4, long_pre_samples=640, batch=16) if LOCAL else
        dict(pre_samples=300000, ft_epochs=40, nopre_epochs=80, long_ft_epochs=120, long_pre_samples=900000, batch=32))
JOBS = LR.experiment_matrix(RUNS, BASE, SIZES)
avail = max(1, len(GPUS)) * (DEADLINE_JOBS - COMBO_RESERVE_MIN * 60 - time.time())
need = sum(LR.estimate_s(j, BENCH, N_TRAIN_LINES) for j in JOBS)          # every experiment, priority 1-3
if not LOCAL and 0.88 * avail < need:
    # quality first: rather than shrink the experiments, take up to one extra hour (Kaggle allows 12 h per session)
    extra = min(3600.0, (need / 0.88 - avail) / max(1, len(GPUS)))
    DEADLINE_JOBS += extra; DEADLINE_ALL += extra; avail += max(1, len(GPUS)) * extra
    log(f'GPU slower than planned -> budget extended by {extra / 60:.0f} min (to {(DEADLINE_ALL - T_START) / 3600:.1f} h) '
        f'to keep every experiment at full size')
# quality first: budgets only shrink if the GPU is much slower than expected, and GROW (up to 2x) if it is faster
SCALE = min(2.0, max(0.5, 0.88 * avail / need))
LR.apply_scale(JOBS, SCALE, BASE)
BASE_SCALED = dict(BASE, pre_samples=int(BASE['pre_samples'] * SCALE), ft_epochs=max(4 if not LOCAL else 1, int(round(BASE['ft_epochs'] * SCALE))))
log(f'[{clock()}] GPU throughput (training lines/s): ' + ', '.join(f"{k} {v['lines_per_s']:,.0f}" for k, v in BENCH.items()))
log(f'all experiments need ~{need / 60:.0f} GPU-min at base size; {avail / 60:.0f} GPU-min available before the final models '
    f'-> every budget x {SCALE:.2f} (all jobs scaled equally, so comparisons stay fair)')
log(f"{'job':10s} {'P':>2s} {'hypothesis':12s} {'pretrain lines':>15s} {'ft epochs':>9s} {'est. min':>8s}  what changes")
for j in sorted(JOBS, key=lambda j: j['priority']):
    log(f"{j['tag']:10s} {j['priority']:>2d} {','.join(j['hyp']):12s} {(j['pretrain'] or {}).get('samples', 0):>15,} "
        f"{(j['finetune'] or {}).get('epochs', 0):>9} {LR.estimate_s(j, BENCH, N_TRAIN_LINES) / 60:8.1f}  {j['desc']}")
""")

code(r"""
# Cell 13 - THE RUN: every job, two at a time (one per GPU), highest priority first, streamed below as [job] lines.
# Each job logs its loss, speed, blank share, predicted length, and after every epoch the validation CER, a sample
# reading and the error mix. A dashboard is printed every 5 minutes. Nothing here needs watching.
log(f'[{clock()}] starting the scheduler; jobs must finish by {time.strftime("%H:%M:%S", time.localtime(DEADLINE_JOBS))} '
    f'(local time of this machine); priority-3 jobs only start if they leave {COMBO_RESERVE_MIN} min for the combination and FINAL models')
SCHED = LR.Scheduler(JOBS, GPUS or [0, 1], f'{CODE}/job.py', WORK, RUNS, DEADLINE_JOBS, BENCH, N_TRAIN_LINES, log,
                     poll=2 if LOCAL else 10, renderer=RENDERER, reserve={3: COMBO_RESERVE_MIN * 60},
                     cpu=LOCAL, threads=2, tails=[(RENDER_LOG, 'render')])
STATE = SCHED.run()
log(f'[{clock()}] main phase finished:', {s: [t for t, v in STATE.items() if v == s] for s in set(STATE.values())})
for t, why in SCHED.notes.items():
    log(f'   {t}: {why}')
""")

code(r"""
# Cell 14 - notebook 07 has no combination / FINAL phase (every job above answers its own question)
log(f'[{clock()}] notebook 07: no combination phase')

# any time still left goes to the priority-3 jobs that were skipped for lack of time
LEFT = [j for j in JOBS if STATE.get(j['tag']) == 'skipped' and 'time' in SCHED.notes.get(j['tag'], '')]
if LEFT and time.time() < DEADLINE_JOBS - 180:
    log(f'[{clock()}] leftover time {(DEADLINE_JOBS - time.time()) / 60:.1f} min -> retrying skipped jobs: {[j["tag"] for j in LEFT]}')
    SCHED3 = LR.Scheduler(LEFT, GPUS or [0, 1], f'{CODE}/job.py', WORK, RUNS, DEADLINE_JOBS, BENCH, N_TRAIN_LINES, log,
                          poll=2 if LOCAL else 10, renderer=RENDERER, cpu=LOCAL, threads=2, known=STATE, ratios=SCHED.ratios)
    log(f'[{clock()}] leftover phase:', SCHED3.run())
""")

code(r"""
# Cell 15 - FINAL PHASE: verdict for every hypothesis, character-LM decoding (N3), ensemble (N15), space restoration
# for the no-space model (N5), the TEST score (once), figures and REPORT.md
if RENDERER.poll() is None:
    RENDERER.terminate()
log(f'[{clock()}] final phase, {(DEADLINE_ALL - time.time()) / 60:.1f} min left in the budget')
dev = 'cuda' if (GPUS and not LOCAL and torch.cuda.is_available()) else 'cpu'
REP = LF.final_phase(OUT, WORK, META_JSON, log, device=dev, procs=max(1, (os.cpu_count() or 2)),
                     time_budget_s=max(180, DEADLINE_ALL - time.time() - 90), quick=LOCAL)
LF.figures(OUT, LF.load_results(RUNS), log)
RUN_INFO = (f"GPUs: {smi.replace(chr(10), '; ') or 'none'}; budget scale {SCALE:.2f}; raqm shaping {'on' if RAQM else 'OFF'}; "
            f"total time {(time.time() - T_START) / 60:.0f} min")
LF.write_report(OUT, REP, RUN_INFO, log)
""")

code(r"""
# Cell 16 - the report, and one zip with every small output file (download this one)
print(open(f'{OUT}/REPORT.md', encoding='utf-8').read())
with zipfile.ZipFile(f'{OUT}/results_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in ['REPORT.md', 'summary.csv', 'logs.txt', 'render_log.txt', 'lm_grid_val_sel.csv', 'pipelines_val.csv',
              'final_report.json'] + [os.path.relpath(f, OUT) for f in glob.glob(f'{OUT}/test_predictions_*.csv')]:
        if os.path.exists(f'{OUT}/{p}'):
            z.write(f'{OUT}/{p}', p)
    for f in glob.glob(f'{FIGS}/*.png') + [f for f in glob.glob(f'{RUNS}/*/*')
                                           if f.endswith(('.txt', '.csv', '.json', '.png')) and not f.endswith('spec.json')]:
        z.write(f, os.path.relpath(f, OUT))
log(f'[{clock()}] DONE. results_bundle.zip = {os.path.getsize(f"{OUT}/results_bundle.zip") / 2**20:.1f} MB | '
    f'total {(time.time() - T_START) / 60:.0f} min')
""")

md(r"""
## Reading the output
* **`REPORT.md`** answers every question: the TEST headline, a verdict per hypothesis (BETTER / WORSE / INCONCLUSIVE, with
  the difference and its 95% CI), decoding and ensemble results, every job, and the error anatomy (N1/N2).
* **Per job** (`runs/<job>/log.txt`): pretraining logs every 100 steps (loss, learning rate, gradient norm, blank share,
  predicted vs true length, lines/s, memory) and the zero-shot CER on REAL validation lines 4 times; fine-tuning logs every
  epoch (validation CER, no-space CER, WER, BLEU, chrF++, patches below 50%, length ratio, blank share) and every 5 epochs a
  sample reading, the deletion/insertion/substitution mix and the recall of the small marks (space, ा, ्, र, ं, ि, ी, ु, ू, े, ो).
* **Zero-shot realism:** `runs/PT_A60/log.txt` and `runs/PT_D60/log.txt` log the CER on REAL validation lines of a model
  trained on synthetic lines only (4 times during pretraining) - plain vs heavy synthetic.
* REPORT.md's TEST table says which pipeline VAL-SEL picked and how it decodes; checkpoints are `runs/<job>/best.pt`.
* **Reference points:** reading nothing scores ~84 CER; the letter-frequency plateau is a CTC loss of 3.32; on the same
  TEST set notebook 03 scored CER 73.1, SPAN v1 74.4, and the CPU baseline (X3a + character LM) **36.3**.
* Send back `results_bundle.zip` (or `REPORT.md` + `logs.txt`).
""")

nb = dict(cells=CELLS, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                     language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {NB} ({NB.stat().st_size / 1024:.0f} KB, {len(CELLS)} cells)")
