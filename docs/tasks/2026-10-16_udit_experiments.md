# Udit — two experiments for tonight (Fri 16 Oct → Sat morning)

Both are new work (Omkar is doing the answer key and the presentation). Pull `main` first (this file is in the repo: `docs/tasks/2026-10-16_udit_experiments.md`).
**Not in git (licence / book terms) — Omkar shares these on Drive:** `colab_pack.zip` (Kaggle input), the 14
handwritten Folder-18 pages (`folder18_handwritten_pages_for_udit/`, place them under `hhdoc_root/18/` from
`configs/paths.yaml`) and the answer-key folder `assets/folder18_gt/` (copy it into your clone at the same path).
Read `results/README.md` once: it is the whole Stage-1 story in one page.

**About the GPU.** Our training code uses an NVIDIA GPU (CUDA) or the CPU. A Mac's GPU (MPS) cannot be used — CTC loss,
which every model here trains with, does not run on MPS. So: **heavy training → Kaggle on your own account** (free 2× T4,
your weekly GPU quota is unused); **your Mac → the CPU rehearsal and Experiment 2.**

---

## Experiment 1 — "Notebook 06": stronger ensemble members (Kaggle, 2× T4, about 5 h)

**Question.** Notebook 05 showed three changes that each help — augmentation (AUG −4.0 CER), a 2× wider model (BIG −1.9)
and 64-px lines (H64 −1.3) — but never trained them TOGETHER (that run was starved by a bug that is now fixed). Do
BIG+AUG and 64-px+AUG models beat AUG alone, and does an ensemble of them beat our best test CER of 24.55?

**What to change** — only `notebooks/lr05/lr05_run.py`, function `experiment_matrix` (the list of `add(...)` calls):
keep `PT_A60` exactly as it is, delete every other `add(...)`, and add these (same helpers `pre`, `ft`, `PTA`, sizes):
```python
add("AUG_s1", 1, ["N6"], "AUG, seed 1", init=PTA, finetune=ft(aug=True, seed=1), after=["PT_A60"])
add("AUG_s2", 1, ["N6"], "AUG, seed 2", init=PTA, finetune=ft(aug=True, seed=2), after=["PT_A60"])
add("BIG_AUG_s0", 1, ["N11+N6"], "2x wider CRNN + augmentation", size="big",
    pretrain=pre("A", a_mid), finetune=ft(aug=True, seed=0), pools=[("A", 40, a_mid)])
add("BIG_AUG_s1", 2, ["N11+N6"], "2x wider CRNN + augmentation, seed 1", size="big",
    pretrain=pre("A", a_mid), finetune=ft(aug=True, seed=1), pools=[("A", 40, a_mid)])
add("H64_AUG", 2, ["N7+N6"], "64-px lines + augmentation", H=64,
    pretrain=pre("A", min(a_mid, sizes["A_h64"])), finetune=ft(aug=True), pools=[("A", 64, min(a_mid, sizes["A_h64"]))])
add("EM1_AUG", 3, ["H9"], "AUG + one CTC re-alignment of the line labels after 1/3 (Viterbi bug fixed)",
    init=PTA, finetune=ft(aug=True, realign_frac=[1 / 3]), after=["PT_A60"])
add("H64_BIG_AUG", 3, ["N7+N11+N6"], "64-px + 2x wider + augmentation (slowest: ~190 lines/s)", size="big", H=64,
    pretrain=pre("A", min(a_mid, sizes["A_h64"])), finetune=ft(aug=True), pools=[("A", 64, min(a_mid, sizes["A_h64"]))])
```
Then switch off the automatic COMBO/FINAL phase in `build_nb05.py` (search `combo_jobs`; make it schedule nothing) — we
want the time spent on the models above. Rename the output to `notebooks/06_strong_members_GPU.ipynb` and the title.

**Rule: rehearse before using the GPU** (it caught dozens of bugs last time): on your Mac,
`LR05_LOCAL=1 python notebooks/lr05/rehearse_nb05.py` (about 40 min; needs the MoDeTrans parquet at
`../FYP-modi-script-data/MoDeTrans-dataset/train-00000-of-00001.parquet` — download from Hugging Face
`historyHulk/MoDeTrans`). Every job must reach "status ok" in the rehearsal. Then build the notebook and run it on Kaggle:
Accelerator **GPU T4 ×2**, Internet **On**, input **colab_pack.zip** (Omkar sends it; upload as a private dataset).
"Save & Run All", let it run (~5 h, it writes REPORT.md and a results zip at the end).

**What to report** (Saturday morning): the results zip (or at least each job's `result.json`, `history.csv`, `best.pt`)
and the job table from REPORT.md. Success = any new job beats AUG (VAL-REP greedy 36.50; AUG + LM VAL-SEL 24.85).
The ensemble with our existing models is then chosen on VAL-SEL and scored on test ONCE with
`research/root-cause/offline/s6_search.py` (we do that part).

---

## Experiment 2 — read two-column and tilted pages of the unseen book (Mac CPU, about 3 h)

**Question.** On the unseen Folder-18 letters our Reader scores 41.8 CER on single-column letters 12 + 13 but 63.2 on
letter 11, whose pages are written in **two columns** (51, 53, 55, 57, about 75 CER each); letter 12 gets worse as its pages tilt
(40 → 54; the deskew search stops at ±6° and pages 61 and 65 hit exactly −6.0°). How much of the out-of-domain gap is
page layout rather than handwriting?

**What exists.** `research/root-cause/offline/s11_folder18_ood.py` (cut lines → read with the frozen models → score per
letter). The handwritten pages are in the folder Omkar shared; the answer-key files (`assets/folder18_gt/`, outside git —
Omkar sends them) hold the key: letters 3, 5, 8, 12, 13, 15 proofread against the printed pages, letter 11 Tesseract only.
Run from the repo root: `python research/root-cause/offline/s11_folder18_ood.py lines|read|score`.

**What to try** (keep everything else identical, change one thing at a time, score after each):
1. **Tilt:** widen the deskew range from ±6° to ±12° for page-level deskew (`lr05_lib.deskew(g, rng=...)`), re-run
   `lines` → `read` → `score`. Report per-page CER for letter 12 before/after.
2. **Columns:** for pages whose ink has a clear empty vertical gutter, split the page at the gutter and cut lines in each
   column separately; read the left column top to bottom, then the right column. The key must follow the same order: run
   Tesseract on the left and right halves of the PRINTED pages 50, 52, 54, 56 (split at their gutter) instead of the whole
   page. Report letter 11 per-page CER before/after.
3. Check that letters 3, 5, 8, 13, 15 do not get worse (their pages are single-column).

**Deliver:** a branch `feat/folder18-layout` with the code and a short `outputs/reports/folder18_layout.md` (a table:
page, before, after; what changed). No page images or book text in git.

---
