# Omkar — notebook 07 on Kaggle (Fri 16 Oct, night), about 2.5 hours of 2× T4

Different from Udit's notebook 06 (bigger / combined models for the in-domain test). Notebook 07 asks:
1. **Can the Reader be trained to cope with what broke on the unseen book?** Three new ways of varying the real training
   lines, each against the standard light augmentation: **stretch** (wider/narrower writing — our error is worst for very big
   and very small writing), **tilt** (rotation up to ±5°, re-cropped — tilted pages), **strong** (both + blur).
2. **Can rougher synthetic lines break the "synthetic ceiling"?** A model trained only on font lines reads real handwriting at
   ~80 CER. Notebook 07 pretrains a second model on **heavily distorted** font lines and logs its zero-shot CER on real
   lines next to the plain one, then fine-tunes it like the control.

## How to run
1. `git pull` (the notebook is `notebooks/07_robust_reader_GPU.ipynb`; it was rehearsed end to end on this PC's CPU — see
   the commit message).
2. Kaggle → New Notebook → File → Import Notebook → upload `07_robust_reader_GPU.ipynb`.
3. Settings: **Accelerator GPU T4 ×2**, **Internet On**. Add Input → your existing **colab_pack.zip** dataset (the same one
   notebook 05 used).
4. **Save Version → Save & Run All (Commit)**. Close the tab; it runs by itself (~2.5 h) and ends with REPORT.md.
5. When it is done: open the version → **Output** → download everything (it includes `runs/<job>/best.pt`, about 4 MB each).
   Unzip into `notebooks/outputs-of-notebook/07-notebook-kaggle/` on this PC.

## What happens with the results (Claude does this)
- REPORT.md gives, for every question, BETTER / WORSE / INCONCLUSIVE against the control with a 95 % CI (validation), and
  the test score once.
- Unseen pages: `python research/root-cause/offline/s11_folder18_ood.py read_runs notebooks/outputs-of-notebook/07-notebook-kaggle/runs AUG,AUG_STRETCH,AUG_TILT,AUG_STRONG,HEAVY_AUG nb07`
  then `... s11_folder18_ood.py score` → do stretch/tilt help on the unseen Folder-18 letters (41.8 now)?
- Synthetic ceiling: zero-shot CER of `PT_D60` vs `PT_A60` (in their `log.txt`).

If the run fails, send `logs.txt` from the output; nothing else needs watching.
