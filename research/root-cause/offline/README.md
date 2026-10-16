# Offline analyses of notebook 05 (CPU only) — includes the best Stage-1 result (test CER 24.55)

These scripts re-use notebook 05's checkpoints (`notebooks/outputs-of-notebook/05-notebook-kaggle/extracted/runs/*/best.pt`)
on a CPU. They were written and run on 15 Oct 2026 in a Claude-session scratchpad and moved here the same day; only the
file paths changed. Re-running `s6_search.py C` from this folder reproduced TEST 24.55 and a byte-identical
`s6/test_predictions_new_pipeline.csv` (15 Oct 2026).

| script | what it does | result (log) |
|---|---|---|
| `s6_logprobs.py` | log-probabilities of the 12 40-px character models for every VAL and TEST line; integrity check against the notebook's own VAL-REP CER | every model within ±0.04 CER of the T4 run (`logs/s6_logprobs.log`) |
| `s6_search.py A` | forward ensemble selection, scored by LM-decoded VAL-SEL CER | AUG → +BIG → +MIX → +B60, VAL-SEL 23.75 (`logs/s6_A.log`, `s6/stageA.json`) |
| `s6_search.py B` (+`B2`) | LM weight grid and LM order (4–8) on VAL-SEL | order 6–7 best; α 0.3, β 1.25 (`logs/s6_B.log`, `s6/stageB.json`) |
| `s6_search.py C` | VAL-REP check of the notebook pipeline (must give 29.15), then **TEST once** for the new pipeline | **TEST CER 24.55 (95 % CI 23.09–26.15)**, no-space 24.26 (`logs/s6_C.log`, `s6/stageC.json`, `s6/test_predictions_new_pipeline.csv`) |
| `s9_stretch.py` | test-time horizontal stretching of lines (character-density hypothesis), AUG model, greedy | VAL-REP 36.50 → 35.89 (`logs/s9.log`) |
| `span_v1_train_vs_val.py`, `span_v2_train_vs_val.py` | did the paragraph (SPAN) models memorise their training pages? | v2-IAM train 43.6 vs val 68.2 (`logs/span_v*_check.log`) |

Also kept: `logs/nb05_verify_against_cpu.log` (notebook-05 code re-checked against the CPU results before the GPU run)
and `logs/rehearse_nb05_final_build.log` (CPU rehearsal of the final notebook-05 build).

## Re-running
1. Line store + metadata (about 85 MB, git-ignored), built from the MoDeTrans parquet:
   `python notebooks/lr05/verify_against_cpu.py research/root-cause/offline/cache 0` → `cache/work_full/`
   (it also re-checks two CPU checkpoints; a few CPU minutes).
2. `python research/root-cause/offline/s6_logprobs.py` → `cache/lp_*.pkl` (12 × 66 MB, ~20 CPU minutes).
3. `python research/root-cause/offline/s6_search.py A`, then `B`, `B2`, `C` (stage C ≈ 1 minute).

Environment variables: `MODI_S6_CACHE` (folder of the `lp_*.pkl` files, default `cache/`), `MODI_WORK` (line store,
default `cache/work_full`). The BLEU printed by these scripts is sacreBLEU's default **word** BLEU from `lr05_lib.metrics`;
the character BLEU used by the team's metric library is reported in `results/README.md`.
