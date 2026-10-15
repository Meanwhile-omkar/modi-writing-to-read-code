"""22 - Export every Stage-1 Reader pipeline's TEST predictions in the team's results format (contract C1).

For each pipeline this writes results/<run>/ with
  test_predictions.csv   image_id,reference,prediction (UTF-8), one row per MoDeTrans test patch, all runs in the
                         same row order (the test order of assets/metadata/splits.json)
  config.json            what the pipeline is and which saved file the predictions were copied from
  metrics.json           scores recomputed here with notebooks/lr05/lr05_lib.metrics, the code behind every Stage-1 number

Nothing is re-run: predictions are copied from the saved run outputs. The export stops with an error unless
  * the image ids are exactly the 306 test patches of assets/metadata/splits.json,
  * every reference equals text_train of that image in assets/metadata/modetrans_text.csv,
  * the recomputed CER equals the CER logged for that run (to the precision it was logged with).

Note on lr05_lib.metrics: it removes the <FRAC> token before scoring (no test patch contains one) and its BLEU is
sacreBLEU's default WORD BLEU. The team's metric library (src/modi/evaluation/metrics.py) counts <FRAC> as one
character and reports CHARACTER BLEU, the number comparable with the MoDeTrans paper.

usage: python scripts/22_export_stage1_predictions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "notebooks" / "lr05"))
import lr05_lib  # noqa: E402

NB05 = "notebooks/outputs-of-notebook/05-notebook-kaggle/extracted"
X7 = "research/root-cause/experiments/cache"
OUT = CODE / "results"
TEST_SET = "MoDeTrans test split, 306 patches (assets/metadata/splits.json)"

# run name, source file, id column, prediction column, logged test CER, decimals it was logged with, description
RUNS = [
    ("s1_offline_ens4_lm7", "research/root-cause/offline/s6/test_predictions_new_pipeline.csv", "patch", "prediction", 24.55, 2,
     "BEST Stage-1 pipeline. Average of the per-frame probabilities of four 40-px line CRNNs from notebook 05 "
     "(AUG, BIG, MIX, B60), CTC prefix beam search (beam 10, top-8) with a character 7-gram LM (alpha 0.3, beta 1.25) "
     "built from training transcriptions only. Chosen on VAL-SEL; test scored once. research/root-cause/offline/s6_search.py C"),
    ("s1_nb05_final_ens3_lm4", f"{NB05}/test_predictions_ensemble.csv", "patch", "lm_beam", 25.91, 2,
     "Notebook 05's own final pipeline: ensemble AUG+BIG+B60, character 4-gram LM (alpha 0.3, beta 1.0)."),
    ("s1_nb05_final_ens3_greedy", f"{NB05}/test_predictions_ensemble.csv", "patch", "greedy", 32.46, 2,
     "Notebook 05's ensemble AUG+BIG+B60, greedy CTC decoding (no LM)."),
    ("s1_nb05_anchor_lm4", f"{NB05}/test_predictions_anchor.csv", "patch", "lm_beam", 32.20, 2,
     "Notebook 05 anchor (FT_A0_s0: plain synthetic-line pretraining -> real lines, no augmentation), character 4-gram LM (alpha 0.45, beta 1.0)."),
    ("s1_nb05_anchor_greedy", f"{NB05}/test_predictions_anchor.csv", "patch", "greedy", 36.88, 2,
     "Notebook 05 anchor (FT_A0_s0), greedy CTC decoding. Single-model reference for Arm comparisons."),
    ("s1_cpu_x7_lm4", f"{X7}/x7_test_predictions_lm_beam.csv", "patch", "prediction", 36.33, 2,
     "First working CPU baseline (X3a line CRNN, ~40 CPU-minutes of training), character 4-gram LM (alpha 0.3, beta 1.5)."),
    ("s1_cpu_x7_greedy", f"{X7}/x7_test_predictions_greedy.csv", "patch", "prediction", 44.88, 2,
     "First working CPU baseline (X3a), greedy CTC decoding."),
    ("s1_span_v2_iam", "notebooks/outputs-of-notebook/kaggle-output-span-version2/test_predictions_iam_init.txt", "image_id", "prediction", 69.90, 2,
     "Notebook 04, faithful SPAN paragraph model, encoder initialised from the authors' IAM line model. Memorises its training pages."),
    ("s1_span_v2_scratch", "notebooks/outputs-of-notebook/kaggle-output-span-version2/test_predictions_scratch.txt", "image_id", "prediction", 72.56, 2,
     "Notebook 04, faithful SPAN paragraph model trained from scratch. Does not read."),
    ("s1_nb03_strip_crnn", "notebooks/outputs-of-notebook/kaggle-output-v2-03/test_predictions.txt", "image_id", "prediction", 73.1, 1,
     "Notebook 03, lines joined into one strip, CRNN + BiLSTM. Its fixed-height line finder destroyed most inputs."),
    ("s1_span_v1", "notebooks/outputs-of-notebook/kaggle-output-span-version1/test_predictions.txt", "image_id", "prediction", 74.36, 2,
     "First, unfaithful SPAN-style paragraph model. Learned how much text a page holds, not which letters."),
    ("s1_nb02_band_crnn", "notebooks/outputs-of-notebook/kaggle-output-1/test_predictions.txt", "image_id", "prediction", 81.32, 2,
     "Notebook 02, lines cut and stacked into fixed bands, CRNN. Best epoch 1; 74 of 306 predictions empty."),
]


def main():
    splits = json.load(open(CODE / "assets/metadata/splits.json", encoding="utf-8"))["modetrans"]["assignment"]
    test_order = [k for k, v in splits.items() if v == "test"]
    test_ids = set(test_order)
    text = (pd.read_csv(CODE / "assets/metadata/modetrans_text.csv", keep_default_na=False, dtype=str)
            .set_index("image_id")["text_train"].to_dict())
    table = []
    for run, src, id_col, pred_col, logged, dec, desc in RUNS:
        df = pd.read_csv(CODE / src, keep_default_na=False, dtype=str)
        if len(df) != df[id_col].nunique() or set(df[id_col]) != test_ids:
            sys.exit(f"{run}: image ids are not exactly the {len(test_ids)} test patches")
        df = df.set_index(id_col).loc[test_order].reset_index()  # one row order for every run
        ids, refs, hyps = list(df[id_col]), list(df["reference"]), list(df[pred_col])
        bad = [i for i, r in zip(ids, refs) if r != text[i]]
        if bad:
            sys.exit(f"{run}: {len(bad)} references differ from text_train, e.g. {bad[:3]}")
        m = lr05_lib.metrics(refs, hyps)
        tol = 0.5 * 10 ** -dec + 1e-9
        if abs(m["CER"] - logged) > tol:
            sys.exit(f"{run}: recomputed CER {m['CER']:.4f} differs from the logged {logged} (tolerance {tol:.3f})")
        d = OUT / run
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dict(image_id=ids, reference=refs, prediction=hyps)).to_csv(
            d / "test_predictions.csv", index=False, encoding="utf-8", lineterminator="\n")
        json.dump(dict(run=run, stage="1 (Reader baseline, tag stage1-final-baseline)", description=desc,
                       test_set=TEST_SET, source_file=src, source_column=pred_col,
                       exported_by="scripts/22_export_stage1_predictions.py"),
                  open(d / "config.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        json.dump(dict(m, logged_CER=logged, metric_code="notebooks/lr05/lr05_lib.py:metrics (<FRAC> removed; BLEU = word BLEU)"),
                  open(d / "metrics.json", "w", encoding="utf-8"), indent=1)
        table.append((run, m["CER"], logged, m["CER_nospace"], m["patches_below_50"]))
        print(f"{run:28s} CER {m['CER']:6.2f} (logged {logged}) | no-space {m['CER_nospace']:6.2f} | "
              f"patches < 50 %: {m['patches_below_50']:3d}/{m['n']} | OK", flush=True)
    print(f"\nexported {len(table)} runs to {OUT}")


if __name__ == "__main__":
    main()
