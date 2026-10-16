"""X7 - score one line-reader checkpoint on the untouched TEST split, once (no training, no tuning on test).

Reports, for greedy decoding and for character-LM beam decoding with alpha/beta fixed beforehand (chosen on validation
in X6): CER, CER with spaces removed, BLEU (the anchor paper's metric), a 10,000-sample bootstrap 95% CI for CER over
patches, the per-patch CER distribution, and how many patches fall below 50% CER. Saves the predictions.
usage: python x7_test_eval.py --model cache/x4b_model.pt --alpha 0.3 --beta 1.0 --tag x7
"""
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, editdistance, sacrebleu

HERE = Path(__file__).resolve().parent
CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
import linemodel as LM

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--alpha", type=float, default=0.3); ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--data", default="cache/lines_h40.pkl"); ap.add_argument("--tag", default="x7")
args = ap.parse_args()
torch.set_num_threads(10)

# reuse the LM and the beam search from x6 without running x6's experiment
src = (HERE / "x6_lm_decode.py").read_text(encoding="utf-8")
head = src.split("model = LM.load")[0]                                  # everything up to the experiment itself
sys.argv = ["x6", "--model", args.model, "--data", args.data]
ns = {"__name__": "x6", "__file__": str(HERE / "x6_lm_decode.py")}
exec(compile(head, "x6_head", "exec"), ns)
beam_decode, greedy, text, split, order, by_patch = (ns["beam_decode"], ns["greedy"], ns["text"], ns["split"],
                                                     ns["order"], ns["by_patch"])

test = [p for p in order if split[p] == "test"]
model = LM.load(HERE / args.model)
t0 = time.time()
with torch.no_grad():
    cache = {p: [model(LM.to_tensor(L["img"])).log_softmax(-1)[0].numpy()
                 for L in sorted(by_patch[p], key=lambda L: L["idx"])] for p in test}
u = lambda s: s.replace("<FRAC>", "")
ns_ = lambda s: u(s).replace(" ", "")


def evaluate(fn, name):
    refs = [text[p] for p in test]
    hyps = [" ".join(t for t in (fn(lp).strip() for lp in cache[p]) if t) for p in test]
    e = np.array([editdistance.eval(u(r), u(h)) for r, h in zip(refs, hyps)], float)
    L = np.array([max(1, len(u(r))) for r in refs], float)
    e2 = sum(editdistance.eval(ns_(r), ns_(h)) for r, h in zip(refs, hyps)); n2 = sum(len(ns_(r)) for r in refs)
    rng = np.random.default_rng(0); idx = rng.integers(0, len(test), (10000, len(test)))
    boot = 100 * e[idx].sum(1) / L[idx].sum(1)
    per = e / L
    res = dict(decoder=name, CER=100 * e.sum() / L.sum(), CER_CI95=(float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
               CER_nospace=100 * e2 / n2, BLEU=sacrebleu.corpus_bleu(hyps, [refs]).score,
               median_patch_CER=100 * float(np.median(per)), patches_below_50=int((per < 0.5).sum()), n=len(test))
    pd.DataFrame({"patch": test, "reference": refs, "prediction": hyps, "cer": per}).to_csv(
        HERE / "cache" / f"{args.tag}_test_predictions_{name}.csv", index=False, encoding="utf-8")
    return res


r1 = evaluate(greedy, "greedy")
r2 = evaluate(lambda lp: beam_decode(lp, args.alpha, args.beta), "lm_beam")
for r in (r1, r2):
    print(f"TEST ({r['n']} patches) {r['decoder']:8s}: CER {r['CER']:.1f} (95% CI {r['CER_CI95'][0]:.1f}-{r['CER_CI95'][1]:.1f}) | "
          f"no-space {r['CER_nospace']:.1f} | BLEU {r['BLEU']:.2f} | median patch {r['median_patch_CER']:.1f} | "
          f"patches below 50%: {r['patches_below_50']}/{r['n']}")
json.dump([r1, r2], open(HERE / "cache" / f"{args.tag}_test_metrics.json", "w", encoding="utf-8"), indent=2)
row = [args.tag, time.strftime("%Y-%m-%d"), "final", f"\"TEST once: model {args.model}; LM alpha {args.alpha} beta {args.beta} fixed from val\"",
       0, f"{len(test)} test patches", f"greedy {r1['CER']:.2f}", f"{r2['CER']:.2f}", f"{r2['CER_nospace']:.2f}", f"{r2['BLEU']:.2f}",
       f"{(time.time()-t0)/60:.1f}", "", f"\"CI95 {r2['CER_CI95'][0]:.1f}-{r2['CER_CI95'][1]:.1f}; below50 {r2['patches_below_50']}/{r2['n']}; val_CER column = TEST CER here\""]
with open(HERE.parent / "results.csv", "a", encoding="utf-8") as f:
    f.write(",".join(str(v) for v in row) + "\n")
print("logged to results.csv")
