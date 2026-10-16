"""Arm A on TEST, once: notebook 05's real-lines-only control (NOPRE: same CRNN, no synthetic pretraining, 2x the
fine-tuning epochs) scored on the 306 test patches, greedy and with a character 4-gram LM whose weights are chosen on
VAL-SEL only. Written 16 Oct 2026 for the Arm A vs Arm B table; the NOPRE checkpoint had never seen the test set.
Outputs: s10/test_predictions_nopre.csv (patch, reference, greedy, lm_beam) and s10/result.json.
Needs the line store from notebooks/lr05/verify_against_cpu.py (MODI_WORK, default cache/work_full)."""
import json, os, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent                         # research/root-cause/offline
CODE = HERE.parents[2]                                         # FYP-modi-script-code
sys.path.insert(0, str(CODE / "notebooks/lr05"))
import lr05_lib as L

CACHE = os.environ.get("MODI_S6_CACHE", str(HERE / "cache"))
WORK = os.environ.get("MODI_WORK", CACHE + "/work_full")
CKPT = CODE / "notebooks/outputs-of-notebook/05-notebook-kaggle/extracted/runs/NOPRE/best.pt"
OUT = HERE / "s10"
_W = {}


def _init(texts, itos):
    _W["lm"] = L.CharLM(texts, 4); _W["itos"] = itos


def _decode(task):
    lines, a, b = task
    return [(k, L.beam_decode(ti, tl, _W["itos"], _W["lm"], a, b).strip()) for k, ti, tl in lines]


def main():
    torch.set_num_threads(8)
    OUT.mkdir(exist_ok=True)
    M = json.load(open(WORK + "/meta.json", encoding="utf-8"))
    store = L.LineStore(WORK + "/real_h40")
    bp = {}
    for i, inf in enumerate(store.info):
        bp.setdefault(inf["patch"], []).append(i)
    for p in bp:
        bp[p].sort(key=lambda i: store.info[i]["idx"])
    vocab = L.make_vocab(M["symbols"], M["train_texts"], "char")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = L.build_crnn(len(vocab), ck["size"]); m.load_state_dict(ck["model"]); m.eval()
    VS, VR, TE = M["val_sel"], M["val_rep"], M["test"]
    t0 = time.time()
    lps = {}
    with torch.no_grad():
        for p in VS + VR + TE:
            b = bp.get(p, [])
            if b:
                x, lens = L.to_batch([store.img(i) for i in b], 40, round_to=L.STRIDE)
                lp = m(torch.from_numpy(x)).float().log_softmax(-1).numpy()
                for k, i in enumerate(b):
                    lps[i] = lp[k, :lens[k]]
    greedy = lambda P: [" ".join(s for s in (vocab.decode(L.greedy_ids(lps[i])).strip() for i in bp.get(p, [])) if s) for p in P]
    vr_cer = L.metrics([M["text"][p] for p in VR], greedy(VR))["CER"]
    print(f"NOPRE VAL-REP greedy CER {vr_cer:.2f} (notebook on T4: see runs/NOPRE/result.json) | {time.time()-t0:.0f}s", flush=True)
    pool = get_context("spawn").Pool(8, initializer=_init, initargs=(M["train_texts"], vocab.itos))

    def lm(P, a, b):
        lines = [(i, *L.topk_compress(lps[i], 8)) for p in P for i in bp.get(p, [])]
        out = {}
        for res in pool.imap_unordered(_decode, [(lines[s::40], a, b) for s in range(40)]):
            out.update(res)
        return [" ".join(s for s in (out[i] for i in bp.get(p, [])) if s) for p in P]

    grid = {}
    for a in (0.3, 0.45, 0.6):
        for b in (0.5, 1.0, 1.5):
            grid[(a, b)] = L.metrics([M["text"][p] for p in VS], lm(VS, a, b))["CER"]
    a, b = min(grid, key=grid.get)
    print(f"LM weights chosen on VAL-SEL: alpha {a} beta {b} (VAL-SEL {grid[(a, b)]:.2f}) | {time.time()-t0:.0f}s", flush=True)
    te_g, te_lm = greedy(TE), lm(TE, a, b)
    pool.close()
    refs = [M["text"][p] for p in TE]
    res = dict(val_rep_greedy=vr_cer, lm=[a, b, 4], val_sel_grid={str(k): v for k, v in grid.items()},
               test_greedy=L.metrics(refs, te_g), test_lm=L.metrics(refs, te_lm))
    pd.DataFrame(dict(patch=TE, reference=refs, greedy=te_g, lm_beam=te_lm)).to_csv(
        OUT / "test_predictions_nopre.csv", index=False, encoding="utf-8", lineterminator="\n")
    json.dump(res, open(OUT / "result.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"TEST (once) Arm A = NOPRE: greedy CER {res['test_greedy']['CER']:.2f} | with LM {res['test_lm']['CER']:.2f} "
          f"| {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
