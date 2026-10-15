"""Strip 6, step 1 - log-probs of every 40-px character model from notebook 05, for every VAL and TEST line, on CPU.
Integrity check: greedy VAL-REP CER per model must reproduce the notebook's own numbers (it ran fp16 on a T4; CPU is
fp32, so tiny differences are possible)."""
# Moved into the repo on 15 Oct 2026 from a Claude-session scratchpad; only the file paths changed (now repo-relative).
import json, os, pickle, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent                         # research/root-cause/offline
CODE = HERE.parents[2]                                         # FYP-modi-script-code
sys.path.insert(0, str(CODE / "notebooks/lr05"))
import lr05_lib as L

RUNS = str(CODE / "notebooks/outputs-of-notebook/05-notebook-kaggle/extracted/runs")
CACHE = os.environ.get("MODI_S6_CACHE", str(HERE / "cache"))   # log-prob pickles, ~66 MB each (git-ignored)
WORK = os.environ.get("MODI_WORK", CACHE + "/work_full")       # meta.json + real_h40 lines: notebooks/lr05/verify_against_cpu.py
OUT = str(HERE / "s6")                                         # small results, kept in git
os.makedirs(CACHE, exist_ok=True)
TAGS = ["AUG", "BIG", "B60", "A120", "FT_A0_s0", "FT_A0_s1", "FT_A0_s2", "MIX", "FT_LONG", "PRE_LONG", "C60", "A12"]
torch.set_num_threads(10)

if __name__ == "__main__":
    META = json.load(open(WORK + "/meta.json", encoding="utf-8"))
    store = L.LineStore(WORK + "/real_h40")
    bp = {}
    for i, inf in enumerate(store.info):
        bp.setdefault(inf["patch"], []).append(i)
    for p in bp:
        bp[p].sort(key=lambda i: store.info[i]["idx"])
    patches = META["val_sel"] + META["val_rep"] + META["test"]
    vocab = L.make_vocab(META["symbols"], META["train_texts"], "char")
    reported = {t: json.load(open(f"{RUNS}/{t}/result.json", encoding="utf-8"))["val_rep"]["CER"] for t in TAGS}
    for t in TAGS:
        t0 = time.time()
        ck = torch.load(f"{RUNS}/{t}/best.pt", map_location="cpu", weights_only=False)
        m = L.build_crnn(len(vocab), ck["size"]); m.load_state_dict(ck["model"]); m.eval()
        lps = {}
        with torch.no_grad():
            for p in patches:
                b = bp.get(p, [])
                if not b:
                    continue
                x, lens = L.to_batch([store.img(i) for i in b], 40, round_to=L.STRIDE)
                lp = m(torch.from_numpy(x)).float().log_softmax(-1).numpy()
                for k, i in enumerate(b):
                    lps[i] = lp[k, :lens[k]].astype(np.float16)
        pickle.dump(lps, open(f"{CACHE}/lp_{t}.pkl", "wb"))
        hyps = [" ".join(s for s in (vocab.decode(L.greedy_ids(lps[i].astype(np.float32))).strip() for i in bp.get(p, [])) if s)
                for p in META["val_rep"]]
        cer = L.metrics([META["text"][p] for p in META["val_rep"]], hyps)["CER"]
        print(f"{t:9s} VAL-REP greedy CER {cer:6.2f} (notebook on T4: {reported[t]:6.2f}, diff {cer - reported[t]:+.2f}) | {time.time()-t0:.0f}s", flush=True)
    json.dump(dict(bp={p: bp.get(p, []) for p in patches}), open(f"{OUT}/bp.json", "w"))
    print("DONE", flush=True)
