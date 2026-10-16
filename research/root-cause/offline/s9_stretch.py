"""Track 9 - does horizontal character density explain the size effect? Test-time horizontal stretching, no retraining.
AUG model (notebook 05's best single), greedy decoding, all 307 val patches; VAL-SEL picks, VAL-REP reports.
  fixed: every line stretched by s (width x s, height 40)
  adaptive: first pass at s=1 -> estimated characters per line (predicted length x c) -> stretch each line so it has
            `target` frames per character, clipped to [0.5, 2.5]; target and c chosen on VAL-SEL."""
# Moved into the repo on 15 Oct 2026 from a Claude-session scratchpad; only the file paths changed (now repo-relative).
import json, math, os, pickle, sys, time
from pathlib import Path
import numpy as np, torch, cv2
HERE = Path(__file__).resolve().parent                         # research/root-cause/offline
CODE = HERE.parents[2]                                         # FYP-modi-script-code
sys.path.insert(0, str(CODE / "notebooks/lr05"))
import lr05_lib as L
CACHE = os.environ.get("MODI_S6_CACHE", str(HERE / "cache"))   # log-prob pickles, ~66 MB each (git-ignored)
WORK = os.environ.get("MODI_WORK", CACHE + "/work_full")       # meta.json + real_h40 lines: notebooks/lr05/verify_against_cpu.py
RUNS = str(CODE / "notebooks/outputs-of-notebook/05-notebook-kaggle/extracted/runs")
torch.set_num_threads(4)

if __name__ == "__main__":
    M = json.load(open(WORK + "/meta.json", encoding="utf-8")); store = L.LineStore(WORK + "/real_h40")
    bp = {}
    for i, inf in enumerate(store.info): bp.setdefault(inf["patch"], []).append(i)
    for p in bp: bp[p].sort(key=lambda i: store.info[i]["idx"])
    vocab = L.make_vocab(M["symbols"], M["train_texts"], "char")
    ck = torch.load(f"{RUNS}/AUG/best.pt", map_location="cpu", weights_only=False)
    m = L.build_crnn(len(vocab), ck["size"]); m.load_state_dict(ck["model"]); m.eval()
    VS, VR = M["val_sel"], M["val_rep"]

    def read(patches, scale_of):
        hyps = []
        with torch.no_grad():
            for p in patches:
                ls = bp.get(p, [])
                imgs = []
                for i in ls:
                    im = store.img(i); s = scale_of(i)
                    imgs.append(im if abs(s - 1) < 1e-6 else cv2.resize(im, (max(8, int(round(im.shape[1] * s))), 40), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR))
                if not imgs:
                    hyps.append(""); continue
                x, lens = L.to_batch(imgs, 40, round_to=L.STRIDE)
                lp = m(torch.from_numpy(x)).float().log_softmax(-1).numpy()
                hyps.append(" ".join(t for t in (vocab.decode(L.greedy_ids(lp[k, :lens[k]])).strip() for k in range(len(ls))) if t))
        return hyps

    cer = lambda P, H: L.metrics([M["text"][p] for p in P], H)["CER"]
    t0 = time.time()
    base_vs = read(VS, lambda i: 1.0); base_vr = read(VR, lambda i: 1.0)
    print(f"no stretch: VAL-SEL {cer(VS, base_vs):.2f} | VAL-REP {cer(VR, base_vr):.2f} | {time.time()-t0:.0f}s", flush=True)
    for s in (0.7, 1.3, 1.6):
        print(f"fixed stretch x{s}: VAL-SEL {cer(VS, read(VS, lambda i: s)):.2f} | {time.time()-t0:.0f}s", flush=True)
    # first pass: predicted characters per line at s = 1
    first = {}
    with torch.no_grad():
        for p in VS + VR:
            ls = bp.get(p, [])
            if not ls: continue
            x, lens = L.to_batch([store.img(i) for i in ls], 40, round_to=L.STRIDE)
            lp = m(torch.from_numpy(x)).float().log_softmax(-1).numpy()
            for k, i in enumerate(ls):
                first[i] = max(1, len(vocab.decode(L.greedy_ids(lp[k, :lens[k]])).replace(" ", "")))
    best = None
    for target in (3.5, 4.0, 4.5):
        for c in (1.0, 1.2):
            sc = lambda i, t=target, c=c: float(np.clip(t * first.get(i, 1) * c / max(1.0, store.widths[i] / 2), 0.5, 2.5))
            v = cer(VS, read(VS, sc))
            print(f"adaptive target {target} frames/char, length correction {c}: VAL-SEL {v:.2f} | {time.time()-t0:.0f}s", flush=True)
            if best is None or v < best[0]:
                best = (v, target, c)
    v, target, c = best
    sc = lambda i: float(np.clip(target * first.get(i, 1) * c / max(1.0, store.widths[i] / 2), 0.5, 2.5))
    r = read(VR, sc)
    print(f"BEST adaptive (target {target}, c {c}): VAL-SEL {v:.2f} -> VAL-REP {cer(VR, r):.2f} (no stretch {cer(VR, base_vr):.2f}) | {time.time()-t0:.0f}s", flush=True)
    os.makedirs(CACHE, exist_ok=True); pickle.dump(dict(target=target, c=c), open(CACHE + "/s9_best.pkl", "wb"))
