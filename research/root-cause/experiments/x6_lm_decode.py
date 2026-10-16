"""X6 - does a character language model in the decoder recover what the image cannot show? (hypothesis H23)

The text is predictable (char 4-gram: 2.10 nats/char vs 3.31 for letter frequencies) and parts of the label are not
visible at all (spaces 15.6%, vowel length). Here the trained line reader is decoded with a CTC prefix beam search
fused with a character n-gram LM built from TRAINING labels only. LM weight (alpha) and length bonus (beta) are
chosen on the first half of the validation patches and the result is reported on the second half, so the number is
not tuned on the patches it is reported on. No training.
usage: python x6_lm_decode.py --model cache/x4b_model.pt [--order 4] [--beam 10]
"""
import argparse, json, math, pickle, re, sys, time, collections
from pathlib import Path
import numpy as np, pandas as pd, torch
import editdistance

HERE = Path(__file__).resolve().parent
CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
import linemodel as LM

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="cache/x4b_model.pt"); ap.add_argument("--order", type=int, default=4)
ap.add_argument("--beam", type=int, default=10); ap.add_argument("--topk", type=int, default=8)
ap.add_argument("--data", default="cache/lines_h40.pkl"); ap.add_argument("--tag", default="x6")
ap.add_argument("--alphas", default="0.3,0.6,1.0"); ap.add_argument("--betas", default="0.0,0.5,1.0")
args = ap.parse_args()
torch.set_num_threads(10)

split = json.load(open(CODE / "assets/metadata/splits.json", encoding="utf-8"))["modetrans"]["assignment"]
text = pd.read_csv(CODE / "assets/metadata/modetrans_text.csv", keep_default_na=False).set_index("image_id")["text_train"].to_dict()
D = pickle.load(open(HERE / args.data, "rb"))
order = [p["patch"] for p in D["patches"]]
val = [p for p in order if split[p] == "val"][:150]
tune, report = val[:75], val[75:]
by_patch = collections.defaultdict(list)
for L in D["lines"]:
    by_patch[L["patch"]].append(L)


# ---- character n-gram LM with interpolated absolute discounting, trained on TRAINING labels only
class CharLM:
    def __init__(self, texts, n, d=0.75):
        self.n, self.d = n, d
        self.c = [collections.Counter() for _ in range(n + 1)]
        self.ctx = [collections.Counter() for _ in range(n + 1)]
        self.types = [collections.defaultdict(set) for _ in range(n + 1)]
        V = set()
        for t in texts:
            s = ["^"] * (n - 1) + LM.TOK.findall(t) + ["$"]
            V.update(s)
            for k in range(1, n + 1):
                for i in range(k - 1, len(s)):
                    h = tuple(s[i - k + 1:i]); w = s[i]
                    self.c[k][h + (w,)] += 1; self.ctx[k][h] += 1; self.types[k][h].add(w)
        self.V = len(V)

    def logp(self, hist, w):
        p = 1.0 / self.V
        for k in range(1, self.n + 1):
            h = tuple(hist[len(hist) - (k - 1):]) if k > 1 else ()
            if len(h) < k - 1: continue
            cn = self.ctx[k][h]
            if cn == 0: continue
            lam = self.d * len(self.types[k][h]) / cn
            p = max(self.c[k][h + (w,)] - self.d, 0) / cn + lam * p
        return math.log(p)


lm = CharLM([text[p] for p in order if split[p] == "train"], args.order)


def beam_decode(lp, alpha, beta):
    """CTC prefix beam search with shallow LM fusion. lp: (T, C) log-probs."""
    T, C = lp.shape
    hist0 = ("^",) * (args.order - 1)
    beams = {(): (0.0, -1e30)}                       # prefix -> (log p_blank, log p_nonblank)
    lmcache = {}
    def lmscore(prefix, c):
        key = (prefix[-(args.order - 1):] if args.order > 1 else (), c)
        if key not in lmcache:
            hist = list(hist0) + [LM.VOCAB[i] for i in prefix]
            lmcache[key] = lm.logp(hist, LM.VOCAB[c])
        return lmcache[key]
    lse = lambda a, b: max(a, b) + math.log1p(math.exp(-abs(a - b))) if min(a, b) > -1e29 else max(a, b)
    for t in range(T):
        top = np.argsort(lp[t])[-args.topk:]
        nb = collections.defaultdict(lambda: [-1e30, -1e30])
        for prefix, (pb, pnb) in beams.items():
            tot = lse(pb, pnb)
            for c in top:
                p = lp[t, c]
                if c == 0:
                    e = nb[prefix]; e[0] = lse(e[0], tot + p)
                    continue
                last = prefix[-1] if prefix else None
                ext = prefix + (int(c),)
                bonus = alpha * lmscore(prefix, int(c)) + beta
                if c == last:
                    e = nb[ext]; e[1] = lse(e[1], pb + p + bonus)      # repeat only after a blank
                    e2 = nb[prefix]; e2[1] = lse(e2[1], pnb + p)       # or collapse into the same char
                else:
                    e = nb[ext]; e[1] = lse(e[1], tot + p + bonus)
        beams = dict(sorted(((k, (v[0], v[1])) for k, v in nb.items()),
                            key=lambda kv: -lse(*kv[1]))[:args.beam])
    best = max(beams.items(), key=lambda kv: lse(*kv[1]))[0]
    return LM.dec(best)


def greedy(lp):
    s = lp.argmax(1); keep = np.ones_like(s, bool); keep[1:] = s[1:] != s[:-1]; keep &= s != 0
    return LM.dec(s[keep])


model = LM.load(HERE / args.model)
cache = {}
with torch.no_grad():
    for p in val:
        cache[p] = [model(LM.to_tensor(L["img"])).log_softmax(-1)[0].numpy()
                    for L in sorted(by_patch[p], key=lambda L: L["idx"])]


def score(patches, fn):
    u = lambda s: s.replace("<FRAC>", "")
    refs = [text[p] for p in patches]
    hyps = [" ".join(t for t in (fn(lp).strip() for lp in cache[p]) if t) for p in patches]
    e = sum(editdistance.eval(u(r), u(h)) for r, h in zip(refs, hyps)); n = sum(len(u(r)) for r in refs)
    ns = lambda s: u(s).replace(" ", "")
    e2 = sum(editdistance.eval(ns(r), ns(h)) for r, h in zip(refs, hyps)); n2 = sum(len(ns(r)) for r in refs)
    return 100 * e / n, 100 * e2 / n2


t0 = time.time()
g_tune, g_rep = score(tune, greedy), score(report, greedy)
print(f"greedy (no LM):  tune half CER {g_tune[0]:.1f} / no-space {g_tune[1]:.1f}  |  report half CER {g_rep[0]:.1f} / no-space {g_rep[1]:.1f}", flush=True)
best = None
for alpha in (float(v) for v in args.alphas.split(",")):
    for beta in (float(v) for v in args.betas.split(",")):
        c = score(tune, lambda lp: beam_decode(lp, alpha, beta))
        print(f"  tune: alpha {alpha} beta {beta} -> CER {c[0]:.1f} / no-space {c[1]:.1f}  ({time.time()-t0:.0f}s)", flush=True)
        if best is None or c[0] < best[0]:
            best = (c[0], alpha, beta)
_, a, b = best
r = score(report, lambda lp: beam_decode(lp, a, b))
print(f"\nREPORT half (75 val patches never used for tuning): greedy CER {g_rep[0]:.1f} -> LM beam (alpha {a}, beta {b}) CER {r[0]:.1f}"
      f"  | no-space {g_rep[1]:.1f} -> {r[1]:.1f}  | {time.time()-t0:.0f}s", flush=True)
row = [args.tag, time.strftime("%Y-%m-%d"), "H23" if args.tag == "x6" else "H23 N3",
       f"\"CTC prefix beam {args.beam} top{args.topk} + char {args.order}-gram LM (train labels only); alpha {args.alphas} x beta {args.betas} tuned on val[:75]; model {args.model}\"",
       0, "report on val[75:150]", f"greedy {g_rep[0]:.2f}", f"{r[0]:.2f}", f"{r[1]:.2f}", "-", f"{(time.time()-t0)/60:.1f}",
       "", f"\"greedy no-space {g_rep[1]:.2f}; best alpha {a} beta {b}\""]
with open(HERE.parent / "results.csv", "a", encoding="utf-8") as f:
    f.write(",".join(str(v) for v in row) + "\n")
print("logged to results.csv")
