"""X3 - re-split each training patch's label across its lines with CTC forced alignment (hypothesis H9).

X2 split a patch label across lines in proportion to line width, cutting at spaces. Now that a model reads (X4b,
val CER 52.2), its per-frame outputs can place the line boundaries better: the full patch label is force-aligned
(CTC Viterbi) to the patch's lines laid end to end, and the label is cut where the alignment crosses from one line's
frames into the next line's frames, snapped to the nearest space.
Writes cache/lines_h40_realigned.pkl (same format as lines_h40.pkl) and reports how many line labels changed.
usage: python x3_realign_labels.py --model cache/x4b_model.pt
"""
import argparse, json, math, pickle, re, time, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE = Path(__file__).resolve().parent
CODE = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="cache/x4b_model.pt"); ap.add_argument("--out", default="cache/lines_h40_realigned.pkl")
args = ap.parse_args()
torch.set_num_threads(10)

import sys
sys.path.insert(0, str(HERE))
import linemodel as LM                                   # same CRNN as x2_train_lines.py; loads with strict=True

D = pickle.load(open(HERE / "cache/lines_h40.pkl", "rb"))
VOCAB, STOI, TOK = LM.VOCAB, LM.STOI, LM.TOK
text = pd.read_csv(CODE / "assets/metadata/modetrans_text.csv", keep_default_na=False).set_index("image_id")["text_train"].to_dict()
model = LM.load(HERE / args.model)


def viterbi_ctc(logp, labels):
    """Best CTC path of `labels` through log-probs (T, C). Returns, for every frame, the index of the label
    character it is assigned to (or -1 for a blank between characters, attributed to the previous character)."""
    T = logp.shape[0]; L = len(labels)
    S = 2 * L + 1
    ext = np.zeros(S, dtype=int); ext[1::2] = labels
    NEG = -1e30
    dp = np.full((T, S), NEG); bp = np.zeros((T, S), dtype=np.int8)
    dp[0, 0] = logp[0, 0]
    if S > 1: dp[0, 1] = logp[0, ext[1]]
    allow2 = np.zeros(S, bool)
    allow2[3::2] = ext[3::2] != ext[1:-2:2]
    for t in range(1, T):
        prev = dp[t - 1]
        c0 = prev
        c1 = np.concatenate([[NEG], prev[:-1]])
        c2 = np.where(allow2, np.concatenate([[NEG, NEG], prev[:-2]]), NEG)
        stack = np.stack([c0, c1, c2])
        k = stack.argmax(0)
        dp[t] = stack[k, np.arange(S)] + logp[t, ext]
        bp[t] = k
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    path = np.empty(T, dtype=int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= bp[t, s]
    char_of_frame = np.where(path % 2 == 1, path // 2, -1)
    last = -1
    for t in range(T):                                   # blanks belong to the character before them
        if char_of_frame[t] >= 0: last = char_of_frame[t]
        else: char_of_frame[t] = last
    return char_of_frame, float(dp[T - 1, path[-1]])


by_patch = {}
for L in D["lines"]:
    by_patch.setdefault(L["patch"], []).append(L)
t0 = time.time(); changed = total = skipped = 0
new_lines = []
with torch.no_grad():
    for p, ls in by_patch.items():
        ls = sorted(ls, key=lambda L: L["idx"])
        if ls[0]["split"] != "train" or len(ls) == 1:
            new_lines += ls; continue
        label = text[p]
        toks = TOK.findall(label)
        ids = [STOI.get(t, STOI["<unk>"]) for t in toks]
        frames, owner = [], []
        for i, L in enumerate(ls):
            lp = model(LM.to_tensor(L["img"])).log_softmax(-1)[0].numpy()
            frames.append(lp); owner += [i] * lp.shape[0]
        logp = np.concatenate(frames); owner = np.array(owner)
        if len(ids) * 1.0 > logp.shape[0]:
            new_lines += ls; skipped += 1; continue
        cof, score = viterbi_ctc(logp, np.array(ids))
        # the first character index assigned to each line's frames = where that line's text starts
        starts = [0]
        for i in range(1, len(ls)):
            fr = np.where(owner == i)[0]
            ci = cof[fr]; ci = ci[ci >= 0]
            starts.append(int(ci.min()) if len(ci) else starts[-1])
        # snap each start to the nearest space (keep words whole), keep them increasing
        spaces = [j for j, t in enumerate(toks) if t == " "]
        cuts = []
        for st in starts[1:]:
            cand = [s for s in spaces if (not cuts or s > cuts[-1])]
            cuts.append(min(cand, key=lambda s: abs(s + 1 - st)) if cand else (cuts[-1] if cuts else 0))
        pieces, b = [], 0
        for c in cuts:
            pieces.append("".join(toks[b:c]).strip()); b = c + 1
        pieces.append("".join(toks[b:]).strip())
        for L, piece in zip(ls, pieces):
            total += 1
            if piece != L["label"]: changed += 1
            new_lines.append(dict(L, label=piece))
pickle.dump(dict(lines=new_lines, patches=D["patches"], H=D["H"]), open(HERE / args.out, "wb"))
print(f"re-aligned training lines: {changed:,} of {total:,} line labels changed ({100*changed/max(1,total):.0f}%), "
      f"{skipped} patches skipped (label longer than frames) | {time.time()-t0:.0f}s")
