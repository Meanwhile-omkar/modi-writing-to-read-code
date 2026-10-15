"""X2 step 2 - can a small line-level reader learn to read, on CPU, in <= 15 minutes? (hypotheses H8, H9, H22)

Trains a small CRNN (conv stack + BiLSTM + CTC) on the pseudo-labelled line images from x2_build_lines.py and scores
it the SAME way every Kaggle run was scored: on whole validation patches (the first 150 in dataset order), joining
each patch's line predictions with spaces and comparing with the full patch label. Also reports CER with spaces
removed (spaces are not visible in Modi, H3), BLEU (the anchor paper's metric), and CER on 150 TRAINING patches
(memorising vs reading).
Reference points: no-reading baseline ~84 CER; 03 test CER 73.1; SPAN v1 val CER ~75 after 123 GPU epochs.

usage: python x2_train_lines.py [--minutes 13] [--seed 0] [--tag x2a] [--aug 0]
"""
import argparse, json, math, pickle, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import editdistance, sacrebleu

HERE = Path(__file__).resolve().parent
CODE = Path(__file__).resolve().parents[3]
RESULTS = HERE.parent / "results.csv"
ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=13.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="x2a")
ap.add_argument("--aug", type=int, default=0)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--data", default="cache/lines_h40.pkl")
ap.add_argument("--init", default="")
ap.add_argument("--train-data", default="", help="train on this pickle instead (e.g. synthetic lines); evaluation stays on real val patches")
args = ap.parse_args()
torch.set_num_threads(10)
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

D = pickle.load(open(HERE / args.data, "rb"))
cs = json.load(open(CODE / "assets/metadata/charset.json", encoding="utf-8"))
VOCAB = ["<blank>"] + cs["symbols"] + ["<FRAC>", "<unk>"]
STOI = {c: i for i, c in enumerate(VOCAB)}
import re
TOK = re.compile(r"<FRAC>|.", re.S)
enc = lambda s: [STOI.get(t, STOI["<unk>"]) for t in TOK.findall(s)]
dec = lambda ids: "".join(VOCAB[i] for i in ids if i > 0)
text = pd.read_csv(CODE / "assets/metadata/modetrans_text.csv", keep_default_na=False).set_index("image_id")["text_train"].to_dict()

STRIDE = 2                                                       # output frames = width / 2
lines = D["lines"]
for L in lines:
    L["y"] = enc(L["label"])
if args.train_data:
    src = pickle.load(open(HERE / args.train_data, "rb"))["lines"]
    for L in src:
        L["y"] = enc(L["label"])
    train = [L for L in src if 0 < len(L["y"]) and len(L["y"]) * 1.2 + 2 <= L["img"].shape[1] / STRIDE]
    dropped = len(src) - len(train)
    print(f"TRAINING ON {args.train_data} ({len(src):,} lines); evaluating on REAL validation patches", flush=True)
else:
    train = [L for L in lines if L["split"] == "train" and 0 < len(L["y"]) and len(L["y"]) * 1.2 + 2 <= L["img"].shape[1] / STRIDE]
    dropped = sum(1 for L in lines if L["split"] == "train") - len(train)
patch_order = [p["patch"] for p in D["patches"]]                 # parquet order == dataset order
split_of = {p["patch"]: p["split"] for p in D["patches"]}
val_patches = [p for p in patch_order if split_of[p] == "val"][:150]
tr_patches = [p for p in patch_order if split_of[p] == "train"][:150]
by_patch = {}
for L in lines:
    by_patch.setdefault(L["patch"], []).append(L)
print(f"train lines {len(train):,} (dropped {dropped} whose label is too long for the width) | "
      f"eval on {len(val_patches)} val patches and {len(tr_patches)} train patches", flush=True)


class CRNN(nn.Module):
    def __init__(self, n=len(VOCAB), c=(32, 64, 128, 128, 128), hid=128):
        super().__init__()
        L, cin = [], 1
        for i, co in enumerate(c):
            L += [nn.Conv2d(cin, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True)]
            if i == 0: L.append(nn.MaxPool2d(2))              # 40 -> 20, width / 2
            if i in (1, 2): L.append(nn.MaxPool2d((2, 1)))    # 20 -> 10 -> 5
            cin = co
        self.cnn = nn.Sequential(*L)
        self.rnn = nn.LSTM(cin, hid, 2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(2 * hid, n)

    def forward(self, x):
        f = self.cnn(x).mean(2).permute(0, 2, 1)                 # (B, W/2, C)
        return self.fc(self.rnn(f)[0])                            # (B, T, classes)


def augment(img, rng):
    import cv2
    h, w = img.shape
    if rng.random() < 0.5:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-2, 2), rng.uniform(0.92, 1.05))
        M[0, 1] += rng.uniform(-0.15, 0.15)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    if rng.random() < 0.4:
        k = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, k) if rng.random() < 0.5 else cv2.dilate(img, k)
    return img


def batchify(items, aug=False, rng=None):
    W = max(L["img"].shape[1] for L in items)
    W = int(math.ceil(W / STRIDE) * STRIDE)
    x = torch.ones(len(items), 1, 40, W)
    for i, L in enumerate(items):
        im = augment(L["img"], rng) if aug else L["img"]
        x[i, 0, :, :im.shape[1]] = torch.from_numpy(im.astype(np.float32) / 255.0)
    x = (x - 0.5) / 0.5
    lens = torch.tensor([int(math.ceil(L["img"].shape[1] / STRIDE)) for L in items])
    return x, lens


@torch.no_grad()
def read_patches(model, patches):
    model.eval()
    hyps, refs = [], []
    for p in patches:
        ls = sorted(by_patch.get(p, []), key=lambda L: L["idx"])
        if not ls:
            hyps.append(""); refs.append(text[p]); continue
        x, lens = batchify(ls)
        out = model(x).argmax(-1).numpy()
        parts = []
        for i, L in enumerate(ls):
            s = out[i, :lens[i]]
            keep = np.ones_like(s, bool); keep[1:] = s[1:] != s[:-1]; keep &= s != 0
            parts.append(dec(s[keep]).strip())
        hyps.append(" ".join(t for t in parts if t)); refs.append(text[p])
    model.train()
    return refs, hyps


def metrics(refs, hyps):
    u = lambda s: s.replace("<FRAC>", "")
    e = sum(editdistance.eval(u(r), u(h)) for r, h in zip(refs, hyps)); n = sum(len(u(r)) for r in refs)
    ns = lambda s: u(s).replace(" ", "")
    e2 = sum(editdistance.eval(ns(r), ns(h)) for r, h in zip(refs, hyps)); n2 = sum(len(ns(r)) for r in refs)
    per = [editdistance.eval(u(r), u(h)) / max(1, len(u(r))) for r, h in zip(refs, hyps)]
    return dict(CER=100 * e / n, CER_nospace=100 * e2 / n2, BLEU=sacrebleu.corpus_bleu(hyps, [refs]).score,
                best_patch_CER=100 * min(per), patches_below_50=int(sum(c < 0.5 for c in per)),
                pred_len=float(np.mean([len(h) for h in hyps])), ref_len=float(np.mean([len(r) for r in refs])))


model = CRNN()
if args.init:
    model.load_state_dict(torch.load(HERE / args.init, map_location="cpu"))
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
print(f"model {sum(p.numel() for p in model.parameters())/1e6:.2f} M params | lr {args.lr} | batch {args.batch} | aug {args.aug} | budget {args.minutes} min", flush=True)
rng = np.random.default_rng(args.seed)
t0 = time.time(); ep = 0; history = []
order = sorted(range(len(train)), key=lambda i: train[i]["img"].shape[1])    # bucket by width for speed
while time.time() - t0 < args.minutes * 60:
    ep += 1
    batches = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
    random.shuffle(batches)
    tot = n = 0
    for b in batches:
        if time.time() - t0 > args.minutes * 60:
            break
        items = [train[i] for i in b]
        x, lens = batchify(items, aug=bool(args.aug), rng=rng)
        out = model(x).log_softmax(-1).permute(1, 0, 2)
        y = torch.tensor([t for L in items for t in L["y"]]); yl = torch.tensor([len(L["y"]) for L in items])
        loss = F.ctc_loss(out, y, lens, yl, blank=0, zero_infinity=True)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        tot += loss.item(); n += 1
    refs, hyps = read_patches(model, val_patches[:60])
    m = metrics(refs, hyps)
    history.append(dict(epoch=ep, loss=tot / max(1, n), **m, minutes=(time.time() - t0) / 60))
    print(f"ep {ep:2d} | train loss {tot/max(1,n):.3f} | val(60 patches) CER {m['CER']:5.1f}  no-space {m['CER_nospace']:5.1f}  "
          f"BLEU {m['BLEU']:5.1f}  best patch {m['best_patch_CER']:5.1f}  len {m['pred_len']:.0f}/{m['ref_len']:.0f} | "
          f"{(time.time()-t0)/60:4.1f} min", flush=True)
    print(f"   GT  : {refs[0][:90]}\n   PRED: {hyps[0][:90]}", flush=True)

train_minutes = (time.time() - t0) / 60
refs, hyps = read_patches(model, val_patches); mv = metrics(refs, hyps)
refs_t, hyps_t = read_patches(model, tr_patches); mt = metrics(refs_t, hyps_t)
torch.save(model.state_dict(), HERE / "cache" / f"{args.tag}_model.pt")
pd.DataFrame(history).to_csv(HERE / "cache" / f"{args.tag}_history.csv", index=False)
pd.DataFrame({"patch": val_patches, "ref": refs, "hyp": hyps}).to_csv(HERE / "cache" / f"{args.tag}_val_predictions.csv", index=False)
print(f"\nFINAL after {ep} epochs / {train_minutes:.1f} min:")
print(f"  VAL   150 patches: CER {mv['CER']:.1f} | no-space {mv['CER_nospace']:.1f} | BLEU {mv['BLEU']:.2f} | "
      f"patches with CER<50%: {mv['patches_below_50']}/150 | best patch {mv['best_patch_CER']:.1f}")
print(f"  TRAIN 150 patches: CER {mt['CER']:.1f} | no-space {mt['CER_nospace']:.1f} | BLEU {mt['BLEU']:.2f}")
for i in range(3):
    print(f"   GT  : {refs[i][:100]}\n   PRED: {hyps[i][:100]}")
row = [args.tag, time.strftime("%Y-%m-%d"), "H8 H9 H22",
       f"\"CRNN 1.07M conv(32-64-128-128-128)+BiLSTM2x128, lines h40 stride2, Adam lr {args.lr}, batch {args.batch}, aug {args.aug}, {ep} epochs, train={args.train_data or 'real lines'}, init={args.init or 'random'}\"",
       args.seed, f"{len(train)} train lines; eval 150 val patches", f"train-patch CER {mt['CER']:.1f}",
       f"{mv['CER']:.2f}", f"{mv['CER_nospace']:.2f}", f"{mv['BLEU']:.2f}", f"{train_minutes:.1f}", "",
       f"\"patches CER<50%: {mv['patches_below_50']}/150; best patch {mv['best_patch_CER']:.1f}\""]
with open(RESULTS, "a", encoding="utf-8") as f:
    f.write(",".join(str(v) for v in row) + "\n")
print("logged to results.csv")
