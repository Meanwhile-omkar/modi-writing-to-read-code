"""SPAN v1 (finished Kaggle run): did the best checkpoint (epoch 105) memorise the TRAINING patches, or fail on both?
Model code copied verbatim from the v1 notebook (outputs-of-notebook/kaggle-output-span-version1). Greedy decoding,
one patch at a time (no padding), same preprocessing: grey -> height 320 keeping aspect -> (x/255-0.5)/0.5."""
# Moved into the repo on 15 Oct 2026 from a Claude-session scratchpad; only the file paths changed (now repo-relative).
import io, json, math, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torch.nn.functional as F, editdistance
from PIL import Image

CODE = Path(__file__).resolve().parents[3].as_posix()          # FYP-modi-script-code
ROOT = Path(CODE).parent.as_posix()                            # also holds FYP-modi-script-data
torch.set_num_threads(6)


class MixDropout(nn.Module):
    def __init__(s, p=0.5, p2d=0.25):
        super().__init__(); s.d, s.d2 = nn.Dropout(p), nn.Dropout2d(p2d)
    def forward(s, x): return s.d(x) if random.random() < 0.5 else s.d2(x)

def norm(c): return nn.InstanceNorm2d(c, eps=1e-3, affine=True)

class ConvBlock(nn.Module):
    def __init__(s, cin, cout, stride, drop):
        super().__init__()
        s.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        s.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        s.c3 = nn.Conv2d(cout, cout, 3, stride=stride, padding=1)
        s.n, s.dp = norm(cout), MixDropout(drop, drop/2)
    def forward(s, x):
        x = F.relu(s.c1(x)); x = s.dp(x)
        x = F.relu(s.c2(x)); x = s.n(x); x = s.dp(x)
        return F.relu(s.c3(x))

class DSCBlock(nn.Module):
    def __init__(s, c, drop):
        super().__init__()
        s.c1 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c), nn.Conv2d(c, c, 1))
        s.c2 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c), nn.Conv2d(c, c, 1))
        s.n, s.dp = norm(c), MixDropout(drop, drop/2)
    def forward(s, x):
        y = F.relu(s.c1(x)); y = s.dp(y)
        y = F.relu(s.c2(y)); y = s.n(y); y = s.dp(y)
        return x + y

class SPAN(nn.Module):
    def __init__(s, n_classes, d=512, drop=0.5):
        super().__init__()
        s.enc = nn.Sequential(
            ConvBlock(1, 32, (1,1), drop), ConvBlock(32, 64, (2,2), drop),
            ConvBlock(64, 128, (2,2), drop), ConvBlock(128, 256, (2,2), drop),
            ConvBlock(256, d, (2,1), drop), ConvBlock(d, d, (2,1), drop),
            DSCBlock(d, drop), DSCBlock(d, drop), DSCBlock(d, drop), DSCBlock(d, drop))
        s.head = nn.Conv2d(d, n_classes, 1)
    def forward(s, x):
        f = s.enc(x); lo = s.head(f); B, C, Hh, Ww = lo.shape
        return lo.permute(0, 2, 3, 1).reshape(B, Hh * Ww, C), (Hh, Ww)


cs = json.load(open(f"{CODE}/assets/metadata/charset.json", encoding="utf-8"))
VOCAB = ["<blank>"] + cs["symbols"] + ["<FRAC>", "<unk>"]
split = json.load(open(f"{CODE}/assets/metadata/splits.json", encoding="utf-8"))["modetrans"]["assignment"]
text = pd.read_csv(f"{CODE}/assets/metadata/modetrans_text.csv", keep_default_na=False).set_index("image_id")["text_train"].to_dict()
ck = torch.load(f"{CODE}/notebooks/outputs-of-notebook/kaggle-output-span-version1/span_seed42_best.zip", map_location="cpu", weights_only=False)
print("checkpoint keys:", list(ck.keys()) if isinstance(ck, dict) else type(ck), "| epoch", ck.get("epoch") if isinstance(ck, dict) else "?")
sd = ck["model"] if "model" in ck else ck
model = SPAN(len(VOCAB)); model.load_state_dict(sd, strict=True); model.eval()
pq = pd.read_parquet(f"{ROOT}/FYP-modi-script-data/MoDeTrans-dataset/train-00000-of-00001.parquet")
names = list(pq.filename); row = {r.filename: r for r in pq.itertuples()}
tr = [n for n in names if split[n] == "train"][:100]
va = [n for n in names if split[n] == "val"][:100]


def read(n):
    g = np.array(Image.open(io.BytesIO(row[n].image["bytes"])).convert("L"))
    w = max(8, int(round(g.shape[1] * 320 / g.shape[0])))
    im = cv2.resize(g, (min(w, 2048), 320), interpolation=cv2.INTER_AREA)
    W = int(math.ceil(im.shape[1] / 8) * 8)
    x = np.full((320, W), 255, np.uint8); x[:, :im.shape[1]] = im
    xt = torch.from_numpy(x).float().div(255).sub(0.5).div(0.5)[None, None]
    with torch.no_grad():
        seq, _ = model(xt)
    ids = seq.argmax(-1)[0].numpy()
    keep = np.ones_like(ids, bool); keep[1:] = ids[1:] != ids[:-1]; keep &= ids != 0
    return "".join(VOCAB[i] for i in ids[keep])


u = lambda s: s.replace("<FRAC>", "")
for name, ids in (("TRAIN (first 100 train patches)", tr), ("VAL (first 100 val patches)", va)):
    t0 = time.time(); e = n = 0; per = []
    for k in ids:
        h = read(k); r = u(text[k]); d = editdistance.eval(r, u(h)); e += d; n += len(r); per.append(d / len(r))
    print(f"{name}: CER {100*e/n:.1f} | median patch {100*np.median(per):.1f} | best patch {100*min(per):.1f} | "
          f"below 50%: {sum(p < 0.5 for p in per)}/{len(per)} | {time.time()-t0:.0f}s", flush=True)
