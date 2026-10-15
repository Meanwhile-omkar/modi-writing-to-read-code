"""SPAN v2 (faithful, notebook 04): did either best checkpoint memorise its TRAINING patches? Exact v2 evaluation:
whole patch resized to 320x880 (INTER_AREA), standardised with the TRAINING set's mean/std, span_lib.SPAN, greedy
decoding over each sample's own grid (span_lib.greedy_decode). Sanity check: val must reproduce the logged val CER."""
# Moved into the repo on 15 Oct 2026 from a Claude-session scratchpad; only the file paths changed (now repo-relative).
import io, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch, editdistance
from PIL import Image

CODE = Path(__file__).resolve().parents[3].as_posix()          # FYP-modi-script-code
sys.path.insert(0, f"{CODE}/notebooks/span")
import span_lib as L
torch.set_num_threads(8)
H, W = 320, 880
cs = json.load(open(f"{CODE}/assets/metadata/charset.json", encoding="utf-8"))
VOCAB = ["<blank>"] + cs["symbols"] + ["<FRAC>", "<unk>"]
dec = lambda ids: "".join(VOCAB[i] for i in ids if i > 0)
split = json.load(open(f"{CODE}/assets/metadata/splits.json", encoding="utf-8"))["modetrans"]["assignment"]
text = pd.read_csv(f"{CODE}/assets/metadata/modetrans_text.csv", keep_default_na=False).set_index("image_id")["text_train"].to_dict()
pq = pd.read_parquet(f"{CODE}/../FYP-modi-script-data/MoDeTrans-dataset/train-00000-of-00001.parquet")
names = list(pq.filename); row = {r.filename: r for r in pq.itertuples()}
img = lambda n: cv2.resize(np.array(Image.open(io.BytesIO(row[n].image["bytes"])).convert("L")), (W, H), interpolation=cv2.INTER_AREA)
t0 = time.time(); tot = tot2 = 0.0; cnt = 0
for n in names:
    if split[n] == "train":
        a = img(n).astype(np.float64); tot += a.sum(); tot2 += (a * a).sum(); cnt += a.size
mean = tot / cnt; std = float(np.sqrt(max(tot2 / cnt - mean * mean, 1e-12)))
print(f"train mean {mean:.2f} std {std:.2f} ({time.time()-t0:.0f}s)", flush=True)
tr = [n for n in names if split[n] == "train"][:100]
va = [n for n in names if split[n] == "val"][:100]
rs = L.reduced_shape(H, W)
u = lambda s: s.replace("<FRAC>", "")
for tag in ("scratch", "iam"):
    ck = torch.load(f"{CODE}/notebooks/outputs-of-notebook/kaggle-output-span-version2/{tag}_best.zip", map_location="cpu", weights_only=False)
    m = L.SPAN(len(VOCAB), 1, 0.5, ck["cfg"].get("width_mult", 1.0)); m.load_state_dict(ck["model"], strict=True); m.eval()
    for name, ids in (("TRAIN", tr), ("VAL", va)):
        t1 = time.time(); e = n = 0; per = []
        for k in ids:
            x = torch.from_numpy(((img(k).astype(np.float32) - mean) / std))[None, None]
            with torch.no_grad():
                lp = m(x)
            h, _ = L.greedy_decode(lp, [rs], dec)
            r = u(text[k]); d = editdistance.eval(r, u(h[0])); e += d; n += len(r); per.append(d / len(r))
        print(f"{tag:8s} (epoch {ck['epoch']}) {name:5s} first 100 patches: CER {100*e/n:.1f} | median {100*np.median(per):.1f} | "
              f"best {100*min(per):.1f} | below 50%: {sum(p < 0.5 for p in per)}/100 | {time.time()-t1:.0f}s", flush=True)
