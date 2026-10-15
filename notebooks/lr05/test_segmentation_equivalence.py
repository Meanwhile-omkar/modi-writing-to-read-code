"""Check: lr05_lib.patch_to_lines reproduces the CPU-proven line set (research/root-cause cache/lines_h40.pkl)
exactly - same number of lines per patch, same pseudo labels, identical 40-px images."""
import io, json, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lr05_lib as L


def work(args):
    name, b, label = args
    g = np.array(Image.open(io.BytesIO(b)).convert("L"))
    lines, st = L.patch_to_lines(g, label, (40, 64))
    return name, [(x["imgs"][40], x["imgs"][64].shape, x["label"], x["idx"], x["n_lines"]) for x in lines], st


if __name__ == "__main__":
    t0 = time.time()
    pq = pd.read_parquet(CODE.parent / "FYP-modi-script-data/MoDeTrans-dataset/train-00000-of-00001.parquet")
    text = pd.read_csv(CODE / "assets/metadata/modetrans_text.csv", keep_default_na=False).set_index("image_id")["text_train"].to_dict()
    ref = pickle.load(open(CODE / "research/root-cause/experiments/cache/lines_h40.pkl", "rb"))
    by = {}
    for x in ref["lines"]:
        by.setdefault(x["patch"], []).append(x)
    with Pool(8) as pool:
        res = pool.map(work, [(r.filename, r.image["bytes"], text[r.filename]) for r in pq.itertuples()], chunksize=16)
    n_new = sum(len(v) for _, v, _ in res); bad = 0; w64 = []
    for name, new, st in res:
        old = sorted(by.get(name, []), key=lambda x: x["idx"])
        if len(old) != len(new):
            bad += 1; continue
        for o, (img40, shp64, lab, idx, nl) in zip(old, new):
            if not (np.array_equal(o["img"], img40) and o["label"] == lab and o["idx"] == idx and o["n_lines"] == nl):
                bad += 1; break
            w64.append(shp64[1])
    print(f"lines: new {n_new:,} vs CPU {len(ref['lines']):,} | patches that differ in any way: {bad} | "
          f"64-px width median {np.median(w64):.0f}, max {max(w64)} | {time.time()-t0:.0f}s")
