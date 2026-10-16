"""Verify job.py against the CPU-proven results, on the FULL data (all 2,043 patches), on this PC's CPU.

1. EVALUATION: load the X4b and X3a checkpoints (research/root-cause) into job.py with no training. Its greedy
   evaluation on VAL-SEL (= the first 150 validation patches, the CPU evaluation set) must reproduce the logged
   val CER exactly: X4b 52.18, X3a 42.44.
2. LEARNING: fine-tune from X4b with job.py's new training loop (batch 32, warm-up + cosine) for N epochs; val CER
   must fall clearly below 52.18 (X3ctl, the CPU loop, reached 42.67 after 3 epochs at batch 8).
usage: python verify_against_cpu.py <scratch dir> [epochs]
"""
import io, json, os, pathlib, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODE = ROOT / "FYP-modi-script-code"
EXP = CODE / "research/root-cause/experiments"
sys.path.insert(0, str(HERE))


def main():
    import numpy as np, pandas as pd
    from PIL import Image
    import lr05_lib as L
    scr = pathlib.Path(sys.argv[1]); epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    work = scr / "work_full"; runs = scr / "runs_verify"
    work.mkdir(parents=True, exist_ok=True); runs.mkdir(parents=True, exist_ok=True)
    splits = json.load(open(CODE / "assets/metadata/splits.json", encoding="utf-8"))["modetrans"]["assignment"]
    text = pd.read_csv(CODE / "assets/metadata/modetrans_text.csv", keep_default_na=False).set_index("image_id")["text_train"].to_dict()
    cs = json.load(open(CODE / "assets/metadata/charset.json", encoding="utf-8"))
    pq = pd.read_parquet(ROOT / "FYP-modi-script-data/MoDeTrans-dataset/train-00000-of-00001.parquet")
    names = list(pq.filename)
    S = {s: [n for n in names if splits[n] == s] for s in ("train", "val", "test")}
    meta = dict(symbols=cs["symbols"], train_texts=[text[n] for n in S["train"]], text={n: text[n] for n in names},
                val_sel=S["val"][:150], val_rep=S["val"][150:], test=S["test"], train150=S["train"][:150], patch_info={})
    if not (work / "real_h40.json").exists():
        t0 = time.time()
        W = L.LineWriter(str(work / "real_h40"), 40)
        for r in pq.itertuples():
            g = np.array(Image.open(io.BytesIO(r.image["bytes"])).convert("L"))
            lines, st = L.patch_to_lines(g, text[r.filename], (40,))
            meta["patch_info"][r.filename] = st
            for x in lines:
                W.add(x["imgs"][40], x["label"], dict(patch=r.filename, split=splits[r.filename], idx=x["idx"], n_lines=x["n_lines"]))
        W.close()
        json.dump(meta, open(work / "meta.json", "w", encoding="utf-8"), ensure_ascii=False)
        print(f"built {len(W.widths):,} real lines in {time.time() - t0:.0f}s", flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")

    def job(tag, **kw):
        spec = dict(tag=tag, work=str(work), run_dir=str(runs / tag), size="small", H=40, vocab="char", no_space=False,
                    cpu=True, threads=10, deadline=time.time() + 3600, time_cap_s=3600, **kw)
        (runs / tag).mkdir(parents=True, exist_ok=True)
        json.dump(spec, open(runs / tag / "spec.json", "w"))
        subprocess.run([sys.executable, "-u", str(HERE / "job.py"), "--spec", str(runs / tag / "spec.json")], env=env, check=False)
        return json.load(open(runs / tag / "result.json", encoding="utf-8"))

    for tag, ck, want in (("EVAL_X4B", "x4b_model.pt", 52.18), ("EVAL_X3A", "x3a_model.pt", 42.44)):
        r = job(tag, init=str(EXP / "cache" / ck))
        got = r["val_sel"]["CER"]
        print(f"\n### {tag}: job.py VAL-SEL CER {got:.2f} vs logged CPU value {want:.2f} -> "
              f"{'MATCH' if abs(got - want) < 0.01 else 'DIFFERENT'}\n", flush=True)
    if epochs <= 0:
        return
    r = job("FT_FROM_X4B", init=str(EXP / "cache/x4b_model.pt"),
            finetune=dict(epochs=epochs, lr=1e-3, batch=32, aug=False, mix=0.0, realign_at=[], seed=0, log_every=20))
    print(f"\n### FT_FROM_X4B ({epochs} epoch(s) of job.py's loop): VAL-SEL CER {r['val_sel']['CER']:.2f} "
          f"(start 52.18; X3ctl after 3 CPU epochs 42.67) | train-150 {r['train150']['CER']:.2f}", flush=True)


if __name__ == "__main__":
    main()
