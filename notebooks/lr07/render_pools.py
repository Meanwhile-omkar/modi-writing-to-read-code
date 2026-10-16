"""render_pools.py - render the synthetic line pools in the background while the GPUs train.

Pools are rendered one after another and published at milestones (the index file is rewritten
atomically), so a job that needs only the first 12,000 or 60,000 lines of pool A can start long before
all of pool A exists. Text comes from TRAINING labels only (or from a character n-gram model of them),
so no validation or test text can leak into the synthetic data.
usage: python render_pools.py --work DIR --font TTF --plan plan.json --procs 3
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lr05_lib as L

_G = {}


def _init(font, texts, seed):
    _G.update(font=font, texts=texts, sampler=None)
    _G["chunks"] = L.label_chunks(texts, np.random.default_rng(seed))


def _render(task):
    style, text_src, heights, chunk_id, n, seed = task
    rng = np.random.default_rng([seed, chunk_id])
    if text_src == "lm" and _G["sampler"] is None:
        _G["sampler"] = L.TextSampler(_G["texts"], order=4)
    out, tries = [], 0
    while len(out) < n and tries < 3 * n:
        tries += 1
        lab = _G["sampler"].sample(rng) if text_src == "lm" else _G["chunks"][int(rng.integers(len(_G["chunks"])))]
        if not lab:
            continue
        o = L.render_line(lab, rng, _G["font"], heights, no_gaps=(style == "realistic"), headline=(style == "realistic"),
                          heavy=(style == "heavy"))
        if o is not None:
            out.append((o, lab))
    return chunk_id, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--font", required=True)
    ap.add_argument("--plan", required=True); ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=200)
    a = ap.parse_args()
    plan = json.load(open(a.plan, encoding="utf-8"))
    meta = json.load(open(os.path.join(a.work, "meta.json"), encoding="utf-8"))
    texts = [L.normalise(t).replace("<FRAC>", "") for t in meta["train_texts"]]
    print(f"renderer: {len(texts)} training labels, {a.procs} processes, pools {[p['name'] for p in plan]}", flush=True)
    t_all = time.time()
    with Pool(a.procs, initializer=_init, initargs=(a.font, texts, 0)) as pool:
        for P in plan:
            t0 = time.time()
            heights = tuple(sorted(set([40] + ([64] if P.get("h64_limit") else []))))
            W = {40: L.LineWriter(os.path.join(a.work, f"pool_{P['name']}_h40"), 40)}
            if 64 in heights:
                W[64] = L.LineWriter(os.path.join(a.work, f"pool_{P['name']}_h64"), 64)
            n_chunks = int(np.ceil(P["n"] / a.chunk))
            tasks = [(P["style"], P["text"], heights, c, min(a.chunk, P["n"] - c * a.chunk), P["seed"]) for c in range(n_chunks)]
            done, next_report, ms = 0, 0.1, sorted(P.get("milestones", []))
            extra = dict(style=P["style"], text=P["text"], seed=P["seed"])
            for cid, lines in pool.imap(_render, tasks):
                for o, lab in lines:
                    W[40].add(o[40], lab)
                    if 64 in W and W[64].n_lines_ok(P["h64_limit"]):
                        W[64].add(o[64], lab)
                    done += 1
                    if ms and done >= ms[0]:
                        W[40].publish(extra=extra)
                        if 64 in W: W[64].publish(extra=extra)
                        print(f"  pool {P['name']}: milestone {ms[0]:,} lines published ({time.time()-t0:.0f}s)", flush=True)
                        ms.pop(0)
                if done >= next_report * P["n"]:
                    r = done / max(1e-9, time.time() - t0)
                    print(f"  pool {P['name']}: {done:,}/{P['n']:,} lines | {r:.0f} lines/s | "
                          f"ETA {L.fmt_s((P['n'] - done) / max(r, 1e-9))}", flush=True)
                    next_report += 0.1
            for w in W.values():
                w.close(extra=extra)
            wid = np.array(W[40].widths); nch = np.array([len(x) for x in W[40].labels])
            print(f"POOL {P['name']} DONE: {len(wid):,} lines ({P['style']}, text={P['text']}) in {time.time()-t0:.0f}s | "
                  f"width@40 median {np.median(wid):.0f} | chars median {np.median(nch):.0f} | "
                  f"frames/char median {np.median(wid / 2 / np.maximum(1, nch)):.1f}", flush=True)
    print(f"RENDERER FINISHED in {time.time()-t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
