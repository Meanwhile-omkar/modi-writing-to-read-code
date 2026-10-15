"""Strip 6, step 2 - better combinations from notebook 05's checkpoints, offline, no GPU.
  stage A  forward ensemble selection, scored by LM-decoded VAL-SEL CER (alpha 0.3, beta 1.0, order 4 = the notebook's
           ensemble setting), starting from the best single model
  stage B  extended LM grid on VAL-SEL for the best ensembles: alpha 0.3-0.8 x beta 0.25-1.25 (order 4), then orders 5/6
  stage C  VAL-REP for the notebook's ensemble (must reproduce 29.15) and for the new pipeline; TEST once for the new one
Every stage writes its result to s6/ so a time limit cannot lose finished work.  usage: python s6_search.py [A|B|C]"""
# Moved into the repo on 15 Oct 2026 from a Claude-session scratchpad; only the file paths changed (now repo-relative).
import json, math, os, pickle, sys, time
from pathlib import Path
import numpy as np
from multiprocessing import get_context
HERE = Path(__file__).resolve().parent                         # research/root-cause/offline
CODE = HERE.parents[2]                                         # FYP-modi-script-code
sys.path.insert(0, str(CODE / "notebooks/lr05"))
import lr05_lib as L

S6 = str(HERE / "s6")                                          # small results, kept in git
CACHE = os.environ.get("MODI_S6_CACHE", str(HERE / "cache"))   # log-prob pickles, ~66 MB each (git-ignored)
WORK = os.environ.get("MODI_WORK", CACHE + "/work_full")       # meta.json + real_h40 lines: notebooks/lr05/verify_against_cpu.py
TAGS = ["AUG", "BIG", "B60", "A120", "FT_A0_s0", "FT_A0_s1", "FT_A0_s2", "MIX", "FT_LONG", "PRE_LONG", "C60", "A12"]
_W = {}


def _init(texts, itos):
    _W["lm"] = {n: L.CharLM(texts, n) for n in (4, 5, 6, 7, 8)}; _W["itos"] = itos


def _decode(task):
    lines, a, b, n = task                      # lines: list of (key, top_idx, top_lp)
    return [(k, L.beam_decode(ti, tl, _W["itos"], _W["lm"][n], a, b).strip()) for k, ti, tl in lines]


class Ctx:
    def __init__(self, pool):
        self.pool = pool
        self.M = json.load(open(WORK + "/meta.json", encoding="utf-8"))
        self.bp = json.load(open(S6 + "/bp.json"))["bp"]
        self.lp = {t: pickle.load(open(f"{CACHE}/lp_{t}.pkl", "rb")) for t in TAGS}
        self.vocab = L.make_vocab(self.M["symbols"], self.M["train_texts"], "char")
        self.cache = {}

    def ens(self, members, i):
        a = [self.lp[t][i].astype(np.float32) for t in members]
        return a[0] if len(a) == 1 else (np.logaddexp.reduce(np.stack(a), 0) - math.log(len(a)))

    def topk(self, members, patches):
        key = (tuple(sorted(members)), tuple(patches))
        if key not in self.cache:
            self.cache[key] = [(i, *L.topk_compress(self.ens(members, i), 8)) for p in patches for i in self.bp[p]]
        return self.cache[key]

    def greedy_cer(self, members, patches):
        hyps = [" ".join(s for s in (self.vocab.decode(L.greedy_ids(self.ens(members, i))).strip() for i in self.bp[p]) if s) for p in patches]
        return L.metrics([self.M["text"][p] for p in patches], hyps)["CER"], hyps

    def lm(self, members, patches, a, b, n, chunks=40):
        lines = self.topk(members, patches)
        tasks = [(lines[s::chunks], a, b, n) for s in range(chunks)]
        out = {}
        for res in self.pool.imap_unordered(_decode, tasks):
            out.update(res)
        hyps = [" ".join(s for s in (out[i] for i in self.bp[p]) if s) for p in patches]
        return L.metrics([self.M["text"][p] for p in patches], hyps), hyps


def save(name, obj):
    json.dump(obj, open(f"{S6}/{name}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "A"
    M = json.load(open(WORK + "/meta.json", encoding="utf-8"))
    vocab = L.make_vocab(M["symbols"], M["train_texts"], "char")
    ctx_pool = get_context("spawn").Pool(10, initializer=_init, initargs=(M["train_texts"], vocab.itos))
    C = Ctx(ctx_pool); VS, VR, TE = M["val_sel"], M["val_rep"], M["test"]
    t0 = time.time()
    if stage == "A":
        single = {t: C.lm([t], VS, 0.3, 1.0, 4)[0]["CER"] for t in TAGS}
        print("single models, LM-decoded VAL-SEL CER (0.3, 1.0, 4):", {t: round(v, 2) for t, v in sorted(single.items(), key=lambda x: x[1])}, f"{time.time()-t0:.0f}s", flush=True)
        cur = [min(single, key=single.get)]; best = single[cur[0]]; path = [(list(cur), round(best, 2))]
        while True:
            trial = {t: C.lm(cur + [t], VS, 0.3, 1.0, 4)[0]["CER"] for t in TAGS if t not in cur}
            t_best = min(trial, key=trial.get)
            print(f"  + {t_best}: {trial[t_best]:.2f} (current {best:.2f}) | all: {sorted(((round(v, 2), t) for t, v in trial.items()))[:5]} | {time.time()-t0:.0f}s", flush=True)
            if trial[t_best] > best - 0.1 or len(cur) >= 7:
                break
            cur.append(t_best); best = trial[t_best]; path.append((list(cur), round(best, 2)))
        nb = C.lm(["AUG", "BIG", "B60"], VS, 0.3, 1.0, 4)[0]["CER"]
        save("stageA", dict(single=single, path=path, chosen=cur, chosen_cer=best, notebook_ensemble_cer=nb))
        print(f"STAGE A: chosen {cur} -> VAL-SEL {best:.2f} | notebook ensemble AUG+BIG+B60 {nb:.2f} (notebook reported 24.05) | {time.time()-t0:.0f}s", flush=True)
    elif stage == "B":
        A = json.load(open(S6 + "/stageA.json", encoding="utf-8"))
        cands = {"chosen": A["chosen"], "notebook": ["AUG", "BIG", "B60"], "AUG": ["AUG"]}
        res = {}
        for name, mem in cands.items():
            grid = {}
            for a in (0.3, 0.45, 0.6, 0.8):
                for b in (0.25, 0.5, 0.75, 1.0, 1.25):
                    grid[(a, b, 4)] = C.lm(mem, VS, a, b, 4)[0]["CER"]
            ba, bb, _ = min(grid, key=grid.get)
            for n in (5, 6):
                grid[(ba, bb, n)] = C.lm(mem, VS, ba, bb, n)[0]["CER"]
            best = min(grid, key=grid.get)
            res[name] = dict(members=mem, best=list(best), best_cer=grid[best], grid={str(k): round(v, 2) for k, v in grid.items()})
            print(f"STAGE B {name:9s} {mem}: best LM {best} -> VAL-SEL {grid[best]:.2f} | order-4 grid edge check: "
                  f"{sorted(((round(v, 2), k) for k, v in grid.items()))[:4]} | {time.time()-t0:.0f}s", flush=True)
            save("stageB", res)
    elif stage == "B2":
        B = json.load(open(S6 + "/stageB.json", encoding="utf-8"))
        for name, r in B.items():
            a, b, n0 = r["best"]
            for n in (7, 8):
                v = C.lm(r["members"], VS, a, b, n)[0]["CER"]; r["grid"][str((a, b, n))] = round(v, 2)
                print(f"STAGE B2 {name:9s} alpha {a} beta {b} order {n}: VAL-SEL {v:.2f} (best so far {r['best_cer']:.2f} at order {r['best'][2]}) | {time.time()-t0:.0f}s", flush=True)
                if v < r["best_cer"]:
                    r["best"], r["best_cer"] = [a, b, n], v
            save("stageB", B)
    else:
        B = json.load(open(S6 + "/stageB.json", encoding="utf-8"))
        out = {}
        nb_m, _ = C.lm(["AUG", "BIG", "B60"], VR, 0.3, 1.0, 4)
        out["notebook_pipeline_val_rep"] = nb_m["CER"]
        print(f"integrity: notebook pipeline (AUG+BIG+B60, LM 0.3/1.0/4) VAL-REP {nb_m['CER']:.2f} (notebook reported 29.15)", flush=True)
        win = min(B, key=lambda k: B[k]["best_cer"]); mem, (a, b, n) = B[win]["members"], B[win]["best"]
        g_cer, _ = C.greedy_cer(mem, VR)
        m_vr, _ = C.lm(mem, VR, a, b, n)
        out.update(new_pipeline=dict(members=mem, lm=[a, b, n], val_sel=B[win]["best_cer"], val_rep_greedy=g_cer, val_rep_lm=m_vr))
        print(f"NEW PIPELINE {mem} LM {(a, b, n)}: VAL-SEL {B[win]['best_cer']:.2f} | VAL-REP greedy {g_cer:.2f} -> LM {m_vr['CER']:.2f} "
              f"(no-space {m_vr['CER_nospace']:.2f}, BLEU {m_vr['BLEU']:.2f}, chrF++ {m_vr['chrF_pp']:.1f})", flush=True)
        m_te, hyps = C.lm(mem, TE, a, b, n)
        import editdistance
        u = lambda s: s.replace("<FRAC>", "")
        e = np.array([editdistance.eval(u(M["text"][p]), u(h)) for p, h in zip(TE, hyps)], float)
        ln = np.array([max(1, len(u(M["text"][p]))) for p in TE], float)
        idx = np.random.default_rng(0).integers(0, len(TE), (10000, len(TE))); bs = 100 * e[idx].sum(1) / ln[idx].sum(1)
        out["new_pipeline_test"] = dict(m_te, CI95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        import pandas as pd
        pd.DataFrame(dict(patch=TE, reference=[M["text"][p] for p in TE], prediction=hyps)).to_csv(S6 + "/test_predictions_new_pipeline.csv", index=False, encoding="utf-8")
        save("stageC", out)
        print(f"TEST (once, pipeline chosen on VAL-SEL only): CER {m_te['CER']:.2f} (95% CI {np.percentile(bs, 2.5):.2f}-{np.percentile(bs, 97.5):.2f}) | "
              f"no-space {m_te['CER_nospace']:.2f} | BLEU {m_te['BLEU']:.2f} | chrF++ {m_te['chrF_pp']:.1f} | <50%: {m_te['patches_below_50']}/{m_te['n']} | "
              f"notebook FINAL pipeline 25.91 | {time.time()-t0:.0f}s", flush=True)
    ctx_pool.close()


if __name__ == "__main__":
    main()
