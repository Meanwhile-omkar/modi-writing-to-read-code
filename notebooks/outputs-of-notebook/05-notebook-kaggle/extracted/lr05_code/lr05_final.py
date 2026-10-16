"""lr05_final.py - after the jobs: hypothesis verdicts, character-LM decoding (N3), checkpoint ensembles (N15),
space restoration for the no-space model (N5), the one-time TEST score, figures and REPORT.md.

Rules fixed before the run (so nothing is tuned on the numbers it reports):
  * VAL-SEL (first 150 validation patches) picks epochs, the combination, LM weights and the final pipeline.
  * VAL-REP (the remaining validation patches) is only ever reported.
  * TEST is scored once, for exactly two pipelines: the anchor (+LM) and the pipeline VAL-SEL picks (+LM).
"""
import glob, json, math, os, pickle, time
import numpy as np
import lr05_lib as L
from lr05_run import HYPOTHESES


# ======================================================================= results and verdicts
def load_results(runs):
    R = {}
    for p in sorted(glob.glob(os.path.join(runs, "*", "result.json"))):
        if os.path.basename(os.path.dirname(p)).startswith("_"):      # _bench is a measurement, not an experiment
            continue
        try:
            r = json.load(open(p, encoding="utf-8"))
            R[r["tag"]] = r
        except Exception:
            pass
    return R


def preds(runs, tag, which):
    import pandas as pd
    p = os.path.join(runs, tag, f"{which}_predictions.csv")
    return pd.read_csv(p, keep_default_na=False) if os.path.exists(p) else None


def content_gap(runs, tag, which="val_rep"):
    """Reads or babbles? CER against each patch's own transcription vs against the patch next to it in transcription
    LENGTH. A reader is far better on its own patch (CPU line reader: 45.9 vs 78.6, better on 149/150); a model that only
    learned how much text a page holds is not (notebook 03: 73.7 vs 74.1; SPAN v1: 75.0 vs 75.6, a coin flip)."""
    import editdistance
    d = preds(runs, tag, which)
    if d is None or len(d) < 4:
        return None
    refs = [r.replace("<FRAC>", "") for r in d.ref]; hyps = [h.replace("<FRAC>", "") for h in d.hyp]
    order = np.argsort([len(r) for r in refs], kind="stable")
    own, nb = [], []
    for k, i in enumerate(order):
        j = order[k + 1] if k + 1 < len(order) else order[k - 1]
        own.append(editdistance.eval(refs[i], hyps[i]) / max(1, len(refs[i])))
        nb.append(editdistance.eval(refs[j], hyps[i]) / max(1, len(refs[j])))
    own, nb = np.array(own), np.array(nb)
    return dict(own=round(100 * own.mean(), 1), neighbour=round(100 * nb.mean(), 1),
                gap=round(100 * (nb.mean() - own.mean()), 1), own_better_pct=round(100 * float((own < nb).mean()), 1))


def compare(runs, a, b, which="val_rep", nospace=False):
    da, db = preds(runs, a, which), preds(runs, b, which)
    if da is None or db is None or list(da.patch) != list(db.patch):
        return None
    if nospace:
        return L.paired_bootstrap(da.err_nospace, db.err_nospace, da.n_nospace)
    return L.paired_bootstrap(da.err, db.err, da.n)


def seed_noise(R, which="val_rep", key="CER"):
    v = [R[t][which][key] for t in ("FT_A0_s0", "FT_A0_s1", "FT_A0_s2") if t in R and R[t].get("status") == "ok"]
    return (float(np.std(v, ddof=1)) if len(v) >= 2 else None), v


def verdicts(R, runs):
    sd, seeds = seed_noise(R)
    sd_ns, _ = seed_noise(R, key="CER_nospace")
    rows = []
    ok = lambda t: t in R and R[t].get("status") == "ok"
    for hid, q, var, ctl, metric in HYPOTHESES:
        if hid == "FINAL" and not ok("COMBO") and ok("FINAL_s0"):
            # nothing beat the anchor, so FINAL is the anchor recipe trained long: compare it with the anchor
            ctl, q = "FT_A0_s0", "No change beat the anchor, so: the anchor recipe on the whole pool, trained long - how much better than the anchor?"
        if not (ok(var) and ok(ctl)):
            why = f"not run ({var}: {R.get(var, {}).get('status', 'not run')}, {ctl}: {R.get(ctl, {}).get('status', 'not run')})"
            rows.append(dict(id=hid, question=q, variant=var, control=ctl, metric=metric, verdict="NOT TESTED", detail=why))
            continue
        c = compare(runs, var, ctl, nospace=(metric == "CER_nospace"))
        a, b = R[var]["val_rep"][metric], R[ctl]["val_rep"][metric]
        s = sd_ns if metric == "CER_nospace" else sd
        if c is None:
            rows.append(dict(id=hid, question=q, variant=var, control=ctl, metric=metric, verdict="NOT TESTED", detail="no predictions"))
            continue
        d, lo, hi = c
        thr = 2 * s if s is not None else 0.0
        if hi < 0 and d < -thr:
            v = "BETTER"
        elif lo > 0 and d > thr:
            v = "WORSE"
        else:
            v = "INCONCLUSIVE (within noise)"
        rows.append(dict(id=hid, question=q, variant=var, control=ctl, metric=metric, variant_value=round(a, 2),
                         control_value=round(b, 2), delta=round(d, 2), ci95=(round(lo, 2), round(hi, 2)),
                         seed_sd=None if s is None else round(s, 2), verdict=v,
                         detail=f"{metric} {a:.2f} vs {b:.2f}: {d:+.2f} (paired 95% CI {lo:+.2f} to {hi:+.2f}; "
                                f"seed noise sd {'n/a' if s is None else f'{s:.2f}'})"))
    return rows, dict(sd=sd, seeds=seeds)


# ======================================================================= log-probs for LM decoding
def _vocab_for(META, vd):
    return L.make_vocab(META["symbols"], META["train_texts"], vd["kind"], vd["no_space"])


def lm_inputs(runs, work, META, groups, patches, device, log, k=8):
    """groups: {key: [tags]} - one tag = a single model, several = an ensemble (probabilities averaged).
    Returns {key: dict(itos, sep, lines={i: (top_idx, top_lp)}, greedy={i: text}, by_patch={patch: [i]})}."""
    import torch
    out = {}
    stores = {}
    for key, tags in groups.items():
        ms, vocab, H = [], None, None
        for t in tags:
            ck = torch.load(os.path.join(runs, t, "best.pt"), map_location="cpu")
            v = _vocab_for(META, ck["vocab"])
            if vocab is not None and (v.itos != vocab.itos or ck["H"] != H):
                raise ValueError(f"ensemble {key}: {t} has a different alphabet or line height")
            vocab, H = v, ck["H"]
            m = L.build_crnn(len(v), ck["size"]); m.load_state_dict(ck["model"]); m.to(device).eval(); ms.append(m)
        if H not in stores:
            stores[H] = L.LineStore(os.path.join(work, f"real_h{H}"))
        st = stores[H]
        bp = {}
        for i, inf in enumerate(st.info):
            if inf["patch"] in patches:
                bp.setdefault(inf["patch"], []).append(i)
        for p in bp:
            bp[p].sort(key=lambda i: st.info[i]["idx"])
        lines, greedy = {}, {}
        with torch.no_grad():
            for b in bp.values():                      # one patch per batch, as in job.py and the CPU experiments
                x, lens = L.to_batch([st.img(i) for i in b], st.H, round_to=L.STRIDE)
                xt = torch.from_numpy(x).to(device)
                lps = [m(xt).float().log_softmax(-1) for m in ms]
                lp = lps[0] if len(lps) == 1 else torch.logsumexp(torch.stack(lps), 0) - math.log(len(lps))
                lp = lp.cpu().numpy()
                for kk, i in enumerate(b):
                    row = lp[kk, :lens[kk]]
                    lines[i] = L.topk_compress(row, k)
                    greedy[i] = vocab.decode(L.greedy_ids(row))
        out[key] = dict(itos=vocab.itos, sep="" if vocab.no_space else " ", lines=lines, greedy=greedy, by_patch=bp,
                        tags=list(tags), H=H, kind=vocab.kind)
        log(f"  log-probs ready for {key} ({'+'.join(tags)}): {len(lines):,} lines, alphabet {len(vocab.itos)}")
        del ms
    return out


_W = {}


def _winit(pkl, texts, orders):
    _W["D"] = pickle.load(open(pkl, "rb"))
    _W["lm"] = {n: L.CharLM(texts, n) for n in orders}


def _wtask(t):
    key, a, b, n, patches = t
    d, lm = _W["D"][key], _W["lm"][n]
    res = {}
    for p in patches:
        parts = [L.beam_decode(*d["lines"][i], d["itos"], lm, a, b).strip() for i in d["by_patch"].get(p, [])]
        res[p] = d["sep"].join(x for x in parts if x)
    return key, a, b, n, res


def greedy_texts(D, key, patches):
    d = D[key]
    return {p: d["sep"].join(t for t in (d["greedy"][i].strip() for i in d["by_patch"].get(p, [])) if t) for p in patches}


def run_lm(D, keys, cfgs, patches, texts, work, procs, log, chunk=8):
    """Decode `patches` for every (key, alpha, beta, order) in cfgs[key], in parallel. Returns {(key,a,b,n): {p: text}}."""
    from multiprocessing import get_context
    pkl = os.path.join(work, "lm_inputs.pkl")
    pickle.dump({k: D[k] for k in keys}, open(pkl, "wb"))
    orders = sorted({c[2] for k in keys for c in cfgs[k]})
    tasks = [(k, a, b, n, patches[s:s + chunk]) for k in keys for (a, b, n) in cfgs[k] for s in range(0, len(patches), chunk)]
    out = {}
    ctx = get_context("spawn")              # never fork a process that holds a CUDA context
    with ctx.Pool(procs, initializer=_winit, initargs=(pkl, texts, orders)) as pool:
        for k, a, b, n, res in pool.imap_unordered(_wtask, tasks):
            out.setdefault((k, a, b, n), {}).update(res)
    return out


def score(refs, hyps, n_boot=0):
    m = L.metrics(refs, hyps)
    if n_boot:
        import editdistance
        u = lambda s: s.replace("<FRAC>", "")
        e = np.array([editdistance.eval(u(r), u(h)) for r, h in zip(refs, hyps)], float)
        n = np.array([max(1, len(u(r))) for r in refs], float)
        idx = np.random.default_rng(0).integers(0, len(refs), (n_boot, len(refs)))
        bs = 100 * e[idx].sum(1) / n[idx].sum(1)
        m["CER_CI95"] = (round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2))
    return m


# ======================================================================= the whole final phase
def final_phase(OUT, WORK, META, log, device="cuda", procs=4, time_budget_s=900, ensemble_k=3, quick=False):
    import pandas as pd
    t0 = time.time()
    runs = os.path.join(OUT, "runs")
    R = load_results(runs)
    TEXT = META["text"]; VS, VR, TE = META["val_sel"], META["val_rep"], META["test"]
    texts = META["train_texts"]
    report = dict(results=R)
    okR = {t: r for t, r in R.items() if r.get("status") == "ok" and "val_sel" in r}
    log(f"FINAL PHASE: {len(okR)} finished jobs with results ({len(R) - len(okR)} failed or without evaluation)")

    # ---- 1. summary table
    rows = []
    for t, r in sorted(R.items()):
        sp = r.get("spec", {})
        row = dict(tag=t, status=r.get("status"), capped=r.get("capped"), priority=sp.get("priority"), hyp=",".join(r.get("hyp", [])),
                   desc=r.get("desc"), model=sp.get("size"), H=sp.get("H"), targets=sp.get("vocab") + ("/no-space" if sp.get("no_space") else ""),
                   params_M=r.get("params_M"), minutes=r.get("job_minutes"))
        if r.get("pre"):
            row.update(pre_lines_seen=r["pre"]["lines_seen"], pre_steps=f"{r['pre']['steps_done']}/{r['pre']['steps_planned']}")
        if r.get("ft"):
            row.update(ft_epochs=f"{r['ft']['epochs_done']}/{r['ft']['epochs_planned']}", best_epoch=r["ft"]["best_epoch"])
        for w in ("val_sel", "val_rep", "train150"):
            if w in r:
                row[f"{w}_CER"] = round(r[w]["CER"], 2); row[f"{w}_CER_nospace"] = round(r[w]["CER_nospace"], 2)
        if "val_rep" in r:
            row.update(val_rep_BLEU=round(r["val_rep"]["BLEU"], 2), val_rep_chrF=round(r["val_rep"]["chrF_pp"], 1),
                       val_rep_below50=f"{r['val_rep']['patches_below_50']}/{r['val_rep']['n']}")
            cg = content_gap(runs, t)
            if cg:
                r["content_gap"] = cg
                row.update(reads_gap=cg["gap"], own_patch_better_pct=cg["own_better_pct"])
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    with pd.option_context("display.max_columns", 50, "display.width", 250, "display.max_colwidth", 60):
        log(summary.drop(columns=["desc"]).to_string(index=False))

    # ---- 2. verdicts
    V, noise = verdicts(R, runs)
    report.update(verdicts=V, noise=noise)
    log("\nHYPOTHESIS VERDICTS (VAL-REP, never used for any choice; paired bootstrap over the same patches)")
    for v in V:
        log(f"  {v['id']:10s} {v['verdict']:28s} {v['variant']} vs {v['control']}: {v['detail']}")
    log(f"  run-to-run noise of the anchor (3 fine-tune seeds, VAL-REP CER): {noise['seeds']} -> sd {noise['sd']}")

    # ---- 3. which models get LM decoding
    cand = {t: r for t, r in okR.items() if r.get("ft") and not r["spec"].get("no_space")}
    groups = {}
    if "FT_A0_s0" in cand:
        groups["anchor"] = ["FT_A0_s0"]
    best = min(cand, key=lambda t: cand[t]["val_sel"]["primary"]) if cand else None
    if best:
        groups["best_single"] = [best]
    for t in ("COMBO", "FINAL_s0", "FINAL_s1"):
        if t in cand and t != best:
            groups[t.lower()] = [t]
    if best:
        fam = lambda t: (cand[t]["spec"]["H"], cand[t]["spec"]["vocab"])
        same = sorted((t for t in cand if fam(t) == fam(best)), key=lambda t: cand[t]["val_sel"]["primary"])[:ensemble_k]
        if len(same) >= 2:
            groups["ensemble"] = same
    log(f"\nLM decoding for: { {k: v for k, v in groups.items()} } (best single by VAL-SEL: {best})")
    if not groups:
        log("nothing to decode"); return report
    D = lm_inputs(runs, WORK, META, groups, set(VS) | set(VR) | set(TE), device, log)

    # ---- 4. LM grid on VAL-SEL (N3), in parallel on the CPUs
    keys = list(groups)
    if quick:
        grid1 = [(0.3, 1.0, 4), (0.3, 2.0, 4)]
    else:
        # beta (length bonus): 0.5 lost clearly on both CPU models (X6, X6b); 1.5 was best on X3a, 2.0 close
        grid1 = [(a, b, 4) for a in (0.2, 0.3, 0.45) for b in (1.0, 1.5, 2.0, 2.5)]
    lm4 = L.CharLM(texts, 4); d0 = D[keys[0]]; probe = VS[:6]
    t1 = time.time()
    for p in probe:
        for i in d0["by_patch"].get(p, []):
            L.beam_decode(*d0["lines"][i], d0["itos"], lm4, 0.3, 1.0)
    per_patch = (time.time() - t1) / max(1, len(probe))
    est = per_patch * len(VS) * len(keys) * (len(grid1) + 2) / max(1, procs) + 60
    left = time_budget_s - (time.time() - t0)
    log(f"  LM decoding speed: ~{per_patch:.2f}s per patch per setting on one core; full grid would take ~{L.fmt_s(est)}, "
        f"{L.fmt_s(left)} of the final budget left")
    if est > 0.6 * left:
        grid1 = [(0.3, b, 4) for b in (1.0, 1.5, 2.0)]
        keys = [k for k in keys if k in ("anchor", "best_single", "ensemble")]
        log(f"  -> reduced grid {grid1} for {keys}")
    res = run_lm(D, keys, {k: grid1 for k in keys}, VS, texts, WORK, procs, log)
    best_cfg, lm_table = {}, []
    refs_vs = [TEXT[p] for p in VS]
    for k in keys:
        g = score(refs_vs, [greedy_texts(D, k, VS)[p] for p in VS])["CER"]
        lm_table.append(dict(model=k, alpha="greedy", beta="", order="", val_sel_CER=round(g, 2)))
        scores = {}
        for (a, b, n) in grid1:
            c = score(refs_vs, [res[(k, a, b, n)][p] for p in VS])["CER"]
            scores[(a, b, n)] = c
            lm_table.append(dict(model=k, alpha=a, beta=b, order=n, val_sel_CER=round(c, 2)))
        best_cfg[k] = min(scores, key=scores.get)
    # orders 5 and 6 at each model's best alpha/beta
    if not quick and time_budget_s - (time.time() - t0) > 3 * per_patch * len(VS) * len(keys) * 2 / max(1, procs):
        cf = {k: [(best_cfg[k][0], best_cfg[k][1], n) for n in (5, 6)] for k in keys}
        res2 = run_lm(D, keys, cf, VS, texts, WORK, procs, log)
        for k in keys:
            for c in cf[k]:
                v = score(refs_vs, [res2[(k,) + c][p] for p in VS])["CER"]
                lm_table.append(dict(model=k, alpha=c[0], beta=c[1], order=c[2], val_sel_CER=round(v, 2)))
                cur = [x["val_sel_CER"] for x in lm_table if x["model"] == k and (x["alpha"], x["beta"], x["order"]) == best_cfg[k]][0]
                if v < cur:
                    best_cfg[k] = c
    lm_df = pd.DataFrame(lm_table); lm_df.to_csv(os.path.join(OUT, "lm_grid_val_sel.csv"), index=False)
    log("\nLM GRID on VAL-SEL (N3):")
    for k in keys:
        sub = lm_df[lm_df.model == k]
        log(f"  {k:12s} greedy {sub[sub.alpha == 'greedy'].val_sel_CER.iloc[0]:.2f} | best alpha/beta/order {best_cfg[k]} -> "
            f"{min(sub[sub.alpha != 'greedy'].val_sel_CER):.2f} | all: " +
            ", ".join(f"({r.alpha},{r.beta},{r.order}) {r.val_sel_CER}" for r in sub[sub.alpha != 'greedy'].itertuples()))

    # ---- 5. VAL-REP with the chosen settings
    rep = run_lm(D, keys, {k: [best_cfg[k]] for k in keys}, VR, texts, WORK, procs, log)
    refs_vr = [TEXT[p] for p in VR]
    pipe = []
    for k in keys:
        gs = score(refs_vr, [greedy_texts(D, k, VR)[p] for p in VR])
        ls = score(refs_vr, [rep[(k,) + best_cfg[k]][p] for p in VR])
        vs_lm = min(x["val_sel_CER"] for x in lm_table if x["model"] == k and x["alpha"] != "greedy")
        pipe.append(dict(pipeline=k, models="+".join(groups[k]), val_sel_LM_CER=round(vs_lm, 2), lm=best_cfg[k],
                         val_rep_greedy_CER=round(gs["CER"], 2), val_rep_LM_CER=round(ls["CER"], 2),
                         val_rep_LM_CER_nospace=round(ls["CER_nospace"], 2), val_rep_LM_BLEU=round(ls["BLEU"], 2),
                         val_rep_LM_chrF=round(ls["chrF_pp"], 1)))
    pipe_df = pd.DataFrame(pipe); pipe_df.to_csv(os.path.join(OUT, "pipelines_val.csv"), index=False)
    log("\nPIPELINES (greedy vs character-LM beam search):")
    log(pipe_df.to_string(index=False))

    # ---- 6. N5: put spaces back into the no-space model's reading
    ns_res = None
    if "NS" in okR:
        try:
            lm5 = L.CharLM(texts, 5)
            dsel, drep = preds(runs, "NS", "val_sel"), preds(runs, "NS", "val_rep")
            best_g, best_c = None, 1e9
            for g in (-3.0, -2.0, -1.0, 0.0, 1.0):
                c = score(list(dsel.ref), [L.restore_spaces(h, lm5, g) for h in dsel.hyp])["CER"]
                log(f"  NS space restoration gamma {g:+.1f}: VAL-SEL CER {c:.2f}")
                if c < best_c:
                    best_g, best_c = g, c
            rr = score(list(drep.ref), [L.restore_spaces(h, lm5, best_g) for h in drep.hyp])
            ns_res = dict(gamma=best_g, val_sel_CER=round(best_c, 2), val_rep_CER=round(rr["CER"], 2),
                          val_rep_CER_nospace=round(rr["CER_nospace"], 2))
            log(f"  N5: no-space model + LM space restoration -> VAL-REP CER {rr['CER']:.2f} (no-space {rr['CER_nospace']:.2f}); "
                f"anchor greedy VAL-REP CER {okR['FT_A0_s0']['val_rep']['CER'] if 'FT_A0_s0' in okR else 'n/a'}")
        except Exception as e:
            log(f"  NS space restoration failed: {e}")
    report["ns_restoration"] = ns_res

    # ---- 7. TEST, once, for the two pre-registered pipelines
    final_key = min(pipe, key=lambda r: r["val_sel_LM_CER"])["pipeline"]
    test_keys = ["anchor"] if "anchor" in keys else []
    if final_key not in test_keys:
        test_keys.append(final_key)
    tr = run_lm(D, test_keys, {k: [best_cfg[k]] for k in test_keys}, TE, texts, WORK, procs, log)
    refs_te = [TEXT[p] for p in TE]
    test = {}
    for k in test_keys:
        g_h = [greedy_texts(D, k, TE)[p] for p in TE]; l_h = [tr[(k,) + best_cfg[k]][p] for p in TE]
        test[k] = dict(models="+".join(groups[k]), lm=best_cfg[k], greedy=score(refs_te, g_h, 10000), lm_beam=score(refs_te, l_h, 10000))
        pd.DataFrame(dict(patch=TE, reference=refs_te, greedy=g_h, lm_beam=l_h)).to_csv(
            os.path.join(OUT, f"test_predictions_{k}.csv"), index=False, encoding="utf-8")
    report.update(test=test, final_pipeline=final_key, lm_best=best_cfg, pipelines=pipe)
    log(f"\nTEST ({len(TE)} patches, scored once; final pipeline chosen on VAL-SEL = '{final_key}'):")
    for k, v in test.items():
        for dec in ("greedy", "lm_beam"):
            m = v[dec]
            log(f"  {k:12s} {dec:8s} CER {m['CER']:6.2f} (95% CI {m['CER_CI95'][0]}-{m['CER_CI95'][1]}) | no-space {m['CER_nospace']:6.2f} | "
                f"WER {m['WER']:6.2f} | BLEU {m['BLEU']:6.2f} | chrF++ {m['chrF_pp']:5.1f} | <50%: {m['patches_below_50']}/{m['n']}")
    json.dump(report, open(os.path.join(OUT, "final_report.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
    log(f"final phase took {L.fmt_s(time.time() - t0)}")
    return report


# ======================================================================= figures and REPORT.md
def figures(OUT, R, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    runs = os.path.join(OUT, "runs"); fig_dir = os.path.join(OUT, "figures"); os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    for t, r in sorted(R.items()):
        p = os.path.join(runs, t, "history.csv")
        if not os.path.exists(p):
            continue
        h = pd.read_csv(p)
        f = h[h.stage == "ft"]
        if len(f):
            ax[0].plot(f.step, f.primary, label=t, lw=2.2 if t == "FT_A0_s0" else 1.1)
        pr = h[h.stage == "pre"]
        if len(pr):
            ax[1].plot(pr.samples, pr.CER, marker="o", label=t)
    ax[0].axhline(84, color="r", ls=":", label="no-reading baseline ~84"); ax[0].set_ylim(0, 100)
    ax[0].set_title("fine-tuning: val-sel CER per epoch (no-space CER for NS)"); ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=7, ncol=2)
    ax[1].set_title("pretraining: zero-shot CER on REAL val-sel vs synthetic lines seen"); ax[1].set_xlabel("synthetic lines seen")
    ax[1].legend(fontsize=7); ax[1].set_ylim(0, 105)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "curves.png"), dpi=110); plt.close(fig)
    ok = [(t, r) for t, r in R.items() if r.get("status") == "ok" and "val_rep" in r]
    if ok:
        ok.sort(key=lambda x: x[1]["val_rep"]["CER"])
        fig, ax = plt.subplots(figsize=(12, 0.42 * len(ok) + 1.5))
        ax.barh([t for t, _ in ok], [r["val_rep"]["CER"] for _, r in ok], color=["tab:red" if t == "FT_A0_s0" else "tab:blue" for t, _ in ok])
        for k, (t, r) in enumerate(ok):
            ax.text(r["val_rep"]["CER"] + 0.3, k, f"{r['val_rep']['CER']:.1f}", va="center", fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel("VAL-REP CER (greedy, lower is better); red = anchor")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "val_rep_cer.png"), dpi=110); plt.close(fig)
    log(f"figures written to {fig_dir}")


def write_report(OUT, rep, run_info, log):
    R = rep.get("results", {})
    L_ = []
    w = L_.append
    w("# Notebook 05 - round-2 hypotheses at GPU scale: results\n")
    w(f"*Generated automatically at the end of the run. {run_info}*\n")
    w("Rules fixed before the run: VAL-SEL (first 150 validation patches) chose every epoch, LM weight and the final pipeline; "
      "VAL-REP (the rest of validation) is only reported; TEST was scored once, for the anchor and the pipeline VAL-SEL chose.\n")
    if rep.get("test"):
        w("## Headline: TEST\n")
        w("| pipeline | models | decoding | CER (95% CI) | CER no-space | WER | BLEU | chrF++ | patches <50% CER |")
        w("|---|---|---|---|---|---|---|---|---|")
        for k, v in rep["test"].items():
            for dec in ("greedy", "lm_beam"):
                m = v[dec]
                w(f"| {k}{' (FINAL)' if k == rep.get('final_pipeline') else ''} | {v['models']} | {dec}{' ' + str(v['lm']) if dec == 'lm_beam' else ''} | "
                  f"{m['CER']:.2f} ({m['CER_CI95'][0]}-{m['CER_CI95'][1]}) | {m['CER_nospace']:.2f} | {m['WER']:.2f} | {m['BLEU']:.2f} | "
                  f"{m['chrF_pp']:.1f} | {m['patches_below_50']}/{m['n']} |")
        w("\nFor reference: notebook 03 test CER 73.1 (BLEU 0.01); SPAN v1 val CER ~75; CPU line reader (X3a) val CER 42.4 greedy, "
          "~41 with the LM (different validation subset).\n")
    w("## Hypothesis verdicts (VAL-REP)\n")
    w("A change counts as BETTER/WORSE only if the paired-bootstrap 95% CI excludes zero AND the difference is larger than "
      "twice the anchor's seed-to-seed standard deviation.\n")
    w("| id | question | variant vs control | metric | variant | control | difference (95% CI) | verdict |")
    w("|---|---|---|---|---|---|---|---|")
    for v in rep.get("verdicts", []):
        if "delta" in v:
            w(f"| {v['id']} | {v['question']} | {v['variant']} vs {v['control']} | {v['metric']} | {v['variant_value']} | "
              f"{v['control_value']} | {v['delta']:+.2f} ({v['ci95'][0]:+.2f} to {v['ci95'][1]:+.2f}) | **{v['verdict']}** |")
        else:
            w(f"| {v['id']} | {v['question']} | {v['variant']} vs {v['control']} | {v['metric']} | | | | {v['verdict']}: {v['detail']} |")
    nz = rep.get("noise", {})
    w(f"\nRun-to-run noise: anchor VAL-REP CER over fine-tune seeds = {nz.get('seeds')} (sd {nz.get('sd')}).\n")
    if rep.get("pipelines"):
        w("## Decoding (N3) and ensembles (N15), VAL-REP\n")
        w("| pipeline | models | LM (alpha, beta, order) chosen on VAL-SEL | greedy CER | LM CER | LM CER no-space | LM BLEU |")
        w("|---|---|---|---|---|---|---|")
        for p in rep["pipelines"]:
            w(f"| {p['pipeline']} | {p['models']} | {p['lm']} | {p['val_rep_greedy_CER']} | {p['val_rep_LM_CER']} | "
              f"{p['val_rep_LM_CER_nospace']} | {p['val_rep_LM_BLEU']} |")
    if rep.get("ns_restoration"):
        n = rep["ns_restoration"]
        w(f"\n**N5 space restoration:** no-space model + LM spaces (gamma {n['gamma']}): VAL-REP CER {n['val_rep_CER']} "
          f"(no-space {n['val_rep_CER_nospace']}).\n")
    w("## Every job\n")
    w("*Reads?* = CER against each patch's own transcription vs against a patch with the same amount of text (VAL-REP). "
      "A reader is far better on its own patch (CPU line reader: own patch better on 149/150); a model that only learned "
      "how much text a page holds is a coin flip (notebook 03: 153/306; SPAN v1: 174/306).\n")
    w("| job | status | what changed | val-sel CER | val-rep CER | val-rep no-space | train-150 CER | reads? own vs same-length (own better) | epochs (best) | minutes |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for t, r in sorted(R.items(), key=lambda x: (x[1].get("val_rep", {}).get("CER", 999))):
        ft = r.get("ft") or {}
        cg = r.get("content_gap")
        reads = f"{cg['own']} vs {cg['neighbour']} ({cg['own_better_pct']}%)" if cg else "-"
        w(f"| {t} | {r.get('status')}{' (time-capped)' if r.get('capped') else ''} | {r.get('desc', '')} | "
          f"{r.get('val_sel', {}).get('CER', float('nan')):.2f} | {r.get('val_rep', {}).get('CER', float('nan')):.2f} | "
          f"{r.get('val_rep', {}).get('CER_nospace', float('nan')):.2f} | {r.get('train150', {}).get('CER', float('nan')):.2f} | "
          f"{reads} | {ft.get('epochs_done', '-')} ({ft.get('best_epoch', '-')}) | {r.get('job_minutes', '-')} |")
    a = R.get("FT_A0_s0", {})
    if a.get("n1n2"):
        w("\n## Error anatomy of the anchor (N1, N2)\n")
        an = a.get("anatomy", {})
        w(f"Errors on all validation: deletions {an.get('deletions_pct')}%, insertions {an.get('insertions_pct')}%, "
          f"substitutions {an.get('substitutions_pct')}%. Most missed: {an.get('most_missed')}. Top confusions: {an.get('top_confusions', [])[:8]}.\n")
        w("| patch feature | Spearman with CER | Spearman with deletion rate | CER by quartile (low -> high) |")
        w("|---|---|---|---|")
        for f, v in a["n1n2"].items():
            if isinstance(v, dict):
                w(f"| {f} | {v['spearman_cer']:+.2f} | {v['spearman_del']:+.2f} | {v['cer_by_quartile']} |")
    w("\n## Files\n`summary.csv` (every job), `runs/<job>/` (log.txt, history.csv, result.json, predictions, best.pt), "
      "`lm_grid_val_sel.csv`, `pipelines_val.csv`, `test_predictions_*.csv`, `figures/`, `logs.txt` (everything printed).\n")
    open(os.path.join(OUT, "REPORT.md"), "w", encoding="utf-8").write("\n".join(L_))
    log(f"wrote {os.path.join(OUT, 'REPORT.md')}")
