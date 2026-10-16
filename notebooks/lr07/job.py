"""job.py - one experiment of notebook 05, in its own process, pinned to one GPU.

A job = [pretrain on a synthetic pool] -> [fine-tune on real lines] -> evaluate. It logs everything to stdout
(streamed into the notebook with a [tag] prefix) and to its run directory: log.txt, history.csv,
result.json, val_sel/val_rep/train150 predictions, best.pt (and pre.pt for shared pretraining).
A crash still writes result.json (status 'failed' + traceback), so the notebook carries on with the rest.
usage: python job.py --spec spec.json [--gpu 0]
"""
import argparse, csv, json, math, os, random, sys, time, traceback

ap = argparse.ArgumentParser()
ap.add_argument("--spec", required=True)
ap.add_argument("--gpu", default=None)
args = ap.parse_args()
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)          # before torch is imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import lr05_lib as L

S = json.load(open(args.spec, encoding="utf-8"))
RUN = S["run_dir"]; os.makedirs(RUN, exist_ok=True)
log = L.Tee(os.path.join(RUN, "log.txt"))
dev = torch.device("cuda" if torch.cuda.is_available() and not S.get("cpu") else "cpu")
AMP = dev.type == "cuda"
torch.backends.cudnn.benchmark = False
if dev.type == "cpu":
    torch.set_num_threads(int(S.get("threads", 2)))
T_JOB = time.time()
DEADLINE = min(float(S.get("deadline", 1e18)), T_JOB + float(S.get("time_cap_s", 1e9)))
EVAL_RESERVE = float(S.get("eval_reserve_s", 75))            # stop training this long before the deadline
RESULT = dict(tag=S["tag"], hyp=S.get("hyp", []), desc=S.get("desc", ""), status="running", capped=False,
              spec={k: v for k, v in S.items() if k not in ("work", "run_dir")})


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=AMP)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=AMP)


def autocast():
    return torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=AMP)


def gpu_mem():
    return f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if dev.type == "cuda" else "cpu"


# ======================================================================= data
W = S["work"]
META = json.load(open(os.path.join(W, "meta.json"), encoding="utf-8"))
TEXT = META["text"]
VAL_SEL, VAL_REP, TR150 = META["val_sel"], META["val_rep"], META["train150"]
PINFO = META["patch_info"]
H = int(S["H"])
vocab = L.make_vocab(META["symbols"], META["train_texts"], S["vocab"], S.get("no_space", False))
real = L.LineStore(os.path.join(W, f"real_h{H}"))
by_patch = {}
for i, inf in enumerate(real.info):
    by_patch.setdefault(inf["patch"], []).append(i)
for p in by_patch:
    by_patch[p].sort(key=lambda i: real.info[i]["idx"])
TRAIN_LINES = [i for i, inf in enumerate(real.info) if inf["split"] == "train"]
TRAIN_PATCHES = sorted({real.info[i]["patch"] for i in TRAIN_LINES})
SHOW = VAL_SEL[0]                                           # the patch whose reading is printed every evaluation


@torch.no_grad()
def predict(model, store, groups, want_lp=False, bs=64):
    """Greedy reading of every line (and optionally its log-probs). Returns (texts, lps, blank share).
    groups = lists of line indices, one list per PATCH: each patch is read as its own batch, padded only to its
    widest line rounded to the stride - exactly how the CPU experiments read, so their numbers reproduce. (A BiLSTM's
    output depends slightly on how much padding follows a line, so batch composition must not vary between runs.)"""
    model.eval()
    out, lps, blank, tot = {}, {}, 0, 0
    for g in groups:
        for s in range(0, len(g), bs):
            b = g[s:s + bs]
            if not b:
                continue
            x, lens = L.to_batch([store.img(i) for i in b], store.H, round_to=L.STRIDE)
            with autocast():
                o = model(torch.from_numpy(x).to(dev))
            lp = o.float().log_softmax(-1).cpu().numpy()
            for k, i in enumerate(b):
                row = lp[k, :lens[k]]
                am = row.argmax(-1); blank += int((am == 0).sum()); tot += len(am)
                out[i] = vocab.decode(L.greedy_ids(row))
                if want_lp:
                    lps[i] = row
    model.train()
    return out, lps, blank / max(1, tot)


def eval_patches(model, patches, save=None):
    pred, _, blank = predict(model, real, [by_patch.get(p, []) for p in patches])
    sep = "" if vocab.no_space else " "
    hyps = [sep.join(t for t in (pred[i].strip() for i in by_patch.get(p, [])) if t) for p in patches]
    refs = [TEXT[p] for p in patches]
    m = L.metrics(refs, hyps)
    m["blank_pct"] = 100 * blank
    m["primary"] = m["CER_nospace"] if vocab.no_space else m["CER"]
    if save:
        import editdistance
        u = lambda s: s.replace("<FRAC>", ""); ns = lambda s: u(s).replace(" ", "")
        with open(os.path.join(RUN, save), "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f); wr.writerow(["patch", "ref", "hyp", "err", "n", "err_nospace", "n_nospace"])
            for p, r, h in zip(patches, refs, hyps):
                wr.writerow([p, r, h, editdistance.eval(u(r), u(h)), len(u(r)), editdistance.eval(ns(r), ns(h)), len(ns(r))])
    return m, refs, hyps


def fmt_m(m):
    return (f"CER {m['CER']:5.1f} | no-space {m['CER_nospace']:5.1f} | WER {m['WER']:5.1f} | BLEU {m['BLEU']:5.2f} | "
            f"chrF++ {m['chrF_pp']:5.1f} | median patch {m['median_patch_CER']:5.1f} | <50%: {m['patches_below_50']}/{m['n']} | "
            f"len {m['len_ratio']:.2f}x | blank {m['blank_pct']:4.1f}%")


HIST = []


def record(stage, step, samples, loss, lr, m, extra=None):
    row = dict(stage=stage, step=step, samples=samples, loss=round(loss, 4) if loss == loss else None, lr=lr,
               CER=round(m["CER"], 3), CER_nospace=round(m["CER_nospace"], 3), primary=round(m["primary"], 3),
               WER=round(m["WER"], 2), BLEU=round(m["BLEU"], 3), chrF_pp=round(m["chrF_pp"], 2),
               below50=m["patches_below_50"], len_ratio=round(m["len_ratio"], 3), blank_pct=round(m["blank_pct"], 2),
               minutes=round((time.time() - T_JOB) / 60, 2), **(extra or {}))
    HIST.append(row)
    with open(os.path.join(RUN, "history.csv"), "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in HIST for k in r)))
        wr.writeheader(); wr.writerows(HIST)


# ======================================================================= training machinery
class Stage:
    """Adam + linear warm-up + cosine decay to 2% of the peak, AMP on GPU, gradient clipping at 5 (as on CPU)."""

    def __init__(self, name, model, lr, total_steps, warmup=200, log_every=100):
        self.name, self.model, self.total = name, model, max(1, int(total_steps))
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        wu = min(warmup, max(1, self.total // 10))
        f = lambda s: min(1.0, (s + 1) / wu) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / self.total))))
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, f)
        self.scaler = make_scaler()
        self.step = self.seen = self.skipped = 0
        self.ema = None; self.gmax = 0.0; self.t0 = time.time(); self.win = (time.time(), 0)
        self.log_every = log_every; self.last_log = time.time()

    def update(self, batch):
        x, lens, y, yl = batch
        x = x.to(dev, non_blocking=True)
        with autocast():
            out = self.model(x)
        lp = out.float().log_softmax(-1)
        loss = F.ctc_loss(lp.permute(1, 0, 2), y.to(dev), lens, yl, blank=0, zero_infinity=True)
        self.opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            self.skipped += 1; self.step += 1; self.sched.step(); return
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        g = float(nn.utils.clip_grad_norm_(self.model.parameters(), 5.0))
        self.scaler.step(self.opt); self.scaler.update(); self.sched.step()
        if not math.isfinite(g):
            self.skipped += 1
        else:
            self.gmax = max(self.gmax, g)
        lv = float(loss.item())
        self.ema = lv if self.ema is None else 0.98 * self.ema + 0.02 * lv
        self.step += 1; self.seen += x.shape[0]
        if self.step % self.log_every == 0 or time.time() - self.last_log > 90:
            am = lp.detach().argmax(-1)
            valid = torch.arange(am.shape[1], device=am.device)[None, :] < lens.to(am.device)[:, None]
            blank = float(((am == 0) & valid).sum()) / max(1.0, float(valid.sum()))
            nz = ((am != 0) & valid); chg = torch.ones_like(am, dtype=torch.bool); chg[:, 1:] = am[:, 1:] != am[:, :-1]
            plen = float((nz & chg).sum()) / max(1, x.shape[0]); tlen = float(yl.float().mean())
            t1, n1 = self.win; rate = (self.seen - n1) / max(1e-9, time.time() - t1); self.win = (time.time(), self.seen)
            el = time.time() - self.t0; eta = el / max(1, self.step) * (self.total - self.step)
            log(f"[{self.name} {self.step:>6}/{self.total} {100*self.step/self.total:3.0f}%] loss {lv:6.3f} (ema {self.ema:6.3f}) | "
                f"lr {self.sched.get_last_lr()[0]:.2e} | |g| {g:6.2f} (max {self.gmax:6.1f}) | blank {100*blank:4.1f}% | "
                f"len {plen:5.1f}/{tlen:5.1f} | {rate:6.0f} lines/s | {self.seen:>8,} lines | {L.fmt_s(el)} el, ETA {L.fmt_s(eta)} | "
                f"mem {gpu_mem()}" + (f" | {self.skipped} overflow steps skipped (normal for a few early AMP steps)"
                                      if self.skipped else ""))
            self.last_log = time.time()
        return lv


def batches_from(store, idx_batches, targets, aug=False, rng=None, pin=False):
    """Generator of (x, lens, y, ylens) tensors for the given batches of store indices."""
    for b in idx_batches:
        imgs = [store.img(i) for i in b]
        if aug:
            imgs = [L.augment_line(im, rng, aug if isinstance(aug, str) else "light") for im in imgs]
        x, lens = L.to_batch(imgs, store.H)
        ys = [targets[i] for i in b]
        y = np.fromiter((t for yy in ys for t in yy), np.int64)
        yl = np.array([len(yy) for yy in ys], np.int64)
        xt = torch.from_numpy(x)
        if pin:
            xt = xt.pin_memory()
        yield xt, torch.from_numpy(lens), torch.from_numpy(y), torch.from_numpy(yl)


def time_left():
    return DEADLINE - EVAL_RESERVE - time.time()


# ======================================================================= stages
def pretrain(model, P):
    seed_all(P.get("seed", 0))
    pool = L.LineStore(os.path.join(W, f"pool_{P['pool']}_h{H}"), P.get("limit"))
    tg = [vocab.encode(lab) for lab in pool.labels]
    ok = np.array([i for i in range(len(pool)) if L.line_ok(len(tg[i]), pool.widths[i])], np.int64)
    if len(ok) == 0:
        raise RuntimeError(f"pool {P['pool']} has no usable lines")
    batch = int(P.get("batch", 32)); steps = max(1, int(P["samples"]) // batch)
    fpc = pool.widths[ok] / 2 / np.maximum(1, np.array([len(tg[i]) for i in ok]))
    log(f"PRETRAIN on pool {P['pool']} (first {len(pool):,} lines; style {pool.meta.get('style')}, text {pool.meta.get('text')}) | "
        f"usable {len(ok):,} (dropped {len(pool) - len(ok)} too long for CTC) | {steps:,} steps x {batch} = {steps*batch:,} lines "
        f"= {steps*batch/max(1,len(ok)):.1f} passes | targets/line median {np.median([len(tg[i]) for i in ok]):.0f} | "
        f"CTC frames per target median {np.median(fpc):.1f}")
    st = Stage("pre", model, P.get("lr", 1e-3), steps, log_every=int(P.get("log_every", 100)))
    rng = np.random.default_rng(P.get("seed", 0) + 17)

    def gen():
        n = 0
        while n < steps:
            for b in L.batch_plan(pool.widths, ok, batch, rng):
                if n >= steps:
                    return
                yield b; n += 1

    marks = set(int(steps * k / max(1, P.get("evals", 4))) for k in range(1, int(P.get("evals", 4)) + 1))
    stream = L.Prefetch(batches_from(pool, gen(), tg, pin=dev.type == "cuda"))
    for bt in stream:
        st.update(bt)
        if st.step in marks or st.step == steps:
            m, refs, hyps = eval_patches(model, VAL_SEL)
            record("pre", st.step, st.seen, st.ema if st.ema is not None else float("nan"), st.sched.get_last_lr()[0], m)
            log(f"  zero-shot REAL val ({len(VAL_SEL)} patches) after {st.seen:,} synthetic lines: {fmt_m(m)}")
            j = VAL_SEL.index(SHOW)
            log(f"     GT  : {refs[j][:110]}\n     PRED: {hyps[j][:110]}")
        if time_left() < 0:
            log(f"  TIME CAP: pretraining stopped at step {st.step:,}/{steps:,} ({100*st.step/steps:.0f}% of plan)")
            RESULT["capped"] = True
            break
    RESULT["pre"] = dict(pool=P["pool"], limit=P.get("limit"), steps_done=st.step, steps_planned=steps, lines_seen=st.seen,
                         minutes=round((time.time() - st.t0) / 60, 2), final_loss_ema=st.ema, skipped=st.skipped,
                         lines_per_s=round(st.seen / max(1e-9, time.time() - st.t0), 1))
    log(f"PRETRAIN done: {st.step:,} steps, {st.seen:,} lines in {L.fmt_s(time.time()-st.t0)} "
        f"({RESULT['pre']['lines_per_s']:.0f} lines/s), loss ema {st.ema:.3f}")


def realign(model, line_labels):
    """X3 at GPU scale: re-split every multi-line TRAINING patch's label with the current model."""
    t0 = time.time()
    pats = [p for p in TRAIN_PATCHES if len(by_patch[p]) > 1]
    changed = total = skipped = 0; dl = []
    for s in range(0, len(pats), 48):
        chunk = pats[s:s + 48]
        _, lps, _ = predict(model, real, [by_patch[p] for p in chunk], want_lp=True)
        for p in chunk:
            ls = by_patch[p]
            pieces = L.realign_patch([lps[i] for i in ls], TEXT[p], vocab)
            if pieces is None:
                skipped += 1; continue
            for i, piece in zip(ls, pieces):
                total += 1
                if piece != line_labels[i]:
                    changed += 1; dl.append(abs(len(piece) - len(line_labels[i])))
                line_labels[i] = piece
    msg = dict(patches=len(pats), skipped=skipped, lines=total, changed=changed,
               changed_pct=round(100 * changed / max(1, total), 1), median_abs_len_change=float(np.median(dl)) if dl else 0.0,
               seconds=round(time.time() - t0, 1))
    log(f"  RE-ALIGNED line labels (CTC forced alignment): {changed:,}/{total:,} changed ({msg['changed_pct']}%), "
        f"median |length change| {msg['median_abs_len_change']:.0f} chars, {skipped} patches skipped | {msg['seconds']}s")
    return msg


def finetune(model, Fs):
    seed_all(Fs.get("seed", 0))
    epochs = int(Fs["epochs"]); batch = int(Fs.get("batch", 32))
    line_labels = list(real.labels)                        # mutable: re-alignment rewrites them
    rng = np.random.default_rng(Fs.get("seed", 0) + 29)
    mix = float(Fs.get("mix", 0.0))
    mix_store = mix_tg = mix_ok = None
    if mix > 0:
        mix_store = L.LineStore(os.path.join(W, f"pool_{Fs.get('mix_pool', 'A')}_h{H}"), Fs.get("mix_limit"))
        mix_tg = [vocab.encode(l) for l in mix_store.labels]
        mix_ok = np.array([i for i in range(len(mix_store)) if L.line_ok(len(mix_tg[i]), mix_store.widths[i])], np.int64)

    def targets():
        tg = {i: vocab.encode(line_labels[i]) for i in TRAIN_LINES}
        ok = np.array([i for i in TRAIN_LINES if L.line_ok(len(tg[i]), real.widths[i])], np.int64)
        return tg, ok

    tg, ok = targets()
    per_epoch = int(math.ceil(len(ok) / batch)) * (2 if mix > 0 else 1)
    realign_at = set(int(e) for e in Fs.get("realign_at", []))
    if realign_at and vocab.no_space:
        log("  re-alignment needs word spaces to snap to; not applicable to no-space targets - skipped")
        realign_at = set()
    log(f"FINE-TUNE on real lines: {len(ok):,} usable training lines (dropped {len(TRAIN_LINES) - len(ok)}) | {epochs} epochs x "
        f"{per_epoch} steps (batch {batch}{', +1 synthetic batch per real batch' if mix > 0 else ''}) | "
        f"augmentation {(Fs['aug'] if isinstance(Fs['aug'], str) else 'light') if Fs.get('aug') else 'off'} | re-align after epochs {sorted(realign_at) or 'never'} | "
        f"select by val {'no-space ' if vocab.no_space else ''}CER")
    st = Stage("ft", model, Fs.get("lr", 1e-3), epochs * per_epoch, warmup=100, log_every=int(Fs.get("log_every", 60)))
    best = dict(primary=1e9, epoch=0, state=None)
    ra_log = []
    for ep in range(1, epochs + 1):
        t_ep = time.time()
        plan = L.batch_plan(real.widths, ok, batch, rng)
        if mix > 0:
            syn = L.batch_plan(mix_store.widths, mix_ok[rng.permutation(len(mix_ok))[:len(ok)]], batch, rng)

            def interleave():
                for k, b in enumerate(plan):
                    yield ("r", b)
                    if k < len(syn):
                        yield ("s", syn[k])
            src = interleave()

            def gen():
                for kind, b in src:
                    if kind == "r":
                        yield next(batches_from(real, [b], tg, aug=Fs.get("aug"), rng=rng, pin=dev.type == "cuda"))
                    else:
                        yield next(batches_from(mix_store, [b], mix_tg, pin=dev.type == "cuda"))
            stream = L.Prefetch(gen())
        else:
            stream = L.Prefetch(batches_from(real, plan, tg, aug=Fs.get("aug"), rng=rng, pin=dev.type == "cuda"))
        tot = n = 0
        for bt in stream:
            lv = st.update(bt)
            if lv is not None:
                tot += lv; n += 1
            if time_left() < 0:
                break
        m, refs, hyps = eval_patches(model, VAL_SEL)
        improved = m["primary"] < best["primary"]
        if improved:
            best.update(primary=m["primary"], epoch=ep, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        record("ft", ep, st.seen, tot / max(1, n), st.sched.get_last_lr()[0], m, dict(best=round(best["primary"], 3)))
        log(f"ep {ep:3d}/{epochs} | train loss {tot/max(1,n):6.3f} | val {fmt_m(m)} | best {best['primary']:5.1f}@{best['epoch']} | "
            f"{time.time()-t_ep:5.1f}s/epoch | {L.fmt_s(time.time()-T_JOB)} job time" + ("  *BEST*" if improved else ""))
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            j = VAL_SEL.index(SHOW)
            log(f"     GT  : {refs[j][:110]}\n     PRED: {hyps[j][:110]}")
            an = L.edit_anatomy(refs, hyps)
            log(f"     errors: deletions {an['deletions_pct']}% insertions {an['insertions_pct']}% substitutions {an['substitutions_pct']}% | "
                f"recall of key marks {an['focus_char_recall']}")
        if ep in realign_at and time_left() > 60:
            ra_log.append(dict(epoch=ep, **realign(model, line_labels)))
            tg, ok = targets()
        if time_left() < 0:
            log(f"  TIME CAP: fine-tuning stopped after epoch {ep}/{epochs}")
            RESULT["capped"] = True
            break
    RESULT["ft"] = dict(epochs_done=ep, epochs_planned=epochs, best_epoch=best["epoch"], best_val_sel_primary=best["primary"],
                        minutes=round((time.time() - st.t0) / 60, 2), lines_seen=st.seen, skipped=st.skipped,
                        lines_per_s=round(st.seen / max(1e-9, time.time() - st.t0), 1), realign=ra_log,
                        train_lines=int(len(ok)), dropped=int(len(TRAIN_LINES) - len(ok)))
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    log(f"FINE-TUNE done: best val(sel) {'no-space ' if vocab.no_space else ''}CER {best['primary']:.2f} at epoch {best['epoch']} | "
        f"{L.fmt_s(time.time()-st.t0)}")


def stroke_stats(store, idxs):
    """Ink share and median horizontal ink run (stroke thickness proxy, px at this line height)."""
    inks, runs = [], []
    for i in idxs:
        b = store.img(i) < 128
        inks.append(b.mean())
        d = np.diff(np.pad(b.astype(np.int8), ((0, 0), (1, 1))), axis=1)
        st, en = np.where(d == 1), np.where(d == -1)
        if len(st[1]):
            runs.append(float(np.median(en[1] - st[1])))
    return float(np.mean(inks)) if inks else float("nan"), float(np.median(runs)) if runs else float("nan")


def n1n2(patches, refs, hyps):
    """N1: do deletions grow where there are few CTC frames per character?  N2: do errors cluster by writing
    size (line pitch), number of lines, ink density or stroke thickness?"""
    from rapidfuzz.distance import Levenshtein
    rows = []
    for p, r, h in zip(patches, refs, hyps):
        r2, h2 = r.replace("<FRAC>", "\x01"), h.replace("<FRAC>", "\x01")
        ops = Levenshtein.editops(r2, h2)
        n = max(1, len(r2))
        frames = sum(math.ceil(real.widths[i] / 2) for i in by_patch.get(p, []))
        ink, stroke = stroke_stats(real, by_patch.get(p, []))
        rows.append(dict(patch=p, cer=100 * len(ops) / n, del_rate=100 * sum(o.tag == "delete" for o in ops) / n,
                         ins_rate=100 * sum(o.tag == "insert" for o in ops) / n,
                         frames_per_char=frames / max(1, len(r2.replace(" ", ""))), pitch=PINFO[p]["pitch"],
                         n_lines=PINFO[p]["n_lines"], label_len=len(r2), ink=ink, stroke=stroke))
    import pandas as pd
    d = pd.DataFrame(rows); d.to_csv(os.path.join(RUN, "n1n2_per_patch.csv"), index=False)
    out = dict(n=len(d))
    for f in ["frames_per_char", "pitch", "n_lines", "label_len", "ink", "stroke"]:
        q = pd.qcut(d[f].rank(method="first"), 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
        out[f] = dict(spearman_cer=round(L.spearman(d[f], d.cer), 3), spearman_del=round(L.spearman(d[f], d.del_rate), 3),
                      cer_by_quartile={str(k): round(v, 1) for k, v in d.groupby(q, observed=True).cer.mean().items()},
                      del_by_quartile={str(k): round(v, 1) for k, v in d.groupby(q, observed=True).del_rate.mean().items()})
    return out


def inside_figure(model, patches, path, n=4):
    """What the network does inside, frame by frame: under each line image, the probability of blank (grey) and of
    its best character (red). Each red spike is where it writes a character; a line with no spikes reads nothing."""
    try:
        import warnings
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = [by_patch[p] for p in patches if by_patch.get(p)][:n]
        if not groups:
            return
        idxs = [g[0] for g in groups]                      # first line of each patch, read with its own patch
        _, lps, _ = predict(model, real, groups, want_lp=True)
        fig, axes = plt.subplots(2 * len(idxs), 1, figsize=(16, 2.2 * len(idxs)), squeeze=False)
        conf = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k, i in enumerate(idxs):
                img, pr = real.img(i), np.exp(lps[i])
                ax = axes[2 * k, 0]
                ax.imshow(img, cmap="gray", aspect="auto", extent=(0, img.shape[1], img.shape[0], 0)); ax.axis("off")
                emit = pr[:, 1:].max(1)[pr.argmax(1) != 0]
                conf += list(emit)
                ax.set_title(f"{real.info[i]['patch']} line {real.info[i]['idx'] + 1}: {len(emit)} characters emitted, "
                             f"mean confidence {emit.mean() if len(emit) else 0:.2f}, blank on {100 * (pr.argmax(1) == 0).mean():.0f}% of frames",
                             fontsize=8, loc="left")
                ax2 = axes[2 * k + 1, 0]; x = np.arange(len(pr)) * L.STRIDE
                ax2.plot(x, pr[:, 0], color="0.6", lw=0.8, label="blank")
                ax2.plot(x, pr[:, 1:].max(1), color="tab:red", lw=0.9, label="best character")
                ax2.set_xlim(0, img.shape[1]); ax2.set_ylim(0, 1.02); ax2.legend(fontsize=7, loc="upper right")
        fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)
        log(f"  inside the network: {len(conf)} characters emitted on {len(idxs)} val lines, mean confidence "
            f"{np.mean(conf) if conf else 0:.2f} -> {os.path.basename(path)}")
    except Exception as e:
        log(f"  (inside-the-network figure skipped: {e})")


def final_eval(model):
    t0 = time.time()
    res = {}
    allv = VAL_SEL + VAL_REP
    m1, r1, h1 = eval_patches(model, VAL_SEL, save="val_sel_predictions.csv")
    m2, r2, h2 = eval_patches(model, VAL_REP, save="val_rep_predictions.csv")
    m3, _, _ = eval_patches(model, TR150, save="train150_predictions.csv")
    res.update(val_sel=m1, val_rep=m2, train150=m3)
    res["anatomy"] = L.edit_anatomy(r1 + r2, h1 + h2)
    res["n1n2"] = n1n2(allv, r1 + r2, h1 + h2)
    log("=" * 110)
    log(f"FINAL [{S['tag']}] best checkpoint, greedy decoding")
    log(f"  VAL-SEL ({len(VAL_SEL)} patches, used to pick the epoch): {fmt_m(m1)}")
    log(f"  VAL-REP ({len(VAL_REP)} patches, never used for any choice): {fmt_m(m2)}")
    log(f"  TRAIN   ({len(TR150)} patches; reading vs memorising): {fmt_m(m3)}")
    a = res["anatomy"]
    log(f"  errors on all val: {a['errors']:,} = deletions {a['deletions_pct']}% / insertions {a['insertions_pct']}% / "
        f"substitutions {a['substitutions_pct']}%")
    log(f"  most missed: {a['most_missed']}")
    log(f"  top confusions: {a['top_confusions'][:10]}")
    log(f"  recall of key marks: {a['focus_char_recall']}")
    for f, v in res["n1n2"].items():
        if isinstance(v, dict):
            log(f"  N1/N2 {f:16s}: Spearman with CER {v['spearman_cer']:+.2f}, with deletion rate {v['spearman_del']:+.2f} | "
                f"CER by quartile {v['cer_by_quartile']}")
    for k in range(min(3, len(r2))):
        log(f"   GT  : {r2[k][:110]}\n   PRED: {h2[k][:110]}")
    inside_figure(model, VAL_REP, os.path.join(RUN, "inside_alignment.png"))
    log(f"  (final evaluation {time.time()-t0:.0f}s)")
    log("=" * 110)
    return res


def bench():
    """Throughput of each model shape on this GPU, so the notebook can size every job to the time budget."""
    out = {}
    for cfg in S["bench"]:
        st_store = L.LineStore(os.path.join(W, f"real_h{cfg['H']}"))
        bp = {}
        for i, inf in enumerate(st_store.info):
            bp.setdefault(inf["patch"], []).append(i)
        model = L.build_crnn(len(vocab), cfg["size"]).to(dev)
        seed_all(0)
        tg = {i: vocab.encode(st_store.labels[i]) for i in range(len(st_store)) if st_store.info[i]["split"] == "train"}
        ok = np.array([i for i in tg if L.line_ok(len(tg[i]), st_store.widths[i])], np.int64)
        n_warm, n_time = int(cfg.get("warm", 5)), int(cfg.get("steps", 25))
        plan = L.batch_plan(st_store.widths, ok, 32, np.random.default_rng(0))
        plan = (plan * ((n_warm + n_time) // max(1, len(plan)) + 1))[:n_warm + n_time]     # cycle if the set is tiny
        stg = Stage("bench", model, 1e-3, 1000, log_every=10 ** 9)
        it = iter(L.Prefetch(batches_from(st_store, plan, tg, pin=dev.type == "cuda")))
        for _ in range(n_warm):
            stg.update(next(it))
        if dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.time(); n0 = stg.seen
        for _ in range(n_time):
            stg.update(next(it))
        if dev.type == "cuda": torch.cuda.synchronize()
        rate = (stg.seen - n0) / max(1e-9, time.time() - t0)
        model.eval()
        t1 = time.time()
        idxs = [i for p in VAL_SEL for i in bp.get(p, [])]
        with torch.no_grad():
            for s in range(0, len(idxs), 64):
                b = idxs[s:s + 64]
                x, _ = L.to_batch([st_store.img(i) for i in b], st_store.H)
                with autocast():
                    model(torch.from_numpy(x).to(dev))
        if dev.type == "cuda": torch.cuda.synchronize()
        ev = time.time() - t1
        key = f"{cfg['size']}_h{cfg['H']}"
        out[key] = dict(lines_per_s=round(rate, 1), eval_s=round(ev + 1.0, 2),
                        params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 3), mem=gpu_mem())
        log(f"BENCH {key}: {rate:,.0f} training lines/s (batch 32, AMP {'on' if AMP else 'off'}) | val forward pass "
            f"{ev:.1f}s | {out[key]['params_M']} M params | peak mem {gpu_mem()}")
        del model
        if dev.type == "cuda": torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(RUN, "bench.json"), "w"), indent=2)
    RESULT["bench"] = out


def main():
    log(f"JOB {S['tag']} | hypotheses {S.get('hyp')} | {S.get('desc','')}")
    log(f"  device {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}) | torch {torch.__version__} | "
        f"AMP {'on' if AMP else 'off'} | deadline in {L.fmt_s(DEADLINE - time.time())}")
    log(f"  model {S.get('size')} | line height {H} px | targets {vocab.describe()} | init {os.path.basename(S.get('init') or '') or 'random'}")
    if S.get("bench"):
        bench(); RESULT["status"] = "ok"; return
    seed_all(int((S.get("pretrain") or {}).get("seed", 0)))
    model = L.build_crnn(len(vocab), S["size"]).to(dev)
    RESULT["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)
    log(f"  {RESULT['params_M']} M parameters | {len(TRAIN_LINES):,} real training lines in {len(TRAIN_PATCHES):,} patches | "
        f"val-sel {len(VAL_SEL)} / val-rep {len(VAL_REP)} patches")
    if S.get("init"):
        sd = torch.load(S["init"], map_location="cpu")
        model.load_state_dict(sd["model"] if "model" in sd else sd, strict=True)
        log(f"  loaded {S['init']} (strict)")
    if S.get("pretrain"):
        pretrain(model, S["pretrain"])
        if S.get("save_pre"):
            torch.save(dict(model=model.state_dict(), size=S["size"], H=H, vocab=vocab.describe()), os.path.join(RUN, "pre.pt.tmp"))
            os.replace(os.path.join(RUN, "pre.pt.tmp"), os.path.join(RUN, "pre.pt"))
            log(f"  saved {os.path.join(RUN, 'pre.pt')}")
    if S.get("finetune"):
        finetune(model, S["finetune"])
    torch.save(dict(model=model.state_dict(), size=S["size"], H=H, vocab=vocab.describe(), tag=S["tag"]),
               os.path.join(RUN, "best.pt"))
    RESULT.update(final_eval(model))
    RESULT["status"] = "ok"


if __name__ == "__main__":
    try:
        main()
    except Exception:
        RESULT["status"] = "failed"; RESULT["error"] = traceback.format_exc()
        log("JOB FAILED:\n" + RESULT["error"])
    RESULT["job_minutes"] = round((time.time() - T_JOB) / 60, 2)
    if dev.type == "cuda":
        RESULT["peak_gpu_mem"] = gpu_mem()
    with open(os.path.join(RUN, "result.json.tmp"), "w", encoding="utf-8") as f:
        json.dump(RESULT, f, indent=2, ensure_ascii=False, default=str)
    os.replace(os.path.join(RUN, "result.json.tmp"), os.path.join(RUN, "result.json"))
    log(f"JOB {S['tag']} {RESULT['status'].upper()} in {RESULT['job_minutes']} min")
    sys.exit(0 if RESULT["status"] == "ok" else 1)
