"""lr05_run.py - the experiment matrix, the time planner and the two-GPU scheduler of notebook 07 (copied from notebook 05).

Every job changes ONE thing relative to the anchor (FT_A0_s0 = 60k plain synthetic lines -> 40 real epochs),
so each comparison answers one question. Jobs run as separate processes, one per GPU; a free GPU always
takes the highest-priority job whose inputs (a synthetic pool, a pretrained checkpoint) are ready, and a
job is only started if its time estimate fits before the deadline.
"""
import json, math, os, subprocess, sys, time
import lr05_lib as L

# ---------------------------------------------------------------- the questions (also used by the report)
HYPOTHESES = [
    # id, question, variant, control, metric  (notebook 07)
    ("N6", "Light augmentation of the real lines - better than the anchor? (replicates notebook 05)", "AUG", "FT_A0_s0", "CER"),
    ("R2", "Add a horizontal stretch (character density) to the augmentation - better than light AUG?", "AUG_STRETCH", "AUG", "CER"),
    ("R3", "Add rotation up to 5 degrees (tilted pages) to the augmentation - better than light AUG?", "AUG_TILT", "AUG", "CER"),
    ("R4", "Stretch + tilt + blur together - better than light AUG?", "AUG_STRONG", "AUG", "CER"),
    ("R1", "Pretrain on HEAVILY distorted synthetic lines instead of plain ones - better after the same fine-tuning?", "HEAVY_AUG", "AUG", "CER"),
]


def experiment_matrix(runs, base, sizes):
    """Notebook 07: augmentations aimed at unseen pages (stretch, tilt) and heavier synthetic distortion, each against
    the same controls (anchor, light AUG). sizes: pool A (plain synthetic) and pool D (heavy synthetic)."""
    pre = lambda pool, limit, samples=None: dict(pool=pool, limit=limit, samples=int(samples or base["pre_samples"]),
                                                 lr=1e-3, batch=base["batch"], evals=4, seed=0)
    ft = lambda **kw: dict(dict(epochs=base["ft_epochs"], lr=1e-3, batch=base["batch"], aug=False, mix=0.0,
                                realign_frac=[], seed=0), **kw)
    PTA, PTD = os.path.join(runs, "PT_A60", "pre.pt"), os.path.join(runs, "PT_D60", "pre.pt")
    a_mid, d_mid = sizes["A_mid"], sizes["D_mid"]
    J = []

    def add(tag, pri, hyp, desc, size="small", H=40, vocab="char", no_space=False, init=None, pretrain=None,
            finetune=None, save_pre=False, after=(), pools=()):
        J.append(dict(tag=tag, priority=pri, hyp=list(hyp), desc=desc, size=size, H=H, vocab=vocab, no_space=no_space,
                      init=init, pretrain=pretrain, finetune=finetune, save_pre=save_pre, after=list(after),
                      pools=[list(p) for p in pools]))

    add("PT_A60", 1, ["anchor"], f"shared pretraining: {a_mid:,} plain synthetic lines (zero-shot CER logged)",
        pretrain=pre("A", a_mid), save_pre=True, pools=[("A", 40, a_mid)])
    add("PT_D60", 1, ["R1"], f"shared pretraining: {d_mid:,} HEAVY synthetic lines (zero-shot CER logged)",
        pretrain=pre("D", d_mid), save_pre=True, pools=[("D", 40, d_mid)])
    add("FT_A0_s0", 1, ["anchor"], "ANCHOR: PT_A60 -> real lines, seed 0", init=PTA, finetune=ft(seed=0), after=["PT_A60"])
    add("AUG", 1, ["N6"], "anchor + light augmentation (notebook 05's best single change) = the control here",
        init=PTA, finetune=ft(aug=True), after=["PT_A60"])
    add("AUG_STRETCH", 1, ["R2"], "light augmentation + horizontal stretch 0.75-1.4x", init=PTA, finetune=ft(aug="stretch"),
        after=["PT_A60"])
    add("AUG_TILT", 1, ["R3"], "light augmentation + rotation up to 5 degrees, re-cropped", init=PTA, finetune=ft(aug="tilt"),
        after=["PT_A60"])
    add("HEAVY_AUG", 1, ["R1"], "heavy-synthetic pretraining (PT_D60) -> real lines with light augmentation", init=PTD,
        finetune=ft(aug=True), after=["PT_D60"])
    add("AUG_STRONG", 2, ["R4"], "stretch + tilt + blur", init=PTA, finetune=ft(aug="strong"), after=["PT_A60"])
    add("FT_A0_s1", 2, ["seeds"], "anchor, fine-tune seed 1 (run-to-run noise)", init=PTA, finetune=ft(seed=1), after=["PT_A60"])
    add("AUG_s1", 2, ["seeds"], "light AUG, seed 1 (noise of the control)", init=PTA, finetune=ft(aug=True, seed=1),
        after=["PT_A60"])
    return J


def estimate_s(j, bench, n_train_lines):
    b = bench.get(f"{j['size']}_h{j['H']}")
    if b is None:
        # never assume the fastest model's speed (notebook 05: big@64px was estimated at small@40px speed, 705 lines/s,
        # but runs at ~190 - COMBO/FINAL were starved). Use the SLOWEST measured configuration and say so.
        b = min(bench.values(), key=lambda v: v["lines_per_s"])
        print(f"WARNING: no benchmark for {j['size']}@{j['H']}px - estimating {j['tag']} at the slowest measured speed "
              f"({b['lines_per_s']:.0f} lines/s)", flush=True)
    thr, ev = max(1.0, b["lines_per_s"]), b["eval_s"]
    t = 40.0                                                             # start-up + loading
    if j.get("pretrain"):
        t += j["pretrain"]["samples"] / thr + j["pretrain"]["evals"] * ev
    if j.get("finetune"):
        Fz = j["finetune"]
        t += Fz["epochs"] * n_train_lines * (2 if Fz.get("mix", 0) > 0 else 1) / thr + Fz["epochs"] * ev
        t += len(Fz.get("realign_frac", [])) * (6 * ev + 15)
    t += 4 * ev + 15                                                     # final evaluation
    return t * (1.06 if j["vocab"] == "akshara" else 1.0)


def apply_scale(jobs, scale, base):
    for j in jobs:
        if j.get("pretrain"):
            j["pretrain"]["samples"] = max(j["pretrain"]["batch"] * 20, int(j["pretrain"]["samples"] * scale))
        if j.get("finetune"):
            j["finetune"]["epochs"] = max(min(4, j["finetune"]["epochs"]), int(round(j["finetune"]["epochs"] * scale)))
    return jobs


def finalize_spec(j, work, runs, deadline, est):
    s = dict(j)
    s.update(work=work, run_dir=os.path.join(runs, j["tag"]), deadline=deadline, time_cap_s=est * 1.8 + 180)
    if s.get("finetune"):
        Fz = dict(s["finetune"]); E = Fz["epochs"]
        Fz["realign_at"] = sorted(set(max(1, min(E - 1, int(round(f * E)))) for f in Fz.pop("realign_frac", [])))
        s["finetune"] = Fz
    return s


class Scheduler:
    def __init__(self, jobs, gpus, script, work, runs, deadline, bench, n_train_lines, log, poll=10.0,
                 renderer=None, reserve=None, cpu=False, threads=2, tails=(), known=None, ratios=None):
        self.jobs = {j["tag"]: j for j in jobs}
        self.order = [j["tag"] for j in sorted(jobs, key=lambda j: j["priority"])]
        self.state = {t: "pending" for t in self.order}
        self.gpus, self.script, self.work, self.runs = list(gpus), script, work, runs
        self.deadline, self.bench, self.ntl, self.log, self.poll = deadline, bench, n_train_lines, log, poll
        self.renderer, self.reserve = renderer, (reserve or {})
        self.cpu, self.threads = cpu, threads
        self.running = {}                         # gpu -> dict(tag, proc, fh, pos, t0, est)
        self.t_dash = 0.0
        self.notes = {}
        self.tails = [dict(path=p, tag=t, pos=0) for p, t in tails]     # other logs to stream (the renderer)
        self.known = dict(known or {})            # states of jobs finished by an earlier scheduler pass
        self.ratios = ratios if ratios is not None else []    # actual / estimated time of finished jobs (shared)

    def factor(self):
        """Correction learned from finished jobs: median of actual/estimated time, clamped to 0.5-2x."""
        r = sorted(self.ratios)
        return min(2.0, max(0.5, r[len(r) // 2])) if len(r) >= 2 else 1.0

    def est(self, tag):
        return estimate_s(self.jobs[tag], self.bench, self.ntl) * self.factor()

    def _deps(self, tag):
        j = self.jobs[tag]
        for t in j["after"]:
            st = self.state.get(t) or self.known.get(t, "missing")
            if st in ("failed", "skipped", "missing"):
                return "dead", f"needs {t}, which {st}"
            if st != "ok":
                return "wait", f"waiting for {t}"
        for name, H, need in j["pools"]:
            base = os.path.join(self.work, f"pool_{name}_h{H}")
            if not L.store_ready(base, need):
                if self.renderer is not None and self.renderer.poll() is not None:
                    return "dead", f"pool {name} (h{H}) never reached {need:,} lines"
                return "wait", f"waiting for pool {name} >= {need:,} lines"
        return "ready", ""

    def _launch(self, tag, gpu):
        est = self.est(tag)
        spec = finalize_spec(self.jobs[tag], self.work, self.runs, self.deadline, est)
        if self.cpu:
            spec.update(cpu=True, threads=self.threads)
        os.makedirs(spec["run_dir"], exist_ok=True)
        sp = os.path.join(spec["run_dir"], "spec.json")
        json.dump(spec, open(sp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        out = os.path.join(spec["run_dir"], "stdout.txt")
        fh = open(out, "w", encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        p = subprocess.Popen([sys.executable, "-u", self.script, "--spec", sp] + ([] if self.cpu else ["--gpu", str(gpu)]),
                             stdout=fh, stderr=subprocess.STDOUT, env=env)
        self.running[gpu] = dict(tag=tag, proc=p, fh=fh, path=out, pos=0, t0=time.time(), est=est,
                                 est_raw=estimate_s(self.jobs[tag], self.bench, self.ntl))
        self.state[tag] = "running"
        self.log(f">>> START {tag} on {'cpu-slot' if self.cpu else 'GPU'} {gpu} | P{self.jobs[tag]['priority']} | "
                 f"{self.jobs[tag]['desc']} | estimated {L.fmt_s(est)} | {L.fmt_s(self.deadline - time.time())} left")

    def _drain(self, r):
        try:
            with open(r["path"], "r", encoding="utf-8", errors="replace") as f:
                f.seek(r["pos"]); chunk = f.read(); r["pos"] = f.tell()
        except FileNotFoundError:
            return
        for line in chunk.splitlines():
            self.log(f"[{r['tag']}] {line}")

    def _reap(self):
        for gpu, r in list(self.running.items()):
            if r["proc"].poll() is None:
                if time.time() > self.deadline + 600:              # backstop; jobs obey the deadline themselves
                    r["proc"].kill()
                continue
            self._drain(r); r["fh"].close()
            res = os.path.join(self.runs, r["tag"], "result.json")
            ok = os.path.exists(res) and json.load(open(res, encoding="utf-8")).get("status") == "ok"
            self.state[r["tag"]] = "ok" if ok else "failed"
            took = time.time() - r["t0"]
            if ok:
                self.ratios.append(took / max(1.0, r["est_raw"]))
            self.log(f"<<< {'DONE' if ok else 'FAILED'} {r['tag']} in {L.fmt_s(took)} (estimated {L.fmt_s(r['est'])}, "
                     f"ratio {took / max(1, r['est']):.2f}) | time estimates now x{self.factor():.2f} (learned from {len(self.ratios)} jobs)")
            del self.running[gpu]

    def dashboard(self, force=False):
        if not force and time.time() - self.t_dash < 300:
            return
        self.t_dash = time.time()
        cnt = {s: sum(1 for v in self.state.values() if v == s) for s in ("ok", "failed", "skipped", "running", "pending")}
        self.log("-" * 110)
        self.log(f"DASHBOARD {time.strftime('%H:%M:%S')} | {L.fmt_s(self.deadline - time.time())} left before the deadline | "
                 f"done {cnt['ok']} | failed {cnt['failed']} | skipped {cnt['skipped']} | running {cnt['running']} | waiting {cnt['pending']}")
        for gpu, r in self.running.items():
            self.log(f"   slot {gpu}: {r['tag']} running {L.fmt_s(time.time() - r['t0'])} of ~{L.fmt_s(r['est'])}")
        waiting = [f"{t} ({self._deps(t)[1] or 'ready'})" for t in self.order if self.state[t] == "pending"]
        if waiting:
            self.log(f"   queue: {', '.join(waiting[:12])}")
        self.log(f"   {L.gpu_status()}")
        try:
            import psutil
            vm = psutil.virtual_memory(); du = psutil.disk_usage(self.work)
            self.log(f"   RAM {vm.used / 2**30:.1f}/{vm.total / 2**30:.1f} GB | disk free {du.free / 2**30:.0f} GB | CPU load {os.getloadavg()[0]:.1f}"
                     if hasattr(os, "getloadavg") else f"   RAM {vm.used / 2**30:.1f}/{vm.total / 2**30:.1f} GB")
        except Exception:
            pass
        self.log("-" * 110)

    def run(self):
        while True:
            self._reap()
            for gpu in self.gpus:
                if gpu in self.running:
                    continue
                for tag in self.order:
                    if self.state[tag] != "pending":
                        continue
                    st, why = self._deps(tag)
                    if st == "dead":
                        self.state[tag] = "skipped"; self.notes[tag] = why
                        self.log(f"--- SKIP {tag}: {why}")
                        continue
                    left = self.deadline - self.reserve.get(self.jobs[tag]["priority"], 0) - time.time()
                    if self.est(tag) > left:
                        if st == "ready" or self.est(tag) > self.deadline - time.time():
                            self.state[tag] = "skipped"; self.notes[tag] = f"not enough time (needs ~{L.fmt_s(self.est(tag))}, {L.fmt_s(max(0, left))} left)"
                            self.log(f"--- SKIP {tag}: {self.notes[tag]}")
                        continue
                    if st == "ready":
                        self._launch(tag, gpu)
                        break
            for r in list(self.running.values()) + self.tails:
                self._drain(r)
            self.dashboard()
            if not self.running and not any(s == "pending" for s in self.state.values()):
                break
            if not self.running and all(self._deps(t)[0] == "wait" for t in self.order if self.state[t] == "pending"):
                if time.time() > self.deadline:
                    for t in self.order:
                        if self.state[t] == "pending":
                            self.state[t] = "skipped"; self.notes[t] = "deadline passed while waiting"
                    break
            time.sleep(self.poll)
        self.dashboard(force=True)
        return dict(self.state)


# ---------------------------------------------------------------- the adaptive combination
def combo_jobs(results, base, sizes, runs, margin, log):
    """Build COMBO from every factor whose variant beat its control on VAL-SEL by more than `margin` CER points
    (VAL-REP stays untouched for the report). Returns (jobs, explanation)."""
    def sel(tag):
        r = results.get(tag)
        return r["val_sel"]["primary"] if r and r.get("status") == "ok" else None
    anchor = sel("FT_A0_s0")
    if anchor is None:
        return [], ["anchor missing - no combination"]
    why, win = [], {}
    pools = {"A60": anchor, "A120": sel("A120"), "B60": sel("B60"), "C60": sel("C60")}
    best_pool = min((k for k in pools if pools[k] is not None), key=lambda k: pools[k])
    if best_pool != "A60" and pools[best_pool] < anchor - margin:
        win["pool"] = best_pool
    for factor, tag in [("big", "BIG"), ("akshara", "AKS"), ("h64", "H64"), ("mix", "MIX"), ("aug", "AUG")]:
        v = sel(tag)
        if v is not None and v < anchor - margin:
            win[factor] = True
    em = {k: sel(k) for k in ("EM1", "EM2")}
    em_best = min((k for k in em if em[k] is not None), key=lambda k: em[k], default=None)
    if em_best and em[em_best] < anchor - margin:
        win["realign"] = em_best
    for k, v in sorted(pools.items()):
        why.append(f"pool {k}: val-sel CER {v if v is None else round(v, 2)}")
    for tag in ("BIG", "AKS", "H64", "MIX", "AUG", "EM1", "EM2"):
        why.append(f"{tag}: val-sel {sel(tag) if sel(tag) is None else round(sel(tag), 2)} vs anchor {anchor:.2f}")
    why.append(f"winning changes (better than the anchor by > {margin:.2f} CER on val-sel): {win or 'none'}")
    # COMBO uses exactly the subset the winning experiment used; FINAL uses the WHOLE pool of the winning style
    pool_name, limit = {"A60": ("A", sizes["A_mid"]), "A120": ("A", sizes["A_big"]), "B60": ("B", sizes["B_mid"]),
                        "C60": ("C", sizes["C_mid"])}[win.get("pool", "A60")]
    H = 64 if win.get("h64") else 40
    if H == 64:
        pool_name, limit = "A", min(sizes["A_mid"], sizes["A_h64"])
    full = sizes["A_h64"] if H == 64 else sizes[pool_name]

    def recipe(lim, seed):
        pre = dict(pool=pool_name, limit=lim, samples=int(base["pre_samples"]), lr=1e-3, batch=base["batch"], evals=4, seed=seed)
        ft = dict(epochs=base["ft_epochs"], lr=1e-3, batch=base["batch"], aug=bool(win.get("aug")),
                  mix=0.5 if win.get("mix") else 0.0, mix_pool="A", mix_limit=sizes["A_mid"], seed=seed,
                  realign_frac={"EM1": [1 / 3], "EM2": [1 / 3, 2 / 3]}.get(win.get("realign"), []))
        common = dict(size="big" if win.get("big") else "small", H=H, vocab="akshara" if win.get("akshara") else "char",
                      no_space=False, init=None, save_pre=False, after=[], pools=[[pool_name, H, lim]])
        if ft["mix"] > 0:
            common["pools"].append(["A", H, sizes["A_mid"]])
        return pre, ft, common

    desc = "COMBINATION of " + (", ".join(f"{k}={v}" for k, v in win.items()) if win else "nothing (no change beat the anchor)")
    jobs = []
    if win:
        pre, ft, common = recipe(limit, 0)
        jobs.append(dict(tag="COMBO", priority=4, hyp=["COMBO"], desc=desc + " - normal length", pretrain=pre, finetune=ft, **common))
    for s in (0, 1):
        pre, ft, common = recipe(full, s)
        jobs.append(dict(tag=f"FINAL_s{s}", priority=4, hyp=["FINAL"],
                         desc=f"FINAL model, seed {s}: {'the winning recipe' if win else 'the anchor recipe'} on the whole "
                              f"{pool_name} pool ({full:,} lines), trained as long as the time allows", pretrain=pre, finetune=ft, **common))
    return jobs, why


def fill_to(j, target_s, bench, ntl, factor=1.0, max_ft_mult=2.5):
    """Lengthen a job (never shorten it) until its estimated duration is ~target_s: fine-tuning up to max_ft_mult x
    its epochs, all remaining time to pretraining (synthetic data is unlimited, real data is not)."""
    est = estimate_s(j, bench, ntl) * factor
    if est >= target_s:
        return j
    if j.get("finetune"):
        j["finetune"]["epochs"] = int(round(j["finetune"]["epochs"] * min(target_s / est, max_ft_mult)))
    if j.get("pretrain"):
        b = bench.get(f"{j['size']}_h{j['H']}") or bench.get("small_h40")
        rest = target_s - estimate_s(dict(j, pretrain=dict(j["pretrain"], samples=0)), bench, ntl) * factor
        j["pretrain"]["samples"] = max(j["pretrain"]["samples"], int(rest / factor * b["lines_per_s"]))
    return j
