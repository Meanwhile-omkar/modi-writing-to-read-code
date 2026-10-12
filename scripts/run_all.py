"""Overnight orchestrator: runs the audit steps in order with SANITY GATES.

Why gates: during development the blank-page metric silently produced a
degenerate result (every page scored ~0, then 26% of pages were flagged "blank").
That was only caught by eye, after a full pass had already been spent. Each gate
below encodes a judgement already made, so a regression fails in seconds instead
of wasting a whole run.

A gate failure STOPS the pipeline and writes the reason to the log. It never
"fixes" anything by itself: changing a metric or a threshold is a judgement call
and project rule 6 says those go to a human, not to an automated guess.

Usage:
    python scripts/run_all.py                 # all steps, skipping completed ones
    python scripts/run_all.py --from 03       # resume from a step
    python scripts/run_all.py --only 06       # one step
    python scripts/run_all.py --force         # re-run even if outputs exist
    python scripts/run_all.py --overlap-full  # step 06 full run (not just pilot)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from modi.utils import console  # noqa: F401,E402
from modi.utils.paths import P  # noqa: E402

import pandas as pd  # noqa: E402

PY = str(CODE_ROOT / ".venv" / "Scripts" / "python.exe")
LOG_DIR = CODE_ROOT / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Sanity gates: each returns (ok: bool, message: str)
# --------------------------------------------------------------------------
def gate_manifest():
    f = P.processed / "hhdoc_manifest.csv"
    if not f.exists():
        return False, "hhdoc_manifest.csv missing"
    df = pd.read_csv(f, dtype={"folder": str})
    if len(df) < 3000:
        return False, f"only {len(df)} rows, expected ~3352"
    bad = df["md5"].isna().sum()
    if bad > 10:
        return False, f"{bad} unreadable images (expected ~1)"
    return True, f"{len(df)} rows, {bad} unreadable"


def gate_edge_density():
    df = pd.read_csv(P.processed / "hhdoc_manifest.csv", dtype={"folder": str})
    if "edge_density" not in df.columns:
        return False, "edge_density column missing"
    ed = pd.to_numeric(df["edge_density"], errors="coerce").dropna()
    # The historical failure: Otsu on a skewed gradient histogram collapsed every
    # page to ~0. Guard against any metric that has no spread or sits at a bound.
    if ed.std() < 0.01:
        return False, f"edge_density has no spread (std={ed.std():.6f}) - degenerate metric"
    if ed.mean() < 0.005:
        return False, f"edge_density collapsed to ~0 (mean={ed.mean():.6f})"
    if ed.max() > 0.99:
        return False, f"edge_density saturated (max={ed.max():.4f})"
    return True, f"mean={ed.mean():.3f} std={ed.std():.3f} min={ed.min():.4f} max={ed.max():.3f}"


def gate_dedup():
    df = pd.read_csv(P.processed / "hhdoc_manifest.csv", dtype={"folder": str})
    for c in ("dup_group", "dup_type", "keep"):
        if c in df.columns:
            continue
        return False, f"column {c} missing"
    removed = int((df["keep"] == False).sum())  # noqa: E712
    if removed == 0:
        return False, "no duplicates found at all - expected ~210 (13<->15, 20<->22)"
    if removed > len(df) * 0.25:
        return False, f"{removed} of {len(df)} marked duplicate - implausibly high"
    return True, f"{removed} duplicates marked, {int(df['keep'].sum())} kept"


def gate_quality_flags():
    df = pd.read_csv(P.processed / "hhdoc_manifest.csv", dtype={"folder": str})
    for c in ("flag_blank", "flag_non_modi_printed", "flag_unusual_shape"):
        if c not in df.columns:
            return False, f"column {c} missing"
    nblank = int(df["flag_blank"].sum())
    # A blank rate above ~10% means the metric is mis-firing, which is exactly the
    # bug this gate exists to catch.
    if nblank > len(df) * 0.10:
        return False, (f"flag_blank={nblank} ({nblank/len(df):.1%} of pages) - "
                       "implausible, the blank metric is probably degenerate again")
    nprinted = int(df["flag_non_modi_printed"].sum())
    if nprinted == 0:
        return False, "folder 18 (printed Devanagari) not flagged"
    return True, (f"blank={nblank} printed={nprinted} "
                  f"unusual_shape={int(df['flag_unusual_shape'].sum())}")


def gate_text_audit():
    f = P.reports / "modetrans_text_audit.md"
    if not f.exists() or f.stat().st_size < 1000:
        return False, "modetrans_text_audit.md missing or too small"
    return True, f"{f.stat().st_size//1024} KB report"


def gate_text_norm():
    f = P.processed / "modetrans_text.csv"
    if not f.exists():
        return False, "modetrans_text.csv missing"
    df = pd.read_csv(f)
    if len(df) != 2043:
        return False, f"{len(df)} rows, expected 2043"
    # text_norm must not be empty where text_raw was not
    raw = df["text_raw"].fillna("").astype(str)
    norm = df["text_norm"].fillna("").astype(str)
    lost = int(((raw.str.strip() != "") & (norm.str.strip() == "")).sum())
    if lost:
        return False, f"{lost} rows lost all text during normalisation"
    cs = P.processed / "charset.json"
    if not cs.exists():
        return False, "charset.json missing"
    n = json.loads(cs.read_text(encoding="utf-8"))["n_symbols"]
    if not (30 <= n <= 300):
        return False, f"charset has {n} symbols - implausible"
    return True, f"{len(df)} rows, {n} symbols, {int(df['needs_review'].sum())} need review"


def gate_hchar():
    f = P.processed / "hchar_manifest.csv"
    if not f.exists():
        return False, "hchar_manifest.csv missing"
    df = pd.read_csv(f)
    if len(df) < 500000:
        return False, f"{len(df)} rows, expected ~576k"
    return True, f"{len(df)} rows, {df['class'].nunique()} classes"


def gate_overlap():
    f = P.processed / "modetrans_hhdoc_matches.csv"
    if not f.exists():
        return True, "full overlap not run (pilot only) - skipping gate"
    df = pd.read_csv(f)
    nm = int((df["status"] == "match").sum())
    return True, f"{len(df)} patches, {nm} matched"


def gate_splits():
    f = P.processed / "splits.json"
    if not f.exists():
        return False, "splits.json missing"
    s = json.loads(f.read_text(encoding="utf-8"))
    mt = s.get("modetrans", {}).get("counts", {})
    tot = sum(mt.values())
    if tot != 2043:
        return False, f"MoDeTrans splits cover {tot} patches, expected 2043"
    if mt.get("train", 0) < tot * 0.5:
        return False, f"train split too small: {mt}"
    return True, f"modetrans={mt} hhdoc={s.get('hhdoc',{}).get('counts')}"


def gate_report():
    f = P.reports / "data_audit.md"
    t = P.reports / "DATA_DECISIONS_TODO.md"
    if not f.exists():
        return False, "data_audit.md missing"
    if not t.exists():
        return False, "DATA_DECISIONS_TODO.md missing"
    return True, f"audit {f.stat().st_size//1024} KB, todo {t.stat().st_size//1024} KB"


def gate_audit_fixes():
    import pandas as pd
    man = pd.read_csv(P.processed / "hhdoc_manifest.csv", dtype={"folder": str})
    if "flag_true_blank" not in man.columns or man["md5"].isna().any():
        return False, "step 14 columns missing or unhashed pages remain"
    g = man.groupby("md5")["keep"].agg(["size", "sum"])
    bad = int(((g["size"] > 1) & (g["sum"] != 1)).sum())
    if bad:
        return False, f"{bad} exact-duplicate groups do not keep exactly one copy"
    return True, f"kept {int(man['keep'].sum())} / {len(man)} pages"


def gate_text_decisions():
    import pandas as pd
    t = pd.read_csv(P.processed / "modetrans_text.csv", keep_default_na=False)
    if "text_train" not in t.columns:
        return False, "text_train column missing"
    if t["text_train"].str.contains(r"[0-9\[\]?\n]").any():
        return False, "text_train still contains ASCII digits, editorial marks or newlines"
    return True, f"{len(t)} rows with text_train"


def gate_folder18_labels():
    import pandas as pd
    lab = pd.read_csv(P.processed / "folder18_labels.csv")
    if len(lab) != 88 or lab["page_type"].isna().any():
        return False, "folder18_labels.csv incomplete"
    return True, lab["page_type"].value_counts().to_dict().__repr__()


# step id -> (script args, output that marks it done, gate)
STEPS = [
    ("01",  ["scripts/01_hhdoc_inventory.py"],          P.processed / "hhdoc_manifest.csv", gate_manifest),
    ("01b", ["scripts/01b_hhdoc_add_edge_density.py"],  None,                               gate_edge_density),
    ("02",  ["scripts/02_hhdoc_dedup.py"],              None,                               gate_dedup),
    ("03",  ["scripts/03_hhdoc_quality_flags.py"],      None,                               gate_quality_flags),
    ("04",  ["scripts/04_modetrans_text_audit.py"],     P.reports / "modetrans_text_audit.md", gate_text_audit),
    ("05",  ["scripts/05_modetrans_text_normalise.py"], P.processed / "modetrans_text.csv", gate_text_norm),
    ("07",  ["scripts/07_hchar_manifest.py"],           P.processed / "hchar_manifest.csv", gate_hchar),
    ("06",  ["scripts/06_modetrans_hhdoc_overlap.py"],  None,                               gate_overlap),
    ("14",  ["scripts/14_audit_fixes.py"],              P.reports / "audit_fixes.md",      gate_audit_fixes),
    ("15",  ["scripts/15_text_decisions.py"],           None,                               gate_text_decisions),
    ("16",  ["scripts/16_folder18_labels.py"],          P.processed / "folder18_labels.csv", gate_folder18_labels),
    ("08",  ["scripts/08_make_splits.py"],              P.processed / "splits.json",        gate_splits),
    ("09",  ["scripts/09_audit_report.py"],             P.reports / "data_audit.md",        gate_report),
]


def run_step(sid, args, log_path, extra=None):
    cmd = [PY] + args + (extra or [])
    print(f"    $ {' '.join(cmd)}")
    with open(log_path, "w", encoding="utf-8") as lf:
        env_py = str(CODE_ROOT / "src")
        import os
        env = dict(os.environ, PYTHONPATH=env_py, PYTHONIOENCODING="utf-8")
        proc = subprocess.run(cmd, cwd=str(CODE_ROOT), stdout=lf,
                              stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--overlap-full", action="store_true")
    args = ap.parse_args()

    order = [s[0] for s in STEPS]
    started = args.start is None
    t_all = time.time()
    results = []

    print(f"=== MODI data audit pipeline — {datetime.now():%Y-%m-%d %H:%M:%S} ===")
    for sid, sargs, out, gate in STEPS:
        if args.only and sid != args.only:
            continue
        if not args.only:
            if not started:
                if sid == args.start:
                    started = True
                else:
                    continue

        extra = ["--full"] if (sid == "06" and args.overlap_full) else None
        skip = (not args.force) and out is not None and out.exists()

        print(f"\n[{sid}] {'SKIP (output exists)' if skip else 'RUN'}")
        t0 = time.time()
        rc = 0
        if not skip:
            log_path = LOG_DIR / f"step_{sid}.log"
            rc = run_step(sid, sargs, log_path, extra)
            if rc != 0:
                print(f"    FAILED rc={rc} — see {log_path}")
                results.append((sid, "FAILED", time.time() - t0, f"rc={rc}"))
                print("\nPipeline stopped: step failed.")
                break

        ok, msg = gate()
        dt = time.time() - t0
        if ok:
            print(f"    GATE OK  ({dt:.1f}s): {msg}")
            results.append((sid, "ok", dt, msg))
        else:
            print(f"    GATE FAIL ({dt:.1f}s): {msg}")
            results.append((sid, "GATE FAIL", dt, msg))
            print("\nPipeline stopped: a sanity gate failed. This needs a human "
                  "decision — see the message above. Nothing was auto-corrected.")
            break

    print(f"\n=== summary (total {time.time()-t_all:.1f}s) ===")
    for sid, status, dt, msg in results:
        print(f"  {sid:4s} {status:10s} {dt:8.1f}s  {msg}")

    summary = LOG_DIR / "run_all_summary.json"
    summary.write_text(json.dumps(
        [{"step": s, "status": st, "seconds": round(d, 1), "detail": m}
         for s, st, d, m in results], indent=2), encoding="utf-8")
    print(f"\nwrote {summary}")


if __name__ == "__main__":
    main()
