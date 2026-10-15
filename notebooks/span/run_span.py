"""run_span.py - train ONE SPAN variant in its own process, on the GPU it is pinned to.

The notebook starts one of these per GPU, so a from-scratch model and an IAM-initialised model can
train side by side in the same session and be compared on equal terms.
"""
import argparse, json, os, sys, csv

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--init", default="none", choices=["none", "iam"])
ap.add_argument("--ckpt", default="")
ap.add_argument("--gpu", default=None)
ap.add_argument("--device", default="cuda")
ap.add_argument("--deadline", type=float, default=None)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--smoke-n", type=int, default=20)
ap.add_argument("--width-mult", type=float, default=1.0)      # 1.0 is the real network; smaller is for plumbing tests only
ap.add_argument("--max-epochs", type=int, default=None)
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--limit-train", type=int, default=None)      # plumbing tests only
ap.add_argument("--limit-val", type=int, default=None)
args = ap.parse_args()
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu              # must be set before torch is imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import span_lib as L

data = L.load_data(args.data)
cfg = dict(seed=42, batch_size=4, lr=1e-4, dropout=0.5, aug_p=0.2, use_aug=True, max_epochs=5000,
           patience=500, min_delta=0.05, num_workers=args.workers, grad_clip=1000.0,
           width_mult=args.width_mult)
if args.smoke:
    # memorise a handful of patches. lr 1e-3 and dropout 0.1 are deliberate: a memorisation test
    # wants updates, not regularisation (the first Kaggle smoke test inherited the main learning
    # rate by accident, took 140 epochs to leave the prior, and failed a healthy pipeline).
    ids = data["splits"]["train"][:args.smoke_n]
    cfg.update(lr=1e-3, batch_size=2, dropout=0.1, use_aug=False, max_epochs=500, patience=500)
    train_ids, val_ids, stop_at = ids, ids, 5.0
    run_dir = f"{args.out}/runs/smoke_{args.name}"
else:
    train_ids = data["splits"]["train"][:args.limit_train]
    val_ids = data["splits"]["val"][:args.limit_val or 150]
    stop_at = None
    run_dir = f"{args.out}/runs/{args.name}"
if args.max_epochs:
    cfg["max_epochs"] = args.max_epochs

print(f"init={args.init} device={args.device} train={len(train_ids)} val={len(val_ids)} "
      f"batch={cfg['batch_size']} lr={cfg['lr']} dropout={cfg['dropout']} aug={cfg['use_aug']}", flush=True)
model, state = L.fit(cfg, data, run_dir, train_ids, val_ids, args.device, init=args.init,
                     pretrained_path=args.ckpt or None, deadline=args.deadline, stop_at_cer=stop_at,
                     log=lambda s: print(s, flush=True))

if args.smoke:
    rows = list(csv.DictReader(open(f"{run_dir}/log.csv")))
    best = min(rows, key=lambda r: float(r["val_CER"]))
    res = dict(variant=args.name, epochs=len(rows), best_epoch=int(best["epoch"]),
               best_CER=float(best["val_CER"]), loss_at_best=float(best["train_loss"]),
               blank_at_best=float(best["blank_slot_pct"]))
    res["passed"] = res["best_CER"] <= 10.0 and res["loss_at_best"] <= 2.5 and res["blank_at_best"] <= 90.0
    json.dump(res, open(f"{run_dir}/result.json", "w"), indent=2)
    print(f"SMOKE RESULT {json.dumps(res)}", flush=True)
