"""span_lib.py - SPAN reimplemented for MODI-SYNTH.

SPAN = Simple Predict & Align Network (Coquenet, Chatelain, Paquet; ICDAR 2021, arXiv 2102.08742).
The architecture, loss and training recipe follow the paper and the authors' public code
(github.com/FactoDeepLearning/SPAN, CeCILL-C). This file is a reimplementation, not a copy. Layer
and parameter names are kept identical to theirs on purpose, so that their published weights
(CC-BY-4.0, zenodo.org/records/7244546) load with strict=True - which doubles as a check that this
network really is the same network.

What is taken from them, and where:
  encoder   6 ConvBlocks 1>32>64>128>256>512>512 with strides (1,1)(2,2)(2,2)(2,2)(2,1)(2,1), then
            4 depthwise-separable DSC blocks with residual connections.  paper section 3.1, models.py
  blocks    conv > ReLU > conv > ReLU > InstanceNorm > conv, dropout at ONE of three random places
            per forward pass.                                           paper section 3.1, models.py
  decoder   a single 5x5 convolution to (classes) channels, log-softmax.  paper section 3.2
  loss      CTC on the 2D grid flattened row by row, per sample cropped to its own valid region,
            normalised by label length.                                  paper section 3.2, trainer
  optimiser Adam, lr 1e-4, no scheduler, fp32, dropout 0.5, augmentation 0.2 each.  main_pg_ctc.py
  input     standardised with the training set's own mean and std, AFTER augmentation.
"""
import os, sys, json, math, time, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import cv2

# ------------------------------------------------------------------ geometry
# five stride-2 steps shrink the height (blocks 2-6), three shrink the width (blocks 2-4).
# a 3x3 convolution with padding 1 and stride 2 turns n into ceil(n / 2).
def reduced_shape(h, w):
    for _ in range(5):
        h = (h + 1) // 2
    for _ in range(3):
        w = (w + 1) // 2
    return h, w


# ------------------------------------------------------------------ model
class DepthSepConv2D(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1)):
        super().__init__()
        self.depth_conv = nn.Conv2d(cin, cin, (3, 3), stride=stride, padding=(1, 1), groups=cin)
        self.point_conv = nn.Conv2d(cin, cout, (1, 1))

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class MixDropout(nn.Module):
    # either ordinary dropout or whole-channel dropout, chosen at random on every call
    def __init__(self, p=0.4, p2d=0.2):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.dropout2d = nn.Dropout2d(p2d)

    def forward(self, x):
        return self.dropout(x) if random.random() < 0.5 else self.dropout2d(x)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1), dropout=0.4):
        super().__init__()
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.conv3 = nn.Conv2d(cout, cout, 3, padding=1, stride=stride)
        self.norm_layer = nn.InstanceNorm2d(cout, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout, dropout / 2)

    def forward(self, x):
        pos = random.randint(1, 3)                 # dropout goes in exactly one of three places
        x = self.activation(self.conv1(x))
        if pos == 1:
            x = self.dropout(x)
        x = self.activation(self.conv2(x))
        if pos == 2:
            x = self.dropout(x)
        x = self.norm_layer(x)
        x = self.activation(self.conv3(x))
        if pos == 3:
            x = self.dropout(x)
        return x


class DSCBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1), dropout=0.4):
        super().__init__()
        self.activation = nn.ReLU()
        self.conv1 = DepthSepConv2D(cin, cout)
        self.conv2 = DepthSepConv2D(cout, cout)
        self.conv3 = DepthSepConv2D(cout, cout, stride=stride)
        self.norm_layer = nn.InstanceNorm2d(cout, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout, dropout / 2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.activation(self.conv1(x))
        if pos == 1:
            x = self.dropout(x)
        x = self.activation(self.conv2(x))
        if pos == 2:
            x = self.dropout(x)
        x = self.norm_layer(x)
        x = self.conv3(x)                          # no ReLU after the third separable conv
        if pos == 3:
            x = self.dropout(x)
        return x


class FCNEncoder(nn.Module):
    def __init__(self, in_ch=1, dropout=0.5, width_mult=1.0):
        super().__init__()
        ch = [max(2, int(round(c * width_mult))) for c in (32, 64, 128, 256, 512, 512)]
        self.init_blocks = nn.ModuleList([
            ConvBlock(in_ch, ch[0], (1, 1), dropout),
            ConvBlock(ch[0], ch[1], (2, 2), dropout),
            ConvBlock(ch[1], ch[2], (2, 2), dropout),
            ConvBlock(ch[2], ch[3], (2, 2), dropout),
            ConvBlock(ch[3], ch[4], (2, 1), dropout),
            ConvBlock(ch[4], ch[5], (2, 1), dropout),
        ])
        self.blocks = nn.ModuleList([DSCBlock(ch[5], ch[5], (1, 1), dropout) for _ in range(4)])
        self.out_channels = ch[5]

    def forward(self, x):
        for b in self.init_blocks:
            x = b(x)
        for b in self.blocks:
            xt = b(x)
            x = x + xt if x.size() == xt.size() else xt
        return x


class Decoder(nn.Module):
    def __init__(self, cin, n_classes):
        super().__init__()
        self.end_conv = nn.Conv2d(cin, n_classes, kernel_size=(5, 5), padding=(2, 2))

    def forward(self, x):
        return F.log_softmax(self.end_conv(x), dim=1)


class SPAN(nn.Module):
    def __init__(self, n_classes, in_ch=1, dropout=0.5, width_mult=1.0):
        super().__init__()
        self.encoder = FCNEncoder(in_ch, dropout, width_mult)
        self.decoder = Decoder(self.encoder.out_channels, n_classes)

    def forward(self, x):
        return self.decoder(self.encoder(x))       # (B, classes, rows, cols) log-probabilities


def load_pretrained_encoder(model, path, log=print):
    """Initialise the encoder from the authors' IAM weights.

    Their network reads 3-channel images (a grey image copied three times). A 1-channel network
    gives the identical result if the first convolution's weights are summed over the input
    channels, so that is what is done: it is exact, not an approximation.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = None
    for key in ("encoder_state_dict", "encoder", "model_state_dict"):
        if isinstance(ck, dict) and key in ck:
            sd = ck[key]
            break
    if sd is None and isinstance(ck, dict) and all(torch.is_tensor(v) for v in ck.values()):
        sd = ck
    assert sd is not None, f"no encoder weights found; checkpoint keys: {list(ck)[:12]}"
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    w = sd["init_blocks.0.conv1.weight"]
    if w.shape[1] == 3 and model.encoder.init_blocks[0].conv1.weight.shape[1] == 1:
        sd["init_blocks.0.conv1.weight"] = w.sum(dim=1, keepdim=True)
    missing, unexpected = model.encoder.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"encoder mismatch: missing {missing[:4]} unexpected {unexpected[:4]}"
    log(f"loaded IAM-pretrained encoder from {os.path.basename(path)} "
        f"({sum(p.numel() for p in model.encoder.parameters())/1e6:.1f} M parameters, strict match)")


# ------------------------------------------------------------------ loss and decoding
def ctc_loss_spatial(logp, targets, rshapes):
    """CTC on the 2D grid. Each sample is cut to its own valid (rows, cols) region, flattened row by
    row, and its loss is divided by its own label length (SPAN's 'mean' reduction). Returns the mean
    over the batch and the per-sample values."""
    B, C = logp.shape[0], logp.shape[1]
    seqs, lens = [], []
    for i in range(B):
        rh, rw = rshapes[i]
        seqs.append(logp[i, :, :rh, :rw].reshape(C, rh * rw).t())      # (T_i, C)
        lens.append(rh * rw)
    padded = pad_sequence(seqs)                                         # (T_max, B, C)
    tl = torch.tensor([len(t) for t in targets], dtype=torch.long)
    il = torch.tensor(lens, dtype=torch.long)
    per = F.ctc_loss(padded.float(), torch.cat(targets).to(padded.device), il, tl,
                     blank=0, reduction="none", zero_infinity=True)
    return (per / tl.to(per.device).clamp(min=1)).mean(), per


def greedy_decode(logp, rshapes, decode_ids):
    ids = logp.argmax(1).cpu().numpy()
    hyps, blank_hit, blank_tot = [], 0, 0
    for i, (rh, rw) in enumerate(rshapes):
        s = ids[i, :rh, :rw].reshape(-1)
        blank_hit += int((s == 0).sum())
        blank_tot += s.size
        keep = np.ones_like(s, dtype=bool)
        keep[1:] = s[1:] != s[:-1]
        keep &= s != 0
        hyps.append(decode_ids(s[keep]))
    return hyps, blank_hit / max(1, blank_tot)


# ------------------------------------------------------------------ metrics (same as notebooks 01-03)
def compute_metrics(refs, hyps, n_boot=0, seed=0):
    import editdistance, sacrebleu
    u = lambda s: s.replace("<FRAC>", "")
    e = np.array([editdistance.eval(u(r), u(h)) for r, h in zip(refs, hyps)], float)
    L = np.array([max(1, len(u(r))) for r in refs], float)
    we = np.array([editdistance.eval(r.split(), h.split()) for r, h in zip(refs, hyps)], float)
    wl = np.array([max(1, len(r.split())) for r in refs], float)
    per = e / L
    out = dict(CER=100 * e.sum() / L.sum(), WER=100 * we.sum() / wl.sum(),
               median_CER=100 * float(np.median(per)), catastrophic_pct=100 * float((per > 0.5).mean()),
               chrF_pp=sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score,
               BLEU=sacrebleu.corpus_bleu(hyps, [refs]).score, n=len(refs))
    if n_boot:
        rg = np.random.default_rng(seed)
        idx = rg.integers(0, len(refs), (n_boot, len(refs)))
        b = 100 * e[idx].sum(1) / L[idx].sum(1)
        out["CER_CI95"] = (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
    return out


# ------------------------------------------------------------------ augmentation (SPAN: each at p=0.2)
def augment(img, rng, p=0.2):
    h, w = img.shape
    if rng.random() < p:                                    # resolution: scale by 0.75-1.0
        f = rng.uniform(0.75, 1.0)
        img = cv2.resize(img, (max(8, int(math.ceil(w * f))), max(8, int(math.ceil(h * f)))),
                         interpolation=cv2.INTER_LINEAR)
        h, w = img.shape
    if rng.random() < p:                                    # perspective, distortion scale 0-0.3
        s = rng.uniform(0.0, 0.3)
        dx, dy = int(s * (w // 2)), int(s * (h // 2))
        dst = np.float32([[rng.integers(0, dx + 1), rng.integers(0, dy + 1)],
                          [rng.integers(w - dx - 1, w), rng.integers(0, dy + 1)],
                          [rng.integers(w - dx - 1, w), rng.integers(h - dy - 1, h)],
                          [rng.integers(0, dx + 1), rng.integers(h - dy - 1, h)]])
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                                  flags=cv2.INTER_LINEAR, borderValue=255)
    elif rng.random() < p:                                  # elastic distortion along one axis
        mag, k = int(rng.integers(1, 21)), int(rng.integers(1, 4))
        mx, my = (mag, 1) if rng.integers(2) == 0 else (1, mag)
        gx = rng.uniform(-mx, mx, (k + 1, k + 1)).astype(np.float32)
        gy = rng.uniform(-my, my, (k + 1, k + 1)).astype(np.float32)
        fx = cv2.resize(gx, (w, h), interpolation=cv2.INTER_CUBIC)
        fy = cv2.resize(gy, (w, h), interpolation=cv2.INTER_CUBIC)
        mapx, mapy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        img = cv2.remap(img, mapx + fx, mapy + fy, cv2.INTER_LINEAR, borderValue=255)
    elif rng.random() < p:                                  # random projective transform, one axis
        dw, dh = (0.10 * w, 0.0) if rng.integers(2) == 0 else (0.0, 0.15 * h)
        sh = lambda d: rng.uniform(-d, d)
        dst = np.float32([[sh(dw), sh(dh)], [w - 1 - sh(dw), sh(dh)],
                          [w - 1 - sh(dw), h - 1 - sh(dh)], [sh(dw), h - 1 - sh(dh)]])
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                                  flags=cv2.INTER_LINEAR, borderValue=255)
    if rng.random() < p:                                    # dilation or erosion, kernel 1-3
        k = np.ones((int(rng.integers(1, 4)), int(rng.integers(1, 4))), np.uint8)
        img = cv2.erode(img, k) if rng.integers(2) == 0 else cv2.dilate(img, k)
    # contrast and brightness: SPAN draws the factor from 0.01-1. Measured on our patches that leaves
    # about 3% of augmented pages with under 10% of their original contrast - effectively blank, yet
    # still carrying a full label, which teaches the model to invent text. The floor is raised to
    # 0.3. This is the ONE deliberate deviation from SPAN's augmentation, and it comes from the
    # image audit (scripts/20_span_input_audit.py), not from taste.
    if rng.random() < p:                                    # contrast (blend with the mean)
        f, m = rng.uniform(0.3, 1.0), float(img.mean())
        img = np.clip(m + f * (img.astype(np.float32) - m), 0, 255).astype(np.uint8)
    if rng.random() < p:                                    # brightness
        img = np.clip(img.astype(np.float32) * rng.uniform(0.3, 1.0), 0, 255).astype(np.uint8)
    if rng.random() < p:                                    # sign flipping
        img = 255 - img
    return img


# ------------------------------------------------------------------ data
class Pages(torch.utils.data.Dataset):
    def __init__(self, data, ids, train, aug_p):
        self.d, self.ids, self.train, self.aug_p = data, list(ids), train, aug_p

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        name = self.ids[i]
        img = np.array(self.d["imgs"][self.d["idx"][name]])            # (H, W) uint8, copied out of the memmap
        if self.train and self.aug_p > 0:
            seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
            img = augment(img, np.random.default_rng(seed), self.aug_p)
        lab = self.d["labels"][name]
        rh, rw = reduced_shape(*img.shape)
        need = 2 * len(lab) + 1                                        # SPAN's 'CTC_pg' constraint
        if need > rh * rw:
            new_w = int(math.ceil(need / rh)) * 8
            if new_w > img.shape[1]:
                img = np.pad(img, ((0, 0), (0, new_w - img.shape[1])), constant_values=255)
            rh, rw = reduced_shape(*img.shape)
        x = (img.astype(np.float32) - self.d["mean"]) / self.d["std"]   # standardise AFTER augmentation
        return torch.from_numpy(x)[None], torch.as_tensor(lab, dtype=torch.long), (rh, rw), name


def collate(batch):
    xs, ys, rs, ns = zip(*batch)
    H = max(x.shape[1] for x in xs)
    W = max(x.shape[2] for x in xs)
    out = torch.zeros(len(xs), 1, H, W)             # 0 is the mean intensity once standardised: SPAN's padding value
    for i, x in enumerate(xs):
        out[i, :, :x.shape[1], :x.shape[2]] = x
    return out, list(ys), list(rs), list(ns)


def load_data(path):
    meta = json.load(open(path, encoding="utf-8"))
    d = dict(meta)
    d["imgs"] = np.load(meta["npy"], mmap_mode="r")
    d["idx"] = {n: i for i, n in enumerate(meta["names"])}
    vocab = meta["vocab"]
    d["decode_ids"] = lambda ids: "".join(vocab[i] for i in ids if i > 0)
    return d


def make_loader(data, ids, train, batch_size, aug_p, workers, pin):
    return torch.utils.data.DataLoader(
        Pages(data, ids, train, aug_p), batch_size=batch_size, shuffle=train,
        drop_last=bool(train) and len(ids) > batch_size, num_workers=workers, collate_fn=collate,
        pin_memory=pin, persistent_workers=workers > 0)


@torch.no_grad()
def evaluate(model, loader, data, device):
    model.eval()
    refs, hyps, names, bl, nb = [], [], [], 0.0, 0
    for x, ys, rs, ns in loader:
        logp = model(x.to(device))
        h, b = greedy_decode(logp, rs, data["decode_ids"])
        hyps += h
        refs += [data["text"][n] for n in ns]
        names += ns
        bl += b
        nb += 1
    return refs, hyps, names, bl / max(1, nb)


# ------------------------------------------------------------------ training
def _save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)                 # never leave a half-written checkpoint behind


def fit(cfg, data, run_dir, train_ids, val_ids, device, init="none", pretrained_path=None,
        deadline=None, stop_at_cer=None, verbose_every=1, log=print):
    os.makedirs(run_dir, exist_ok=True)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    n_classes = len(data["vocab"])
    model = SPAN(n_classes, 1, cfg["dropout"], cfg.get("width_mult", 1.0))
    opt_state, state = None, dict(epoch=0, best_cer=1e9, best_epoch=0, bad_epochs=0, sessions=0)
    last = f"{run_dir}/last.pt"
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt_state = ck["opt"]; state = ck["state"]
        state["sessions"] = state.get("sessions", 0) + 1
        log(f"RESUMED at epoch {state['epoch'] + 1} (session #{state['sessions'] + 1}); "
            f"best val CER so far {state['best_cer']:.2f} at epoch {state['best_epoch']}")
    elif init == "iam":
        load_pretrained_encoder(model, pretrained_path, log)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], amsgrad=False)
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    json.dump(dict(cfg, init=init), open(f"{run_dir}/config.json", "w"), indent=2)

    pin = str(device).startswith("cuda")
    tl = make_loader(data, train_ids, True, cfg["batch_size"], cfg["aug_p"] if cfg["use_aug"] else 0.0,
                     cfg["num_workers"], pin)
    vl = make_loader(data, val_ids, False, cfg["batch_size"], 0.0, cfg["num_workers"], pin)
    logf = f"{run_dir}/log.csv"
    COLS = ["epoch", "train_loss", "val_CER", "val_WER", "val_median_CER", "val_catastrophic_pct",
            "val_chrF_pp", "blank_slot_pct", "pred_len", "gt_len", "grad_norm", "weight_norm",
            "skipped", "sec", "elapsed_h"]
    if not os.path.exists(logf):
        csv.writer(open(logf, "w", newline="")).writerow(COLS)
    gt_len = float(np.mean([len(data["text"][n]) for n in val_ids]))
    t_start = time.time(); last_sec = 0.0

    while state["epoch"] < cfg["max_epochs"] and state["bad_epochs"] < cfg["patience"]:
        if deadline is not None and time.time() + max(60.0, 1.15 * last_sec) > deadline:
            log(f"TIME BUDGET reached at epoch {state['epoch']}; last.pt is saved. Attach this "
                f"notebook's output as an input and commit again to continue at epoch {state['epoch'] + 1}.")
            break
        t0 = time.time(); model.train(); tot = nb = skipped = 0; gmax = 0.0
        for x, ys, rs, ns in tl:
            logp = model(x.to(device, non_blocking=True))
            loss, _ = ctc_loss_spatial(logp, [y.to(device) for y in ys], rs)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); skipped += 1; continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # SPAN does not clip gradients. The limit here is so high it only catches a genuine
            # explosion, and a non-finite gradient (loss finite, gradient not) skips the step.
            g = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1000.0)))
            if not math.isfinite(g):
                opt.zero_grad(set_to_none=True); skipped += 1; continue
            gmax = max(gmax, g)
            opt.step()
            tot += loss.item(); nb += 1
        train_loss = tot / max(1, nb)
        wnorm = float(sum(p.detach().float().pow(2).sum() for p in model.parameters()).sqrt())
        if skipped > 0.05 * len(tl):
            log(f"ABORTED: {skipped}/{len(tl)} batches gave a non-finite loss (weight norm {wnorm:,.0f}).")
            break
        refs, hyps, _, blank = evaluate(model, vl, data, device)
        m = compute_metrics(refs, hyps)
        state["epoch"] += 1
        improved = m["CER"] < state["best_cer"] - cfg["min_delta"]
        if improved:
            state.update(best_cer=m["CER"], best_epoch=state["epoch"], bad_epochs=0)
            _save({"model": model.state_dict(), "cfg": cfg, "epoch": state["epoch"], "val": m},
                  f"{run_dir}/best.pt")
        else:
            state["bad_epochs"] += 1
        _save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state}, last)
        sec = time.time() - t0; last_sec = sec
        pred_len = float(np.mean([len(h) for h in hyps]))
        csv.writer(open(logf, "a", newline="")).writerow(
            [state["epoch"], round(train_loss, 4), round(m["CER"], 2), round(m["WER"], 2),
             round(m["median_CER"], 2), round(m["catastrophic_pct"], 1), round(m["chrF_pp"], 2),
             round(100 * blank, 1), round(pred_len, 1), round(gt_len, 1), round(gmax, 2),
             round(wnorm, 1), skipped, round(sec), round((time.time() - t_start) / 3600, 3)])
        if state["epoch"] % verbose_every == 0 or improved:
            left = ""
            if deadline is not None:
                left = f" | ~{max(0, (deadline - time.time()) / max(sec, 1)):.0f} epochs left this session"
            tag = "  *BEST*" if improved else f"  ({state['bad_epochs']}/{cfg['patience']} stale)"
            log(f"ep {state['epoch']:4d} | loss {train_loss:6.3f} | CER {m['CER']:6.2f} "
                f"(med {m['median_CER']:5.1f}) | WER {m['WER']:6.2f} | chrF {m['chrF_pp']:5.1f} | "
                f"best {state['best_cer']:6.2f}@{state['best_epoch']:<4d} | blank {100 * blank:4.1f}% | "
                f"len {pred_len:5.1f}/{gt_len:.0f} | g {gmax:5.1f} w {wnorm:6.0f} | {sec:4.0f}s{left}{tag}")
            log(f"   GT  : {refs[0][:85]}")
            log(f"   PRED: {hyps[0][:85]}")
        if stop_at_cer is not None and m["CER"] <= stop_at_cer:
            log(f"reached CER {m['CER']:.2f}, stopping"); break
    log(f"done: {state['epoch']} epochs, best val CER {state['best_cer']:.2f} at epoch {state['best_epoch']}")
    return model, state


# ------------------------------------------------------------------ data preparation
def build_cache(pil_images, n, H, W, path):
    """Scale every patch to H x W once and keep them all in one uint8 array on disk. Nothing is cut,
    split, reordered or discarded: this is the entire image pipeline."""
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(n, H, W))
    for i, pil in enumerate(pil_images):
        arr[i] = cv2.resize(np.array(pil.convert("L")), (W, H), interpolation=cv2.INTER_AREA)
    arr.flush()
    return arr


def train_stats(arr, rows):
    """Mean and standard deviation of the TRAINING images only, as SPAN standardises its input."""
    tot = tot2 = 0.0
    for r in rows:
        a = np.asarray(arr[r], dtype=np.float64)
        tot += a.sum(); tot2 += (a * a).sum()
    n = len(rows) * arr.shape[1] * arr.shape[2]
    mean = tot / n
    return float(mean), float(math.sqrt(max(tot2 / n - mean * mean, 1e-12)))


# ------------------------------------------------------------------ one process per GPU
def launch_jobs(jobs, script, data_json, out, ckpt, deadline, extra_args=(), poll=15.0, log=print,
                stop_when=None):
    """Start one training process per job, each pinned to its own GPU, and stream their output.

    Separate processes (not threads, not DataParallel) so that two different models can train at the
    same time on the two T4s a Kaggle session provides. Each writes to its own run directory, so a
    crash in one cannot touch the other. If stop_when() becomes true, the remaining processes are
    ended (the smoke test uses this: once one model has proved the pipeline, waiting for the slower
    one only leaves a GPU idle).
    """
    import subprocess
    os.makedirs(f"{out}/runs", exist_ok=True)
    procs = []
    for j in jobs:
        path = f"{out}/runs/{j['tag']}.stdout"
        fh = open(path, "w", encoding="utf-8")
        cmd = [sys.executable, "-u", script, "--data", data_json, "--out", out, "--name", j["name"],
               "--init", j["init"], "--deadline", str(deadline)]
        cmd += (["--ckpt", ckpt] if ckpt else []) + (["--smoke"] if j.get("smoke") else []) + list(extra_args)
        env = dict(os.environ)
        if j.get("gpu") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(j["gpu"])
        procs.append(dict(job=j, p=subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env),
                          path=path, fh=fh, pos=0))

    def drain(P):
        with open(P["path"], "r", encoding="utf-8", errors="replace") as f:
            f.seek(P["pos"]); chunk = f.read(); P["pos"] = f.tell()
        for line in chunk.splitlines():
            log(f"[{P['job']['tag']}] {line}")

    while any(P["p"].poll() is None for P in procs):
        time.sleep(poll)
        for P in procs:
            drain(P)
        if stop_when is not None and stop_when():
            log("stop condition met - ending the processes that are still running")
            for P in procs:
                if P["p"].poll() is None:
                    P["p"].terminate()
            time.sleep(5)
            for P in procs:
                if P["p"].poll() is None:
                    P["p"].kill()
            break
        if time.time() > deadline + 900:                    # children obey the deadline; this is the backstop
            for P in procs:
                if P["p"].poll() is None:
                    P["p"].kill()
    for P in procs:
        drain(P); P["fh"].close()
    return {P["job"]["tag"]: P["p"].returncode for P in procs}
