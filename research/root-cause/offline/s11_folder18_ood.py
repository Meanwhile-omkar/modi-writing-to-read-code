"""Out-of-domain test on the Folder-18 book (Illustrative Modi Documents), strict letters 3, 5, 8, 11, 12, 13, 15.
Written 16 Oct 2026. CPU only. Everything it writes goes to assets/folder18_gt/ood/ (git-ignored: the page images
and the book's text must not be published).

  python s11_folder18_ood.py lines      cut the 14 handwritten pages into lines with the Stage-1 line finder
  python s11_folder18_ood.py read       read the lines with the frozen Stage-1 Readers
  python s11_folder18_ood.py score [answers.txt]
        score per letter with the team metric library against the answer key: the proofread files of
        modi.evaluation.folder18_gt (Tesseract `mar` draft, proofread for some letters), or a human transcription
        typed in the numbered format of folder18_transcription_pack/answers.txt
  python s11_folder18_ood.py overlap    how much of each letter's text already occurs in MoDeTrans transcriptions

Page handling: each page is shrunk to half size, the text block is found from the ink, and lr05_lib.segment cuts it
into lines (per-page deskew and line pitch). Bands whose ink covers < 30 % of the block width (stamps, page numbers)
are skipped. No page layout was set by hand. Lines are read in top-to-bottom order; pages of a letter are joined.
"""
import json, math, pickle, re, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]
sys.path.insert(0, str(CODE / "notebooks/lr05")); sys.path.insert(0, str(CODE / "src"))
import lr05_lib as L
from modi.utils.paths import P

OUT = CODE / "assets/folder18_gt/ood"
RUNS = CODE / "notebooks/outputs-of-notebook/05-notebook-kaggle/extracted/runs"
SCALE, MIN_WIDTH = 0.5, 0.30
FACSIMILE = {3: [15], 5: [21], 8: [41], 11: [51, 53, 55, 57], 12: [59, 61, 63, 65], 13: [67, 69], 15: [77]}
PACK_ORDER = [(3, 14), (5, 20), (8, 40), (11, 50), (11, 52), (11, 54), (11, 56), (12, 58), (12, 60), (12, 62), (12, 64),
              (13, 66), (13, 68), (15, 76)]           # numbering of folder18_transcription_pack (01-14)
_W = {}


def meta():
    """Alphabet and training texts exactly as the Stage-1 models saw them."""
    import pandas as pd
    split = json.load(open(CODE / "assets/metadata/splits.json", encoding="utf-8"))["modetrans"]["assignment"]
    t = pd.read_csv(CODE / "assets/metadata/modetrans_text.csv", keep_default_na=False, dtype=str)
    cs = json.load(open(CODE / "assets/metadata/charset.json", encoding="utf-8"))
    pq_order = [r.image_id for r in t.itertuples()]
    return cs["symbols"], [r.text_train for r in t.itertuples() if split.get(r.image_id) == "train"], pq_order


def text_box(g):
    ink = g < 128
    h, w = ink.shape
    ink[:, :int(0.03 * w)] = False; ink[:, int(0.97 * w):] = False; ink[:int(0.03 * h)] = False; ink[int(0.97 * h):] = False
    cols, rows = ink.sum(0), ink.sum(1)
    c = np.where(cols > 0.02 * cols.max())[0]; r = np.where(rows > 0.02 * rows.max())[0]
    return int(r[0]), int(r[-1]) + 1, int(c[0]), int(c[-1]) + 1


def cut_lines():
    OUT.mkdir(parents=True, exist_ok=True)
    res, stats = {}, {}
    for p in [p for ps in FACSIMILE.values() for p in ps]:
        g = np.array(Image.open(P.hhdoc_root / f"18/page{p}.jpg").convert("L"))
        g = cv2.resize(g, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        y0, y1, x0, x1 = text_box(g)
        g2, bands, pitch, conf, ang = L.segment(g[y0:y1, x0:x1])
        lines, skipped = [], 0
        for a, b in bands:
            crop, w = L.crop_line(g2, a, b, (40,))
            if crop is None:
                continue
            if w < MIN_WIDTH * g2.shape[1]:
                skipped += 1; continue
            lines.append(crop[40])
        res[p] = lines; stats[p] = dict(lines=len(lines), skipped=skipped, angle=float(ang), pitch=float(pitch))
        print(f"page {p}: {len(lines)} lines ({skipped} skipped), angle {ang:+.1f}, pitch {pitch:.0f}", flush=True)
    pickle.dump(res, open(OUT / "lines.pkl", "wb")); json.dump(stats, open(OUT / "line_stats.json", "w"), indent=1)


def _init(texts, itos):
    _W["lm"] = {n: L.CharLM(texts, n) for n in (6, 7)}; _W["itos"] = itos


def _decode(task):
    items, a, b, n = task
    return [(k, L.beam_decode(ti, tl, _W["itos"], _W["lm"][n], a, b).strip()) for k, ti, tl in items]


def read_lines():
    torch.set_num_threads(8)
    symbols, train_texts, _ = meta()
    vocab = L.make_vocab(symbols, train_texts, "char")
    pages = pickle.load(open(OUT / "lines.pkl", "rb"))
    keys = [(p, i) for p in pages for i in range(len(pages[p]))]
    lp = {}
    for tag in ("AUG", "BIG", "MIX", "B60", "FT_A0_s0"):
        ck = torch.load(RUNS / tag / "best.pt", map_location="cpu", weights_only=False)
        m = L.build_crnn(len(vocab), ck["size"]); m.load_state_dict(ck["model"]); m.eval()
        with torch.no_grad():
            for p in pages:
                x, lens = L.to_batch(pages[p], 40, round_to=L.STRIDE)
                out = m(torch.from_numpy(x)).float().log_softmax(-1).numpy()
                for i in range(len(pages[p])):
                    lp[(tag, p, i)] = out[i, :lens[i]]
    ens = {k: np.logaddexp.reduce(np.stack([lp[(t,) + k] for t in ("AUG", "BIG", "MIX", "B60")]), 0) - math.log(4) for k in keys}
    pool = get_context("spawn").Pool(8, initializer=_init, initargs=(train_texts, vocab.itos))

    def lm(src, a, b, n):
        items = [(k, *L.topk_compress(src[k], 8)) for k in keys]
        out = {}
        for r in pool.imap_unordered(_decode, [(items[s::32], a, b, n) for s in range(32)]):
            out.update(r)
        return out
    res = {"best": lm(ens, 0.3, 1.25, 7), "aug_lm": lm({k: lp[("AUG",) + k] for k in keys}, 0.45, 1.0, 6),
           "anchor_greedy": {k: vocab.decode(L.greedy_ids(lp[("FT_A0_s0",) + k])).strip() for k in keys}}
    pool.close()
    out = {name: {str(p): [r[(p, i)] for i in range(len(pages[p]))] for p in pages} for name, r in res.items()}
    json.dump(out, open(OUT / "predictions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def parse_answers(path):
    """Numbered human transcription (=== 01 ... === 14) -> {letter: text}. Two-column rows 'left | right' are read
    row by row, like the line finder reads the handwritten page."""
    pages, cur = {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*===\s*(\d+)", line)
        if m:
            cur = int(m.group(1)); pages[cur] = []
        elif cur is not None and line.strip():
            pages[cur].append(" ".join(part.strip() for part in line.split("|") if part.strip()))
    letters = {}
    for k, (n, _) in enumerate(PACK_ORDER, 1):
        if pages.get(k):
            letters.setdefault(n, []).extend(pages[k])
    return {n: " ".join(ls) for n, ls in letters.items()}


def references(answers=None):
    from modi.evaluation.folder18_gt import OUT as GT, join, to_train, is_proofread
    clean = to_train()
    refs, tier = {}, {}
    for n in FACSIMILE:
        proof = (GT / f"proofread/letter{n:02d}.txt").read_text(encoding="utf-8")
        refs[n] = clean(join(proof.splitlines())); tier[n] = "proofread by Claude" if is_proofread(proof) else "Tesseract only"
    if answers:
        for n, t in parse_answers(answers).items():
            refs[n] = clean(t); tier[n] = "human transcription"
    return refs, tier


def overlap(refs, K=20):
    """Share of each letter's K-character pieces (spaces removed) that occur in MoDeTrans TRAINING transcriptions."""
    symbols, train_texts, _ = meta()
    grams = {s[i:i + K] for s in (t.replace(" ", "") for t in train_texts) for i in range(len(s) - K + 1)}
    out = {}
    for n, r in refs.items():
        s = r.replace(" ", ""); g = [s[i:i + K] for i in range(len(s) - K + 1)]
        out[n] = 100 * sum(x in grams for x in g) / max(1, len(g))
    return out


def score(answers=None):
    from modi.evaluation.metrics import bleu, cer
    refs, tier = references(answers)
    ov = overlap(refs)
    pred = json.load(open(OUT / "predictions.json", encoding="utf-8"))
    strip = lambda s: re.sub(r"[\s.,?()\-:;'\"|!]", "", s)
    report = {}
    for name, pages in pred.items():
        hyp = {n: " ".join(" ".join(l for l in pages[str(p)] if l) for p in ps) for n, ps in FACSIMILE.items()}
        rows = {n: dict(key=tier[n], overlap_pct=round(ov[n], 1), chars=len(strip(refs[n])), CER=cer([refs[n]], [hyp[n]]),
                        CER_letters_only=cer([strip(refs[n])], [strip(hyp[n])]),
                        own_vs_other=[cer([strip(refs[n])], [strip(hyp[n])]),
                                      float(np.mean([cer([strip(refs[m])], [strip(hyp[n])]) for m in refs if m != n]))])
                for n in refs}
        # unseen text: at most 5 % of the letter's 20-character pieces occur in MoDeTrans training transcriptions
        # (shared greeting formulas); the leaking letters 3, 5, 8, 15 have 22-31 %
        clean = [n for n in refs if ov[n] <= 5.0]
        agg = lambda sel: dict(letters=sel, CER=cer([refs[n] for n in sel], [hyp[n] for n in sel]),
                               CER_letters_only=cer([strip(refs[n]) for n in sel], [strip(hyp[n]) for n in sel]),
                               char_BLEU=bleu([refs[n] for n in sel], [hyp[n] for n in sel]))
        report[name] = dict(per_letter=rows, unseen_text_letters=agg(clean), all_letters=agg(list(refs)))
        print(f"\n=== {name}")
        for n, r in rows.items():
            print(f"letter {n:2d} | key: {r['key']:22s} | text in MoDeTrans train {r['overlap_pct']:5.1f} % | {r['chars']:5d} chars | "
                  f"CER {r['CER']:5.1f} | letters-only {r['CER_letters_only']:5.1f} | own vs other {r['own_vs_other'][0]:.0f} vs {r['own_vs_other'][1]:.0f}")
        u, a = report[name]["unseen_text_letters"], report[name]["all_letters"]
        print(f"UNSEEN-TEXT letters {u['letters']}: CER {u['CER']:.1f} | letters-only {u['CER_letters_only']:.1f} | char BLEU {u['char_BLEU']:.1f}")
        print(f"all 7 letters: CER {a['CER']:.1f} | letters-only {a['CER_letters_only']:.1f}")
    json.dump(report, open(OUT / "scores.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "lines":
        cut_lines()
    elif cmd == "read":
        read_lines()
    elif cmd == "overlap":
        print(overlap(references()[0]))
    else:
        score(sys.argv[2] if len(sys.argv) > 2 else None)
