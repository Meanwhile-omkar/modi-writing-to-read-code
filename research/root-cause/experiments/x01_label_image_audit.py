"""X0.1 - does each label describe the text actually visible in its image? (hypothesis H1)

Picks the patches whose label length is most out of line with the amount of ink on the page (most and fewest
characters per 1,000 ink pixels) plus random controls, and renders each image beside its label, with the counts
needed to judge a gross mismatch (e.g. one short line of writing carrying a 250-character label).
No training. Output: figures/x01_*.png and a printed table.
"""
import io, re, textwrap, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "FYP-modi-script-code"
OUT = CODE / "research" / "root-cause" / "figures"
SEED = 0
plt.rcParams["font.family"] = ["Nirmala UI", "DejaVu Sans"]          # Devanagari-capable on Windows

pq = pd.read_parquet(ROOT / "FYP-modi-script-data/MoDeTrans-dataset/train-00000-of-00001.parquet")
lab = pd.read_csv(CODE / "assets/metadata/modetrans_text.csv", keep_default_na=False).set_index("image_id")["text_train"]
tok = re.compile(r"<FRAC>|.", re.S)
rows = []
for r in pq.itertuples():
    g = np.array(Image.open(io.BytesIO(r.image["bytes"])).convert("L"))
    ink = int((g < 128).sum())
    t = lab[r.filename]
    rows.append(dict(name=r.filename, ink=ink, chars=len(tok.findall(t)), words=len(t.split()), text=t, img=g))
d = pd.DataFrame(rows)
d["chars_per_kpx"] = d.chars / (d.ink / 1000)

rng = np.random.default_rng(SEED)
groups = {
    "high": d.nlargest(6, "chars_per_kpx"),      # many characters for little ink: is the label too long?
    "low": d.nsmallest(6, "chars_per_kpx"),      # few characters for much ink: is the label too short?
    "random": d.iloc[rng.choice(len(d), 12, replace=False)],
}
for gname, g in groups.items():
    n = len(g)
    fig, axes = plt.subplots(n, 2, figsize=(20, 2.6 * n), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, (_, r) in zip(np.atleast_2d(axes), g.iterrows()):
        ax[0].imshow(r.img, cmap="gray", vmin=0, vmax=255); ax[0].axis("off")
        ax[0].set_title(f"{r['name']}  ink {r.ink:,} px  |  {r.chars} chars, {r.words} words  |  "
                        f"{r.chars_per_kpx:.2f} chars per 1k ink px", fontsize=9, loc="left")
        ax[1].axis("off")
        ax[1].text(0, 1, "\n".join(textwrap.wrap(r.text, 62)), va="top", fontsize=10.5, wrap=True)
    fig.tight_layout()
    fig.savefig(OUT / f"x01_{gname}.png", dpi=72)
    plt.close(fig)
    print(f"\n[{gname}]")
    for _, r in g.iterrows():
        print(f"  {r['name']:>9}  ink {r.ink:>7,}  chars {r.chars:>4}  words {r.words:>3}  ratio {r.chars_per_kpx:5.2f}")
print("\nwrote", sorted(p.name for p in OUT.glob("x01_*.png")))
