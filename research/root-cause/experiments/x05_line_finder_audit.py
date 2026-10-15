"""X0.5 - did notebook 03's line finder cut real text lines apart? (hypothesis H21)

Runs the exact line finder 03 shipped with (deskew +-8 deg, core threshold 0.08, valley split, growth cap 0.35;
code in scripts/19_image_pipeline_audit.py) on the same 24 patches as X0.1, draws the detected band boundaries,
and shows the strip 03 actually fed its model. Band boundaries that run THROUGH a line of writing mean that line
was split, and its two halves were laid side by side in the strip.
Output: figures/x05_bands_{random,low,high}.png and a printed table of bands per patch.
"""
import io, re, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "FYP-modi-script-code"
OUT = CODE / "research" / "root-cause" / "figures"
plt.rcParams["font.family"] = ["Nirmala UI", "DejaVu Sans"]

# the exact functions notebook 03 used (same thresholds), taken from the audit script without running its main()
src = (CODE / "scripts" / "19_image_pipeline_audit.py").read_text(encoding="utf-8").split("def main()")[0]
ns = {"__file__": str(CODE / "scripts" / "19_image_pipeline_audit.py")}
exec(compile(src, "audit19", "exec"), ns)
find_lines_new, line_crops, assemble = ns["find_lines_new"], ns["line_crops"], ns["assemble"]
print("03 line-finder settings: LINE_THR", ns["LINE_THR"], "GROW_CAP", ns["GROW_CAP"], "SKEW_RANGE", ns["SKEW_RANGE"])
MED_LINE = 56.0                                     # measured by 03 on its first 300 patches

pq = pd.read_parquet(ROOT / "FYP-modi-script-data/MoDeTrans-dataset/train-00000-of-00001.parquet")
row = {r.filename: r for r in pq.itertuples()}
groups = {   # the same patches X0.1 audited
    "random": ["715.jpg", "1028.jpg", "1934.jpg", "324.jpg", "1562.jpg", "132.jpg",
               "1493.jpg", "1073.jpg", "353.jpg", "653.jpg", "838.jpg", "1136.jpg"],
    "low": ["869.jpg", "1367.jpg", "868.jpg", "1375.jpg", "858.jpg", "701.jpg"],
    "high": ["1116.jpg", "1090.jpg", "1193.jpg", "1087.jpg", "20.jpg", "1142.jpg"],
}
for gname, names in groups.items():
    fig, axes = plt.subplots(len(names), 2, figsize=(22, 2.7 * len(names)), gridspec_kw={"width_ratios": [1, 1.6]})
    print(f"\n[{gname}]")
    for ax, n in zip(axes, names):
        g = np.array(Image.open(io.BytesIO(row[n].image["bytes"])).convert("L"))
        g2, lines, ang = find_lines_new(g, MED_LINE)
        raw, crops = line_crops(g2, lines)
        strip, capped = assemble(crops)
        heights = [b - a for a, b in lines]
        ax[0].imshow(g2, cmap="gray", vmin=0, vmax=255); ax[0].axis("off")
        for k, (a, b) in enumerate(lines):
            col = "lime" if k % 2 == 0 else "magenta"
            ax[0].axhspan(a, b, color=col, alpha=0.12)
            ax[0].axhline(a, color=col, lw=1.2); ax[0].axhline(b - 1, color=col, lw=1.2, ls=":")
        ax[0].set_title(f"{n}: {len(lines)} bands detected, heights {heights}, deskew {ang:+.1f} deg", fontsize=8, loc="left")
        ax[1].imshow(strip[:, :2600], cmap="gray", vmin=0, vmax=255, aspect="auto"); ax[1].axis("off")
        ax[1].set_title(f"the strip 03 fed its model (first 2600 of {strip.shape[1]} px): bands laid end to end", fontsize=8, loc="left")
        print(f"  {n:>9}  bands {len(lines)}  heights {heights}")
    fig.tight_layout(); fig.savefig(OUT / f"x05_bands_{gname}.png", dpi=70); plt.close(fig)
print("\nwrote", sorted(p.name for p in OUT.glob("x05_*.png")))
