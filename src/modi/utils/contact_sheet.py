"""Build contact sheets (grids of small thumbnails) for human review.

Thumbnails are <= 256 px long side (disk rule 4). Never writes full-res copies.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image, ImageDraw


def make_contact_sheet(
    items: Iterable[Tuple[Path, str]],
    out_path: Path,
    thumb: int = 200,
    cols: int = 6,
    pad: int = 6,
    label_h: int = 16,
    max_items: int = 300,
) -> int:
    """items: iterable of (image_path, caption). Returns number of cells drawn."""
    items = list(items)[:max_items]
    if not items:
        return 0
    n = len(items)
    rows = math.ceil(n / cols)
    cell_w = thumb + pad
    cell_h = thumb + label_h + pad
    W = cols * cell_w + pad
    H = rows * cell_h + pad
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (path, cap) in enumerate(items):
        r, c = divmod(idx, cols)
        x0 = pad + c * cell_w
        y0 = pad + r * cell_h
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb, thumb), Image.BILINEAR)
                ox = x0 + (thumb - im.width) // 2
                oy = y0 + (thumb - im.height) // 2
                sheet.paste(im, (ox, oy))
        except Exception:
            draw.rectangle([x0, y0, x0 + thumb, y0 + thumb], outline="red")
            draw.text((x0 + 4, y0 + 4), "ERR", fill="red")
        draw.text((x0 + 2, y0 + thumb + 2), cap[:40], fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, "JPEG", quality=70)
    return n
