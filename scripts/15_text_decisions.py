"""15 - apply the open text decisions -> modetrans_text.csv (text_train) + charset.json

Run after 05. text_raw and text_norm stay unchanged; the recognizer trains on text_train.

Decisions (taken at the data review, Oct 2026):
1. ASCII digit '6' -> Devanagari '६'. Every occurrence sits inside Devanagari numerals
   (e.g. '८6२', '७6१') or in a numeral slot, so it is a keyboard slip for '६'.
   Any other ASCII digit maps to its Devanagari counterpart by the same rule.
2. Editorial markup is not written on the manuscript, so the model cannot see it:
   '?' is dropped (the scribe never wrote it; checked on 494.jpg); ' [x]' becomes 'x' joined
   to the preceding word, because the editor is completing that word
   (744.jpg: 'हर जिन [स] भरून' -> 'हर जिनस भरून'). text_raw/text_norm keep the marks.
3. Internal newlines -> single space (training is at patch level, not line level).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modi.utils import console  # noqa: F401,E402
from modi.utils.paths import P  # noqa: E402

ASCII_TO_DEVA = str.maketrans("0123456789", "०१२३४५६७८९")
FRAC = "<FRAC>"


def to_train(s: str) -> str:
    s = s.translate(ASCII_TO_DEVA)
    s = s.replace("?", "")
    s = re.sub(r"\s*\[([^\]]*)\]", r"\1", s)  # editor completes the word: "जिन [स]" -> "जिनस"
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    path = P.processed / "modetrans_text.csv"
    t = pd.read_csv(path, keep_default_na=False)
    t["text_train"] = t["text_norm"].map(to_train)
    changed = int((t["text_train"] != t["text_norm"]).sum())
    t.to_csv(path, index=False, encoding="utf-8")

    chars = sorted(set("".join(t["text_train"].str.replace(FRAC, "", regex=False))))
    cs_path = P.processed / "charset.json"
    cs = json.loads(cs_path.read_text(encoding="utf-8"))
    removed = sorted(set(cs["symbols"]) - set(chars))
    cs.update({"symbols": chars, "n_symbols": len(chars), "built_from": "text_train (step 15)"})
    cs["notes"]["newline"] = "newlines are converted to spaces in text_train (step 15)"
    cs["notes"]["step15_removed"] = removed
    cs_path.write_text(json.dumps(cs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"text_train written; rows changed vs text_norm: {changed}")
    print(f"charset: {len(chars)} symbols (+{len(cs['special_tokens'])} special); removed {removed}")
    assert not t["text_train"].str.contains(r"[0-9\[\]?\n]").any()


if __name__ == "__main__":
    main()
