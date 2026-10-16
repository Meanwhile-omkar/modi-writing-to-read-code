"""Shared line-reader model and helpers for the root-cause experiments.

The CRNN here is architecturally identical to the one in x2_train_lines.py (same layers, same names), so
checkpoints saved by x2a/x2b/x4a/x4b load into it with strict=True.
"""
import json, re
from pathlib import Path
import torch, torch.nn as nn

CODE = Path(__file__).resolve().parents[3]
cs = json.load(open(CODE / "assets/metadata/charset.json", encoding="utf-8"))
VOCAB = ["<blank>"] + cs["symbols"] + ["<FRAC>", "<unk>"]
STOI = {c: i for i, c in enumerate(VOCAB)}
TOK = re.compile(r"<FRAC>|.", re.S)
STRIDE = 2


def enc(s):
    return [STOI.get(t, STOI["<unk>"]) for t in TOK.findall(s)]


def dec(ids):
    return "".join(VOCAB[i] for i in ids if i > 0)


class CRNN(nn.Module):
    def __init__(self, n=len(VOCAB), c=(32, 64, 128, 128, 128), hid=128):
        super().__init__()
        L, cin = [], 1
        for i, co in enumerate(c):
            L += [nn.Conv2d(cin, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True)]
            if i == 0: L.append(nn.MaxPool2d(2))              # 40 -> 20, width / 2
            if i in (1, 2): L.append(nn.MaxPool2d((2, 1)))    # 20 -> 10 -> 5
            cin = co
        self.cnn = nn.Sequential(*L)
        self.rnn = nn.LSTM(cin, hid, 2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(2 * hid, n)

    def forward(self, x):
        f = self.cnn(x).mean(2).permute(0, 2, 1)                 # (B, W/2, C)
        return self.fc(self.rnn(f)[0])                            # (B, T, classes)


def load(path):
    m = CRNN()
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    return m.eval()


def to_tensor(img):
    import numpy as np
    x = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None]
    return (x - 0.5) / 0.5
