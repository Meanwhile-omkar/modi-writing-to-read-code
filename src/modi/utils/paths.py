"""Central path loader. Every script reads paths from configs/paths.yaml via this module.

Usage:
    from modi.utils.paths import P
    P.hhdoc_root            # pathlib.Path to raw HHDoc root
    P.processed             # processed artifacts dir (created on access)
    P.reports               # outputs/reports dir (created)
    P.figures               # outputs/figures dir (created)
"""
from __future__ import annotations

import functools
from pathlib import Path

import yaml

# This file lives at: FYP-modi-script-code/src/modi/utils/paths.py
# Code root (where configs/ lives) is 3 parents up.
CODE_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = CODE_ROOT / "configs" / "paths.yaml"


def _resolve(p: str) -> Path:
    """Resolve a path from paths.yaml relative to the code root."""
    pp = Path(p)
    if not pp.is_absolute():
        pp = (CODE_ROOT / pp).resolve()
    return pp


class _Paths:
    def __init__(self) -> None:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            self._cfg = yaml.safe_load(fh)

    # ---- raw (read-only) ----
    @property
    def modetrans_parquet(self) -> Path:
        return _resolve(self._cfg["raw"]["modetrans_parquet"])

    @property
    def modetrans_images(self) -> Path:
        return _resolve(self._cfg["raw"]["modetrans_images"])

    @property
    def modetrans_meta(self) -> Path:
        return _resolve(self._cfg["raw"]["modetrans_meta"])

    @property
    def hchar_root(self) -> Path:
        return _resolve(self._cfg["raw"]["hchar_root"])

    @property
    def hhdoc_root(self) -> Path:
        return _resolve(self._cfg["raw"]["hhdoc_root"])

    # ---- derived (writable) ----
    @property
    def data_root(self) -> Path:
        return _resolve(self._cfg["data_root"])

    @property
    def processed(self) -> Path:
        d = _resolve(self._cfg["processed"])
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def outputs(self) -> Path:
        d = _resolve(self._cfg["outputs"])
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def reports(self) -> Path:
        d = self.outputs / "reports"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def figures(self) -> Path:
        d = self.outputs / "figures"
        d.mkdir(parents=True, exist_ok=True)
        return d


@functools.lru_cache(maxsize=1)
def get_paths() -> _Paths:
    return _Paths()


P = get_paths()
