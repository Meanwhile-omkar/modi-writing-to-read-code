"""Force UTF-8 stdout/stderr so Devanagari/Modi text and symbols (sigma, ZWNJ, etc.)
print correctly on Windows consoles that default to cp1252. Import for side effect:

    from modi.utils import console  # noqa: F401
"""
from __future__ import annotations

import sys

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
