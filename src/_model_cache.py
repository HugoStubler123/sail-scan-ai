"""Global YOLO model cache.

Ultralytics ``YOLO(path)`` re-reads the ``.pt`` file from disk every
call. On a CPU-only Streamlit Cloud box with many per-bbox inferences
that alone burns 30-60 seconds per analysis.

``get_yolo(path)`` memoises the loaded model keyed by absolute path, so
the first call loads it and every subsequent call is O(1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

_CACHE: Dict[str, Any] = {}


def get_yolo(path: str) -> Optional[Any]:
    """Return a YOLO instance for ``path``, loading once and caching.

    Returns ``None`` if the file doesn't exist or ultralytics isn't
    importable — callers should treat that as "skip this step".
    """
    p = str(Path(path).resolve())
    if p in _CACHE:
        return _CACHE[p]
    if not Path(p).exists():
        return None
    try:
        from ultralytics import YOLO
    except Exception:
        return None
    try:
        m = YOLO(p)
    except Exception:
        return None
    _CACHE[p] = m
    return m


def clear_cache() -> None:
    _CACHE.clear()
