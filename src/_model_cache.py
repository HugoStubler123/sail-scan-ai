"""Global YOLO model cache + inference-mode guard.

Two wins this buys us:

  1. Caching: ``YOLO(path)`` re-reads the .pt file from disk on every
     call. ``get_yolo(path)`` memoises by absolute path.
  2. Inference mode: all downstream calls on the returned model run
     under ``torch.inference_mode`` (via a small wrapper) which skips
     autograd storage — a measurable CPU RAM saving on Streamlit Cloud.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, Optional

_CACHE: Dict[str, Any] = {}


class _InferenceYOLO:
    """Thin wrapper: delegates to the underlying YOLO object but forces
    inference mode on ``__call__`` and ``predict`` to drop autograd
    tensors that Ultralytics otherwise keeps around.
    """

    def __init__(self, model):
        self._m = model

    def __call__(self, *args, **kwargs):
        try:
            import torch
            with torch.inference_mode():
                return self._m(*args, **kwargs)
        except Exception:
            return self._m(*args, **kwargs)

    def predict(self, *args, **kwargs):
        try:
            import torch
            with torch.inference_mode():
                return self._m.predict(*args, **kwargs)
        except Exception:
            return self._m.predict(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._m, name)


def get_yolo(path: str) -> Optional[Any]:
    """Return a YOLO instance for ``path``, loading once and caching."""
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
    wrapped = _InferenceYOLO(m)
    _CACHE[p] = wrapped
    return wrapped


def clear_cache() -> None:
    _CACHE.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def trim_memory() -> None:
    """Light cleanup between photos: Python GC + torch empty_cache. Keeps
    the YOLO cache warm (we WANT those loaded) but releases activation
    tensors, numpy temporaries and matplotlib figures.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass
