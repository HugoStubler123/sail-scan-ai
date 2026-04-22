"""Sail-boundary shape analysis (side-quest of the stage report).

Two things live here:

* :func:`head_from_mask` — return the topmost pixel (smallest y) of the
  SAM2 sail mask as the head point. This replaces the curvature-peak
  heuristic used elsewhere.
* :func:`fit_edge_spline` + :func:`compute_max_depth` — fit a smooth
  spline along luff / leech polylines and report the maximum
  perpendicular deviation from the chord connecting the polyline
  endpoints, as a percentage of that chord length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class EdgeDepth:
    """Result of :func:`compute_max_depth`."""

    chord_start: np.ndarray     # (2,) first polyline vertex
    chord_end: np.ndarray       # (2,) last polyline vertex
    chord_length_px: float
    spline: np.ndarray          # (N, 2) fitted spline, image coords
    max_depth_px: float
    max_depth_pct: float        # max_depth_px / chord_length_px * 100
    max_depth_point: np.ndarray # (2,) point on spline that is deepest
    max_depth_foot: np.ndarray  # (2,) corresponding foot on the chord
    max_depth_position_pct: float  # chord-coordinate of max-depth foot (0 = start, 100 = end)


def resplit_luff_leech_at_head(
    sail,
    head_point: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-extract luff and leech polylines so the split happens AT ``head_point``.

    The default :func:`split_contour_luff_leech` splits at a curvature peak
    which may drift from the topmost pixel — but the user's convention is
    "head = topmost pixel, luff = tack→head, leech = head→clew".

    This function walks the contour from the nearest vertex to the user's
    head through the known tack and clew, producing polylines that meet
    exactly at ``head_point``.

    Returns:
        (luff_polyline, leech_polyline, head_on_contour)
    """
    from src.utils.geometry import _open_arc, _extract_arc_avoiding

    contour = sail.contour
    if contour is None or len(contour) < 3:
        return sail.luff_polyline, sail.leech_polyline, head_point

    # Snap the user-provided head to the nearest contour vertex
    head_d = np.linalg.norm(contour - head_point, axis=1)
    head_idx = int(np.argmin(head_d))

    tack_d = np.linalg.norm(contour - sail.tack_point, axis=1)
    tack_idx = int(np.argmin(tack_d))
    clew_d = np.linalg.norm(contour - sail.clew_point, axis=1)
    clew_idx = int(np.argmin(clew_d))

    if head_idx in (tack_idx, clew_idx):
        return sail.luff_polyline, sail.leech_polyline, head_point

    is_open = float(np.linalg.norm(contour[0] - contour[-1])) > 50.0
    try:
        if is_open:
            luff = _open_arc(contour, tack_idx, head_idx)
            leech = _open_arc(contour, head_idx, clew_idx)
        else:
            luff = _extract_arc_avoiding(contour, tack_idx, head_idx, clew_idx)
            leech = _extract_arc_avoiding(contour, head_idx, clew_idx, tack_idx)
    except Exception:
        return sail.luff_polyline, sail.leech_polyline, head_point

    if luff is None or leech is None or len(luff) < 2 or len(leech) < 2:
        return sail.luff_polyline, sail.leech_polyline, head_point

    return luff.astype(np.float64), leech.astype(np.float64), contour[head_idx].astype(np.float64)


def head_from_mask(mask: np.ndarray) -> np.ndarray:
    """Return the (x, y) pixel of the sail mask with the smallest y.

    If several columns tie for the minimum y (e.g. a perfectly flat top),
    the median x among them is returned so the result is deterministic
    and stays inside the mask.
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return np.array([mask.shape[1] / 2.0, 0.0], dtype=np.float64)
    y_min = int(ys.min())
    xs_at_top = xs[ys == y_min]
    x_med = float(np.median(xs_at_top))
    return np.array([x_med, float(y_min)], dtype=np.float64)


def fit_edge_spline(
    polyline: np.ndarray,
    n_samples: int = 120,
    smoothing: Optional[float] = None,
) -> np.ndarray:
    """Smooth-fit the polyline (B-spline) and return a dense sampling.

    Uses scipy.interpolate.splprep on the full polyline. If splprep fails
    (too few points, collinear data) we fall back to a cubic polynomial
    of x(t), y(t) from t in [0, 1].
    """
    if polyline is None or len(polyline) < 3:
        return np.zeros((0, 2), dtype=np.float32)

    # Dedup consecutive duplicates
    pl = polyline.astype(np.float64)
    deltas = np.diff(pl, axis=0)
    keep = np.concatenate([[True], np.linalg.norm(deltas, axis=1) > 1e-6])
    pl = pl[keep]
    if len(pl) < 3:
        return pl.astype(np.float32)

    try:
        from scipy.interpolate import splprep, splev  # type: ignore

        s = smoothing if smoothing is not None else max(len(pl) * 2.0, 5.0)
        tck, _ = splprep([pl[:, 0], pl[:, 1]], s=s, k=min(3, len(pl) - 1))
        t_grid = np.linspace(0.0, 1.0, n_samples)
        xs, ys = splev(t_grid, tck)
        return np.column_stack([xs, ys]).astype(np.float32)
    except Exception:
        # Polynomial fallback
        t = np.linspace(0.0, 1.0, len(pl))
        t_grid = np.linspace(0.0, 1.0, n_samples)
        deg = min(5, len(pl) - 1)
        px = np.polyfit(t, pl[:, 0], deg=deg)
        py = np.polyfit(t, pl[:, 1], deg=deg)
        xs = np.polyval(px, t_grid)
        ys = np.polyval(py, t_grid)
        return np.column_stack([xs, ys]).astype(np.float32)


def compute_max_depth(spline: np.ndarray) -> Optional[EdgeDepth]:
    """Max perpendicular distance from the spline to its endpoint chord.

    Args:
        spline: (N, 2) fitted spline points, image coordinates, ordered.

    Returns:
        ``EdgeDepth`` or ``None`` if the spline is degenerate.
    """
    if spline is None or len(spline) < 3:
        return None
    start = spline[0].astype(np.float64)
    end = spline[-1].astype(np.float64)
    chord = end - start
    chord_len = float(np.linalg.norm(chord))
    if chord_len < 1e-3:
        return None
    chord_unit = chord / chord_len

    # Perpendicular signed distances of each spline vertex from the chord
    rel = spline.astype(np.float64) - start
    proj = rel @ chord_unit                              # scalar along chord
    proj_pts = start[None, :] + np.outer(proj, chord_unit)
    perp = np.linalg.norm(spline.astype(np.float64) - proj_pts, axis=1)

    j = int(np.argmax(perp))
    return EdgeDepth(
        chord_start=start,
        chord_end=end,
        chord_length_px=chord_len,
        spline=spline.astype(np.float32),
        max_depth_px=float(perp[j]),
        max_depth_pct=float(perp[j] / chord_len * 100.0),
        max_depth_point=spline[j].astype(np.float64),
        max_depth_foot=proj_pts[j],
        max_depth_position_pct=float(np.clip(proj[j] / chord_len * 100.0, 0.0, 100.0)),
    )


def analyze_sail_edges(
    luff_polyline: np.ndarray,
    leech_polyline: np.ndarray,
    n_samples: int = 160,
) -> Tuple[Optional[EdgeDepth], Optional[EdgeDepth]]:
    """Convenience: spline-fit both polylines and return their max depths."""
    luff_spline = fit_edge_spline(luff_polyline, n_samples=n_samples)
    leech_spline = fit_edge_spline(leech_polyline, n_samples=n_samples)
    return compute_max_depth(luff_spline), compute_max_depth(leech_spline)
