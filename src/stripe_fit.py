"""Consensus-spline fit for sail stripes (production module).

Computes 5 candidate splines (cubic / Bernstein-4-param / smoothing spline /
polynomial-3 / NACA), takes the chord-space median perpendicular offset d(t)
across methods, Gaussian-smooths it (sigma ≈ 1.5 % chord), then reprojects to
image coordinates.

Public API
----------
fit_consensus_spline(detection_points, luff_ep, leech_ep, ...)
    -> Optional[np.ndarray]   # (N, 2) in image coords

The logic is extracted from ``build_spline_fit_diag.py::_render_panel`` so
both the diagnostic and the production pipeline share the same code path.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.ndimage import gaussian_filter1d

from src.physics import constrained_bspline_fit
from src.flexible_fit import fit_naca_style

logger = logging.getLogger(__name__)


# ── candidate spline helpers ──────────────────────────────────────────────────

def _fit_cubic_spline(
    pts_sorted: np.ndarray,
    luff_ep: np.ndarray,
    leech_ep: np.ndarray,
) -> Optional[np.ndarray]:
    """Arc-length parameterised cubic spline."""
    all_pts = np.vstack([luff_ep, pts_sorted, leech_ep]).astype(np.float64)
    diffs = np.diff(all_pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg_lens)])
    if t[-1] < 1e-6:
        return None
    t_norm = t / t[-1]
    _, unique_idx = np.unique(t_norm, return_index=True)
    if len(unique_idx) < 2:
        return None
    t_u = t_norm[unique_idx]
    pts_u = all_pts[unique_idx]
    try:
        cs_x = CubicSpline(t_u, pts_u[:, 0])
        cs_y = CubicSpline(t_u, pts_u[:, 1])
    except Exception:
        return None
    t_dense = np.linspace(0.0, 1.0, 200)
    return np.column_stack([cs_x(t_dense), cs_y(t_dense)])


def _fit_bernstein_4param(
    pts_sorted: np.ndarray,
    luff_ep: np.ndarray,
    leech_ep: np.ndarray,
    keypoint_confidences: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Physics 4-param Bernstein camber model (src/physics.py)."""
    try:
        _, _, spline_pts = constrained_bspline_fit(
            pts_sorted,
            luff_ep.astype(np.float64),
            leech_ep.astype(np.float64),
            keypoint_confidences=keypoint_confidences,
        )
        return spline_pts.astype(np.float64)
    except Exception as exc:
        logger.debug("Bernstein fit failed: %s", exc)
        return None


def _fit_smoothing_spline(
    pts_sorted: np.ndarray,
    luff_ep: np.ndarray,
    leech_ep: np.ndarray,
    keypoint_confidences: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Chord-space UnivariateSpline with pinned endpoints (weight=50)."""
    chord_vec = leech_ep.astype(np.float64) - luff_ep.astype(np.float64)
    chord_len = float(np.linalg.norm(chord_vec))
    if chord_len < 1e-6:
        return None
    chord_unit = chord_vec / chord_len
    normal_unit = np.array([-chord_unit[1], chord_unit[0]])

    pts = pts_sorted.astype(np.float64)
    rel = pts - luff_ep
    t_pts = rel @ chord_unit / chord_len
    d_pts = rel @ normal_unit

    valid = (t_pts > 0.005) & (t_pts < 0.995)
    t_int = t_pts[valid]
    d_int = d_pts[valid]

    if keypoint_confidences is not None and len(keypoint_confidences) == len(pts_sorted):
        kp_w = np.clip(np.asarray(keypoint_confidences, dtype=np.float64)[valid], 0.2, 1.0)
    else:
        kp_w = np.ones(int(valid.sum()))

    t_all = np.concatenate([[0.0], t_int, [1.0]])
    d_all = np.concatenate([[0.0], d_int, [0.0]])
    w_all = np.concatenate([[50.0], kp_w, [50.0]])

    # Deduplicate strictly increasing
    keep = [0]
    for i in range(1, len(t_all)):
        if t_all[i] - t_all[keep[-1]] > 5e-4:
            keep.append(i)
    if len(keep) < 4:
        return None
    t_all = t_all[keep]
    d_all = d_all[keep]
    w_all = w_all[keep]

    try:
        k = min(3, len(t_all) - 1)
        s_val = 0.5 * chord_len * len(t_all)
        spline_fn = UnivariateSpline(t_all, d_all, w=w_all, s=s_val, k=k)
    except Exception:
        return None

    t_fine = np.linspace(0.0, 1.0, 200)
    d_fine = spline_fn(t_fine)
    d_fine = np.clip(d_fine, -0.25 * chord_len, 0.25 * chord_len)

    spline_xy = (
        np.outer(1.0 - t_fine, luff_ep)
        + np.outer(t_fine, leech_ep)
        + np.outer(d_fine, normal_unit)
    )
    return spline_xy.astype(np.float64)


def _fit_polynomial(
    pts_sorted: np.ndarray,
    luff_ep: np.ndarray,
    leech_ep: np.ndarray,
    keypoint_confidences: Optional[np.ndarray],
    degree: int = 3,
) -> Optional[np.ndarray]:
    """Weighted polynomial d(t) in chord coords, endpoints pinned."""
    chord_vec = leech_ep.astype(np.float64) - luff_ep.astype(np.float64)
    chord_len = float(np.linalg.norm(chord_vec))
    if chord_len < 1e-6:
        return None
    chord_unit = chord_vec / chord_len
    normal_unit = np.array([-chord_unit[1], chord_unit[0]])

    pts = pts_sorted.astype(np.float64)
    rel = pts - luff_ep
    t_pts = rel @ chord_unit / chord_len
    d_pts = rel @ normal_unit

    if keypoint_confidences is not None and len(keypoint_confidences) == len(pts_sorted):
        kp_w = np.clip(np.asarray(keypoint_confidences, dtype=np.float64), 0.2, 1.0)
    else:
        kp_w = np.ones(len(pts_sorted))

    PIN_W = 1000.0
    t_all = np.concatenate([[0.0], t_pts, [1.0]])
    d_all = np.concatenate([[0.0], d_pts, [0.0]])
    w_all = np.concatenate([[PIN_W], kp_w, [PIN_W]])

    A = np.column_stack([t_all ** k for k in range(degree + 1)])
    W = np.sqrt(w_all)
    try:
        coeffs, *_ = np.linalg.lstsq(A * W[:, None], d_all * W, rcond=None)
    except Exception:
        return None

    t_fine = np.linspace(0.0, 1.0, 200)
    A_fine = np.column_stack([t_fine ** k for k in range(degree + 1)])
    d_fine = A_fine @ coeffs
    d_fine = np.clip(d_fine, -0.25 * chord_len, 0.25 * chord_len)

    spline_xy = (
        np.outer(1.0 - t_fine, luff_ep)
        + np.outer(t_fine, leech_ep)
        + np.outer(d_fine, normal_unit)
    )
    return spline_xy.astype(np.float64)


def _fit_naca(
    pts_sorted: np.ndarray,
    luff_ep: np.ndarray,
    leech_ep: np.ndarray,
    keypoint_confidences: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """NACA-style A·t^alpha·(1-t)^beta single-peak camber."""
    try:
        _, _, spline_pts = fit_naca_style(
            pts_sorted,
            luff_ep.astype(np.float64),
            leech_ep.astype(np.float64),
            keypoint_confidences=keypoint_confidences,
        )
        return spline_pts.astype(np.float64)
    except Exception as exc:
        logger.debug("NACA fit failed: %s", exc)
        return None


# ── public API ────────────────────────────────────────────────────────────────

def fit_consensus_spline(
    detection_points: np.ndarray,
    luff_ep: np.ndarray,
    leech_ep: np.ndarray,
    keypoint_confidences: Optional[np.ndarray] = None,
    smoothing_sigma_frac: float = 0.015,
    n_samples: int = 200,
) -> Optional[np.ndarray]:
    """Compute the smoothed-consensus spline (production path).

    Steps:
    1. Sort detection points along the chord axis.
    2. Compute 5 candidate splines (cubic / Bernstein / smoothing-spline /
       polynomial-3 / NACA), all anchored at ``luff_ep`` and ``leech_ep``.
    3. Resample each to a ``n_samples``-point chord grid t ∈ [0, 1].
    4. Project each sample to perpendicular offset d(t) in chord coords.
    5. Take the pointwise median of d(t) across all valid methods.
    6. Gaussian-smooth d(t) with sigma = smoothing_sigma_frac * n_samples.
    7. Reproject to image (x, y) coordinates.
    8. Pin the first and last samples exactly to luff_ep / leech_ep.

    Parameters
    ----------
    detection_points:
        (N, 2) array of detected stripe sample positions in image coords.
    luff_ep, leech_ep:
        (2,) endpoints (already fused / snapped to the SAM polylines).
    keypoint_confidences:
        Optional (N,) per-point confidence weights (same order as
        ``detection_points``).  Passed to the Bernstein, smoothing-spline,
        polynomial, and NACA fits.
    smoothing_sigma_frac:
        Gaussian sigma as a fraction of ``n_samples``.  Default 0.015
        → sigma = 3 samples on a 200-sample grid (≈ 1.5 % chord).
    n_samples:
        Number of output samples.

    Returns
    -------
    np.ndarray of shape (n_samples, 2) or None if fewer than 2 candidate
    splines could be computed.
    """
    pts = np.asarray(detection_points, dtype=np.float64)
    if pts is None or len(pts) < 3:
        return None

    luff = np.asarray(luff_ep, dtype=np.float64)
    leech = np.asarray(leech_ep, dtype=np.float64)

    chord_vec = leech - luff
    chord_len = float(np.linalg.norm(chord_vec))
    if chord_len < 1e-6:
        return None

    # Sort detection points along chord axis
    chord_unit = chord_vec / chord_len
    t_pts = (pts - luff) @ chord_unit / chord_len
    order = np.argsort(t_pts)
    pts_sorted = pts[order]

    kp_confs: Optional[np.ndarray] = None
    if keypoint_confidences is not None:
        kp_arr = np.asarray(keypoint_confidences, dtype=np.float64)
        if len(kp_arr) == len(pts):
            kp_confs = kp_arr[order]

    # 2. Compute 5 candidate splines
    sp_cubic  = _fit_cubic_spline(pts_sorted, luff, leech)
    sp_bern   = _fit_bernstein_4param(pts_sorted, luff, leech, kp_confs)
    sp_smooth = _fit_smoothing_spline(pts_sorted, luff, leech, kp_confs)
    sp_poly   = _fit_polynomial(pts_sorted, luff, leech, kp_confs, degree=3)
    sp_naca   = _fit_naca(pts_sorted, luff, leech, kp_confs)

    valid = [sp for sp in (sp_cubic, sp_bern, sp_smooth, sp_poly, sp_naca)
             if sp is not None and len(sp) >= 2]

    if len(valid) < 2:
        # Fall back to whichever single method succeeded
        if len(valid) == 1:
            return valid[0].astype(np.float64)
        return None

    # 3-5. Chord-space median of d(t)
    normal_unit = np.array([-chord_unit[1], chord_unit[0]], dtype=np.float64)
    t_grid = np.linspace(0.0, 1.0, n_samples)

    stacks: list[np.ndarray] = []
    for sp in valid:
        rel = sp.astype(np.float64) - luff
        t_sp = (rel @ chord_unit) / chord_len
        d_sp = rel @ normal_unit
        sort_idx = np.argsort(t_sp)
        t_sorted = t_sp[sort_idx]
        d_sorted = d_sp[sort_idx]
        keep = np.concatenate(([True], np.diff(t_sorted) > 1e-6))
        t_unique = t_sorted[keep]
        d_unique = d_sorted[keep]
        if len(t_unique) < 2:
            continue
        d_interp = np.interp(t_grid, t_unique, d_unique)
        stacks.append(d_interp)

    if len(stacks) < 2:
        return valid[0].astype(np.float64) if valid else None

    # 6. Median then Gaussian smooth
    d_med = np.median(np.stack(stacks, axis=0), axis=0)  # (n_samples,)
    sigma = max(1.0, smoothing_sigma_frac * n_samples)
    d_smooth = gaussian_filter1d(d_med, sigma=sigma, mode="nearest")

    # 7. Reproject to image coords
    consensus = (
        luff[None, :]
        + t_grid[:, None] * (chord_len * chord_unit)[None, :]
        + d_smooth[:, None] * normal_unit[None, :]
    )

    # 8. Pin endpoints exactly
    consensus[0]  = luff
    consensus[-1] = leech

    return consensus.astype(np.float64)
