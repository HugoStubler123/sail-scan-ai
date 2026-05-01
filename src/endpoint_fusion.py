"""Robust 2-D endpoint fusion for sail stripe endpoints.

Combines up to 4 candidate endpoints (A, B, Combined, D) into a single
best estimate using a Weiszfeld geometric-median anchor followed by
confidence-weighted inlier averaging.

Algorithm:
1. Weiszfeld geometric median (~20 iterations) — L1-optimal center,
   up to 50 % breakdown point vs. 0 % for arithmetic mean.
2. MAD-based inlier gate: keep candidates within max(2.5*MAD, 0.08*chord).
3. Confidence-weighted mean of inliers.  Method D gets a 1.5× anchor
   weight when its per-end confidence exceeds 0.4 (user observation:
   "purple cross is the best most of the time").
4. Fused confidence: (sum_inlier_conf / sum_all_conf) * mean_inlier_conf.

Reference techniques:
  - Weiszfeld (1937); Vardi & Zhang (2000) corrected iteration.
  - MAD-based outlier rejection (Rousseeuw & Leroy, "Robust Regression
    and Outlier Detection", 1987).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from src.types import SailBoundary

logger = logging.getLogger(__name__)


# ── Weiszfeld geometric median ────────────────────────────────────────────────

def _geometric_median(
    points: np.ndarray,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 20,
    eps: float = 1e-5,
) -> np.ndarray:
    """Weighted geometric median via Weiszfeld iteration.

    Parameters
    ----------
    points:  (N, 2) array of candidate positions.
    weights: (N,) positive weights; uniform if None.
    max_iter: maximum Weiszfeld iterations.
    eps:      convergence threshold (pixel units).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 1:
        return pts[0].copy()
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 1e-12, None)
    # Initialise at (weighted) centroid
    med = (pts * w[:, None]).sum(axis=0) / w.sum()
    for _ in range(max_iter):
        d = np.linalg.norm(pts - med, axis=1)           # (N,)
        nz = d > 1e-9
        if not nz.any():
            break
        inv_d = np.where(nz, 1.0 / np.where(nz, d, 1.0), 0.0)
        wi = w * inv_d
        new_med = (pts * wi[:, None]).sum(axis=0) / wi.sum()
        if float(np.linalg.norm(new_med - med)) < eps:
            med = new_med
            break
        med = new_med
    return med


# ── Per-method confidence helpers ─────────────────────────────────────────────

def confidence_method_a(
    detection_extreme: np.ndarray,
    polyline_snap: np.ndarray,
) -> float:
    """High when the detection extreme is already close to the polyline."""
    dist = float(np.linalg.norm(np.asarray(detection_extreme) - np.asarray(polyline_snap)))
    return float(1.0 / (1.0 + dist))


def confidence_method_b(
    left_run: np.ndarray,
    chord_length: float,
) -> float:
    """High when the last-N points are well-aligned (low line-fit residual)."""
    if left_run is None or len(left_run) < 2 or chord_length < 1.0:
        return 0.2
    x = left_run[:, 0]; y = left_run[:, 1]
    run_len = float(np.linalg.norm(left_run[-1] - left_run[0]))
    if run_len < 1.0:
        return 0.2
    if np.ptp(x) >= np.ptp(y):
        m, b = np.polyfit(x, y, 1)
        resid = float(np.std(y - (m * x + b)))
    else:
        m, b = np.polyfit(y, x, 1)
        resid = float(np.std(x - (m * y + b)))
    alignment_bonus = min(1.0, run_len / (chord_length + 1e-6))
    return float((1.0 / (1.0 + resid / run_len)) * (0.5 + 0.5 * alignment_bonus))


def confidence_combined(
    a_ep: Optional[np.ndarray],
    b_ep: Optional[np.ndarray],
    chord_length: float,
) -> float:
    """High when A and B agree."""
    if a_ep is None or b_ep is None or chord_length < 1.0:
        return 0.3
    dist = float(np.linalg.norm(np.asarray(a_ep) - np.asarray(b_ep)))
    return float(1.0 - min(1.0, dist / chord_length))


def confidence_method_d(
    ml_ep: np.ndarray,
    combined_ep: np.ndarray,
    model_conf: float = 0.5,
    moved_min_px: float = 5.0,
    moved_max_px: float = 30.0,
) -> float:
    """High when D actually moved from Combined (refiner found a feature).

    If D == Combined (refiner abstained), confidence is halved so the
    fusion does not anchor on a no-op.
    """
    if ml_ep is None or combined_ep is None:
        return 0.0
    move = float(np.linalg.norm(np.asarray(ml_ep) - np.asarray(combined_ep)))
    # Bonus for being in the "sweet spot" of movement (5-30 px)
    if move < moved_min_px:
        move_bonus = 0.5
    elif move <= moved_max_px:
        move_bonus = 1.0
    else:
        move_bonus = 0.7  # large move — possible hallucination
    return float(np.clip(model_conf * move_bonus, 0.0, 1.0))


# ── Main fusion function ───────────────────────────────────────────────────────

def fuse_endpoint_candidates(
    candidates: List[Tuple[np.ndarray, float]],
    chord_length: float,
    d_index: Optional[int] = None,
    d_anchor_weight: float = 1.5,
    mad_sigma: float = 2.5,
    mad_min_frac: float = 0.08,
) -> Tuple[np.ndarray, float]:
    """Robust 2-D fusion of endpoint candidates.

    Parameters
    ----------
    candidates:       List of (point_xy, confidence) pairs.  Any pair with
                      None point or confidence <= 0 is silently dropped.
    chord_length:     Chord length of the stripe in pixels (used to scale the
                      MAD inlier gate floor).
    d_index:          Index into `candidates` of Method D; that candidate gets
                      an extra `d_anchor_weight` multiplier in the weighted
                      average when its confidence > 0.4.
    d_anchor_weight:  Multiplier applied to Method D in the weighted average.
    mad_sigma:        Inlier gate: keep points within mad_sigma * MAD of the
                      geometric median.
    mad_min_frac:     Floor for the MAD gate as a fraction of chord_length.

    Returns
    -------
    (fused_point, fused_confidence)
        fused_confidence in [0, 1].
    """
    # 1. Filter valid candidates
    valid: List[Tuple[np.ndarray, float, int]] = []
    for i, (pt, conf) in enumerate(candidates):
        if pt is None:
            continue
        pt_arr = np.asarray(pt, dtype=np.float64)
        if not np.all(np.isfinite(pt_arr)):
            continue
        c = float(conf)
        if c <= 0.0:
            c = 1e-3  # keep with minimal weight
        valid.append((pt_arr, c, i))

    if not valid:
        # Degenerate: return zero with zero confidence
        return np.zeros(2, dtype=np.float64), 0.0

    if len(valid) == 1:
        return valid[0][0].copy(), float(valid[0][1]) * 0.5

    pts_arr = np.stack([v[0] for v in valid])  # (M, 2)
    confs   = np.array([v[1] for v in valid])  # (M,)
    orig_idx = [v[2] for v in valid]

    # 2. Geometric median (Weiszfeld) — breakdown-resistant central tendency
    gmed = _geometric_median(pts_arr, weights=confs)

    # 3. MAD-based inlier gate
    dists = np.linalg.norm(pts_arr - gmed, axis=1)   # (M,)
    mad = float(np.median(dists))
    gate = max(mad_sigma * mad, mad_min_frac * chord_length, 1.0)
    inlier_mask = dists <= gate

    # 4. Build per-candidate weights (confidence × D anchor)
    weights = confs.copy()
    if d_index is not None:
        for j, oi in enumerate(orig_idx):
            if oi == d_index and confs[j] > 0.4:
                weights[j] *= d_anchor_weight

    # 5. Fused point from inliers
    n_inliers = int(inlier_mask.sum())

    if n_inliers >= 2:
        w_in = weights[inlier_mask]
        p_in = pts_arr[inlier_mask]
        fused = (p_in * w_in[:, None]).sum(axis=0) / w_in.sum()
        mean_conf_in = float(confs[inlier_mask].mean())
    elif n_inliers == 1:
        idx1 = int(np.argmax(inlier_mask))
        fused = pts_arr[idx1].copy()
        mean_conf_in = float(confs[idx1]) * 0.7
    else:
        # All candidates are outliers of each other → fall back to gmed
        fused = gmed
        mean_conf_in = float(confs.mean()) * 0.5

    # 6. Fused confidence
    sum_all   = float(confs.sum())
    sum_in    = float(confs[inlier_mask].sum()) if n_inliers > 0 else float(confs.min())
    ratio     = float(np.clip(sum_in / (sum_all + 1e-12), 0.0, 1.0))
    fused_conf = float(np.clip(ratio * mean_conf_in, 0.0, 1.0))

    return fused, fused_conf


# ── High-level wrapper ────────────────────────────────────────────────────────

def compute_fused_endpoints(
    detection_points: np.ndarray,
    luff_polyline: np.ndarray,
    leech_polyline: np.ndarray,
    image_bgr: Optional[np.ndarray] = None,
    sail_boundary: Optional["SailBoundary"] = None,
    use_ml_refiner: bool = True,
    keypoint_confidences: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
    """Fuse up to 4 candidate endpoints per end into a single robust estimate.

    Returns ``(luff_ep, leech_ep, conf_luff, conf_leech)`` or ``None`` when
    there are insufficient detection points or polylines.

    Method pipeline
    ---------------
    A — geometric snap (project detection extremes to nearest polyline point).
    B — linear extrapolation from the last-N aligned detection points.
    Combined — outlier-robust average of A and B (from stripe_endpoints.py).
    D (optional) — ML two-stage refiner. Invoked only when ``use_ml_refiner``
        is True AND both ``image_bgr`` and ``sail_boundary`` are provided AND
        the required model files exist.

    All four candidates are then passed to :func:`fuse_endpoint_candidates`
    (Weiszfeld geometric median + MAD inlier gate + confidence-weighted mean).
    """
    from src.stripe_endpoints import compute_stripe_endpoints_full, _select_aligned_run

    pts = np.asarray(detection_points, dtype=np.float64)
    if pts is None or len(pts) < 3:
        return None

    # Sort along chord (x-axis approximation)
    order = np.argsort(pts[:, 0])
    pts_sorted = pts[order]
    kp_confs_sorted: Optional[np.ndarray] = None
    if keypoint_confidences is not None:
        kc = np.asarray(keypoint_confidences, dtype=np.float64)
        if len(kc) == len(pts):
            kp_confs_sorted = kc[order]

    # A + B + Combined via stripe_endpoints
    ep_full = compute_stripe_endpoints_full(
        pts_sorted,
        luff_polyline,
        leech_polyline,
        keypoint_confidences=kp_confs_sorted,
    )
    if ep_full is None:
        return None

    a_l, a_r = ep_full.get("method_a", (None, None))
    b_l, b_r = ep_full.get("method_b", (None, None))
    combined = ep_full.get("combined")

    chord_len = float(np.linalg.norm(pts_sorted[-1] - pts_sorted[0])) \
                if len(pts_sorted) >= 2 else 1.0

    comb_luff_ep = np.asarray(combined[0]) if combined is not None else None
    comb_leech_ep = np.asarray(combined[1]) if combined is not None else None

    # Align A and B to luff/leech using Combined proximity
    def _align_to_luff(
        left_ep: Optional[np.ndarray],
        right_ep: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if left_ep is None or right_ep is None:
            return left_ep, right_ep
        if comb_luff_ep is None or comb_leech_ep is None:
            return left_ep, right_ep
        l = np.asarray(left_ep, dtype=np.float64)
        r = np.asarray(right_ep, dtype=np.float64)
        d_l_luff = float(np.linalg.norm(l - comb_luff_ep))
        d_r_luff = float(np.linalg.norm(r - comb_luff_ep))
        if d_l_luff <= d_r_luff:
            return l, r
        return r, l

    a_luff_ep, a_leech_ep = _align_to_luff(a_l, a_r)
    b_luff_ep, b_leech_ep = _align_to_luff(b_l, b_r)

    # Determine which detection extreme corresponds to luff
    cloud_left_is_luff = (
        a_luff_ep is not None and a_l is not None
        and np.allclose(a_luff_ep, np.asarray(a_l, dtype=np.float64))
    )

    left_run  = _select_aligned_run(pts_sorted)
    right_run = _select_aligned_run(pts_sorted[::-1])[::-1]
    luff_run  = left_run  if cloud_left_is_luff else right_run
    leech_run = right_run if cloud_left_is_luff else left_run

    luff_extreme  = pts_sorted[0]  if cloud_left_is_luff else pts_sorted[-1]
    leech_extreme = pts_sorted[-1] if cloud_left_is_luff else pts_sorted[0]

    # Per-method confidences
    conf_a_l = confidence_method_a(luff_extreme, a_luff_ep)   if a_luff_ep  is not None else 0.0
    conf_b_l = confidence_method_b(luff_run, chord_len)        if b_luff_ep  is not None else 0.0
    conf_c_l = confidence_combined(a_luff_ep, b_luff_ep, chord_len)

    conf_a_r = confidence_method_a(leech_extreme, a_leech_ep) if a_leech_ep is not None else 0.0
    conf_b_r = confidence_method_b(leech_run, chord_len)       if b_leech_ep is not None else 0.0
    conf_c_r = confidence_combined(a_leech_ep, b_leech_ep, chord_len)

    # Method D — ML refiner (optional)
    d_luff_ep:  Optional[np.ndarray] = None
    d_leech_ep: Optional[np.ndarray] = None
    conf_d_l = 0.0
    conf_d_r = 0.0
    d_idx = None  # index into candidates list for d_anchor_weight

    if use_ml_refiner and image_bgr is not None and sail_boundary is not None:
        # Lazy import to avoid circular deps and heavy model loading at import time
        try:
            from src.pipeline_v7 import refine_endpoints_two_stage
            import yaml
            from pathlib import Path

            cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
            cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
            det_cfg = cfg.get("detection", {})

            raw_luff  = np.asarray(comb_luff_ep,  dtype=np.float32) if comb_luff_ep  is not None else np.asarray(luff_extreme,  dtype=np.float32)
            raw_leech = np.asarray(comb_leech_ep, dtype=np.float32) if comb_leech_ep is not None else np.asarray(leech_extreme, dtype=np.float32)

            ml_luff, ml_leech, _ = refine_endpoints_two_stage(
                raw_luff, raw_leech, image_bgr, sail_boundary, det_cfg,
                max_nudge_px=30.0,
            )
            d_luff_ep  = np.asarray(ml_luff,  dtype=np.float64)
            d_leech_ep = np.asarray(ml_leech, dtype=np.float64)

            combined_for_d = comb_luff_ep if comb_luff_ep is not None else luff_extreme
            conf_d_l = confidence_method_d(d_luff_ep, combined_for_d)
            combined_for_d_r = comb_leech_ep if comb_leech_ep is not None else leech_extreme
            conf_d_r = confidence_method_d(d_leech_ep, combined_for_d_r)
            d_idx = 3  # index of Method D in the 4-element candidates list
        except Exception as exc:
            logger.debug("ML refiner (Method D) skipped: %s", exc)

    # Build candidate lists and fuse
    luff_candidates: List[Tuple[Optional[np.ndarray], float]] = [
        (a_luff_ep,  conf_a_l),
        (b_luff_ep,  conf_b_l),
        (comb_luff_ep, conf_c_l),
        (d_luff_ep,  conf_d_l),
    ]
    leech_candidates: List[Tuple[Optional[np.ndarray], float]] = [
        (a_leech_ep,  conf_a_r),
        (b_leech_ep,  conf_b_r),
        (comb_leech_ep, conf_c_r),
        (d_leech_ep,  conf_d_r),
    ]

    fused_luff,  conf_luff  = fuse_endpoint_candidates(luff_candidates,  chord_len, d_index=d_idx)
    fused_leech, conf_leech = fuse_endpoint_candidates(leech_candidates, chord_len, d_index=d_idx)

    return fused_luff, fused_leech, conf_luff, conf_leech
