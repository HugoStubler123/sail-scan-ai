"""Endpoint recovery via tangent extension to the SAM mask polylines.

User-specified algorithm (no precomputed luff_ep / leech_ep — endpoints
are derived purely from the detection points + the SAM polylines):

    Method A — shape prior. Fit a low-degree polynomial in chord-space
    (perpendicular offset vs parametric position along the leftmost-to-
    rightmost detection chord). Compute analytical tangent at each end
    of the detection cloud. Cast a ray from each end along that tangent
    until it crosses the union of luff and leech polylines. Crossing
    point becomes the candidate endpoint.

    Method B — local linear extrapolation. Take the last 3+ relatively
    aligned detection points at each end, fit a line through them,
    extrapolate the line until it crosses the polyline union. Crossing
    point becomes the candidate endpoint.

    Combine: average the two candidates. Fall back to whichever is
    available when one method fails.

The function returns ``(left_ep, right_ep)`` labelled by detection-cloud
order (left = leftmost in chord coords, right = rightmost). The caller
assigns the luff/leech labels by polyline proximity.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def _polyline_intersect_ray(
    origin: np.ndarray,
    direction: np.ndarray,
    polyline: np.ndarray,
    max_t: float = 1e6,
) -> Optional[np.ndarray]:
    """First positive-t intersection of origin + t·direction with polyline."""
    if polyline is None or len(polyline) < 2:
        return None
    d = direction.astype(np.float64)
    dn = float(np.linalg.norm(d))
    if dn < 1e-9:
        return None
    d = d / dn
    o = origin.astype(np.float64)
    poly = np.asarray(polyline, dtype=np.float64)

    best_t = max_t
    best_pt: Optional[np.ndarray] = None
    for i in range(len(poly) - 1):
        a = poly[i]; b = poly[i + 1]
        ab = b - a
        det = d[0] * (-ab[1]) - d[1] * (-ab[0])
        if abs(det) < 1e-9:
            continue
        rhs = a - o
        t = (-ab[1] * rhs[0] + ab[0] * rhs[1]) / det
        s = (d[0] * rhs[1] - d[1] * rhs[0]) / det
        if t < 0.0 or not (-1e-3 <= s <= 1.0 + 1e-3):
            continue
        if t < best_t:
            best_t = t
            best_pt = o + t * d
    return best_pt


def _ray_to_any_polyline(
    origin: np.ndarray,
    direction: np.ndarray,
    polylines: List[np.ndarray],
) -> Optional[np.ndarray]:
    best = None
    best_t = np.inf
    for pl in polylines:
        hit = _polyline_intersect_ray(origin, direction, pl)
        if hit is None:
            continue
        t = float(np.linalg.norm(hit - origin))
        if t < best_t:
            best_t = t
            best = hit
    return best


def _chord_frame(pts: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Return (chord_start, chord_unit, chord_normal, chord_length)."""
    if len(pts) < 2:
        return None
    chord_start = pts[0].astype(np.float64)
    chord_end = pts[-1].astype(np.float64)
    vec = chord_end - chord_start
    L = float(np.linalg.norm(vec))
    if L < 1.0:
        return None
    u = vec / L
    n = np.array([-u[1], u[0]], dtype=np.float64)
    return chord_start, u, n, L


def _nearest_point_on_polyline(
    point: np.ndarray,
    polyline: np.ndarray,
) -> Optional[np.ndarray]:
    """Project ``point`` onto the nearest segment of ``polyline``.

    Returns the closest point on any segment (not just vertices).
    """
    if polyline is None or len(polyline) < 2:
        return None
    p = point.astype(np.float64)
    poly = np.asarray(polyline, dtype=np.float64)
    best_d = np.inf
    best_pt: Optional[np.ndarray] = None
    for i in range(len(poly) - 1):
        a = poly[i]; b = poly[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            proj = a
        else:
            t = float((p - a) @ ab / L2)
            t = max(0.0, min(1.0, t))
            proj = a + t * ab
        d = float(np.linalg.norm(proj - p))
        if d < best_d:
            best_d = d
            best_pt = proj
    return best_pt


def _method_a_shape_prior(
    pts_sorted: np.ndarray,
    polylines: List[np.ndarray],
    weights: Optional[np.ndarray] = None,
    degree: int = 3,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Geometric snap: project the first and last detection points onto
    the nearest SAM polyline.

    This replaces the former polynomial-tangent ray-cast (Method A) which
    diverged when the degree-3 polynomial flared at the chord ends.
    Option C from the design spec: no extrapolation — direct projection.

    ``polylines`` is [luff_polyline, leech_polyline]. The left detection
    extreme is projected onto whichever polyline is closest, and similarly
    for the right extreme.
    """
    if pts_sorted is None or len(pts_sorted) < 2 or not polylines:
        return None, None

    left_pt = pts_sorted[0].astype(np.float64)
    right_pt = pts_sorted[-1].astype(np.float64)

    def _snap(p: np.ndarray) -> Optional[np.ndarray]:
        best_d = np.inf
        best_proj = None
        for pl in polylines:
            proj = _nearest_point_on_polyline(p, pl)
            if proj is None:
                continue
            d = float(np.linalg.norm(proj - p))
            if d < best_d:
                best_d = d
                best_proj = proj
        return best_proj

    left_ep = _snap(left_pt)
    right_ep = _snap(right_pt)
    return left_ep, right_ep


def _select_aligned_run(
    pts: np.ndarray, n_min: int = 3, n_max: int = 24,
    max_residual_frac: float = 0.05,
) -> np.ndarray:
    """Take the first ``n_min..n_max`` points and pick the largest k for
    which a line fit has per-point residual < ``max_residual_frac`` of
    the line length. Returns the chosen sub-array (≥2 points).

    User spec: Method B should use AT LEAST 3 points; if more points are
    well-aligned (residual still below threshold) use them. The cap is
    ``min(n_max, len(pts) - 1)`` — leave at least one point as a sanity
    floor and allow up to n_max=24 well-aligned points to be used.
    """
    if len(pts) < 2:
        return pts
    cap = min(n_max, max(n_min, len(pts) - 1))
    best_k = min(n_min, len(pts))
    for k in range(cap, n_min - 1, -1):
        sub = pts[:k]
        if len(sub) < 2:
            continue
        x = sub[:, 0]; y = sub[:, 1]
        if x.max() - x.min() < 1.0 and y.max() - y.min() < 1.0:
            continue
        # Fit y = m*x + b OR x = m*y + b depending on which spans more
        if np.ptp(x) >= np.ptp(y):
            m, b = np.polyfit(x, y, 1)
            resid = float(np.std(y - (m * x + b)))
            line_len = float(np.linalg.norm(sub[-1] - sub[0]))
            if line_len > 1.0 and resid / line_len < max_residual_frac:
                best_k = k
                break
        else:
            m, b = np.polyfit(y, x, 1)
            resid = float(np.std(x - (m * y + b)))
            line_len = float(np.linalg.norm(sub[-1] - sub[0]))
            if line_len > 1.0 and resid / line_len < max_residual_frac:
                best_k = k
                break
    return pts[:best_k]


def _line_ray_from_run(
    sub: np.ndarray, going_left: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (origin, direction) of the linear extrapolation ray."""
    if len(sub) < 2:
        return None, None
    diffs = sub[-1] - sub[0]
    norm = float(np.linalg.norm(diffs))
    if norm < 1e-3:
        return None, None
    direction = diffs / norm
    if going_left:
        origin = sub[0].astype(np.float64)
        direction = -direction
    else:
        origin = sub[-1].astype(np.float64)
    return origin, direction


def _method_b_linear_run(
    pts_sorted: np.ndarray,
    polylines: List[np.ndarray],
    n_min: int = 3,
    n_max: int = 6,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Linear extrapolation from first/last aligned run."""
    left_run = _select_aligned_run(pts_sorted, n_min=n_min, n_max=n_max)
    right_run = _select_aligned_run(pts_sorted[::-1], n_min=n_min, n_max=n_max)[::-1]

    o_l, d_l = _line_ray_from_run(left_run, going_left=True)
    o_r, d_r = _line_ray_from_run(right_run, going_left=False)

    left_ep = _ray_to_any_polyline(o_l, d_l, polylines) if o_l is not None else None
    right_ep = _ray_to_any_polyline(o_r, d_r, polylines) if o_r is not None else None
    return left_ep, right_ep


def _combine_with_outlier_rejection(
    a_ep: Optional[np.ndarray],
    b_ep: Optional[np.ndarray],
    origin: Optional[np.ndarray],
    direction: Optional[np.ndarray],
    chord_length: Optional[float] = None,
    disagree_frac: float = 0.10,
    overshoot_ratio: float = 2.5,
) -> Optional[np.ndarray]:
    """Combine Method A (shape prior) and Method B (linear) endpoints.

    Default behaviour: average A and B. Only one is rejected when it is a
    *clear* outlier:

    * If only one is valid, use it.
    * If A overshoots far past B along the tangent direction
      (tan_a > ``overshoot_ratio`` × tan_b, both positive), A is the
      unreliable one (polynomial flare) — use B.
    * If B overshoots far past A along the tangent direction
      (tan_b > ``overshoot_ratio`` × tan_a, both positive), B is the
      unreliable one (linear extrapolation runs off the sail) — use A.
    * Otherwise: average A and B. This is the default for both close and
      moderately-disagreeing candidates — the average is visually correct
      when both are plausible.
    """
    if a_ep is None and b_ep is None:
        return None
    if a_ep is None:
        return b_ep.astype(np.float64)
    if b_ep is None:
        return a_ep.astype(np.float64)

    a = a_ep.astype(np.float64)
    b = b_ep.astype(np.float64)

    # Outlier rejection: compare tangent-direction distances from the
    # linear-run origin. If one candidate is much farther along the
    # extrapolation ray than the other, it has overshot the sail edge.
    if origin is not None and direction is not None:
        d = direction.astype(np.float64)
        dn = float(np.linalg.norm(d))
        if dn >= 1e-9:
            d = d / dn
            o = origin.astype(np.float64)
            tan_a = float((a - o) @ d)
            tan_b = float((b - o) @ d)
            if tan_a > 0 and tan_b > 0:
                if tan_a > overshoot_ratio * tan_b:
                    # A overshooting — trust B alone
                    return b
                if tan_b > overshoot_ratio * tan_a:
                    # B overshooting — trust A alone
                    return a

    # Default: average both candidates. Both are plausible estimates of
    # where the stripe meets the sail edge; the average minimises bias
    # from either method's assumptions.
    return 0.5 * (a + b)


def compute_stripe_endpoints_full(
    detection_points: np.ndarray,
    luff_polyline: Optional[np.ndarray],
    leech_polyline: Optional[np.ndarray],
    keypoint_confidences: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """Compute endpoints by both methods, return ALL outputs for
    diagnostic visualization.

    Returns a dict with:
      method_a: (left_ep, right_ep) or (None, None)
      method_b: (left_ep, right_ep) or (None, None)
      combined: (luff_ep, leech_ep) labelled by polyline proximity
      method_a_origins: (origin_left, origin_right) tangent ray origins
      method_a_directions: (dir_left, dir_right) tangent ray directions
      method_b_origins: same for method B
      method_b_directions: same for method B

    or None if there are not enough points / no polylines.
    """
    pts = np.asarray(detection_points, dtype=np.float64)
    if pts is None or len(pts) < 3:
        return None
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    if keypoint_confidences is not None and len(keypoint_confidences) == len(pts):
        weights = np.asarray(keypoint_confidences, dtype=np.float64)[order]
    else:
        weights = None

    polylines = [p for p in (luff_polyline, leech_polyline) if p is not None and len(p) >= 2]
    if not polylines:
        return None

    # Method A: geometric snap — project first/last detection points to
    # nearest polyline. No ray-casting, no polynomial extrapolation.
    a_left, a_right = _method_a_shape_prior(pts, polylines, weights=weights)
    # Origins for diagnostic display: the detection extremes themselves
    a_origins = (pts[0] if a_left is not None else None,
                 pts[-1] if a_right is not None else None)
    a_directions = (None, None)

    # Method B geometry
    b_left, b_right = _method_b_linear_run(pts, polylines)
    left_run = _select_aligned_run(pts)
    right_run = _select_aligned_run(pts[::-1])[::-1]
    bo_l, bd_l = _line_ray_from_run(left_run, going_left=True)
    bo_r, bd_r = _line_ray_from_run(right_run, going_left=False)
    b_origins = (bo_l, bo_r)
    b_directions = (bd_l, bd_r)

    # Outlier-robust combination: average Method A and Method B when they
    # agree; trust Method A (shape prior) when they disagree — Method B's
    # linear extrapolation cannot see curvature and creates discontinuities
    # at curving stripe ends. Pass chord_length so the disagreement
    # threshold scales with the stripe's overall extent.
    chord_len = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) >= 2 else None
    left_origin, left_dir = _line_ray_from_run(left_run, going_left=True)
    right_origin, right_dir = _line_ray_from_run(right_run, going_left=False)

    left_ep = _combine_with_outlier_rejection(
        a_left, b_left, origin=left_origin, direction=left_dir,
        chord_length=chord_len,
    )
    right_ep = _combine_with_outlier_rejection(
        a_right, b_right, origin=right_origin, direction=right_dir,
        chord_length=chord_len,
    )
    if left_ep is None or right_ep is None:
        return {
            "method_a": (a_left, a_right),
            "method_b": (b_left, b_right),
            "combined": None,
            "method_a_origins": a_origins,
            "method_a_directions": a_directions,
            "method_b_origins": b_origins,
            "method_b_directions": b_directions,
        }

    def _dist_to_polyline(p, polyline):
        if polyline is None or len(polyline) < 1:
            return np.inf
        diffs = polyline - p
        return float(np.min(np.linalg.norm(diffs, axis=1)))

    if luff_polyline is not None and leech_polyline is not None:
        d_left_luff = _dist_to_polyline(left_ep, luff_polyline)
        d_left_leech = _dist_to_polyline(left_ep, leech_polyline)
        d_right_luff = _dist_to_polyline(right_ep, luff_polyline)
        d_right_leech = _dist_to_polyline(right_ep, leech_polyline)
        left_is_luff = (d_left_luff + d_right_leech) <= (d_left_leech + d_right_luff)
    else:
        left_is_luff = True

    if left_is_luff:
        combined = (left_ep.astype(np.float64), right_ep.astype(np.float64))
    else:
        combined = (right_ep.astype(np.float64), left_ep.astype(np.float64))

    return {
        "method_a": (a_left, a_right),
        "method_b": (b_left, b_right),
        "combined": combined,
        "method_a_origins": a_origins,
        "method_a_directions": a_directions,
        "method_b_origins": b_origins,
        "method_b_directions": b_directions,
    }


def compute_stripe_endpoints(
    detection_points: np.ndarray,
    luff_polyline: Optional[np.ndarray],
    leech_polyline: Optional[np.ndarray],
    keypoint_confidences: Optional[np.ndarray] = None,
    average_when_both: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Thin wrapper around ``compute_stripe_endpoints_full`` that returns
    only the combined ``(luff_ep, leech_ep)`` pair.
    """
    full = compute_stripe_endpoints_full(
        detection_points, luff_polyline, leech_polyline,
        keypoint_confidences=keypoint_confidences,
    )
    if full is None or full["combined"] is None:
        return None
    return full["combined"]
