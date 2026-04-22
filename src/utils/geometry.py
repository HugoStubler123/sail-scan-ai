"""Geometric utility functions for polyline operations and intersections.

These utilities are used throughout the pipeline for:
- Splitting sail contours into luff and leech edges
- Finding intersections between stripe curves and sail boundaries
- Validating point proximity to polylines
"""

import numpy as np
from typing import Tuple, List


def _compute_contour_curvature(
    contour: np.ndarray,
    k: int = 7,
    smooth_sigma: float = 3.0
) -> np.ndarray:
    """Compute curvature at each point of a closed contour.

    Smooths the contour with Gaussian filter, then computes the turning
    angle at each point using k-neighborhood vectors.

    Args:
        contour: Ordered contour points (N x 2)
        k: Neighborhood size for tangent estimation
        smooth_sigma: Gaussian smoothing sigma

    Returns:
        Absolute curvature array (N,)
    """
    from scipy.ndimage import gaussian_filter1d

    n = len(contour)
    if n < 2 * k + 1:
        k = max(1, n // 4)

    # Smooth contour coordinates (wrap mode for closed contour)
    x_smooth = gaussian_filter1d(contour[:, 0], sigma=smooth_sigma, mode='wrap')
    y_smooth = gaussian_filter1d(contour[:, 1], sigma=smooth_sigma, mode='wrap')

    curvature = np.zeros(n)
    for i in range(n):
        # Forward and backward vectors using k-neighborhood
        i_prev = (i - k) % n
        i_next = (i + k) % n

        v_back = np.array([x_smooth[i] - x_smooth[i_prev],
                           y_smooth[i] - y_smooth[i_prev]])
        v_fwd = np.array([x_smooth[i_next] - x_smooth[i],
                          y_smooth[i_next] - y_smooth[i]])

        norm_back = np.linalg.norm(v_back)
        norm_fwd = np.linalg.norm(v_fwd)

        if norm_back < 1e-9 or norm_fwd < 1e-9:
            curvature[i] = 0.0
            continue

        v_back = v_back / norm_back
        v_fwd = v_fwd / norm_fwd

        # Turning angle between the two tangent vectors
        cos_angle = np.clip(np.dot(v_back, v_fwd), -1.0, 1.0)
        curvature[i] = np.arccos(cos_angle)

    return curvature


def find_sail_corners(
    contour: np.ndarray
) -> Tuple[int, int, int]:
    """Find head, tack, and clew corners using curvature analysis.

    Args:
        contour: Ordered contour points (N x 2)

    Returns:
        (head_idx, tack_idx, clew_idx) indices into contour
    """
    from scipy.signal import find_peaks

    curvature = _compute_contour_curvature(contour)
    n = len(contour)

    # Find peaks on doubled curvature to handle wrap-around peaks
    # then map back to original indices, prioritizing higher curvature
    min_dist = max(n // 6, 3)
    doubled = np.concatenate([curvature, curvature])
    raw_peaks, _ = find_peaks(doubled, distance=min_dist)
    # Map to original range
    mapped = raw_peaks % n
    # Sort by curvature (descending) so highest-curvature peaks get priority
    curv_at_mapped = curvature[mapped]
    order = np.argsort(curv_at_mapped)[::-1]
    mapped_sorted = mapped[order]
    # Deduplicate: keep highest curvature, reject nearby duplicates
    seen = set()
    peaks = []
    for p in mapped_sorted:
        p = int(p)
        # Check no nearby peak already added (circular distance)
        too_close = False
        for s in seen:
            dist = min(abs(p - s), n - abs(p - s))
            if dist < min_dist:
                too_close = True
                break
        if not too_close:
            seen.add(p)
            peaks.append(p)
    peaks = np.array(peaks)

    if len(peaks) < 3:
        # Not enough peaks — fall back to position-based
        return None

    # Sort peaks by curvature value (descending) and take top 3
    peak_curvatures = curvature[peaks]
    sorted_peak_indices = np.argsort(peak_curvatures)[::-1]
    top3 = peaks[sorted_peak_indices[:3]]

    # Head = topmost peak (min y in image coords — sails always have head at top)
    y_coords = contour[top3, 1]
    head_local = np.argmin(y_coords)
    head_idx = top3[head_local]

    # The other two are bottom corners
    others = [top3[i] for i in range(3) if i != head_local]
    corner_a_idx, corner_b_idx = others[0], others[1]

    tack_idx, clew_idx = _classify_luff_leech(
        contour, head_idx, corner_a_idx, corner_b_idx
    )

    return head_idx, tack_idx, clew_idx


def _classify_luff_leech(
    contour: np.ndarray,
    head_idx: int,
    corner_a_idx: int,
    corner_b_idx: int
) -> Tuple[int, int]:
    """Classify two bottom corners as tack (luff side) and clew (leech side).

    The luff (tack->head) is straighter than the leech (head->clew) because
    the luff is under mast tension.

    Uses arc-avoiding logic to compare only the actual edges (not the path
    through the third corner).

    Args:
        contour: Ordered contour points (N x 2)
        head_idx: Index of head corner
        corner_a_idx: Index of first bottom corner
        corner_b_idx: Index of second bottom corner

    Returns:
        (tack_idx, clew_idx)
    """
    def arc_straightness(arc: np.ndarray) -> float:
        """Compute chord_length / arc_length for a contour arc."""
        if len(arc) < 2:
            return 1.0

        chord_length = np.linalg.norm(arc[-1] - arc[0])
        diffs = np.diff(arc, axis=0)
        arc_length = np.sum(np.linalg.norm(diffs, axis=1))

        if arc_length < 1e-9:
            return 1.0
        return chord_length / arc_length

    # Extract the actual edge from each corner to head, avoiding the other corner
    arc_a = _extract_arc_avoiding(contour, corner_a_idx, head_idx, corner_b_idx)
    arc_b = _extract_arc_avoiding(contour, corner_b_idx, head_idx, corner_a_idx)

    s_a = arc_straightness(arc_a)
    s_b = arc_straightness(arc_b)

    # Luff is straighter (under mast tension)
    if abs(s_a - s_b) > 0.02:
        # Clear difference — straighter edge is luff
        if s_a > s_b:
            return corner_a_idx, corner_b_idx  # a=tack, b=clew
        else:
            return corner_b_idx, corner_a_idx  # b=tack, a=clew
    else:
        # Tie-breaker: leftmost bottom point = tack (conventional sail orientation)
        if contour[corner_a_idx, 0] < contour[corner_b_idx, 0]:
            return corner_a_idx, corner_b_idx
        else:
            return corner_b_idx, corner_a_idx


def _position_based_corners(
    contour: np.ndarray
) -> Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Original position-based corner detection (fallback).

    Returns:
        (head_idx, tack_idx, clew_idx, head_point, tack_point, clew_point)
    """
    x_coords = contour[:, 0]
    y_coords = contour[:, 1]
    min_y, max_y = y_coords.min(), y_coords.max()

    head_idx = np.argmin(y_coords)
    head_point = contour[head_idx]

    bottom_mask = y_coords > (min_y + 0.7 * (max_y - min_y))
    bottom_indices = np.where(bottom_mask)[0]
    if len(bottom_indices) == 0:
        tack_idx = np.argmin(x_coords)
    else:
        tack_idx = bottom_indices[np.argmin(x_coords[bottom_indices])]
    tack_point = contour[tack_idx]

    bottom_indices_clew = np.where(bottom_mask)[0]
    if len(bottom_indices_clew) == 0:
        clew_idx = np.argmax(x_coords)
    else:
        clew_idx = bottom_indices_clew[np.argmax(x_coords[bottom_indices_clew])]
    clew_point = contour[clew_idx]

    return head_idx, tack_idx, clew_idx, head_point, tack_point, clew_point


def _find_head_on_open_contour(contour: np.ndarray) -> int:
    """Return the index of the head (topmost high-curvature peak).

    For an open sail contour (endpoints on image edges, the foot is
    off-screen), the head is the point that is (a) not at an endpoint
    and (b) has the lowest y coordinate among high-curvature peaks. If no
    clear peak exists, fall back to the lowest-y interior point.
    """
    from scipy.signal import find_peaks

    curv = _compute_contour_curvature(contour)
    n = len(contour)
    margin = max(n // 10, 3)
    if n <= 2 * margin:
        return int(np.argmin(contour[:, 1]))

    interior = slice(margin, n - margin)
    interior_curv = curv[interior]
    peaks, _ = find_peaks(interior_curv, distance=max(n // 8, 3))
    if len(peaks) == 0:
        return int(margin + np.argmin(contour[interior, 1]))
    peak_indices = peaks + margin
    top_peaks = peak_indices[
        np.argsort(curv[peak_indices])[-5:][::-1]
    ]
    # Among top-curvature peaks, pick the one with smallest y (topmost).
    return int(top_peaks[np.argmin(contour[top_peaks, 1])])


def _open_arc(
    contour: np.ndarray, idx_from: int, idx_to: int
) -> np.ndarray:
    """Slice an open polyline between two indices (inclusive)."""
    lo, hi = min(idx_from, idx_to), max(idx_from, idx_to)
    arc = contour[lo:hi + 1]
    if idx_from > idx_to:
        arc = arc[::-1]
    return arc


def _extract_arc_avoiding(
    contour: np.ndarray,
    idx_from: int,
    idx_to: int,
    avoid_idx: int
) -> np.ndarray:
    """Extract contour arc from idx_from to idx_to, choosing the path that avoids avoid_idx.

    On a closed contour there are two paths between any two points. This picks
    the one that does NOT pass through the third corner (avoid_idx).

    Args:
        contour: Ordered contour points (N x 2)
        idx_from: Start index
        idx_to: End index
        avoid_idx: Index that must NOT appear on the chosen arc

    Returns:
        Contour points along the correct arc (M x 2), oriented from idx_from to idx_to
    """
    n = len(contour)

    # Check if avoid_idx lies on the forward arc (idx_from → idx_from+1 → ... → idx_to)
    # Forward arc visits indices in increasing order (with wrap-around)
    if idx_from <= idx_to:
        avoid_on_fwd = (idx_from < avoid_idx < idx_to)
    else:
        avoid_on_fwd = (avoid_idx > idx_from or avoid_idx < idx_to)

    if not avoid_on_fwd:
        # Forward arc is safe (doesn't contain avoid_idx)
        if idx_from <= idx_to:
            return contour[idx_from:idx_to + 1]
        else:
            return np.vstack([contour[idx_from:], contour[:idx_to + 1]])
    else:
        # Reverse arc is safe: go idx_to → idx_to+1 → ... → idx_from, then flip
        if idx_to <= idx_from:
            arc = contour[idx_to:idx_from + 1]
        else:
            arc = np.vstack([contour[idx_to:], contour[:idx_from + 1]])
        return arc[::-1]


def split_contour_luff_leech(
    contour: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split an ordered sail contour into luff, leech, and foot polylines.

    Uses curvature-based corner detection to identify head, tack, and clew,
    then classifies luff vs leech by edge straightness (luff is straighter
    due to mast tension). Falls back to position-based heuristics on failure.

    For each edge, the arc that does NOT pass through the third corner is
    selected, ensuring correct direction regardless of contour traversal order.

    Args:
        contour: Ordered contour points (N x 2) in image coordinates [x, y]

    Returns:
        Tuple of (luff_polyline, leech_polyline, foot_polyline,
                  head_point, tack_point, clew_point)
        - luff_polyline: Points from tack to head (M1 x 2)
        - leech_polyline: Points from head to clew (M2 x 2)
        - foot_polyline: Points from clew to tack (M3 x 2)
        - head_point: Top corner coordinates (2,)
        - tack_point: Bottom-left corner (2,)
        - clew_point: Bottom-right corner (2,)
    """
    if len(contour) < 3:
        raise ValueError("Contour must have at least 3 points")

    # When the contour is an OPEN polyline (endpoints far apart), the sail
    # mask was clipped by the image frame. Treat the two endpoints as the
    # bottom corners (tack/clew) — the visible sail starts and ends there —
    # and find the head as the highest-curvature peak on the open path.
    is_open_contour = (
        float(np.linalg.norm(contour[0] - contour[-1])) > 50.0
    )

    corners = None
    if is_open_contour:
        head_idx = _find_head_on_open_contour(contour)
        end_a, end_b = 0, len(contour) - 1
        head_point = contour[head_idx]
        # Classify endpoints as tack (luff side) vs clew (leech side) by
        # straightness of the arc from endpoint to head (luff is straighter).
        tack_idx, clew_idx = _classify_luff_leech(
            contour, head_idx, end_a, end_b
        )
        tack_point = contour[tack_idx]
        clew_point = contour[clew_idx]
    else:
        try:
            corners = find_sail_corners(contour)
        except Exception:
            pass
        if corners is not None:
            head_idx, tack_idx, clew_idx = corners
            head_point = contour[head_idx]
            tack_point = contour[tack_idx]
            clew_point = contour[clew_idx]

    if corners is None and not is_open_contour:
        # Fallback to position-based
        head_idx, tack_idx, clew_idx, head_point, tack_point, clew_point = \
            _position_based_corners(contour)

    if is_open_contour:
        # Linear slice between corners — contour is not closed.
        luff_polyline = _open_arc(contour, tack_idx, head_idx)
        leech_polyline = _open_arc(contour, head_idx, clew_idx)
        # No foot edge visible in the image — synthesise a straight line
        # between tack and clew for downstream consumers.
        foot_polyline = np.vstack([tack_point, clew_point])
    else:
        luff_polyline = _extract_arc_avoiding(contour, tack_idx, head_idx, clew_idx)
        leech_polyline = _extract_arc_avoiding(contour, head_idx, clew_idx, tack_idx)
        foot_polyline = _extract_arc_avoiding(contour, clew_idx, tack_idx, head_idx)

    return luff_polyline, leech_polyline, foot_polyline, head_point, tack_point, clew_point


def polyline_curve_intersection(
    polyline: np.ndarray,
    spline_points: np.ndarray,
    tolerance: float = 1.0
) -> List[np.ndarray]:
    """Find intersection points between a polyline and a densely-sampled curve.

    Checks each segment of the polyline against each segment of the spline
    to find intersection points using line-line intersection.

    Args:
        polyline: Polyline points (N x 2)
        spline_points: Densely sampled curve points (M x 2)
        tolerance: Minimum distance between distinct intersections

    Returns:
        List of intersection points (each is shape (2,)), sorted by x-coordinate
    """
    intersections = []

    # Check each polyline segment against each spline segment
    for i in range(len(polyline) - 1):
        p1, p2 = polyline[i], polyline[i + 1]

        for j in range(len(spline_points) - 1):
            s1, s2 = spline_points[j], spline_points[j + 1]

            # Line-line intersection using parametric form
            intersection = _line_segment_intersection(p1, p2, s1, s2)
            if intersection is not None:
                # Check if this intersection is far enough from existing ones
                if len(intersections) == 0 or all(
                    np.linalg.norm(intersection - existing) > tolerance
                    for existing in intersections
                ):
                    intersections.append(intersection)

    # Sort by x-coordinate
    if intersections:
        intersections.sort(key=lambda pt: pt[0])

    return intersections


def _line_segment_intersection(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray
) -> np.ndarray | None:
    """Find intersection point between two line segments.

    Uses parametric line representation:
    P = p1 + t * (p2 - p1), t in [0, 1]
    Q = q1 + s * (q2 - q1), s in [0, 1]

    Args:
        p1, p2: Endpoints of first segment
        q1, q2: Endpoints of second segment

    Returns:
        Intersection point (2,) if segments intersect, None otherwise
    """
    r = p2 - p1
    s = q2 - q1
    qp = q1 - p1

    cross_rs = np.cross(r, s)

    # Parallel or coincident
    if abs(cross_rs) < 1e-10:
        return None

    t = np.cross(qp, s) / cross_rs
    u = np.cross(qp, r) / cross_rs

    # Check if intersection is within both segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        return p1 + t * r

    return None


def point_on_polyline(
    point: np.ndarray,
    polyline: np.ndarray,
    tolerance: float = 5.0
) -> bool:
    """Check if a point is within tolerance distance of any polyline segment.

    Args:
        point: Point coordinates (2,)
        polyline: Polyline points (N x 2)
        tolerance: Maximum distance in pixels

    Returns:
        True if point is within tolerance of any segment, False otherwise
    """
    if len(polyline) < 2:
        return False

    for i in range(len(polyline) - 1):
        p1, p2 = polyline[i], polyline[i + 1]

        # Distance from point to line segment
        dist = _point_to_segment_distance(point, p1, p2)
        if dist <= tolerance:
            return True

    return False


def _point_to_segment_distance(
    point: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray
) -> float:
    """Calculate minimum distance from point to line segment.

    Args:
        point: Point coordinates (2,)
        seg_start: Segment start point (2,)
        seg_end: Segment end point (2,)

    Returns:
        Minimum distance in pixels
    """
    # Vector from start to end
    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)

    # Degenerate segment (point)
    if seg_len_sq < 1e-10:
        return np.linalg.norm(point - seg_start)

    # Project point onto line (clamped to segment)
    t = max(0, min(1, np.dot(point - seg_start, seg_vec) / seg_len_sq))
    projection = seg_start + t * seg_vec

    return np.linalg.norm(point - projection)
