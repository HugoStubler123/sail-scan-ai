"""Endpoint detection module for sail stripe analysis.

This module computes where each stripe meets the sail boundary by:
1. Fitting a line through interior keypoints to determine stripe direction
2. Projecting that line to find intersections with luff and leech polylines
3. Falling back to horizontal projection or nearest-point if needed

Linear projection is used instead of B-spline extrapolation because
short stripe fragments cause cubic splines to diverge wildly outside
the data range. A linear projection is stable regardless of fragment length.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

from src.types import StripeDetection, SailBoundary
from src.utils.geometry import point_on_polyline


def find_endpoints(
    detection: StripeDetection,
    sail_boundary: SailBoundary,
    extrapolation_factor: float = 1.5
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Find luff and leech endpoints for a stripe.

    Preferred path: when ``detection.polygon`` is available (e.g. from the
    Roboflow stripe-segmentation model) we intersect the polygon's edges
    with the SAM2 luff and leech polylines. This places each endpoint at
    the real stripe-boundary crossing.

    Fallback: snap the outermost interior keypoint to the nearest point
    on the respective polyline.

    Args:
        detection: Stripe detection containing interior keypoints.
        sail_boundary: Sail boundary with luff/leech polylines.
        extrapolation_factor: Unused, kept for API compatibility.

    Returns:
        Tuple of (luff_endpoint, leech_endpoint) or None.
    """
    points = detection.points
    if len(points) < 2:
        return None
    if np.std(points[:, 0]) < 1e-6:
        return None

    # Classify which extreme kp is on the luff side vs leech side
    first_pt, last_pt = points[0], points[-1]
    luff_centroid = np.mean(sail_boundary.luff_polyline, axis=0)
    d_first_luff = np.linalg.norm(first_pt - luff_centroid)
    d_last_luff = np.linalg.norm(last_pt - luff_centroid)
    if d_first_luff < d_last_luff:
        luff_near_pt, leech_near_pt = first_pt, last_pt
    else:
        luff_near_pt, leech_near_pt = last_pt, first_pt

    # Optional: when a Roboflow polygon is attached, intersect its bottom
    # (draft-side) edge with the SAM2 luff / leech polylines.
    polygon = getattr(detection, 'polygon', None)
    use_poly_bottom = (
        polygon is not None
        and len(polygon) >= 3
        and getattr(find_endpoints, '_use_polygon_bottom_intersection', False)
    )
    if use_poly_bottom:
        luff_ep = polygon_bottom_polyline_intersection(
            polygon, sail_boundary.luff_polyline, luff_near_pt
        )
        leech_ep = polygon_bottom_polyline_intersection(
            polygon, sail_boundary.leech_polyline, leech_near_pt
        )
        if luff_ep is not None and leech_ep is not None:
            return (luff_ep, leech_ep)

    # When a Roboflow polygon is attached AND the detection.points
    # endpoints came from the polygon (i.e. are the polygon's leftmost/
    # rightmost points, not kp interior), trust them. Compare the naive
    # polyline-snap distance: if it's large, the polyline is too short
    # (e.g. luff stops at the head while the top stripe's luff endpoint
    # lies above the head) and snapping would worsen the result.
    snap_luff = _nearest_boundary_point(
        luff_near_pt, sail_boundary.luff_polyline
    )
    snap_leech = _nearest_boundary_point(
        leech_near_pt, sail_boundary.leech_polyline
    )

    # A snap that moves the endpoint more than ``max_snap_px`` almost
    # always lands on the wrong part of the sail boundary (e.g. the head
    # when a stripe's luff end lies above it). Trust the detection's
    # own endpoint in that case. ``max_snap_px`` is tighter when the
    # detection carries a polygon (high confidence) and looser when we
    # only have kp/seg interior points (which are ~10-20% inside the
    # real stripe end).
    # Only refuse long snaps when we have a polygon (trustworthy
    # endpoint source). For kp-only / seg-only detections the snap is
    # still our best bet, even if sometimes wrong.
    has_polygon = polygon is not None and len(polygon) >= 3
    if has_polygon:
        # When a polygon is attached, the detection's endpoints are
        # trustworthy. Snap only if the polyline target is within 50 px
        # (the polygon extremes are at the sail boundary already;
        # anything larger than that is pathological — usually the
        # polyline doesn't reach the stripe's y-level, so it snaps to
        # an irrelevant feature like the head).
        max_snap_px = 50.0
        if snap_luff is None or np.linalg.norm(snap_luff - luff_near_pt) > max_snap_px:
            snap_luff = luff_near_pt
        if snap_leech is None or np.linalg.norm(snap_leech - leech_near_pt) > max_snap_px:
            snap_leech = leech_near_pt

    if snap_luff is not None and snap_leech is not None:
        return (snap_luff, snap_leech)

    luff_endpoint = _nearest_boundary_point(
        luff_near_pt, sail_boundary.luff_polyline
    )
    leech_endpoint = _nearest_boundary_point(
        leech_near_pt, sail_boundary.leech_polyline
    )
    if luff_endpoint is None or leech_endpoint is None:
        return None
    return (luff_endpoint, leech_endpoint)


def _project_stripe_onto_polyline(
    stripe_points: np.ndarray,
    polyline: np.ndarray,
    toward: np.ndarray,
    max_y_deviation_ratio: float = 0.25,
) -> Optional[np.ndarray]:
    """Find the polyline point the stripe's line passes through.

    We least-squares fit the stripe as ``y = a·x + b``, then for each
    polyline vertex compute the perpendicular distance to that line and
    pick the polyline point that minimises it. A soft guard rejects
    candidates whose y is far from the stripe's y-range (prevents
    grabbing the head of a diagonal luff above/below the stripe).
    """
    if len(stripe_points) < 2 or len(polyline) < 2:
        return None

    xs = stripe_points[:, 0]
    ys = stripe_points[:, 1]
    if np.std(xs) < 1e-6:
        return None
    slope, intercept = np.polyfit(xs, ys, 1)

    # Only consider polyline vertices whose y is within ±ratio · chord_length
    # of the stripe's y range — keeps the candidate on the same stripe.
    y_min, y_max = float(ys.min()), float(ys.max())
    chord = float(
        np.linalg.norm(stripe_points[0] - stripe_points[-1])
    ) or 100.0
    y_band = max_y_deviation_ratio * chord
    mask = (polyline[:, 1] >= y_min - y_band) & (
        polyline[:, 1] <= y_max + y_band
    )
    candidates = polyline[mask]
    if len(candidates) < 1:
        candidates = polyline  # fall back to whole polyline

    # Perpendicular distance from candidate to line y=slope·x+intercept
    # |slope·x - y + intercept| / sqrt(slope^2 + 1)
    num = np.abs(slope * candidates[:, 0] - candidates[:, 1] + intercept)
    denom = np.sqrt(slope * slope + 1.0)
    perp_dist = num / denom

    # Among the closest (by perp distance) candidates, pick the one
    # closest to ``toward`` — resolves left-vs-right endpoint.
    best_idx_sorted = np.argsort(perp_dist)
    top_k = candidates[best_idx_sorted[: max(1, min(10, len(candidates)))]]
    dists_to_toward = np.linalg.norm(top_k - toward, axis=1)
    return top_k[int(np.argmin(dists_to_toward))]


def _extend_stripe_line_to_polyline(
    stripe_points: np.ndarray,
    polyline: np.ndarray,
    toward: np.ndarray,
) -> Optional[np.ndarray]:
    """Fit a line through all kp points and intersect with ``polyline``.

    We least-squares fit y = a·x + b through the stripe's interior kpts,
    then build a ray starting at ``toward`` and walking outward along the
    line until it hits the polyline. Works for curved stripes because the
    line is global (averaged over all kpts) and we start the ray at the
    already-outermost kpt.
    """
    if len(stripe_points) < 2 or len(polyline) < 2:
        return None

    xs = stripe_points[:, 0]
    ys = stripe_points[:, 1]
    if np.std(xs) < 1e-6:
        return None
    slope, intercept = np.polyfit(xs, ys, 1)

    # Decide ray direction: from the mean to ``toward``.
    mean = np.mean(stripe_points, axis=0)
    direction = toward - mean
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    unit = direction / norm

    # Walk along the fitted line in the ray direction. Re-project so the
    # ray truly follows y=ax+b: parameterise by x.
    # If unit[0] ≈ 0, the ray is nearly vertical — unusual for a stripe.
    if abs(unit[0]) < 1e-3:
        return None

    stripe_length = float(np.linalg.norm(
        stripe_points[0] - stripe_points[-1]
    ))
    reach = max(stripe_length, 200.0) * 2.0
    sign = 1.0 if unit[0] > 0 else -1.0
    x_end = float(toward[0]) + sign * reach
    y_end = slope * x_end + intercept
    ray_start = np.array([float(toward[0]), slope * toward[0] + intercept])
    ray_end = np.array([x_end, y_end])

    intersections: list[np.ndarray] = []
    for j in range(len(polyline) - 1):
        pt = _segment_intersection(
            ray_start, ray_end, polyline[j], polyline[j + 1]
        )
        if pt is not None:
            intersections.append(pt)

    if not intersections:
        return None
    arr = np.asarray(intersections)
    dists = np.linalg.norm(arr - toward, axis=1)
    return arr[int(np.argmin(dists))]


def _extrapolate_to_polyline(
    stripe_points: np.ndarray,
    polyline: np.ndarray,
    toward: np.ndarray,
) -> Optional[np.ndarray]:
    """Extrapolate the stripe through ``toward`` and find the polyline crossing.

    Uses the outermost kp points near ``toward`` (the closest 3 points) to
    compute a local direction vector, then walks a ray outward along that
    direction and returns the first polyline intersection. If none, returns
    the nearest polyline point to the ray's end as a last resort.
    """
    if len(stripe_points) < 2 or len(polyline) < 2:
        return None

    # Order stripe points by distance to ``toward`` and take the outermost
    # 3 (or fewer) — these define the local slope at the stripe end.
    dists = np.linalg.norm(stripe_points - toward, axis=1)
    order = np.argsort(dists)
    end_pts = stripe_points[order[:3]]
    if len(end_pts) < 2:
        return None

    # The direction vector points from the INNER end_pts toward ``toward``.
    inner = np.mean(stripe_points[order[-3:]], axis=0)
    direction = toward - inner
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    direction = direction / norm

    # Ray: start at ``toward`` and walk out up to 3x the stripe length.
    stripe_length = float(np.linalg.norm(
        stripe_points[0] - stripe_points[-1]
    ))
    ray_end = toward + direction * max(stripe_length, 200.0) * 2.0

    intersections: list[np.ndarray] = []
    for j in range(len(polyline) - 1):
        pt = _segment_intersection(toward, ray_end, polyline[j], polyline[j + 1])
        if pt is not None:
            intersections.append(pt)

    if intersections:
        arr = np.asarray(intersections)
        dists_to_toward = np.linalg.norm(arr - toward, axis=1)
        return arr[int(np.argmin(dists_to_toward))]

    # Also try ray going the other direction (in case we picked the wrong side)
    ray_end_rev = toward - direction * max(stripe_length, 200.0) * 2.0
    for j in range(len(polyline) - 1):
        pt = _segment_intersection(toward, ray_end_rev, polyline[j], polyline[j + 1])
        if pt is not None:
            intersections.append(pt)
    if intersections:
        arr = np.asarray(intersections)
        dists_to_toward = np.linalg.norm(arr - toward, axis=1)
        return arr[int(np.argmin(dists_to_toward))]

    return _nearest_boundary_point(toward, polyline)


def _polygon_contour_crossings(
    polygon: np.ndarray,
    contour: np.ndarray,
    y_center: Optional[float] = None,
    y_band: float = 80.0,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Intersect the polygon's bottom edge with the full sail contour.

    Returns the two crossings farthest apart within ``y_center ± y_band``
    (the stripe's local y-range). Fallback used when the classified
    luff/leech polyline doesn't reach the stripe.
    """
    bottom = _polygon_bottom_edge(polygon)
    if len(bottom) < 2 or len(contour) < 2:
        return None, None

    hits: list[np.ndarray] = []
    contour_close = (
        contour
        if np.allclose(contour[0], contour[-1])
        else np.vstack([contour, contour[0:1]])
    )
    for i in range(len(bottom) - 1):
        for j in range(len(contour_close) - 1):
            pt = _segment_intersection(
                bottom[i], bottom[i + 1],
                contour_close[j], contour_close[j + 1],
            )
            if pt is not None:
                if y_center is None or abs(pt[1] - y_center) <= y_band:
                    hits.append(pt)

    if len(hits) < 2:
        return (hits[0] if hits else None), None

    arr = np.asarray(hits)
    # Pick the two intersections farthest apart (x-wise) — one per side.
    best_i, best_j, best_d = 0, 0, -1.0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            d = abs(float(arr[i, 0] - arr[j, 0]))
            if d > best_d:
                best_d = d
                best_i, best_j = i, j
    return arr[best_i], arr[best_j]


def _polygon_bottom_edge(polygon: np.ndarray, num_bins: int = 80) -> np.ndarray:
    """Return the lower envelope (max-y per x bin) of a stripe polygon.

    Constructs a discrete lower envelope by binning polygon points by x
    and keeping the maximum-y point in each non-empty bin. Returns a
    left-to-right ordered polyline. This is the curve that hugs the
    deeper (high-y) side of a thin stripe polygon.
    """
    if len(polygon) < 3:
        return polygon
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    if x_max - x_min < 1e-6:
        return polygon
    edges = np.linspace(x_min, x_max, num_bins + 1)
    bins = np.digitize(xs, edges) - 1
    bins = np.clip(bins, 0, num_bins - 1)

    envelope: list[np.ndarray] = []
    for b in range(num_bins):
        mask = bins == b
        if not np.any(mask):
            continue
        idx = int(np.argmax(ys[mask]))
        candidates = np.where(mask)[0]
        envelope.append(polygon[candidates[idx]])

    if len(envelope) < 2:
        return polygon
    arr = np.asarray(envelope)
    return arr[np.argsort(arr[:, 0])]


def polygon_bottom_polyline_intersection(
    polygon: np.ndarray,
    polyline: np.ndarray,
    toward: np.ndarray,
) -> Optional[np.ndarray]:
    """Intersect the polygon's bottom edge with a sail polyline.

    For each segment of the bottom chain and each segment of the
    polyline, compute the line-segment intersection. Returns the
    intersection closest to ``toward`` (the kp-based outer endpoint
    used to disambiguate luff-side vs leech-side).
    """
    bottom = _polygon_bottom_edge(polygon)
    if len(bottom) < 2 or len(polyline) < 2:
        return None

    intersections: list[np.ndarray] = []
    for i in range(len(bottom) - 1):
        for j in range(len(polyline) - 1):
            pt = _segment_intersection(
                bottom[i], bottom[i + 1], polyline[j], polyline[j + 1]
            )
            if pt is not None:
                intersections.append(pt)

    if not intersections:
        return None
    arr = np.asarray(intersections)
    dists = np.linalg.norm(arr - toward, axis=1)
    return arr[int(np.argmin(dists))]


def _polygon_polyline_intersection(
    polygon: np.ndarray,
    polyline: np.ndarray,
    centerline_ref: np.ndarray,
) -> Optional[np.ndarray]:
    """Return the polygon×polyline intersection closest to ``centerline_ref``.

    Enumerates every edge of the (closed) polygon and every segment of
    the polyline, gathers all intersection points, and picks the one
    minimising euclidean distance to the stripe's centerline midpoint —
    which, for a horizontal stripe, naturally lies near the real
    luff/leech crossing.

    Returns ``None`` if the polygon and polyline do not intersect.
    """
    if len(polygon) < 3 or len(polyline) < 2:
        return None

    # Close the polygon (if not already closed)
    poly = polygon
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0:1]])

    intersections: list[np.ndarray] = []
    for i in range(len(poly) - 1):
        p1 = poly[i]
        p2 = poly[i + 1]
        for j in range(len(polyline) - 1):
            q1 = polyline[j]
            q2 = polyline[j + 1]
            pt = _segment_intersection(p1, p2, q1, q2)
            if pt is not None:
                intersections.append(pt)

    if not intersections:
        return None

    arr = np.asarray(intersections)
    dists = np.linalg.norm(arr - centerline_ref, axis=1)
    return arr[int(np.argmin(dists))]


def _line_polyline_nearest_intersection(
    line_start: np.ndarray,
    line_end: np.ndarray,
    polyline: np.ndarray,
    near_point: np.ndarray
) -> Optional[np.ndarray]:
    """Find where a line segment intersects a polyline, closest to near_point."""
    intersections = []

    for i in range(len(polyline) - 1):
        p1, p2 = polyline[i], polyline[i + 1]
        pt = _segment_intersection(line_start, line_end, p1, p2)
        if pt is not None:
            intersections.append(pt)

    if not intersections:
        return None

    return min(intersections, key=lambda pt: np.linalg.norm(pt - near_point))


def _segment_intersection(
    p1: np.ndarray, p2: np.ndarray,
    q1: np.ndarray, q2: np.ndarray
) -> Optional[np.ndarray]:
    """Find intersection point between two line segments (parametric form)."""
    r = p2 - p1
    s = q2 - q1
    qp = q1 - p1

    # 2D cross product: r[0]*s[1] - r[1]*s[0]
    cross_rs = float(r[0] * s[1] - r[1] * s[0])
    if abs(cross_rs) < 1e-10:
        return None

    t = float(qp[0] * s[1] - qp[1] * s[0]) / cross_rs
    u = float(qp[0] * r[1] - qp[1] * r[0]) / cross_rs

    if 0 <= t <= 1 and 0 <= u <= 1:
        return p1 + t * r
    return None


def _horizontal_polyline_intersection(
    y_val: float,
    polyline: np.ndarray
) -> Optional[np.ndarray]:
    """Find where a horizontal line at y=y_val crosses the polyline."""
    intersections = []
    for i in range(len(polyline) - 1):
        p1, p2 = polyline[i], polyline[i + 1]
        y1, y2 = p1[1], p2[1]

        if (y1 <= y_val <= y2) or (y2 <= y_val <= y1):
            if abs(y2 - y1) < 1e-6:
                intersections.append((p1 + p2) / 2.0)
            else:
                t = (y_val - y1) / (y2 - y1)
                x_val = p1[0] + t * (p2[0] - p1[0])
                intersections.append(np.array([x_val, y_val]))

    return intersections[0] if intersections else None


def _nearest_boundary_point(
    point: np.ndarray,
    polyline: np.ndarray
) -> Optional[np.ndarray]:
    """Find the nearest point on polyline to the given point."""
    if len(polyline) == 0:
        return None
    dists = np.sqrt(np.sum((polyline - point) ** 2, axis=1))
    idx = np.argmin(dists)
    return polyline[idx].copy()


def validate_stripe_coverage(
    luff_endpoint: np.ndarray,
    leech_endpoint: np.ndarray,
    sail_boundary: SailBoundary,
    tolerance: float = 5.0
) -> bool:
    """Validate that a stripe spans from luff to leech with sufficient coverage.

    Args:
        luff_endpoint: Endpoint on luff side
        leech_endpoint: Endpoint on leech side
        sail_boundary: Sail boundary for validation
        tolerance: Distance tolerance in pixels

    Returns:
        True if stripe is valid (endpoints on boundaries, sufficient span)
    """
    # Check that luff_endpoint is on luff_polyline
    on_luff = point_on_polyline(
        luff_endpoint,
        sail_boundary.luff_polyline,
        tolerance=tolerance
    )

    if not on_luff:
        return False

    # Check that leech_endpoint is on leech_polyline
    on_leech = point_on_polyline(
        leech_endpoint,
        sail_boundary.leech_polyline,
        tolerance=tolerance
    )

    if not on_leech:
        return False

    # Check that chord spans at least 60% of sail width at that height
    chord_length = np.linalg.norm(leech_endpoint - luff_endpoint)

    # Estimate sail width at the stripe's height
    # Use the average y-coordinate of the endpoints
    stripe_y = (luff_endpoint[1] + leech_endpoint[1]) / 2.0

    # Find sail width at this height from tack/clew distance
    tack = sail_boundary.tack_point
    clew = sail_boundary.clew_point
    head = sail_boundary.head_point

    # Simple linear interpolation of width based on height
    # Width decreases linearly from foot to head
    foot_width = np.linalg.norm(clew - tack)
    head_y = head[1]
    foot_y = max(tack[1], clew[1])

    if abs(foot_y - head_y) < 1e-6:
        # Degenerate sail
        expected_width = foot_width
    else:
        # Interpolate width based on height
        height_ratio = (stripe_y - head_y) / (foot_y - head_y)
        height_ratio = np.clip(height_ratio, 0.0, 1.0)
        expected_width = foot_width * height_ratio

    # Require chord to be at least 30% of expected width
    min_required_width = 0.3 * expected_width

    if chord_length < min_required_width:
        return False

    return True


def detect_endpoints_model(
    image: np.ndarray,
    config: Optional[dict],
) -> np.ndarray:
    """Run the trained YOLO endpoint-pose model and return an (N, 3) array.

    Columns are ``[x, y, conf]``. Returns an empty (0, 3) array if the model
    is unavailable or produces no detections.
    """
    if config is None:
        config = {}

    model_path = config.get('endpoint_model_path', 'stripe_endpoints_v1.pt')
    if not Path(model_path).exists():
        fallback = config.get('endpoint_model_fallback', 'stripe_endpoints_v1.pt')
        if Path(fallback).exists():
            model_path = fallback
        else:
            return np.zeros((0, 3), dtype=np.float32)

    try:
        from ultralytics import YOLO
    except ImportError:
        return np.zeros((0, 3), dtype=np.float32)

    min_conf = float(config.get('endpoint_min_confidence', 0.1))
    imgsz = int(config.get('endpoint_imgsz', 1280))
    model = YOLO(model_path)
    results = model(image, conf=min_conf, verbose=False, imgsz=imgsz)
    r = results[0]

    if r.keypoints is None or r.keypoints.xy is None or len(r.keypoints.xy) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    xy = r.keypoints.xy.cpu().numpy()  # (N, 1, 2)
    if r.boxes is not None and r.boxes.conf is not None:
        confs = r.boxes.conf.cpu().numpy()
    else:
        confs = np.ones(len(xy), dtype=np.float32)

    rows = []
    for i in range(len(xy)):
        x, y = float(xy[i, 0, 0]), float(xy[i, 0, 1])
        if x == 0 and y == 0:
            continue
        rows.append([x, y, float(confs[i])])

    if not rows:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


_CROP_REFINER_CACHE: dict = {}


def _get_crop_refiner(model_path: str):
    if model_path in _CROP_REFINER_CACHE:
        return _CROP_REFINER_CACHE[model_path]
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    if not Path(model_path).exists():
        return None
    model = YOLO(model_path)
    _CROP_REFINER_CACHE[model_path] = model
    return model


def refine_endpoint_with_crop_model(
    geometric_endpoint: np.ndarray,
    image: np.ndarray,
    model_path: str,
    crop_size: int = 256,
    imgsz: int = 256,
    min_confidence: float = 0.1,
    max_offset_px: float = 60.0,
) -> np.ndarray:
    """Stage-2 endpoint refinement by YOLO-pose on a crop.

    Crops a ``crop_size × crop_size`` window centred on the current
    endpoint, runs a specialist pose model, takes the highest-confidence
    keypoint prediction inside the crop, and projects it back to image
    coordinates. If the refinement would move the endpoint more than
    ``max_offset_px`` it's rejected (likely picked a different stripe's
    endpoint) and the original point is kept.
    """
    model = _get_crop_refiner(model_path)
    if model is None:
        return geometric_endpoint

    h, w = image.shape[:2]
    half = crop_size // 2
    cx = int(round(geometric_endpoint[0]))
    cy = int(round(geometric_endpoint[1]))
    tl_x = max(0, min(w - crop_size, cx - half))
    tl_y = max(0, min(h - crop_size, cy - half))
    crop = image[tl_y:tl_y + crop_size, tl_x:tl_x + crop_size]
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        padded = np.zeros((crop_size, crop_size, 3), dtype=image.dtype)
        padded[: crop.shape[0], : crop.shape[1]] = crop
        crop = padded

    try:
        results = model(crop, conf=min_confidence, verbose=False, imgsz=imgsz)
    except Exception:
        return geometric_endpoint
    r = results[0]
    if r.keypoints is None or r.keypoints.xy is None or len(r.keypoints.xy) == 0:
        return geometric_endpoint

    xy = r.keypoints.xy.cpu().numpy()  # (N, 1, 2)
    confs = (
        r.boxes.conf.cpu().numpy()
        if r.boxes is not None and r.boxes.conf is not None
        else np.ones(len(xy))
    )
    best_idx = int(np.argmax(confs))
    kp_x, kp_y = float(xy[best_idx, 0, 0]), float(xy[best_idx, 0, 1])
    if kp_x == 0 and kp_y == 0:
        return geometric_endpoint

    refined = np.array([tl_x + kp_x, tl_y + kp_y], dtype=np.float32)
    if np.linalg.norm(refined - geometric_endpoint) > max_offset_px:
        return geometric_endpoint
    return refined


def refine_endpoint_with_model(
    geometric_endpoint: np.ndarray,
    polyline: np.ndarray,
    model_endpoints: np.ndarray,
    max_distance: float = 60.0,
    max_polyline_offset: float = 25.0,
    max_y_gap: float = 60.0,
) -> np.ndarray:
    """Snap a geometric endpoint to the nearest model-detected endpoint.

    Falls back to the original geometric point when no model endpoint is
    close enough. The model endpoint is then projected onto the given
    polyline so the result is guaranteed to sit on the sail boundary.

    ``max_y_gap`` is a separate tighter constraint on the vertical gap —
    endpoints from a different stripe on the same sail may be close in
    euclidean distance but far in y, so we reject those.

    Args:
        geometric_endpoint: Current endpoint (x, y) from geometric snap.
        polyline: Luff or leech polyline (M, 2).
        model_endpoints: Endpoint model output (N, 3): x, y, confidence.
        max_distance: Max euclidean distance (px) from geometric endpoint.
        max_polyline_offset: Max distance (px) from polyline to accept.
        max_y_gap: Max vertical distance to stay on the same stripe.

    Returns:
        Refined endpoint (2,).
    """
    if model_endpoints.shape[0] == 0 or len(polyline) == 0:
        return geometric_endpoint

    dxy = model_endpoints[:, :2] - geometric_endpoint
    dists = np.linalg.norm(dxy, axis=1)
    dy = np.abs(dxy[:, 1])
    candidate_mask = (dists <= max_distance) & (dy <= max_y_gap)
    if not np.any(candidate_mask):
        return geometric_endpoint

    order = np.argsort(dists[candidate_mask])
    candidate_indices = np.where(candidate_mask)[0][order]

    for idx in candidate_indices:
        cand = model_endpoints[idx, :2]
        snapped = _nearest_boundary_point(cand, polyline)
        if snapped is None:
            continue
        if np.linalg.norm(snapped - cand) <= max_polyline_offset:
            return snapped

    return geometric_endpoint


def _polyline_at_x(polyline: np.ndarray, x_target: float) -> Optional[np.ndarray]:
    """Interpolate a polyline to find the (x, y) point at a given x.

    Returns the closest in-range point if x_target falls within the
    polyline's x extent, else the nearest endpoint.
    """
    if polyline is None or len(polyline) < 2:
        return None
    order = np.argsort(polyline[:, 0])
    xs = polyline[order, 0]
    ys = polyline[order, 1]
    if x_target <= xs[0]:
        return polyline[order[0]]
    if x_target >= xs[-1]:
        return polyline[order[-1]]
    y_target = float(np.interp(x_target, xs, ys))
    return np.array([x_target, y_target], dtype=np.float64)


def find_endpoints_for_bbox(
    bbox: Tuple[float, float, float, float],
    sail_boundary: SailBoundary,
    detection: Optional[StripeDetection] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Bbox-anchored endpoint finder.

    Places the luff endpoint on the luff polyline at the bbox's luff-side
    x coordinate, and the leech endpoint on the leech polyline at the
    bbox's leech-side x. Which side is "luff" vs "leech" is decided by
    comparing the bbox's centre x to the luff polyline centroid — whichever
    end of the bbox is closer to the luff centroid is the luff side.

    Falls back to the detection-based logic if the polylines don't reach
    the bbox's x-range.
    """
    x1, y1, x2, y2 = bbox
    if sail_boundary.luff_polyline is None or len(sail_boundary.luff_polyline) < 2:
        return None
    if sail_boundary.leech_polyline is None or len(sail_boundary.leech_polyline) < 2:
        return None

    luff_mean_x = float(np.mean(sail_boundary.luff_polyline[:, 0]))
    leech_mean_x = float(np.mean(sail_boundary.leech_polyline[:, 0]))

    # Which side of the bbox is closest to the luff? We pick the bbox edge
    # (left or right) whose x is nearest the luff centroid.
    dist_left_luff = abs(x1 - luff_mean_x)
    dist_right_luff = abs(x2 - luff_mean_x)
    if dist_left_luff <= dist_right_luff:
        x_luff_target = float(x1)   # left edge of bbox → luff
        x_leech_target = float(x2)  # right edge → leech
    else:
        x_luff_target = float(x2)
        x_leech_target = float(x1)

    luff_ep = _polyline_at_x(sail_boundary.luff_polyline, x_luff_target)
    leech_ep = _polyline_at_x(sail_boundary.leech_polyline, x_leech_target)
    if luff_ep is None or leech_ep is None:
        return None

    # Clamp y of the endpoints to the bbox's y-range (with 15 px padding)
    # so they don't jump far vertically on noisy polylines.
    pad_y = max(15.0, 0.4 * (y2 - y1))
    y_lo = y1 - pad_y
    y_hi = y2 + pad_y
    luff_ep = np.array([luff_ep[0], np.clip(luff_ep[1], y_lo, y_hi)])
    leech_ep = np.array([leech_ep[0], np.clip(leech_ep[1], y_lo, y_hi)])

    return luff_ep, leech_ep


def endpoints_from_detection(
    detection: StripeDetection,
    bbox: Tuple[float, float, float, float],
    sail_boundary: Optional[SailBoundary] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Use the detection's own extremes as endpoints (no polyline snap).

    The NACA fit stretches when the endpoints are pushed out to the
    luff / leech polylines while the detected keypoints only cover the
    middle of the chord. Using the detection's own leftmost / rightmost
    points keeps the fit tight against the data — the "raw keypoints"
    view in the report looks great; we want that quality in Stage 4.

    Strategy:
      * If ``detection.polygon`` is available, use the polygon-bottom
        edge's leftmost & rightmost points (these mark the true stripe
        start/end as seen by v7).
      * Else fall back to detection.points[0] / detection.points[-1]
        (which are already sorted by x).
      * "Luff" vs "leech" is decided by which side of the bbox is closer
        to the luff polyline centroid (so the ordering matches the
        report's convention).
    """
    pts = detection.points
    if pts is None or len(pts) < 2:
        return None

    # When the detection already has dense interior points (e.g. from a
    # seg-on-crop skeleton or a fused polygon-bottom + kp), those
    # extremes ARE the physical stripe ends — use them directly and
    # don't re-derive from the polygon (which would add the band margin
    # back in).
    if len(pts) >= 6:
        ep_a = pts[np.argmin(pts[:, 0])]
        ep_b = pts[np.argmax(pts[:, 0])]
    elif getattr(detection, "polygon", None) is not None and len(detection.polygon) >= 3:
        try:
            from src.polygon_fusion import polygon_bottom_edge
            img_shape = (
                int(sail_boundary.mask.shape[0]) if sail_boundary is not None else 10000,
                int(sail_boundary.mask.shape[1]) if sail_boundary is not None else 10000,
            )
            bottom = polygon_bottom_edge(detection.polygon, img_shape, n_samples=24)
            ep_a = bottom[0]
            ep_b = bottom[-1]
        except Exception:
            ep_a = pts[np.argmin(pts[:, 0])]
            ep_b = pts[np.argmax(pts[:, 0])]
    else:
        ep_a = pts[np.argmin(pts[:, 0])]
        ep_b = pts[np.argmax(pts[:, 0])]

    # Decide luff vs leech by proximity to the luff polyline centroid
    # (fallback: bbox centre x).
    if sail_boundary is not None and sail_boundary.luff_polyline is not None:
        luff_cx = float(np.mean(sail_boundary.luff_polyline[:, 0]))
    else:
        luff_cx = 0.5 * (bbox[0] + bbox[2])

    if abs(ep_a[0] - luff_cx) <= abs(ep_b[0] - luff_cx):
        luff_ep = np.asarray(ep_a, dtype=np.float64)
        leech_ep = np.asarray(ep_b, dtype=np.float64)
    else:
        luff_ep = np.asarray(ep_b, dtype=np.float64)
        leech_ep = np.asarray(ep_a, dtype=np.float64)

    return luff_ep, leech_ep


def dedup_stripe_endpoints(
    stripes: List[Tuple[StripeDetection, np.ndarray, np.ndarray]],
    min_separation_px: float = 15.0,
) -> List[Tuple[StripeDetection, np.ndarray, np.ndarray]]:
    """Ensure no two stripes share endpoints within ``min_separation_px``.

    If two stripes' luff endpoints (or leech endpoints) are closer than
    the threshold, nudge the shorter one slightly upward / downward along
    the polyline so the points become unique.
    """
    if len(stripes) <= 1:
        return stripes
    out = [(d, luff.copy(), leech.copy()) for d, luff, leech in stripes]
    for attr in (1, 2):  # 1 = luff, 2 = leech
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                pi, pj = out[i][attr], out[j][attr]
                if np.linalg.norm(pi - pj) < min_separation_px:
                    # Nudge the one whose y is lower (bottom stripe) downward
                    if pi[1] < pj[1]:
                        pj[1] += min_separation_px
                    else:
                        pi[1] += min_separation_px
    return out


def process_all_stripes(
    detections: List[StripeDetection],
    sail_boundary: SailBoundary,
    config: Optional[dict] = None,
    image: Optional[np.ndarray] = None,
) -> List[Tuple[StripeDetection, np.ndarray, np.ndarray]]:
    """Process all stripe detections and return only valid ones with endpoints.

    Filters out stripes that:
    - Cannot be projected to both sail edges
    - Have chord angle too steep (> 35 degrees from horizontal)

    Args:
        detections: List of stripe detections
        sail_boundary: Sail boundary for endpoint computation
        config: Optional configuration dict

    Returns:
        List of tuples: (detection, luff_endpoint, leech_endpoint)
        Only includes stripes that pass all validation checks
    """
    max_chord_angle = 60.0  # degrees from horizontal (relaxed from 35)
    valid_stripes = []

    # Feature toggle: polygon-bottom × polyline intersection (off by default)
    find_endpoints._use_polygon_bottom_intersection = bool(
        (config or {}).get('use_polygon_bottom_intersection', False)
    )
    # Feature toggle: ML two-stage endpoint refinement (off by default)
    _enable_crop_refiner = bool(
        (config or {}).get('endpoint_crop_refiner_enabled', False)
        and image is not None
    )

    use_endpoint_refiner = bool(
        (config or {}).get('endpoint_refiner_enabled', False) and image is not None
    )
    model_endpoints = (
        detect_endpoints_model(image, config) if use_endpoint_refiner
        else np.zeros((0, 3), dtype=np.float32)
    )

    for detection in detections:
        # Find endpoints
        endpoints = find_endpoints(detection, sail_boundary)

        if endpoints is None:
            continue

        luff_endpoint, leech_endpoint = endpoints

        # Stage-2: crop-specialist refinement (runs AFTER geometric snap)
        if _enable_crop_refiner:
            crop_model = str((config or {}).get(
                'endpoint_crop_model_path', 'stripe_endpoint_crop_26x.pt'
            ))
            crop_size = int((config or {}).get('endpoint_crop_size', 256))
            max_off = float((config or {}).get(
                'endpoint_crop_max_offset_px', 60.0
            ))
            luff_endpoint = refine_endpoint_with_crop_model(
                luff_endpoint, image, crop_model,
                crop_size=crop_size, imgsz=crop_size,
                max_offset_px=max_off,
            )
            leech_endpoint = refine_endpoint_with_crop_model(
                leech_endpoint, image, crop_model,
                crop_size=crop_size, imgsz=crop_size,
                max_offset_px=max_off,
            )

        if model_endpoints.shape[0] > 0:
            max_d = float((config or {}).get(
                'endpoint_refiner_max_distance_px', 60.0
            ))
            max_y = float((config or {}).get(
                'endpoint_refiner_max_y_gap_px', 40.0
            ))
            max_poly = float((config or {}).get(
                'endpoint_refiner_max_polyline_offset_px', 25.0
            ))
            luff_endpoint = refine_endpoint_with_model(
                luff_endpoint,
                sail_boundary.luff_polyline,
                model_endpoints,
                max_distance=max_d,
                max_polyline_offset=max_poly,
                max_y_gap=max_y,
            )
            leech_endpoint = refine_endpoint_with_model(
                leech_endpoint,
                sail_boundary.leech_polyline,
                model_endpoints,
                max_distance=max_d,
                max_polyline_offset=max_poly,
                max_y_gap=max_y,
            )

        # Reject degenerate chords (both endpoints at same location)
        chord_vec = leech_endpoint - luff_endpoint
        chord_length = np.linalg.norm(chord_vec)
        if chord_length < 5.0:
            continue

        # Reject stripes where chord is too steep
        chord_angle = abs(np.degrees(np.arctan2(chord_vec[1], chord_vec[0])))
        if chord_angle > 90:
            chord_angle = 180 - chord_angle
        if chord_angle > max_chord_angle:
            continue

        # Validate endpoints are within image bounds
        h, w = sail_boundary.mask.shape
        if not (0 <= luff_endpoint[0] < w and 0 <= luff_endpoint[1] < h):
            continue
        if not (0 <= leech_endpoint[0] < w and 0 <= leech_endpoint[1] < h):
            continue

        # Check endpoints are near the sail mask (relaxed from 10px to 50px)
        if not _point_near_mask(luff_endpoint, sail_boundary.mask, radius=50):
            continue
        if not _point_near_mask(leech_endpoint, sail_boundary.mask, radius=50):
            continue

        valid_stripes.append((detection, luff_endpoint, leech_endpoint))

    # Deduplicate stripes with near-identical endpoints
    valid_stripes = _dedup_stripes(valid_stripes, endpoint_tolerance=20.0)

    return valid_stripes


def _point_near_mask(
    point: np.ndarray,
    mask: np.ndarray,
    radius: int = 10
) -> bool:
    """Check if a point is within radius pixels of any True pixel in mask.

    Args:
        point: (x, y) coordinates
        mask: Binary mask (H, W) bool
        radius: Search radius in pixels

    Returns:
        True if any mask pixel within radius is True
    """
    h, w = mask.shape
    x, y = int(round(point[0])), int(round(point[1]))

    y_lo = max(0, y - radius)
    y_hi = min(h, y + radius + 1)
    x_lo = max(0, x - radius)
    x_hi = min(w, x + radius + 1)

    return np.any(mask[y_lo:y_hi, x_lo:x_hi])


def _dedup_stripes(
    valid_stripes: List[Tuple[StripeDetection, np.ndarray, np.ndarray]],
    endpoint_tolerance: float = 20.0
) -> List[Tuple[StripeDetection, np.ndarray, np.ndarray]]:
    """Remove duplicate stripes with near-identical endpoints.

    For each pair of stripes, if both luff AND leech endpoints are within
    endpoint_tolerance pixels, keep the one with higher detection confidence.

    Args:
        valid_stripes: List of (detection, luff_endpoint, leech_endpoint)
        endpoint_tolerance: Maximum distance in pixels to consider duplicates

    Returns:
        Deduplicated list
    """
    if len(valid_stripes) <= 1:
        return valid_stripes

    keep = [True] * len(valid_stripes)

    for i in range(len(valid_stripes)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(valid_stripes)):
            if not keep[j]:
                continue

            det_i, luff_i, leech_i = valid_stripes[i]
            det_j, luff_j, leech_j = valid_stripes[j]

            luff_dist = np.linalg.norm(luff_i - luff_j)
            leech_dist = np.linalg.norm(leech_i - leech_j)

            if luff_dist < endpoint_tolerance and leech_dist < endpoint_tolerance:
                # Duplicate — keep higher confidence
                if det_j.confidence > det_i.confidence:
                    keep[i] = False
                else:
                    keep[j] = False

    return [s for s, k in zip(valid_stripes, keep) if k]
