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


def smooth_polyline(
    polyline: np.ndarray, window: int = 9, polyorder: int = 3,
) -> np.ndarray:
    """Smooth a polyline with Savitzky-Golay so small kinks disappear.

    Operates separately on x and y. Window size and polynomial order are
    capped to the polyline length so short polylines are returned as-is.
    """
    if polyline is None or len(polyline) < 5:
        return polyline
    try:
        from scipy.signal import savgol_filter
    except Exception:
        return polyline
    n = len(polyline)
    win = min(window, n if n % 2 == 1 else n - 1)
    if win < 5:
        return polyline
    if win % 2 == 0:
        win -= 1
    po = min(polyorder, win - 1)
    sx = savgol_filter(polyline[:, 0].astype(np.float64), win, po)
    sy = savgol_filter(polyline[:, 1].astype(np.float64), win, po)
    return np.column_stack([sx, sy]).astype(polyline.dtype)


def cleanup_sail_mask(
    mask: np.ndarray,
    kernel_frac: float = 0.020,
    smooth_frac: float = 0.012,
) -> np.ndarray:
    """Close small bites, fill interior holes, and smooth zigzag-y mask
    boundaries from SAM2.

    Pipeline:
      1. Morphological close (kernel ~ ``kernel_frac`` of max(H, W)).
         Closes small bites and gaps.
      2. Hole fill via flood-fill of the exterior.
      3. Gaussian-blur-then-rethreshold (sigma ~ ``smooth_frac`` of
         max(H, W)). This rounds off jagged boundary pixels — the
         blue zigzag the user sees on the leech — without changing
         the global silhouette.
    """
    import cv2

    m = mask.astype(np.uint8)
    if m.sum() < 100:
        return mask
    H, W = m.shape
    k = max(5, int(round(kernel_frac * max(H, W))))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    inv = (closed == 0).astype(np.uint8) * 255
    border = np.full((H + 2, W + 2), 255, np.uint8)
    border[1:-1, 1:-1] = inv
    cv2.floodFill(border, None, (0, 0), 128)
    holes = (border[1:-1, 1:-1] == 255).astype(np.uint8)
    filled = (closed | holes).astype(np.uint8)

    # Boundary smoothing: blur as 0/255 then rethreshold. Removes
    # high-frequency zigzag without shifting the global mask area.
    sigma_px = max(2.0, smooth_frac * max(H, W))
    ksize = max(3, int(round(sigma_px * 4)))
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(
        (filled * 255).astype(np.uint8), (ksize, ksize), sigma_px,
    )
    smoothed = (blurred >= 128).astype(np.uint8)

    return smoothed.astype(bool) if mask.dtype == bool else smoothed


def position_corners_from_mask(
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick head, tack and clew from absolute mask geometry (post-rotation).

    Assumes the image has already been oriented so the head is at the top
    of the frame (use :func:`detect_sail_orientation` first). This avoids
    the curvature-peak labeling that misfires on cropped or partial sail
    masks.

    * ``head``  = topmost mask pixel (median x at min y)
    * ``tack``  = leftmost mask pixel within the bottom 30 % of mask rows
    * ``clew``  = rightmost mask pixel within the bottom 30 % of mask rows
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        H, W = mask.shape
        return (
            np.array([W / 2.0, 0.0], dtype=np.float64),
            np.array([0.0, H - 1.0], dtype=np.float64),
            np.array([W - 1.0, H - 1.0], dtype=np.float64),
        )

    y_min, y_max = int(ys.min()), int(ys.max())
    head_x = float(np.median(xs[ys == y_min]))
    head = np.array([head_x, float(y_min)], dtype=np.float64)

    band_top = y_min + int(0.7 * (y_max - y_min))
    band_mask = ys >= band_top
    if band_mask.sum() < 5:
        band_mask = ys >= (y_min + int(0.5 * (y_max - y_min)))
    band_xs = xs[band_mask]; band_ys = ys[band_mask]

    tack_idx = int(np.argmin(band_xs))
    clew_idx = int(np.argmax(band_xs))
    tack = np.array([float(band_xs[tack_idx]),
                       float(band_ys[tack_idx])], dtype=np.float64)
    clew = np.array([float(band_xs[clew_idx]),
                       float(band_ys[clew_idx])], dtype=np.float64)
    return head, tack, clew


def head_from_mask(mask: np.ndarray) -> np.ndarray:
    """Return the best estimate of the sail head from a binary mask.

    Strategy: simplify the contour to a triangle (Douglas-Peucker with
    large epsilon) and pick the vertex with the smallest interior angle
    that lies opposite the longest edge (the foot). This is the true
    apex of the triangular sail silhouette — the sharpest corner where
    luff and leech meet.

    Fallbacks (in order):
    1. Triangle-apex from simplified contour (primary).
    2. Curvature-peak in the upper 40% of the mask height.
    3. Topmost pixel (median-x at min-y).
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return np.array([mask.shape[1] / 2.0, 0.0], dtype=np.float64)
    H, W = mask.shape
    y_min = int(ys.min())
    xs_at_top = xs[ys == y_min]
    x_top = float(np.median(xs_at_top))
    topmost_pixel = np.array([x_top, float(y_min)], dtype=np.float64)

    head = _head_triangle_apex(mask)
    if head is not None:
        return head
    head = _head_curvature_upper_band(mask)
    if head is not None:
        return head
    return topmost_pixel


def _head_triangle_apex(mask: np.ndarray) -> Optional[np.ndarray]:
    """Simplify the sail-mask contour to a 3-vertex polygon and return
    the apex (vertex with smallest interior angle, opposite the longest
    edge = the foot).

    Uses Douglas-Peucker (cv2.approxPolyDP) with a decreasing epsilon
    until exactly 3 vertices remain. If that fails, picks the 3
    highest-curvature peaks from the contour.
    """
    try:
        import cv2
        from skimage import measure
    except Exception:
        return None

    H, W = mask.shape
    padded = np.zeros((H + 2, W + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask.astype(np.uint8)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return None
    # largest contour, (row, col) → (x, y), undo padding
    raw = max(contours, key=len)
    raw_xy = (np.fliplr(raw) - np.array([1.0, 1.0])).astype(np.float32)

    # Convert to cv2 format for approxPolyDP
    cv2_contour = raw_xy.reshape(-1, 1, 2)
    perimeter = cv2.arcLength(cv2_contour, closed=True)

    # Try decreasing epsilon to get a triangle
    tri: Optional[np.ndarray] = None
    for frac in [0.10, 0.07, 0.05, 0.04, 0.03, 0.02]:
        eps = frac * perimeter
        approx = cv2.approxPolyDP(cv2_contour, eps, closed=True)
        if len(approx) == 3:
            tri = approx.reshape(3, 2).astype(np.float64)
            break
        if len(approx) < 3:
            break

    if tri is None or len(tri) != 3:
        # Fallback: pick 3 highest-curvature peaks from the contour
        from src.utils.geometry import _compute_contour_curvature
        contour_xy = raw_xy.astype(np.float64)
        if len(contour_xy) < 9:
            return None
        curv = _compute_contour_curvature(contour_xy, k=7, smooth_sigma=3.0)
        # Find 3 peaks that are well-separated (> 5% of contour apart)
        min_dist = max(3, len(contour_xy) // 20)
        peaks = []
        used = np.zeros(len(curv), dtype=bool)
        for _ in range(3):
            idx = int(np.argmax(curv * (~used).astype(float)))
            peaks.append(idx)
            lo = max(0, idx - min_dist)
            hi = min(len(used), idx + min_dist + 1)
            used[lo:hi] = True
        if len(peaks) != 3:
            return None
        tri = contour_xy[np.array(peaks)]

    # Find the longest edge (= the foot), then pick the opposite vertex
    d01 = float(np.linalg.norm(tri[0] - tri[1]))
    d12 = float(np.linalg.norm(tri[1] - tri[2]))
    d20 = float(np.linalg.norm(tri[2] - tri[0]))
    longest = np.argmax([d01, d12, d20])
    # vertex opposite the longest edge
    apex_idx = (longest + 2) % 3  # edge 0→1 is opposite vertex 2, etc.
    # edge 0: verts 0,1 → opposite = 2; edge 1: verts 1,2 → opposite = 0; edge 2: verts 2,0 → opposite = 1
    apex_map = {0: 2, 1: 0, 2: 1}
    apex = tri[apex_map[longest]]

    # Sanity: apex should be in the upper half of the mask
    ys_all = np.where(mask > 0)[0]
    y_min = int(ys_all.min())
    y_max = int(ys_all.max())
    mid_y = (y_min + y_max) / 2.0
    if float(apex[1]) > mid_y:
        # Apex is in lower half — triangle may be inverted. Pick the
        # vertex with the smallest y (highest in image) as head.
        apex = tri[int(np.argmin(tri[:, 1]))]

    return np.array([float(apex[0]), float(apex[1])], dtype=np.float64)


def _head_curvature_upper_band(mask: np.ndarray) -> Optional[np.ndarray]:
    """Find the sail head as the highest-curvature vertex in the upper
    40 % of the mask's vertical extent.

    The sail head is where luff and leech meet — the sharpest corner of
    the triangular silhouette. It may not be at the very topmost row
    (e.g. when a nearly-horizontal leech or luff crosses the top of the
    frame before reaching the actual apex).

    Returns ``None`` if the contour is too short or no clear peak exists.
    """
    try:
        from skimage import measure
    except Exception:
        return None

    H, W = mask.shape
    padded = np.zeros((H + 2, W + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask.astype(np.uint8)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return None
    contour = max(contours, key=len)
    # skimage gives (row, col) — convert to (x, y) and undo padding offset
    contour = np.fliplr(contour) - np.array([1.0, 1.0])
    if len(contour) < 12:
        return None

    ys_all = np.where(mask > 0)[0]
    if len(ys_all) == 0:
        return None
    y_min = int(ys_all.min())
    y_max = int(ys_all.max())
    height = y_max - y_min
    if height < 1:
        return None

    # Restrict to points in the upper 40 % of the mask height
    upper_thresh = y_min + 0.40 * height
    upper_mask_idx = np.where(contour[:, 1] <= upper_thresh)[0]
    if len(upper_mask_idx) < 6:
        # Upper band too sparse — widen to top 60 %
        upper_thresh = y_min + 0.60 * height
        upper_mask_idx = np.where(contour[:, 1] <= upper_thresh)[0]
    if len(upper_mask_idx) < 4:
        return None

    # Compute turning angles on the full contour for smoothness, then
    # extract values at the upper-band indices.
    from src.utils.geometry import _compute_contour_curvature
    curv = _compute_contour_curvature(contour, k=7, smooth_sigma=3.0)

    upper_curv = curv[upper_mask_idx]
    best_local = int(np.argmax(upper_curv))
    best_idx = int(upper_mask_idx[best_local])
    pt = contour[best_idx]

    # Sanity: the chosen point must have non-trivial curvature (> 0.15 rad ~8°).
    # If the upper band is essentially flat (no corner), fall back.
    if upper_curv[best_local] < 0.15:
        return None

    return np.array([float(pt[0]), float(pt[1])], dtype=np.float64)


def _head_from_visible_ridge(mask: np.ndarray) -> Optional[np.ndarray]:
    """Find the sail head as the highest-curvature peak on the
    sail-vs-sky portion of the mask boundary (the "head ridge").

    Walks the contour, drops segments lying on the image border (those
    are framing artefacts, not real sail edges), then runs the standard
    open-contour curvature peak finder. Returns ``None`` on failure.
    """
    try:
        from skimage import measure
        from src.utils.geometry import _find_head_on_open_contour
    except Exception:
        return None
    H, W = mask.shape
    padded = np.zeros((H + 2, W + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask.astype(np.uint8)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return None
    contour = max(contours, key=len)
    contour = np.fliplr(contour) - np.array([1.0, 1.0])
    edge_tol = 2.0
    on_edge = (
        (contour[:, 0] <= edge_tol)
        | (contour[:, 0] >= W - 1 - edge_tol)
        | (contour[:, 1] <= edge_tol)
        | (contour[:, 1] >= H - 1 - edge_tol)
    )
    if on_edge.all() or not on_edge.any():
        return None
    first_interior = int(np.argmax(~on_edge))
    contour = np.roll(contour, -first_interior, axis=0)
    on_edge = np.roll(on_edge, -first_interior)
    segments: list[np.ndarray] = []
    i = 0
    n = len(contour)
    while i < n:
        if on_edge[i]:
            i += 1
            continue
        j = i
        while j < n and not on_edge[j]:
            j += 1
        segments.append(contour[i:j])
        i = j
    if not segments:
        return None
    ridge = max(segments, key=len)
    if len(ridge) < 8:
        return None
    try:
        head_idx = _find_head_on_open_contour(ridge)
    except Exception:
        return None
    pt = ridge[head_idx]
    return np.array([float(pt[0]), float(pt[1])], dtype=np.float64)


def detect_sail_orientation(mask: np.ndarray) -> int:
    """Return number of 90° CCW rotations needed to put the head at top.

    Returns 0, 1, 2 or 3. Mapping to ``cv2.rotate``:

    * 0 → no rotation
    * 1 → ``cv2.ROTATE_90_COUNTERCLOCKWISE`` (head was on the right)
    * 2 → ``cv2.ROTATE_180`` (head was at the bottom)
    * 3 → ``cv2.ROTATE_90_CLOCKWISE`` (head was on the left)

    The head of a sail is the *narrowest* part of the mask, the foot the
    widest. We pick the rotation that maximises ``bottom_width /
    top_width`` so the silhouette tapers from a thin top to a wide
    bottom.
    """
    import cv2

    if mask is None:
        return 0
    m = mask.astype(np.uint8)
    if m.sum() < 100:
        return 0

    rotations = {
        0: m,
        1: cv2.rotate(m, cv2.ROTATE_90_COUNTERCLOCKWISE),
        2: cv2.rotate(m, cv2.ROTATE_180),
        3: cv2.rotate(m, cv2.ROTATE_90_CLOCKWISE),
    }

    best_k = 0
    best_score = -np.inf
    for k, rm in rotations.items():
        # Crop to the mask's vertical bounding box so padding above/below
        # doesn't dilute the width profile.
        ys = np.where(rm.any(axis=1))[0]
        if len(ys) < 4:
            continue
        y0, y1 = int(ys[0]), int(ys[-1]) + 1
        crop = rm[y0:y1]
        h = crop.shape[0]
        if h < 4:
            continue
        widths = crop.sum(axis=1).astype(np.float64)
        # Top 25 % vs bottom 25 % — average horizontal extent.
        top_band = max(1, h // 4)
        top_w = float(widths[:top_band].mean())
        bot_w = float(widths[-top_band:].mean())
        score = bot_w / max(top_w, 1.0)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


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
