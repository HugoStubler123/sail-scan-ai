"""Fuse Roboflow v7 polygons with keypoint detections.

Insight from domain feedback: the v7 segmentation model outputs a polygon
that wraps the stripe with a margin; the actual drawn stripe lives at the
**bottom** of the polygon (largest y in image coordinates), not at the
polygon's skeleton. We therefore:

1. Rasterise each polygon and take ``max-y per column`` to recover the
   stripe's true lower edge.
2. Match keypoint-model keypoints that fall within the polygon's bbox
   to their nearest point on that bottom edge. Matched kpts become
   high-confidence anchors.
3. Augment with curvature-biased extra samples drawn from the bottom
   edge itself, so curved stripes get more support points than straight
   ones.
4. Clamp the point set between the polygon's left / right extremes so
   downstream endpoint snapping stays honest.

Returns :class:`~src.types.StripeDetection` objects with ``polygon``
preserved for the endpoint stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.types import StripeDetection


@dataclass
class FusionDiagnostics:
    """Artefacts produced by :func:`fuse_polygon_with_keypoints` for reports."""

    bottom_edge: np.ndarray            # (N, 2) lower contour of the polygon
    curvature_samples: np.ndarray      # (K, 2) extra points at high curvature
    matched_kp: np.ndarray             # (M, 2) kp points that matched
    unmatched_kp: np.ndarray           # (U, 2) kp points that did not match
    match_dists: np.ndarray            # (M,) nearest-distance of each match
    bbox: Tuple[int, int, int, int]    # polygon bbox (x1, y1, x2, y2)
    final_points: np.ndarray           # (P, 2) points handed to the fitter
    polygon: np.ndarray                # (V, 2) source polygon


def polygon_bbox(polygon: np.ndarray) -> Tuple[int, int, int, int]:
    x1, y1 = polygon.min(axis=0)
    x2, y2 = polygon.max(axis=0)
    return int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))


def polygon_bottom_edge(
    polygon: np.ndarray,
    image_shape: Tuple[int, int],
    n_samples: int = 24,
    smooth_sigma: float = 1.5,
) -> np.ndarray:
    """Return the lower (max-y per column) edge of a polygon, resampled.

    Args:
        polygon: (V, 2) polygon vertices, float.
        image_shape: ``(H, W)`` shape of the source image (bounds the
            rasterisation).
        n_samples: Number of points to return (evenly spaced in x).
        smooth_sigma: Gaussian-1D smoothing stddev applied to the y
            profile before resampling. Keeps the curve from wobbling on
            individual pixel jags.

    Returns:
        (n_samples, 2) array sorted by x.
    """
    h, w = image_shape
    poly_int = np.round(polygon).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_int], 255)

    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return polygon.astype(np.float32)

    # Per-column max y (bottom edge in image coords)
    x_min, x_max = int(xs.min()), int(xs.max())
    col_max_y = np.full(x_max - x_min + 1, np.nan, dtype=np.float32)
    order = np.argsort(xs)
    xs_sorted, ys_sorted = xs[order], ys[order]
    cur_x = xs_sorted[0]
    cur_max = ys_sorted[0]
    for i in range(1, len(xs_sorted)):
        xi = xs_sorted[i]
        yi = ys_sorted[i]
        if xi == cur_x:
            if yi > cur_max:
                cur_max = yi
            continue
        col_max_y[cur_x - x_min] = cur_max
        cur_x = xi
        cur_max = yi
    col_max_y[cur_x - x_min] = cur_max

    # Fill any NaN gaps by linear interpolation
    valid = ~np.isnan(col_max_y)
    if valid.sum() < 2:
        return polygon.astype(np.float32)
    idx = np.arange(len(col_max_y))
    col_max_y = np.interp(idx, idx[valid], col_max_y[valid]).astype(np.float32)

    # Smooth with edge-reflected padding so the boundary isn't pulled
    # toward zero.
    if smooth_sigma > 0 and len(col_max_y) > 5:
        radius = max(1, int(round(3 * smooth_sigma)))
        kx = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (kx / smooth_sigma) ** 2)
        kernel /= kernel.sum()
        padded = np.concatenate([
            np.full(radius, col_max_y[0]),
            col_max_y,
            np.full(radius, col_max_y[-1]),
        ])
        col_max_y = np.convolve(padded, kernel, mode="valid").astype(np.float32)

    # Resample evenly along x
    x_sample = np.linspace(x_min, x_max, n_samples).astype(np.float32)
    y_sample = np.interp(
        x_sample,
        np.arange(x_min, x_max + 1).astype(np.float32),
        col_max_y.astype(np.float32),
    )
    return np.column_stack([x_sample, y_sample]).astype(np.float32)


def _curvature(pts: np.ndarray) -> np.ndarray:
    """Discrete absolute curvature along a polyline (len == len(pts))."""
    if len(pts) < 3:
        return np.zeros(len(pts), dtype=np.float32)
    d1 = np.gradient(pts[:, 0])
    d2 = np.gradient(pts[:, 1])
    dd1 = np.gradient(d1)
    dd2 = np.gradient(d2)
    denom = (d1 * d1 + d2 * d2) ** 1.5 + 1e-6
    return np.abs(d1 * dd2 - d2 * dd1) / denom


def curvature_weighted_resample(
    edge: np.ndarray,
    n_points: int = 18,
    min_separation_px: float = 8.0,
) -> np.ndarray:
    """Resample ``edge`` denser where absolute curvature is higher.

    Args:
        edge: (N, 2) ordered polyline.
        n_points: Desired number of output samples.
        min_separation_px: Lower bound on the spacing between samples.
    """
    if len(edge) < 3:
        return edge.astype(np.float32)

    # Arc-length parameterisation
    seg = np.linalg.norm(np.diff(edge, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    if arc[-1] < 1.0:
        return edge.astype(np.float32)

    # Density ~ 1 + curvature (normalised), integrated to a CDF
    kappa = _curvature(edge)
    kappa /= (kappa.max() + 1e-6)
    density = 1.0 + 3.0 * kappa  # boost curved areas up to 4×
    density_cdf = np.concatenate([[0.0], np.cumsum(density[:-1] * seg)])
    total = density_cdf[-1]
    if total <= 0:
        return edge.astype(np.float32)
    density_cdf /= total

    # Equispaced in density-CDF → locations on arc
    targets = np.linspace(0.0, 1.0, n_points)
    arc_targets = np.interp(targets, density_cdf, arc)

    xs = np.interp(arc_targets, arc, edge[:, 0])
    ys = np.interp(arc_targets, arc, edge[:, 1])
    out = np.column_stack([xs, ys]).astype(np.float32)

    # Enforce minimum separation
    filtered = [out[0]]
    for p in out[1:]:
        if np.linalg.norm(p - filtered[-1]) >= min_separation_px:
            filtered.append(p)
    return np.asarray(filtered, dtype=np.float32)


def match_keypoints_to_edge(
    kp_points: np.ndarray,
    edge: np.ndarray,
    polygon_bbox_xyxy: Tuple[int, int, int, int],
    max_distance_px: float = 18.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match each kp to its nearest bottom-edge point.

    A kp matches if it is inside the polygon's bbox AND its nearest
    distance to ``edge`` is ≤ ``max_distance_px``.

    Returns:
        (matched_kp, match_dists, unmatched_kp)
    """
    if kp_points is None or len(kp_points) == 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.float32), empty

    x1, y1, x2, y2 = polygon_bbox_xyxy
    in_bbox = (
        (kp_points[:, 0] >= x1 - 2)
        & (kp_points[:, 0] <= x2 + 2)
        & (kp_points[:, 1] >= y1 - 20)
        & (kp_points[:, 1] <= y2 + 20)
    )

    matches: List[np.ndarray] = []
    dists: List[float] = []
    unmatched: List[np.ndarray] = []
    for i, pt in enumerate(kp_points):
        if not in_bbox[i]:
            unmatched.append(pt)
            continue
        d = np.linalg.norm(edge - pt, axis=1)
        j = int(np.argmin(d))
        if d[j] <= max_distance_px:
            # Snap the kp to the bottom edge for consistency
            matches.append(edge[j])
            dists.append(float(d[j]))
        else:
            unmatched.append(pt)

    if matches:
        matched_arr = np.asarray(matches, dtype=np.float32)
    else:
        matched_arr = np.zeros((0, 2), dtype=np.float32)
    if unmatched:
        unmatched_arr = np.asarray(unmatched, dtype=np.float32)
    else:
        unmatched_arr = np.zeros((0, 2), dtype=np.float32)
    return matched_arr, np.asarray(dists, dtype=np.float32), unmatched_arr


def _iou_xyxy(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = max(0.0, (a[2] - a[0]) * (a[3] - a[1])) + max(0.0, (b[2] - b[0]) * (b[3] - b[1])) - inter
    return inter / ua if ua > 0 else 0.0


def clip_polygon_to_bbox(
    polygon: np.ndarray,
    bbox: Tuple[float, float, float, float],
    image_shape: Tuple[int, int],
    pad: int = 6,
) -> Optional[np.ndarray]:
    """Clip a polygon to a bbox via rasterisation.

    Handles arbitrary concave polygons; returns the largest connected
    component of the intersection (or ``None`` if empty).
    """
    h, w = image_shape
    if polygon is None or len(polygon) < 3:
        return None
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [np.round(polygon).astype(np.int32)], 255)

    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)
    bbox_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, thickness=-1)

    clipped = cv2.bitwise_and(poly_mask, bbox_mask)
    if clipped.sum() < 50:
        return None
    contours, _ = cv2.findContours(
        clipped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=lambda c: cv2.contourArea(c))
    if cv2.contourArea(largest) < 50:
        return None
    return largest.reshape(-1, 2).astype(np.float32)


def best_rf_for_bbox(
    bbox: Tuple[float, float, float, float],
    rf_dets: List[StripeDetection],
    min_iou: float = 0.15,
) -> Optional[Tuple[int, StripeDetection, float]]:
    """Pick the RF polygon whose bbox has the highest IoU with ``bbox``.

    Returns ``(index, detection, iou)`` or ``None``.
    """
    best_idx, best_iou = -1, 0.0
    for j, rf in enumerate(rf_dets):
        if rf.polygon is None or len(rf.polygon) < 3:
            continue
        pb = rf.polygon
        rb = (float(pb[:, 0].min()), float(pb[:, 1].min()),
              float(pb[:, 0].max()), float(pb[:, 1].max()))
        iou = _iou_xyxy(bbox, rb)
        if iou > best_iou:
            best_iou = iou
            best_idx = j
    if best_idx < 0 or best_iou < min_iou:
        return None
    return best_idx, rf_dets[best_idx], best_iou


def _kp_in_bbox_points(
    kp_dets: List[StripeDetection],
    bbox: Tuple[float, float, float, float],
    pad: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect kp points whose (x, y) fall inside ``bbox`` (±pad).

    Returns ``(points, confidences)`` as 2D arrays.
    """
    pts_all, conf_all = [], []
    x1, y1, x2, y2 = bbox
    for kp in kp_dets:
        pts = kp.points
        if pts is None or len(pts) == 0:
            continue
        inside = (
            (pts[:, 0] >= x1 - pad)
            & (pts[:, 0] <= x2 + pad)
            & (pts[:, 1] >= y1 - pad)
            & (pts[:, 1] <= y2 + pad)
        )
        if inside.sum() == 0:
            continue
        pts_all.append(pts[inside])
        if kp.keypoint_confidences is not None and len(kp.keypoint_confidences) == len(pts):
            conf_all.append(kp.keypoint_confidences[inside])
        else:
            conf_all.append(np.full(int(inside.sum()), kp.confidence, dtype=np.float32))
    if not pts_all:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return (
        np.concatenate(pts_all).astype(np.float32),
        np.concatenate(conf_all).astype(np.float32),
    )


def best_kp_for_bbox(
    bbox: Tuple[float, float, float, float],
    kp_dets: List[StripeDetection],
    min_points: int = 3,
) -> Optional[StripeDetection]:
    """Build a single StripeDetection from the kp points that lie in ``bbox``.

    All kp detections whose points fall inside the bbox are pooled and
    returned as one StripeDetection, sorted by x. If multiple kp detections
    overlap, their points are merged — typically harmless because the
    bbox already isolates them to one stripe's row.
    """
    points, confidences = _kp_in_bbox_points(kp_dets, bbox)
    if len(points) < min_points:
        return None
    # Sort by x along the stripe
    order = np.argsort(points[:, 0])
    points = points[order]
    confidences = confidences[order]

    # Simple anomaly filter: reject points whose y is > 3 MAD from the median
    if len(points) >= 5:
        med_y = np.median(points[:, 1])
        mad = np.median(np.abs(points[:, 1] - med_y)) + 1e-6
        keep = np.abs(points[:, 1] - med_y) <= 4.0 * mad
        if keep.sum() >= min_points:
            points = points[keep]
            confidences = confidences[keep]

    if len(points) >= 2:
        dx = points[-1, 0] - points[0, 0]
        dy = points[-1, 1] - points[0, 1]
        orient = np.arctan(dy / dx) * 180.0 / np.pi if abs(dx) > 1e-6 else 90.0
    else:
        orient = 0.0
    return StripeDetection(
        points=points,
        confidence=float(confidences.mean()) if len(confidences) else 0.5,
        orientation_deg=float(orient),
        keypoint_confidences=confidences,
    )


def best_seg_for_bbox(
    bbox: Tuple[float, float, float, float],
    seg_dets: List[StripeDetection],
    min_points: int = 5,
) -> Optional[StripeDetection]:
    """Return the seg detection with most points inside ``bbox``."""
    best, best_in = None, 0
    x1, y1, x2, y2 = bbox
    for s in seg_dets:
        pts = s.points
        if pts is None or len(pts) == 0:
            continue
        inside = (
            (pts[:, 0] >= x1 - 10)
            & (pts[:, 0] <= x2 + 10)
            & (pts[:, 1] >= y1 - 10)
            & (pts[:, 1] <= y2 + 10)
        )
        n = int(inside.sum())
        if n >= min_points and n > best_in:
            best_in = n
            clipped = pts[inside]
            order = np.argsort(clipped[:, 0])
            clipped = clipped[order]
            if len(clipped) >= 2:
                dx = clipped[-1, 0] - clipped[0, 0]
                dy = clipped[-1, 1] - clipped[0, 1]
                orient = np.arctan(dy / dx) * 180.0 / np.pi if abs(dx) > 1e-6 else 90.0
            else:
                orient = 0.0
            best = StripeDetection(
                points=clipped.astype(np.float32),
                confidence=s.confidence,
                orientation_deg=float(orient),
            )
    return best


def seg_on_crop(
    image_bgr: np.ndarray,
    sail_mask: np.ndarray,
    bbox: Tuple[float, float, float, float],
    seg_model_path: str,
    pad_ratio: float = 0.08,
    min_conf: float = 0.10,
) -> Optional[StripeDetection]:
    """Run the YOLO seg model ON THE CROP ``bbox ∩ sail_mask``.

    Returns a StripeDetection with:
      * ``points``: skeleton-traced centerline (up to 14 samples)
      * ``polygon``: the predicted instance mask contour

    Higher-quality than reusing full-image seg detections — the model
    sees a tight crop focused on one stripe with no distractors.
    """
    try:
        from ultralytics import YOLO
        from skimage.morphology import skeletonize
    except ImportError:
        return None

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    pad_x = max(6, int(round((x2 - x1) * pad_ratio)))
    pad_y = max(6, int(round((y2 - y1) * pad_ratio)))
    cx1 = max(0, x1 - pad_x); cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x); cy2 = min(h, y2 + pad_y)
    if cx2 - cx1 < 20 or cy2 - cy1 < 8:
        return None

    crop = image_bgr[cy1:cy2, cx1:cx2].copy()
    crop_mask = sail_mask[cy1:cy2, cx1:cx2]
    crop[~crop_mask] = 0   # mask out non-sail pixels

    try:
        from src._model_cache import get_yolo
        model = get_yolo(seg_model_path)
        if model is None:
            return None
        # imgsz=640 halves inference latency vs the default 1024 with
        # no measurable recall loss on ~300 px crops.
        res = model(crop, conf=min_conf, verbose=False, imgsz=640)[0]
    except Exception:
        return None
    if res.masks is None or len(res.masks.data) == 0:
        return None

    # Pick the highest-confidence mask inside the crop
    box_confs = (
        res.boxes.conf.cpu().numpy() if res.boxes is not None
        else np.array([0.3])
    )
    best = int(np.argmax(box_confs))
    inst = res.masks.data[best].cpu().numpy().astype(np.uint8)
    ch, cw = crop.shape[:2]
    if inst.shape != (ch, cw):
        inst = cv2.resize(inst, (cw, ch), interpolation=cv2.INTER_NEAREST)
    inst = (inst > 0) & crop_mask
    if inst.sum() < 30:
        return None

    # Polygon contour (for later polygon-bottom-edge fusion)
    contours, _ = cv2.findContours(
        inst.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
    )
    polygon = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= 40:
            poly_local = largest.reshape(-1, 2).astype(np.float32)
            polygon = poly_local + np.array([cx1, cy1], dtype=np.float32)

    # Skeleton → ordered centerline
    sk = skeletonize(inst)
    ys, xs = np.where(sk)
    if len(xs) < 6:
        return None
    pts = np.column_stack([xs + cx1, ys + cy1]).astype(np.float32)
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    n_sample = min(14, len(pts))
    idx = np.linspace(0, len(pts) - 1, n_sample).astype(int)
    pts = pts[idx]

    dx = pts[-1, 0] - pts[0, 0]
    dy = pts[-1, 1] - pts[0, 1]
    orient = (
        np.arctan(dy / dx) * 180.0 / np.pi if abs(dx) > 1e-6 else 90.0
    )
    return StripeDetection(
        points=pts,
        confidence=float(box_confs[best]),
        orientation_deg=float(orient),
        polygon=polygon,
    )


def dedup_bboxes(
    bboxes: List[np.ndarray],
    iou_threshold: float = 0.35,
    y_center_tol_ratio: float = 0.25,
) -> List[np.ndarray]:
    """Remove overlapping / stacked bboxes.

    Two bboxes collapse if:
      * their IoU exceeds ``iou_threshold``, OR
      * their centers are vertically within ``y_center_tol_ratio`` of the
        taller box's height AND they overlap horizontally by ≥ 40 %.

    Keeps the bbox with the larger area (more likely the primary stripe).
    """
    if len(bboxes) <= 1:
        return bboxes

    def _area(b):
        return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))

    def _iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = _area(a) + _area(b) - inter
        return inter / ua if ua > 0 else 0.0

    def _y_cluster(a, b):
        cy_a = 0.5 * (a[1] + a[3])
        cy_b = 0.5 * (b[1] + b[3])
        h = max(a[3] - a[1], b[3] - b[1])
        if h < 1:
            return False
        if abs(cy_a - cy_b) > h * y_center_tol_ratio:
            return False
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        w_min = min(a[2] - a[0], b[2] - b[0])
        return w_min > 0 and ix / w_min >= 0.4

    keep = [True] * len(bboxes)
    # Sort by y-center first so the walk is consistent
    order = sorted(range(len(bboxes)), key=lambda i: 0.5 * (bboxes[i][1] + bboxes[i][3]))
    for pi in range(len(order)):
        i = order[pi]
        if not keep[i]:
            continue
        for pj in range(pi + 1, len(order)):
            j = order[pj]
            if not keep[j]:
                continue
            if _iou(bboxes[i], bboxes[j]) > iou_threshold or _y_cluster(bboxes[i], bboxes[j]):
                # Keep the larger one
                if _area(bboxes[j]) > _area(bboxes[i]):
                    keep[i] = False
                    break
                else:
                    keep[j] = False
    return [b for b, k in zip(bboxes, keep) if k]


def classical_ridge_in_crop(
    image_bgr: np.ndarray,
    sail_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pad_ratio: float = 0.15,
) -> Optional[StripeDetection]:
    """Last-resort classical (Meijering ridge) detection inside a bbox.

    Used when no ML method fires — rare, but needed for tiny top-of-sail
    stripes that ML misses.
    """
    from skimage.filters import meijering
    from skimage.morphology import skeletonize
    from skimage.measure import label as sk_label, regionprops
    from scipy.spatial import cKDTree

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    pad_x = int(round((x2 - x1) * pad_ratio))
    pad_y = int(round((y2 - y1) * pad_ratio))
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    if cx2 - cx1 < 15 or cy2 - cy1 < 6:
        return None

    crop = image_bgr[cy1:cy2, cx1:cx2].copy()
    crop_mask = sail_mask[cy1:cy2, cx1:cx2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    # Sail-colour sign: mostly light sails → look for dark ridges
    if crop_mask.any():
        median = np.median(gray[crop_mask])
    else:
        median = np.median(gray)
    black_ridges = median > 0.5

    ridges = meijering(gray, sigmas=range(1, 4), black_ridges=black_ridges)
    ridges[~crop_mask] = 0
    if ridges.max() <= 0:
        return None
    threshold = np.percentile(ridges[ridges > 0], 80)
    binary = (ridges > threshold).astype(np.uint8)
    if binary.sum() < 20:
        return None
    # Bridge small horizontal gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    sk = skeletonize(binary > 0)
    if sk.sum() < 8:
        return None
    labeled = sk_label(sk)
    regions = regionprops(labeled)
    if not regions:
        return None
    region = max(regions, key=lambda r: len(r.coords))
    coords = region.coords  # (y, x)
    if len(coords) < 5:
        return None
    pts = np.column_stack([coords[:, 1] + cx1, coords[:, 0] + cy1]).astype(np.float32)

    # Order along x, subsample
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    n_sample = min(12, len(pts))
    idx = np.linspace(0, len(pts) - 1, n_sample).astype(int)
    pts = pts[idx]

    if len(pts) < 2:
        return None
    dx = pts[-1, 0] - pts[0, 0]
    dy = pts[-1, 1] - pts[0, 1]
    orient = np.arctan(dy / dx) * 180.0 / np.pi if abs(dx) > 1e-6 else 90.0
    return StripeDetection(
        points=pts, confidence=0.4,
        orientation_deg=float(orient),
    )


def detect_stripe_in_bbox_full(
    image_bgr: np.ndarray,
    sail_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    kp_model_path: str,
    seg_model_path: str,
    rf_config: Optional[dict] = None,
    pad_ratio: float = 0.10,
    kp_min_conf: float = 0.10,
    seg_min_conf: float = 0.10,
    rf_lower_conf_pct: int = 10,
) -> Tuple[Optional[StripeDetection], Dict]:
    """Full per-bbox detection: crop, mask, run v7 + kp, fuse.

    Strategy:
      1. Crop ``image_bgr`` to ``bbox`` (padded), zero outside
         ``sail_mask``.
      2. Run v7 (Roboflow) on the crop with LOW confidence
         (``rf_lower_conf_pct``). Pick the highest-confidence polygon.
      3. Run kp model on the crop, keep points above ``kp_min_conf``.
      4. Run seg model on the crop as a fallback shape source.
      5. If v7 polygon available, fuse with the kp points (which are
         already restricted to the bbox by construction).
      6. Else, use kp polyline. Else, use seg skeleton.

    Returns ``(stripe_detection_or_None, diag_dict)``. diag_dict has:
        ``bbox``, ``crop_origin``, ``crop_shape``, ``rf_conf`` (float),
        ``kp_count``, ``seg_count``, ``source`` in {"v7_fused", "v7_only",
        "kp_only", "seg_only", "none"}.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return None, {"source": "none", "error": "ultralytics unavailable"}

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    pad_x = int(round((x2 - x1) * pad_ratio))
    pad_y = int(round((y2 - y1) * pad_ratio))
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    if cx2 - cx1 < 20 or cy2 - cy1 < 10:
        return None, {"source": "none", "error": "bbox too small"}

    crop = image_bgr[cy1:cy2, cx1:cx2].copy()
    crop_mask = sail_mask[cy1:cy2, cx1:cx2]
    crop[~crop_mask] = 0  # black outside the sail

    origin = np.array([cx1, cy1], dtype=np.float32)
    diag: Dict = {
        "bbox": (cx1, cy1, cx2, cy2),
        "crop_origin": (float(cx1), float(cy1)),
        "crop_shape": crop.shape[:2],
        "rf_conf": 0.0,
        "kp_count": 0,
        "seg_count": 0,
        "source": "none",
    }

    # --- v7 on crop (lower threshold to give it a chance) ------------------
    rf_polygon: Optional[np.ndarray] = None
    rf_conf = 0.0
    if rf_config is not None and rf_config.get("roboflow_api_key"):
        import tempfile, os
        from src.detection import _get_roboflow_model  # reuse cache

        model = _get_roboflow_model(
            rf_config["roboflow_api_key"],
            rf_config.get("roboflow_workspace", "sailing-project"),
            rf_config.get("roboflow_project", "fleurs-bot"),
            int(rf_config.get("roboflow_version", 7)),
        )
        if model is not None:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                cv2.imwrite(tf.name, crop)
                tmp_path = tf.name
            try:
                result = model.predict(tmp_path, confidence=rf_lower_conf_pct).json()
                preds = result.get("predictions", [])
                if preds:
                    best_pred = max(preds, key=lambda p: p.get("confidence", 0))
                    pts_list = best_pred.get("points") or []
                    if len(pts_list) >= 3:
                        rf_polygon = np.array(
                            [[float(p["x"]) + cx1, float(p["y"]) + cy1]
                             for p in pts_list],
                            dtype=np.float32,
                        )
                        rf_conf = float(best_pred.get("confidence", 0))
            except Exception:
                pass
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    diag["rf_conf"] = rf_conf

    # --- kp on crop --------------------------------------------------------
    kp_det: Optional[StripeDetection] = None
    try:
        kp_model = YOLO(kp_model_path)
        res = kp_model(crop, conf=kp_min_conf, verbose=False)[0]
        if res.keypoints is not None and len(res.keypoints.data) > 0:
            box_confs = (
                res.boxes.conf.cpu().numpy() if res.boxes is not None
                else np.array([0.5])
            )
            best = int(np.argmax(box_confs))
            kp_data = res.keypoints.data[best].cpu().numpy()
            kp_xy = kp_data[:, :2]
            kp_conf = kp_data[:, 2]
            valid = kp_conf > 0.05
            if valid.sum() >= 3:
                pts = kp_xy[valid].astype(np.float32)
                pts[:, 0] += cx1; pts[:, 1] += cy1
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                dx = pts[-1, 0] - pts[0, 0]
                dy = pts[-1, 1] - pts[0, 1]
                orient = (np.arctan(dy / dx) * 180.0 / np.pi
                          if abs(dx) > 1e-6 else 90.0)
                kp_det = StripeDetection(
                    points=pts,
                    confidence=float(box_confs[best]),
                    orientation_deg=float(orient),
                    keypoint_confidences=kp_conf[valid].astype(np.float32),
                )
                diag["kp_count"] = int(valid.sum())
    except Exception:
        pass

    # --- If v7 polygon exists → fuse with kp -------------------------------
    if rf_polygon is not None and len(rf_polygon) >= 3:
        rf_det = StripeDetection(
            points=rf_polygon,  # bottom edge filled later
            confidence=rf_conf,
            orientation_deg=0.0,
            polygon=rf_polygon,
        )
        fused, _ = fuse_polygon_with_keypoints(
            rf_det, kp_det, image_shape=image_bgr.shape[:2]
        )
        diag["source"] = "v7_fused" if kp_det is not None else "v7_only"
        return fused, diag

    # --- kp fallback -------------------------------------------------------
    if kp_det is not None and len(kp_det.points) >= 3:
        diag["source"] = "kp_only"
        return kp_det, diag

    # --- seg fallback ------------------------------------------------------
    try:
        from skimage.morphology import skeletonize
        seg_model = YOLO(seg_model_path)
        res = seg_model(crop, conf=seg_min_conf, verbose=False)[0]
        if res.masks is not None and len(res.masks.data) > 0:
            box_confs = (
                res.boxes.conf.cpu().numpy() if res.boxes is not None
                else np.array([0.3])
            )
            best = int(np.argmax(box_confs))
            inst = res.masks.data[best].cpu().numpy().astype(np.uint8)
            ch, cw = crop.shape[:2]
            if inst.shape != (ch, cw):
                inst = cv2.resize(inst, (cw, ch), interpolation=cv2.INTER_NEAREST)
            inst = (inst > 0) & crop_mask
            if inst.sum() >= 20:
                sk = skeletonize(inst)
                ys, xs = np.where(sk)
                if len(xs) >= 6:
                    pts = np.column_stack([xs + cx1, ys + cy1]).astype(np.float32)
                    order = np.argsort(pts[:, 0])
                    pts = pts[order]
                    n_sample = min(12, len(pts))
                    idx = np.linspace(0, len(pts) - 1, n_sample).astype(int)
                    pts = pts[idx]
                    dx = pts[-1, 0] - pts[0, 0]
                    dy = pts[-1, 1] - pts[0, 1]
                    orient = (np.arctan(dy / dx) * 180.0 / np.pi
                              if abs(dx) > 1e-6 else 90.0)
                    diag["seg_count"] = len(pts)
                    diag["source"] = "seg_only"
                    return StripeDetection(
                        points=pts,
                        confidence=float(box_confs[best]),
                        orientation_deg=float(orient),
                    ), diag
    except Exception:
        pass

    # --- Final fallback: classical Meijering ridge trace in the crop -------
    classical = classical_ridge_in_crop(image_bgr, sail_mask, bbox)
    if classical is not None:
        diag["source"] = "classical"
        return classical, diag

    return None, diag


def detect_stripe_in_bbox(
    image_bgr: np.ndarray,
    sail_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    kp_model_path: str,
    seg_model_path: str,
    pad_ratio: float = 0.08,
    min_conf: float = 0.10,
) -> Optional[StripeDetection]:
    """Run keypoint + seg models on the crop ``bbox ∩ sail_mask``.

    Used as a recovery path when a stripe has a confident bbox (from the
    bbox detector) but no Roboflow polygon — so the fusion layer would
    otherwise drop it. We crop the image to the bbox (with padding),
    zero-out everything outside the sail mask, run kp first (preferred
    shape) and seg as a fallback, and return a single StripeDetection in
    global image coordinates.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return None

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    pad_x = int(round((x2 - x1) * pad_ratio))
    pad_y = int(round((y2 - y1) * pad_ratio))
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    if cx2 - cx1 < 20 or cy2 - cy1 < 10:
        return None

    crop = image_bgr[cy1:cy2, cx1:cx2].copy()
    crop_mask = sail_mask[cy1:cy2, cx1:cx2]

    # Zero anything outside the sail so models don't latch onto sky/rigging
    crop[~crop_mask] = 0

    # Try keypoint model first
    try:
        kp_model = YOLO(kp_model_path)
        res = kp_model(crop, conf=min_conf, verbose=False)[0]
        if res.keypoints is not None and len(res.keypoints.data) > 0:
            # Pick the single detection with highest bbox confidence
            box_confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.array([0.5])
            best = int(np.argmax(box_confs))
            kp_data = res.keypoints.data[best].cpu().numpy()  # (K, 3) x, y, conf
            kp_xy = kp_data[:, :2]
            kp_conf = kp_data[:, 2]
            valid = kp_conf > 0.05
            if valid.sum() >= 3:
                pts = kp_xy[valid].astype(np.float32)
                # Shift back to image coords
                pts[:, 0] += cx1
                pts[:, 1] += cy1
                # Clip to image
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                if len(pts) >= 2:
                    dx = pts[-1, 0] - pts[0, 0]
                    dy = pts[-1, 1] - pts[0, 1]
                    orient = np.arctan(dy / dx) * 180.0 / np.pi if abs(dx) > 1e-6 else 90.0
                else:
                    orient = 0.0
                return StripeDetection(
                    points=pts,
                    confidence=float(box_confs[best]),
                    orientation_deg=float(orient),
                    keypoint_confidences=kp_conf[valid].astype(np.float32),
                )
    except Exception:
        pass

    # Fallback: seg model → skeleton
    try:
        from skimage.morphology import skeletonize
        seg_model = YOLO(seg_model_path)
        res = seg_model(crop, conf=min_conf, verbose=False)[0]
        if res.masks is None or len(res.masks.data) == 0:
            return None
        box_confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.array([0.3])
        best = int(np.argmax(box_confs))
        inst = res.masks.data[best].cpu().numpy().astype(np.uint8)
        ch, cw = crop.shape[:2]
        if inst.shape != (ch, cw):
            inst = cv2.resize(inst, (cw, ch), interpolation=cv2.INTER_NEAREST)
        inst = (inst > 0) & crop_mask
        if inst.sum() < 20:
            return None
        sk = skeletonize(inst)
        ys, xs = np.where(sk)
        if len(xs) < 6:
            return None
        pts = np.column_stack([xs + cx1, ys + cy1]).astype(np.float32)
        # Sort by x, sub-sample to 12 points
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        n_sample = min(12, len(pts))
        idx = np.linspace(0, len(pts) - 1, n_sample).astype(int)
        pts = pts[idx]
        if len(pts) >= 2:
            dx = pts[-1, 0] - pts[0, 0]
            dy = pts[-1, 1] - pts[0, 1]
            orient = np.arctan(dy / dx) * 180.0 / np.pi if abs(dx) > 1e-6 else 90.0
        else:
            orient = 0.0
        return StripeDetection(
            points=pts,
            confidence=float(box_confs[best]),
            orientation_deg=float(orient),
        )
    except Exception:
        return None


def fuse_polygon_with_keypoints(
    rf_det: StripeDetection,
    kp_det: Optional[StripeDetection],
    image_shape: Tuple[int, int],
    n_curve_samples: int = 18,
    kp_match_radius_px: float = 18.0,
) -> Tuple[StripeDetection, FusionDiagnostics]:
    """Build a high-confidence stripe from a v7 polygon + kp matches.

    Strategy
    --------
    * Bottom-edge of the polygon is the primary shape.
    * kp points inside the polygon's bbox that are within
      ``kp_match_radius_px`` of the bottom edge become anchors (very
      high confidence).
    * Extra points are sampled from the bottom edge with density
      proportional to local curvature.
    * Unmatched kp points are kept with low weight for the fitter (they
      may pick up camber the polygon's rasterisation smoothed away).

    Returns a new ``StripeDetection`` plus diagnostics used by the
    report.
    """
    polygon = rf_det.polygon
    if polygon is None or len(polygon) < 3:
        # No polygon — pass kp through
        kp_pts = kp_det.points if kp_det is not None else np.zeros((0, 2), dtype=np.float32)
        empty = np.zeros((0, 2), dtype=np.float32)
        diag = FusionDiagnostics(
            bottom_edge=empty,
            curvature_samples=empty,
            matched_kp=empty,
            unmatched_kp=kp_pts,
            match_dists=np.zeros(0, dtype=np.float32),
            bbox=(0, 0, 0, 0),
            final_points=kp_pts,
            polygon=np.zeros((0, 2), dtype=np.float32),
        )
        return (kp_det if kp_det is not None else rf_det), diag

    bbox = polygon_bbox(polygon)
    bottom = polygon_bottom_edge(polygon, image_shape, n_samples=max(32, n_curve_samples * 2))

    if kp_det is not None and kp_det.points is not None and len(kp_det.points) > 0:
        matched, dists, unmatched = match_keypoints_to_edge(
            kp_det.points, bottom, bbox, max_distance_px=kp_match_radius_px
        )
    else:
        matched = np.zeros((0, 2), dtype=np.float32)
        dists = np.zeros(0, dtype=np.float32)
        unmatched = np.zeros((0, 2), dtype=np.float32)

    curve_samples = curvature_weighted_resample(bottom, n_points=n_curve_samples)

    # Identify the kp coverage range (matched-kp x-span). Curvature
    # samples OUTSIDE this range get promoted to anchor weight — they
    # fill in what the kp model missed on the left or right side of the
    # stripe. Inside the kp coverage they keep the regular 0.7 weight.
    if len(matched) >= 2:
        kp_x_min = float(matched[:, 0].min())
        kp_x_max = float(matched[:, 0].max())
        gap_margin = 12.0  # px — samples within this margin of the kp
                           # coverage still count as "kp-supported"
    else:
        kp_x_min = np.inf
        kp_x_max = -np.inf
        gap_margin = 0.0

    def _curve_weight(pt):
        x = float(pt[0])
        outside = (x < kp_x_min - gap_margin) or (x > kp_x_max + gap_margin)
        return 1.0 if outside else 0.7

    curve_weights = np.array([_curve_weight(p) for p in curve_samples],
                              dtype=np.float32)

    # Merge: matched kp (high-weight) + curvature samples, dedup on
    # proximity, sort by x. Keep the weight associated with each point.
    all_pts = np.vstack([matched, curve_samples])
    all_w = np.concatenate([
        np.full(len(matched), 1.0, dtype=np.float32),
        curve_weights,
    ])
    if len(all_pts) > 1:
        order = np.argsort(all_pts[:, 0])
        all_pts = all_pts[order]
        all_w = all_w[order]
        kept_idx = [0]
        for i in range(1, len(all_pts)):
            if np.linalg.norm(all_pts[i] - all_pts[kept_idx[-1]]) > 3.0:
                kept_idx.append(i)
        merged = all_pts[kept_idx].astype(np.float32)
        kp_weights = all_w[kept_idx].astype(np.float32)
    else:
        merged = all_pts.astype(np.float32)
        kp_weights = all_w.astype(np.float32)
    n_matched = len(matched)

    # Compute orientation from first→last
    if len(merged) >= 2:
        dx = merged[-1, 0] - merged[0, 0]
        dy = merged[-1, 1] - merged[0, 1]
        orient = float(np.degrees(np.arctan2(dy, dx))) if abs(dx) > 1e-6 else 90.0
    else:
        orient = rf_det.orientation_deg

    fused = StripeDetection(
        points=merged if len(merged) > 0 else bottom,
        confidence=max(rf_det.confidence, kp_det.confidence if kp_det else 0.0),
        orientation_deg=orient,
        keypoint_confidences=kp_weights if len(merged) > 0 else None,
        polygon=polygon,
    )

    diag = FusionDiagnostics(
        bottom_edge=bottom,
        curvature_samples=curve_samples,
        matched_kp=matched,
        unmatched_kp=unmatched,
        match_dists=dists,
        bbox=bbox,
        final_points=fused.points,
        polygon=polygon,
    )
    return fused, diag
