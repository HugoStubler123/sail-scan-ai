"""Filter YOLO bboxes to the ones that span the visible sail mask
end-to-end.

Rationale: a sail draft stripe runs from luff to leech, so a real
stripe bbox should cover essentially the full visible width of the
sail at its y-band. Bboxes that cover only a small fraction of the
sail width are partial detections (one stripe seen from a confidence-
limited region of the model) and should be either dropped or merged
into the full-span bbox they belong to.

Lowering the YOLO confidence threshold gives more candidates; this
filter keeps only the geometrically-valid ones.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _sail_x_extent_at_band(sail_mask: np.ndarray, y1: int, y2: int) -> Tuple[int, int]:
    """Return ``(x_min, x_max)`` of the sail mask within rows [y1, y2]."""
    H = sail_mask.shape[0]
    yi1 = max(0, int(y1))
    yi2 = min(H, int(y2) + 1)
    if yi2 <= yi1:
        return -1, -1
    band = sail_mask[yi1:yi2]
    if not band.any():
        return -1, -1
    cols = np.where(band.any(axis=0))[0]
    return int(cols[0]), int(cols[-1])


def is_full_span_bbox(
    bbox: np.ndarray,
    sail_mask: np.ndarray,
    min_span_frac: float = 0.75,
    edge_tol_frac: float = 0.12,
) -> bool:
    """Return True iff ``bbox`` covers the full visible sail width at
    its y-band.

    A bbox qualifies as "full-span" when:
      * its width is at least ``min_span_frac`` of the sail mask's
        width within the bbox y-range,
      * its left edge is within ``edge_tol_frac × sail_width`` of the
        leftmost mask pixel at that band,
      * its right edge is within the same tolerance of the rightmost
        mask pixel.
    """
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    sail_x_min, sail_x_max = _sail_x_extent_at_band(sail_mask, int(y1), int(y2))
    if sail_x_min < 0:
        return False
    sail_width = sail_x_max - sail_x_min
    if sail_width < 50:
        return False
    bbox_width = x2 - x1
    if bbox_width / sail_width < min_span_frac:
        return False
    edge_tol = edge_tol_frac * sail_width
    if x1 > sail_x_min + edge_tol:
        return False
    if x2 < sail_x_max - edge_tol:
        return False
    return True


def filter_full_span_bboxes(
    bboxes: List[np.ndarray],
    sail_mask: np.ndarray,
    min_span_frac: float = 0.75,
    edge_tol_frac: float = 0.12,
) -> List[np.ndarray]:
    """Keep only bboxes that span the visible sail end-to-end."""
    out = []
    for bb in bboxes:
        if is_full_span_bbox(bb, sail_mask, min_span_frac, edge_tol_frac):
            out.append(bb)
    return out


def reject_oversized_bboxes(
    bboxes: List[np.ndarray],
    sail_mask: np.ndarray,
    max_height_frac_of_sail: float = 0.18,
) -> List[np.ndarray]:
    """Drop bboxes whose y-span is unreasonably large compared to the
    sail mask height.

    A YOLO box for a single draft stripe should be a thin horizontal
    band — typically 5-15% of the sail's vertical extent (more if the
    stripe is foreshortened). Any box taller than ``max_height_frac``
    × sail_height is almost certainly multi-stripe contamination
    (YOLO accidentally grouped two adjacent stripes into one box).

    Returns the filtered list. Logs nothing — caller is expected to
    report counts.
    """
    if sail_mask is None:
        return list(bboxes)
    ys, _ = np.where(sail_mask)
    if len(ys) == 0:
        return list(bboxes)
    sail_h = float(ys.max() - ys.min())
    if sail_h < 50.0:
        return list(bboxes)
    limit = max_height_frac_of_sail * sail_h
    return [b for b in bboxes if (float(b[3]) - float(b[1])) <= limit]


def collapse_overlapping_bboxes(
    bboxes: List[np.ndarray],
    iou_thresh: float = 0.40,
) -> List[np.ndarray]:
    """Merge overlapping full-span bboxes (NMS-style by IoU).

    Sorts by area descending; keeps a bbox only if it has IoU below
    ``iou_thresh`` with all already-kept boxes.
    """
    if not bboxes:
        return []
    items = [(float((b[2] - b[0]) * (b[3] - b[1])), i, b) for i, b in enumerate(bboxes)]
    items.sort(key=lambda t: -t[0])
    kept: List[np.ndarray] = []
    for _, _, bb in items:
        bb_arr = np.asarray(bb, dtype=np.float32)
        ok = True
        for k in kept:
            x_inter = max(0.0, min(bb_arr[2], k[2]) - max(bb_arr[0], k[0]))
            y_inter = max(0.0, min(bb_arr[3], k[3]) - max(bb_arr[1], k[1]))
            inter = x_inter * y_inter
            if inter <= 0:
                continue
            a1 = (bb_arr[2] - bb_arr[0]) * (bb_arr[3] - bb_arr[1])
            a2 = (k[2] - k[0]) * (k[3] - k[1])
            union = a1 + a2 - inter
            if union > 0 and (inter / union) >= iou_thresh:
                ok = False
                break
        if ok:
            kept.append(bb_arr)
    return kept
