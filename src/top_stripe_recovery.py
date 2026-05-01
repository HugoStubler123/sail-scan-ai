"""Top-stripe recovery for foreshortened mainsail tops.

The keypoint and bbox YOLO models trained on jib-heavy data miss the
topmost stripe of mainsails because the stripe is severely foreshortened
(small in pixels, near halyard/mast clutter). Two recovery strategies
that operate WITHOUT retraining:

1. ``detect_kp_on_top_crop``: crop the top 30-40% of the segmented sail,
   upscale to 1280, run the kp model. Detections in the upscaled crop
   are remapped to image coordinates. Effectively gives the foreshortened
   stripe 3-4x more pixels at inference time.

2. ``synth_kp_from_bbox_cluster``: when the bbox model finds a fragmented
   cluster of small bboxes (which IS the top stripe, fragmented), DBSCAN
   the clusters, merge, and synthesize 8 keypoints by walking a 1D ridge
   response across the merged region. Promotes existing bbox evidence
   into the same StripeDetection schema downstream code expects.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from src.types import StripeDetection


def detect_kp_on_top_crop(
    image_bgr: np.ndarray,
    sail_mask: Optional[np.ndarray],
    config: dict,
    kp_detect_fn,
    top_frac: float = 0.40,
    target_size: int = 1280,
    min_conf_override: float = 0.10,
) -> List[StripeDetection]:
    """Run the kp model on the top crop of the sail mask, upscaled.

    Parameters
    ----------
    image_bgr : full image
    sail_mask : SAM2 sail mask (bool)
    config : detection config — keypoint_model_path read from here
    kp_detect_fn : callable(image, mask, config) -> List[StripeDetection]
        The original (unpatched) keypoint detector.
    top_frac : fraction of sail height (from top) to crop
    target_size : long-edge size to upscale crop to
    min_conf_override : keypoint conf threshold inside the crop pass
    """
    if sail_mask is None or sail_mask.sum() < 100:
        return []
    ys, xs = np.where(sail_mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    h_sail = y_max - y_min
    w_sail = x_max - x_min
    if h_sail < 50 or w_sail < 50:
        return []

    pad_x = int(0.04 * w_sail)
    pad_y_top = int(0.02 * h_sail)
    pad_y_bot = int(0.05 * h_sail)
    y1 = max(0, y_min - pad_y_top)
    y2 = min(image_bgr.shape[0], y_min + int(top_frac * h_sail) + pad_y_bot)
    x1 = max(0, x_min - pad_x)
    x2 = min(image_bgr.shape[1], x_max + pad_x)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    H_c, W_c = crop.shape[:2]
    scale = float(target_size) / max(H_c, W_c)
    if scale <= 1.0:
        scale = 1.0
        big = crop
    else:
        big = cv2.resize(
            crop, (int(W_c * scale), int(H_c * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    cfg_low = dict(config)
    cfg_low["min_keypoint_confidence"] = float(min_conf_override)
    cfg_low["keypoint_imgsz"] = int(target_size)

    crop_dets = kp_detect_fn(big, None, cfg_low)
    head_band_y = y_min + int(top_frac * h_sail) + pad_y_bot

    out: List[StripeDetection] = []
    for d in crop_dets:
        pts = d.points
        if pts is None or len(pts) < 4:
            continue
        new_pts = pts.astype(np.float64).copy()
        new_pts[:, 0] = new_pts[:, 0] / scale + x1
        new_pts[:, 1] = new_pts[:, 1] / scale + y1
        cy = float(np.mean(new_pts[:, 1]))
        if cy > head_band_y:
            continue
        new_kp_conf = (
            d.keypoint_confidences.copy()
            if d.keypoint_confidences is not None else None
        )
        out.append(StripeDetection(
            points=new_pts.astype(np.float32),
            confidence=float(d.confidence),
            orientation_deg=float(d.orientation_deg),
            keypoint_confidences=new_kp_conf,
        ))
    return out


def _ridge_y_at_x(
    gray: np.ndarray,
    x: int,
    y_lo: int,
    y_hi: int,
    tophat_size: int = 21,
) -> Optional[tuple]:
    """Return (y, response_strength) of the brightest thin ridge in the
    column ``gray[y_lo:y_hi, x]`` after a 1-D top-hat. Returns None if
    the column is too short or the response is degenerate.
    """
    from scipy.ndimage import grey_opening
    if y_hi - y_lo < 6:
        return None
    col = gray[y_lo:y_hi, x].astype(np.float32)
    opened = grey_opening(col, size=tophat_size)
    tophat = col - opened
    if tophat.std() < 1e-3:
        return None
    yi = int(np.argmax(tophat))
    return y_lo + yi, float(tophat[yi])


def synth_kp_from_bbox_cluster(
    image_bgr: np.ndarray,
    sail_mask: Optional[np.ndarray],
    bboxes: List[np.ndarray],
    head_frac: float = 0.30,
    n_keypoints: int = 8,
    min_cluster_size: int = 3,
) -> List[StripeDetection]:
    """Cluster small head-region bboxes, merge each cluster, then sample
    ``n_keypoints`` by 1-D ridge walk across the merged bbox.

    Designed for the failure mode where the bbox model finds the top
    stripe as a tight cluster of overlapping fragments.
    """
    if sail_mask is None or not bboxes:
        return []
    ys, xs = np.where(sail_mask)
    if len(ys) < 100:
        return []
    y_sail_min = int(ys.min())
    y_sail_max = int(ys.max())
    h_sail = y_sail_max - y_sail_min
    head_y_thresh = y_sail_min + int(head_frac * h_sail)

    head_bbs = [
        bb for bb in bboxes
        if 0.5 * (bb[1] + bb[3]) <= head_y_thresh
    ]
    if len(head_bbs) < min_cluster_size:
        return []

    centers = np.array(
        [[0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])] for b in head_bbs],
        dtype=np.float64,
    )
    eps = max(15.0, h_sail * 0.04)

    try:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=eps, min_samples=2).fit(centers).labels_
    except Exception:
        return []

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    out: List[StripeDetection] = []
    for k in set(int(l) for l in labels):
        if k < 0:
            continue
        idxs = np.where(labels == k)[0]
        if len(idxs) < min_cluster_size:
            continue
        cluster = [head_bbs[i] for i in idxs]
        x1 = float(min(b[0] for b in cluster))
        y1 = float(min(b[1] for b in cluster))
        x2 = float(max(b[2] for b in cluster))
        y2 = float(max(b[3] for b in cluster))
        if (x2 - x1) < 30 or (y2 - y1) < 4:
            continue

        pad_y = max(8.0, 0.6 * (y2 - y1))
        y_search_lo = max(0, int(y1 - pad_y))
        y_search_hi = min(gray.shape[0], int(y2 + pad_y))

        # Constrain search to inside the sail mask
        x_starts = np.linspace(x1 + 2, x2 - 2, n_keypoints)
        kp_pts = []
        kp_confs = []
        for xv in x_starts:
            xi = int(round(xv))
            if not (0 <= xi < gray.shape[1]):
                continue
            mask_col = sail_mask[y_search_lo:y_search_hi, xi]
            if not mask_col.any():
                continue
            res = _ridge_y_at_x(gray, xi, y_search_lo, y_search_hi)
            if res is None:
                continue
            yi, strength = res
            if not sail_mask[yi, xi]:
                continue
            kp_pts.append([float(xi), float(yi)])
            kp_confs.append(strength)
        if len(kp_pts) < max(4, n_keypoints // 2):
            continue
        pts = np.array(kp_pts, dtype=np.float32)
        confs = np.array(kp_confs, dtype=np.float32)
        confs = confs / max(float(confs.max()), 1.0)
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        confs = confs[order]

        # Reject cluster if y-coordinates aren't monotonic-ish (a real
        # stripe should have smoothly varying y; noise gives jagged y).
        y_diffs = np.diff(pts[:, 1])
        if np.std(y_diffs) > 0.5 * (y2 - y1):
            continue

        out.append(StripeDetection(
            points=pts,
            confidence=float(np.mean(confs)),
            orientation_deg=0.0,
            keypoint_confidences=confs,
        ))
    return out
