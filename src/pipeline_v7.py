"""Pipeline v7 — seg-crop first detection with two-stage endpoint refine.

Composes existing primitives in a new order per user spec (stages A-I):

    A. SAM2 segmentation       -> sail_boundary (head, luff, leech, clew, tack)
    B. YOLO stripe bbox model  -> list of per-stripe bboxes
    C. YOLO keypoint model     -> 8 kpts per stripe (pooled over image)
    D. New seg model on crop   -> per-bbox polygon (bbox ∩ sail_mask)
       Runs across N color variants of the crop; keeps the highest-conf
       mask (diagnostics for every variant are retained for the report).
    E. Match polygon bottom edge to kp points -> anchors
    F. Add curvature-weighted samples from polygon bottom edge
    G. Two-stage endpoint refiner: stripe_endpoints_v2.pt +
       stripe_endpoint_crop_26x.pt (≤ 30 px nudge each)
    H. Guard endpoints against outliers — if endpoint is >12 px
       perpendicular to nearest stripe point, pull it back
    I. NACA-style spline fit (fallback: regularised Bernstein)

This module does NOT replace build_stage_report.py — it's a sibling used
by build_v7_report.py. It reuses src.polygon_fusion, src.endpoints,
src.flexible_fit and src.analysis unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.types import StripeDetection, SailBoundary
from src.polygon_fusion import (
    FusionDiagnostics,
    best_kp_for_bbox,
    classical_ridge_in_crop,
    fuse_polygon_with_keypoints,
    seg_on_crop,
)
from src.endpoints import (
    endpoints_from_detection,
    detect_endpoints_model,
    refine_endpoint_with_crop_model,
    refine_endpoint_with_model,
    dedup_stripe_endpoints,
)


# ---------------------------------------------------------------------------
# Color / contrast variants used to stress-test the seg model per crop.
# ---------------------------------------------------------------------------

def _gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / max(gamma, 1e-3)
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def _unsharp(img: np.ndarray, amount: float = 1.5) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=2.5)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def _clahe_l(img: np.ndarray, clip: float = 3.5) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def _sat_boost(img: np.ndarray, factor: float = 1.5) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _channel_min(img: np.ndarray) -> np.ndarray:
    m = img.min(axis=2)
    return np.stack([m, m, m], axis=-1)


COLOR_VARIANTS: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("raw",        lambda x: x),
    ("CLAHE-L",    lambda x: _clahe_l(x, 3.5)),
    ("gamma 0.7",  lambda x: _gamma(x, 0.7)),
    ("gamma 1.4",  lambda x: _gamma(x, 1.4)),
    ("sat +50%",   lambda x: _sat_boost(x, 1.5)),
    ("unsharp",    lambda x: _unsharp(x, 1.5)),
    ("channel-min", _channel_min),
]


@dataclass
class SegVariantResult:
    """One attempt of seg_on_crop against a color-variant of the crop."""

    variant_name: str
    detection: Optional[StripeDetection]
    confidence: float
    # raw crop and the preprocessed crop used for inference — kept so the
    # report can visualise exactly what the model saw.
    raw_crop_bgr: np.ndarray
    variant_crop_bgr: np.ndarray
    # crop coords in image frame (so the report can draw the polygon back)
    crop_origin_xy: Tuple[int, int]
    # Composite score & component breakdown, filled by _score_variant.
    composite_score: float = 0.0
    score_breakdown: Optional[Dict[str, float]] = None


@dataclass
class V7StripeResult:
    """Per-bbox result from the v7 pipeline (stages D-I)."""

    bbox: Tuple[float, float, float, float]
    detection: StripeDetection
    fusion: Optional[FusionDiagnostics]
    source: str                                # "seg_crop+kp" | "seg_crop" | "kp" | "classical"
    raw_luff_ep: np.ndarray                    # endpoint_from_detection (pre ML)
    raw_leech_ep: np.ndarray
    luff_ep: np.ndarray                        # post ML refine + guard + SAM snap
    leech_ep: np.ndarray
    endpoint_nudge_px: Tuple[float, float]     # how far each endpoint moved
    endpoint_guarded: Tuple[bool, bool]        # True when guard pulled it back
    endpoint_snapped_to_sam: Tuple[bool, bool] = (False, False)  # True when SAM-snap picked stripe-extreme over refined
    spline_points: Optional[np.ndarray] = None
    spline_meta: Optional[Dict] = None
    # Seg model diagnostics (populated by run_v7_detection_stage)
    seg_variants: List[SegVariantResult] = field(default_factory=list)
    best_seg_variant: Optional[str] = None


def guard_endpoint(
    endpoint: np.ndarray,
    stripe_points: np.ndarray,
    max_perp_px: float = 12.0,
    max_extension_ratio: float = 1.15,
) -> Tuple[np.ndarray, bool]:
    """Pull endpoint back onto the stripe if it strays too far.

    Rejects endpoints that (a) lie more than ``max_perp_px`` from the
    stripe polyline, or (b) extend the detected chord by more than
    ``max_extension_ratio``. In those cases we fall back to the stripe's
    own extreme on the same side.

    Returns (corrected_endpoint, was_guarded).
    """
    if stripe_points is None or len(stripe_points) < 2:
        return endpoint.astype(np.float32), False

    pts = stripe_points.astype(np.float32)
    # Perpendicular distance to the polyline (min segment distance)
    min_d = _min_distance_to_polyline(endpoint, pts)

    # Horizontal extension check: detected x-range
    xs = pts[:, 0]
    x_min, x_max = float(xs.min()), float(xs.max())
    span = max(x_max - x_min, 1.0)
    ep_x = float(endpoint[0])
    # how far outside [x_min, x_max] is the endpoint?
    over = max(x_min - ep_x, ep_x - x_max, 0.0)
    too_far_outside = over > span * (max_extension_ratio - 1.0)

    if min_d > max_perp_px or too_far_outside:
        # Snap to the nearest extreme on the matching side
        if ep_x <= 0.5 * (x_min + x_max):
            anchor_idx = int(np.argmin(pts[:, 0]))
        else:
            anchor_idx = int(np.argmax(pts[:, 0]))
        return pts[anchor_idx].astype(np.float32), True

    return endpoint.astype(np.float32), False


def _min_distance_to_polyline(p: np.ndarray, polyline: np.ndarray) -> float:
    """Min perpendicular distance from point p to polyline segments."""
    if len(polyline) < 2:
        return float(np.linalg.norm(polyline[0] - p)) if len(polyline) else np.inf
    best = np.inf
    for i in range(len(polyline) - 1):
        a, b = polyline[i], polyline[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-9:
            d = float(np.linalg.norm(a - p))
        else:
            t = float((p - a) @ ab / L2)
            t = max(0.0, min(1.0, t))
            proj = a + t * ab
            d = float(np.linalg.norm(proj - p))
        if d < best:
            best = d
    return best


def _project_to_polyline(p: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Return the nearest point ON the polyline to ``p`` (with segment
    interpolation). Falls back to ``p`` if polyline is empty.
    """
    if len(polyline) == 0:
        return p.astype(np.float32)
    if len(polyline) == 1:
        return polyline[0].astype(np.float32)
    best_d = np.inf
    best_proj = polyline[0]
    for i in range(len(polyline) - 1):
        a, b = polyline[i], polyline[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-9:
            proj = a
        else:
            t = float((p - a) @ ab / L2)
            t = max(0.0, min(1.0, t))
            proj = a + t * ab
        d = float(np.linalg.norm(proj - p))
        if d < best_d:
            best_d = d
            best_proj = proj
    return best_proj.astype(np.float32)


def snap_endpoint_to_sam(
    current_endpoint: np.ndarray,
    stripe_points: np.ndarray,
    sam_polyline: np.ndarray,
    side: str,  # "luff" or "leech"
) -> Tuple[np.ndarray, bool]:
    """Select the SAM-polyline point closest to the stripe.

    Two candidates are projected onto the SAM polyline:
      A. the current endpoint (post ML refine + guard)
      B. the stripe's own extreme on the appropriate side (luff = leftmost,
         leech = rightmost by x)

    We pick whichever projection sits closer to the stripe extreme point
    B_raw. Rationale: the stripe's own extreme is the most reliable piece
    of data (it's literally where the detection ends), so the SAM-
    boundary endpoint should minimise displacement from that, not from
    the floating ML-refined point.

    Returns (endpoint_on_sam, was_swapped_to_candidate_B).
    """
    if len(sam_polyline) == 0 or stripe_points is None or len(stripe_points) < 2:
        return current_endpoint.astype(np.float32), False

    pts = stripe_points.astype(np.float32)
    if side == "luff":
        anchor_idx = int(np.argmin(pts[:, 0]))
    else:
        anchor_idx = int(np.argmax(pts[:, 0]))
    stripe_extreme = pts[anchor_idx]

    proj_A = _project_to_polyline(current_endpoint, sam_polyline)
    proj_B = _project_to_polyline(stripe_extreme, sam_polyline)

    d_A = float(np.linalg.norm(proj_A - stripe_extreme))
    d_B = float(np.linalg.norm(proj_B - stripe_extreme))
    if d_B < d_A:
        return proj_B, True
    return proj_A, False


def _seg_on_crop_variants(
    image_bgr: np.ndarray,
    sail_mask: np.ndarray,
    bbox: Tuple[float, float, float, float],
    seg_model_path: str,
    seg_min_conf: float,
    pad_ratio: float = 0.08,
    legacy_model_path: Optional[str] = None,
) -> List[SegVariantResult]:
    """Run seg_on_crop across COLOR_VARIANTS of the crop.

    If ``legacy_model_path`` is provided (e.g. ``stripe_seg_v1.pt``), it
    is tried AS AN ADDITIONAL VARIANT on the raw crop. The composite
    scorer then picks whichever model+variant combination wins — so the
    first seg model contributes on stripes where the new one is sparse.
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    pad_x = max(6, int(round((x2 - x1) * pad_ratio)))
    pad_y = max(6, int(round((y2 - y1) * pad_ratio)))
    cx1 = max(0, x1 - pad_x); cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x); cy2 = min(h, y2 + pad_y)
    if cx2 - cx1 < 20 or cy2 - cy1 < 8:
        return []

    raw_crop = image_bgr[cy1:cy2, cx1:cx2].copy()
    results: List[SegVariantResult] = []
    for name, transform in COLOR_VARIANTS:
        variant_crop = transform(raw_crop.copy())
        buffer = image_bgr.copy()
        buffer[cy1:cy2, cx1:cx2] = variant_crop
        try:
            det = seg_on_crop(
                buffer, sail_mask, bbox,
                seg_model_path=seg_model_path,
                pad_ratio=pad_ratio,
                min_conf=seg_min_conf,
            )
        except Exception:
            det = None
        conf = float(det.confidence) if det is not None else 0.0
        results.append(SegVariantResult(
            variant_name=name,
            detection=det,
            confidence=conf,
            raw_crop_bgr=raw_crop,
            variant_crop_bgr=variant_crop,
            crop_origin_xy=(cx1, cy1),
        ))

    # Extra: LEGACY seg model (stripe_seg_v1.pt) on BOTH the raw crop
    # AND the channel-min variant. channel-min consistently exposes
    # stripe edges better than other preprocessing on dark sails, so
    # giving the legacy model that view catches edge points the new
    # model misses.
    if legacy_model_path and Path(legacy_model_path).exists():
        legacy_variants = [
            ("legacy v1 (raw)", raw_crop),
            ("legacy v1 (ch-min)", _channel_min(raw_crop.copy())),
        ]
        for legacy_name, variant_crop in legacy_variants:
            buffer = image_bgr.copy()
            buffer[cy1:cy2, cx1:cx2] = variant_crop
            try:
                det_legacy = seg_on_crop(
                    buffer, sail_mask, bbox,
                    seg_model_path=legacy_model_path,
                    pad_ratio=pad_ratio,
                    min_conf=seg_min_conf,
                )
            except Exception:
                det_legacy = None
            conf_legacy = float(det_legacy.confidence) if det_legacy is not None else 0.0
            results.append(SegVariantResult(
                variant_name=legacy_name,
                detection=det_legacy,
                confidence=conf_legacy,
                raw_crop_bgr=raw_crop,
                variant_crop_bgr=variant_crop,
                crop_origin_xy=(cx1, cy1),
            ))

    return results


def _merge_second_polygon_bottom(
    primary: StripeDetection,
    secondary_polygon: np.ndarray,
    image_shape: Tuple[int, int],
) -> StripeDetection:
    """Merge a 2nd-shape polygon's bottom-edge points into a primary
    StripeDetection. Only points that extend the x-range beyond the
    primary (i.e. cover the luff or leech side better) are added.
    Existing primary points are kept.
    """
    from src.polygon_fusion import polygon_bottom_edge
    if primary.points is None or len(primary.points) < 2:
        return primary
    p_x = primary.points[:, 0]
    p_min, p_max = float(p_x.min()), float(p_x.max())
    # Small tolerance to prefer expansion only (not duplication)
    tol = 4.0

    try:
        secondary_bottom = polygon_bottom_edge(
            secondary_polygon, image_shape, n_samples=24,
        )
    except Exception:
        return primary

    extra = []
    for p in secondary_bottom:
        x = float(p[0])
        if x < p_min - tol or x > p_max + tol:
            extra.append(p)

    if not extra:
        return primary

    extra_arr = np.asarray(extra, dtype=np.float32)
    merged_pts = np.vstack([primary.points, extra_arr])
    # Sort by x
    order = np.argsort(merged_pts[:, 0])
    merged_pts = merged_pts[order]
    # Dedup on proximity
    kept = [merged_pts[0]]
    for p in merged_pts[1:]:
        if np.linalg.norm(p - kept[-1]) > 3.0:
            kept.append(p)
    merged_pts = np.asarray(kept, dtype=np.float32)

    # Weights: keep existing; new ones get 0.85 (slightly below matched kp,
    # above the typical curvature 0.7 — they're reliable edge anchors).
    if primary.keypoint_confidences is not None and len(primary.keypoint_confidences) == len(primary.points):
        kp_w = primary.keypoint_confidences.astype(np.float32)
        # We lost the per-point mapping after sort+dedup; rebuild
        # approximately by matching each merged_pt to its nearest origin point
        new_w = np.zeros(len(merged_pts), dtype=np.float32)
        for i, p in enumerate(merged_pts):
            # find nearest in primary.points or in extra
            d_primary = np.min(np.linalg.norm(primary.points - p, axis=1))
            d_extra = np.min(np.linalg.norm(extra_arr - p, axis=1)) if len(extra_arr) else np.inf
            if d_primary < d_extra:
                idx = int(np.argmin(np.linalg.norm(primary.points - p, axis=1)))
                new_w[i] = kp_w[idx] if idx < len(kp_w) else 0.85
            else:
                new_w[i] = 0.85
        kp_weights = new_w
    else:
        kp_weights = None

    return StripeDetection(
        points=merged_pts,
        confidence=primary.confidence,
        orientation_deg=primary.orientation_deg,
        keypoint_confidences=kp_weights,
        polygon=primary.polygon,
    )


def _score_variant(
    det: Optional[StripeDetection],
    bbox: Tuple[float, float, float, float],
    luff_polyline: np.ndarray,
    leech_polyline: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """Composite score for a seg variant.

    Components:
      * ``confidence`` (0-1): the YOLO instance mask confidence.
      * ``length``: polygon x-range normalised by bbox width. Long = good
        (stripe spans luff to leech).
      * ``edge_luff`` / ``edge_leech``: exp(-dist / 50 px). 1.0 = polygon
        extreme sits on the SAM polyline; decays fast.
      * ``thinness``: triangular reward for aspect ratio in [4, 20].
        Fat polygons (aspect < 3) get penalised — those usually mean the
        mask blew out into the sail body.

    Weights: conf 0.35, length 0.20, edge_luff 0.15, edge_leech 0.15,
    thinness 0.15. Max score = 1.0.
    """
    if det is None or det.polygon is None or len(det.polygon) < 3:
        return -np.inf, {}

    poly = det.polygon
    conf = float(det.confidence)

    bbox_w = max(float(bbox[2] - bbox[0]), 1.0)
    bbox_h = max(float(bbox[3] - bbox[1]), 1.0)

    # Length — how much of the bbox width the polygon spans
    poly_x_range = float(poly[:, 0].max() - poly[:, 0].min())
    length_score = min(1.0, poly_x_range / bbox_w)

    # Edge proximity — leftmost / rightmost polygon point vs SAM polylines
    leftmost = poly[int(np.argmin(poly[:, 0]))]
    rightmost = poly[int(np.argmax(poly[:, 0]))]
    d_luff = _min_distance_to_polyline(leftmost, luff_polyline) if len(luff_polyline) else 200.0
    d_leech = _min_distance_to_polyline(rightmost, leech_polyline) if len(leech_polyline) else 200.0
    # Swap if the "luff side" is actually to the right of the "leech side"
    if d_luff > d_leech and len(luff_polyline) and len(leech_polyline):
        # Check by comparing to BOTH polylines at both extremes
        d_luff_right = _min_distance_to_polyline(rightmost, luff_polyline)
        d_leech_left = _min_distance_to_polyline(leftmost, leech_polyline)
        if d_luff_right + d_leech_left < d_luff + d_leech:
            d_luff, d_leech = d_luff_right, d_leech_left
    edge_luff = float(np.exp(-d_luff / 50.0))
    edge_leech = float(np.exp(-d_leech / 50.0))

    # Thinness — stripes are thin horizontal bands. Aspect = length/height.
    poly_y_range = max(float(poly[:, 1].max() - poly[:, 1].min()), 1.0)
    aspect = poly_x_range / poly_y_range
    # Triangular reward: 0 at aspect<=2, 1 in [4,20], 0 at >=40
    if aspect < 2:
        thinness = 0.0
    elif aspect < 4:
        thinness = (aspect - 2) / 2.0
    elif aspect <= 20:
        thinness = 1.0
    elif aspect < 40:
        thinness = max(0.0, (40 - aspect) / 20.0)
    else:
        thinness = 0.0

    composite = (
        0.35 * conf
        + 0.20 * length_score
        + 0.15 * edge_luff
        + 0.15 * edge_leech
        + 0.15 * thinness
    )
    breakdown = {
        "conf": conf,
        "length": length_score,
        "edge_luff": edge_luff,
        "edge_leech": edge_leech,
        "thinness": thinness,
        "aspect": aspect,
        "composite": composite,
    }
    return float(composite), breakdown


def detect_stripe_v7_per_bbox(
    image_bgr: np.ndarray,
    sail_mask: np.ndarray,
    bbox: Tuple[float, float, float, float],
    kp_dets: List[StripeDetection],
    seg_model_path: str,
    luff_polyline: Optional[np.ndarray] = None,
    leech_polyline: Optional[np.ndarray] = None,
    legacy_seg_model_path: Optional[str] = None,
    seg_min_conf: float = 0.10,
    kp_match_radius_px: float = 20.0,
) -> Optional[Tuple[
    StripeDetection,
    Optional[FusionDiagnostics],
    str,
    List[SegVariantResult],
    Optional[str],
]]:
    """Run stages D-F on a single bbox.

    D is run across COLOR_VARIANTS of the crop — best (highest confidence)
    wins. The variant sweep's per-variant diagnostics are returned so the
    report can show *what the model saw* for each variant.

    Priority:
      1. seg_on_crop (best variant) + kp fusion  ->  "seg_crop+kp" or "seg_crop"
      2. kp only                                 ->  "kp"
      3. classical ridge                         ->  "classical"
    """
    kp_in_bb = best_kp_for_bbox(bbox, kp_dets)
    luff_poly = (
        luff_polyline if luff_polyline is not None
        else np.zeros((0, 2), dtype=np.float32)
    )
    leech_poly = (
        leech_polyline if leech_polyline is not None
        else np.zeros((0, 2), dtype=np.float32)
    )

    # 1. Seg on crop — sweep color variants, score each one with the
    #    composite SAM-aware metric, pick highest score.
    variants = _seg_on_crop_variants(
        image_bgr, sail_mask, bbox,
        seg_model_path=seg_model_path,
        seg_min_conf=seg_min_conf,
        legacy_model_path=legacy_seg_model_path,
    )
    # Score every variant (including the None-detection ones get -inf)
    for v in variants:
        score, breakdown = _score_variant(v.detection, bbox, luff_poly, leech_poly)
        v.composite_score = score
        v.score_breakdown = breakdown

    valid = [v for v in variants if v.detection is not None]
    best_variant_name: Optional[str] = None
    seg_det = None
    if valid:
        best_variant = max(valid, key=lambda v: v.composite_score)
        seg_det = best_variant.detection
        best_variant_name = best_variant.variant_name

    if seg_det is not None and seg_det.polygon is not None:
        # Pull out channel-min variant (even if it wasn't the winner) and
        # merge its polygon-bottom points with the winner's — this is the
        # "second segment shape to fill edges" step. channel-min
        # consistently extends further toward the sail edges, so its
        # bottom edge on the OUTSIDE of the winner's range provides
        # high-weight anchors for the spline fit.
        chmin_variant = next(
            (v for v in valid if v.variant_name == "channel-min"),
            None,
        )
        chmin_poly = None
        if (chmin_variant is not None
                and chmin_variant is not best_variant
                and chmin_variant.detection is not None
                and chmin_variant.detection.polygon is not None):
            chmin_poly = chmin_variant.detection.polygon

        if kp_in_bb is not None:
            fused, diag = fuse_polygon_with_keypoints(
                seg_det, kp_in_bb, image_shape=image_bgr.shape[:2],
                kp_match_radius_px=kp_match_radius_px,
            )
        else:
            fused, diag = fuse_polygon_with_keypoints(
                seg_det, None, image_shape=image_bgr.shape[:2],
                kp_match_radius_px=kp_match_radius_px,
            )

        # Merge channel-min bottom-edge points for extra edge coverage
        if chmin_poly is not None:
            fused = _merge_second_polygon_bottom(
                fused, chmin_poly, image_bgr.shape[:2],
            )

        source = "seg_crop+kp" if kp_in_bb is not None else "seg_crop"
        if chmin_poly is not None:
            source += "+chmin"
        return fused, diag, source, variants, best_variant_name

    # 2. kp only
    if kp_in_bb is not None and len(kp_in_bb.points) >= 3:
        return kp_in_bb, None, "kp", variants, best_variant_name

    # 3. Classical fallback
    classical = classical_ridge_in_crop(image_bgr, sail_mask, bbox)
    if classical is not None:
        return classical, None, "classical", variants, best_variant_name

    return None


def refine_endpoints_two_stage(
    luff_ep: np.ndarray,
    leech_ep: np.ndarray,
    image_bgr: np.ndarray,
    sail_boundary: SailBoundary,
    config: dict,
    max_nudge_px: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """Stage G — two-stage ML endpoint refinement.

    Stage 1: full-image stripe_endpoints_v2.pt -> candidate (x, y, conf).
             Only applied when the v2 candidate is within max_nudge_px of
             the raw geometric endpoint on the correct polyline.
    Stage 2: stripe_endpoint_crop_26x.pt on a 256 px crop around each
             endpoint — small local nudge (also capped by max_nudge_px).

    Returns (refined_luff, refined_leech, (luff_total_move, leech_total_move)).
    """
    orig_luff = luff_ep.astype(np.float32).copy()
    orig_leech = leech_ep.astype(np.float32).copy()
    out_luff = orig_luff.copy()
    out_leech = orig_leech.copy()

    # --- Stage 1: full-image endpoint model (v2) ---
    v2_path = str(config.get("endpoint_model_path", "stripe_endpoints_v2.pt"))
    if Path(v2_path).exists():
        mdl_eps = detect_endpoints_model(image_bgr, {
            "endpoint_model_path": v2_path,
            "endpoint_model_fallback": config.get(
                "endpoint_model_fallback", "stripe_endpoints_v1.pt"
            ),
            "endpoint_min_confidence": config.get(
                "endpoint_min_confidence", 0.05
            ),
            "endpoint_imgsz": config.get("endpoint_imgsz", 1280),
        })
        if mdl_eps.shape[0] > 0:
            out_luff = refine_endpoint_with_model(
                out_luff, sail_boundary.luff_polyline, mdl_eps,
                max_distance=max_nudge_px,
                max_polyline_offset=40.0,
                max_y_gap=25.0,
            )
            out_leech = refine_endpoint_with_model(
                out_leech, sail_boundary.leech_polyline, mdl_eps,
                max_distance=max_nudge_px,
                max_polyline_offset=40.0,
                max_y_gap=25.0,
            )

    # --- Stage 2: crop specialist ---
    crop_path = str(config.get(
        "endpoint_crop_model_path", "stripe_endpoint_crop_26x.pt"
    ))
    if Path(crop_path).exists():
        crop_size = int(config.get("endpoint_crop_size", 256))
        out_luff = refine_endpoint_with_crop_model(
            out_luff, image_bgr, crop_path,
            crop_size=crop_size, imgsz=crop_size,
            max_offset_px=max_nudge_px,
        )
        out_leech = refine_endpoint_with_crop_model(
            out_leech, image_bgr, crop_path,
            crop_size=crop_size, imgsz=crop_size,
            max_offset_px=max_nudge_px,
        )

    # Enforce the total-nudge budget against the ORIGINAL geometric ep
    if float(np.linalg.norm(out_luff - orig_luff)) > max_nudge_px:
        out_luff = orig_luff
    if float(np.linalg.norm(out_leech - orig_leech)) > max_nudge_px:
        out_leech = orig_leech

    return (
        out_luff,
        out_leech,
        (
            float(np.linalg.norm(out_luff - orig_luff)),
            float(np.linalg.norm(out_leech - orig_leech)),
        ),
    )


def _split_contaminated_detection(
    det: StripeDetection,
    max_perp_std_frac: float = 0.06,
    min_inter_cluster_sigma: float = 3.5,
    max_clusters: int = 4,
) -> List[StripeDetection]:
    """Split a contaminated detection (one StripeDetection covering
    multiple physical stripes) into per-stripe sub-detections.

    Trigger: perpendicular std > ``max_perp_std_frac`` × chord length
    AND a KMeans clustering with k ≥ 2 yields clusters whose mean
    separation is ≥ ``min_inter_cluster_sigma`` × max intra-cluster std
    (the same separability criterion used for the merge — so a clean
    stripe with deep camber will NOT be split, only true multi-stripe
    contamination is).
    """
    pts = det.points
    if pts is None or len(pts) < 6:
        return [det]
    sorted_pts = pts[np.argsort(pts[:, 0])].astype(np.float64)
    chord = sorted_pts[-1] - sorted_pts[0]
    L = float(np.linalg.norm(chord))
    if L < 1.0:
        return [det]
    u = chord / L
    n = np.array([-u[1], u[0]], dtype=np.float64)
    perp = sorted_pts @ n
    perp_std = float(np.std(perp))
    if perp_std / L <= max_perp_std_frac:
        return [det]

    try:
        from sklearn.cluster import KMeans
    except Exception:
        return [det]

    confs = (
        det.keypoint_confidences
        if det.keypoint_confidences is not None and len(det.keypoint_confidences) == len(pts)
        else None
    )
    if confs is not None:
        confs = np.asarray(confs, dtype=np.float32)[np.argsort(pts[:, 0])]

    for k in range(2, max_clusters + 1):
        try:
            km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(perp.reshape(-1, 1))
        except Exception:
            continue
        labels = km.labels_
        cluster_data = []
        ok = True
        for c in range(k):
            mask = labels == c
            if mask.sum() < 3:
                ok = False
                break
            cp = sorted_pts[mask]
            cperp = cp @ n
            cluster_data.append((mask, cp, float(np.mean(cperp)), float(np.std(cperp))))
        if not ok:
            continue
        # Pairwise separation gate: every cluster pair must satisfy
        # |μ1 - μ2| / max(σ1, σ2) >= min_inter_cluster_sigma. If the
        # clusters don't have a clean perpendicular gap, this isn't a
        # multi-stripe contamination, just a curved single stripe.
        separable = True
        for i in range(len(cluster_data)):
            for j in range(i + 1, len(cluster_data)):
                m1, _, mean1, std1 = cluster_data[i]
                m2, _, mean2, std2 = cluster_data[j]
                max_std = max(std1, std2, 0.5)
                if abs(mean1 - mean2) / max_std < min_inter_cluster_sigma:
                    separable = False
                    break
            if not separable:
                break
        if not separable:
            continue
        clusters: List[StripeDetection] = []
        for mask, cp, _, _ in cluster_data:
            cluster_confs = confs[mask] if confs is not None else None
            clusters.append(StripeDetection(
                points=cp.astype(np.float32),
                confidence=float(det.confidence),
                orientation_deg=float(det.orientation_deg),
                keypoint_confidences=cluster_confs,
            ))
        return clusters
    return [det]


def _bbox_of_points(pts: np.ndarray) -> Tuple[float, float, float, float]:
    return (
        float(pts[:, 0].min()), float(pts[:, 1].min()),
        float(pts[:, 0].max()), float(pts[:, 1].max()),
    )


def run_v7_detection_only(
    image_bgr: np.ndarray,
    sail_boundary: SailBoundary,
    bboxes: List[np.ndarray],
    kp_dets: List[StripeDetection],
    seg_model_path: str,
    legacy_seg_model_path: Optional[str] = None,
) -> List[V7StripeResult]:
    """Run stages D-F (detection + fusion) for every bbox.

    Returns one V7StripeResult per CLEAN stripe. Contaminated detections
    (whose points cover multiple physical stripes) are split via
    KMeans on the chord-perpendicular axis. Every result's ``bbox``
    is the tight bbox of its detection points (not the source YOLO
    bbox), so the downstream merge can use bbox identity safely.
    """
    results: List[V7StripeResult] = []
    for bb in bboxes:
        bb_t = tuple(float(v) for v in bb)
        det_out = detect_stripe_v7_per_bbox(
            image_bgr, sail_boundary.mask, bb_t,
            kp_dets=kp_dets,
            seg_model_path=seg_model_path,
            luff_polyline=sail_boundary.luff_polyline,
            leech_polyline=sail_boundary.leech_polyline,
            legacy_seg_model_path=legacy_seg_model_path,
        )
        if det_out is None:
            continue
        det, diag, source, seg_variants, best_variant_name = det_out

        # Split if contaminated. Each clean sub-detection becomes its
        # own V7StripeResult.
        sub_dets = _split_contaminated_detection(det)
        for sub in sub_dets:
            if sub.points is None or len(sub.points) < 3:
                continue
            sub_bb = _bbox_of_points(sub.points)
            raw_ep = endpoints_from_detection(sub, sub_bb, sail_boundary=sail_boundary)
            if raw_ep is None:
                continue
            raw_luff, raw_leech = raw_ep
            results.append(V7StripeResult(
                bbox=sub_bb,
                detection=sub,
                fusion=diag,
                source=source if len(sub_dets) == 1 else f"{source}/split",
                raw_luff_ep=raw_luff.astype(np.float32),
                raw_leech_ep=raw_leech.astype(np.float32),
                luff_ep=raw_luff.astype(np.float32),
                leech_ep=raw_leech.astype(np.float32),
                endpoint_nudge_px=(0.0, 0.0),
                endpoint_guarded=(False, False),
                seg_variants=seg_variants,
                best_seg_variant=best_variant_name,
            ))
    return results


def run_v7_endpoints(
    detections: List[V7StripeResult],
    image_bgr: np.ndarray,
    sail_boundary: SailBoundary,
    config: Optional[dict] = None,
) -> List[V7StripeResult]:
    """Run stages G-H over a list of pre-detected stripes.

    Two endpoint backends:

    * ``endpoint_method="tangent_extension"`` (NEW DEFAULT) — derive
      ``(luff_ep, leech_ep)`` purely from detection points + the SAM
      luff/leech polylines. Combines a degree-3 chord-space polynomial
      ray-cast (shape prior) with a linear extrapolation from the
      first/last 3 aligned detection points; averages both candidates.
      No precomputed endpoint or ML refiner.

    * ``endpoint_method="legacy"`` — original v7 path: detection-extreme
      endpoints + optional ML refiner + perpendicular guard.

    After endpoints are set, ``merge_overlapping_stripes`` fuses results
    that represent the same physical stripe (same chord direction +
    midpoints close in chord-perpendicular distance) so the spline fit
    sees richer point sets.
    """
    import copy as _copy
    from src.stripe_endpoints import compute_stripe_endpoints
    from src.endpoint_fusion import compute_fused_endpoints
    config = config or {}
    method = str(config.get("endpoint_method", "fused"))
    use_ml = bool(config.get("endpoint_use_ml_refiner", True))
    guard_max_perp = float(config.get("endpoint_guard_max_perp_px", 12.0))
    max_nudge = float(config.get("endpoint_max_nudge_px", 30.0))

    out: List[V7StripeResult] = []
    for r_in in detections:
        r = _copy.copy(r_in)           # shallow copy; arrays replaced below

        ep_pair = None
        if method == "fused":
            # Robust 4-method fusion (Weiszfeld + MAD).  ML refiner is
            # included only when explicitly enabled and models are present.
            fused = compute_fused_endpoints(
                r.detection.points,
                luff_polyline=sail_boundary.luff_polyline,
                leech_polyline=sail_boundary.leech_polyline,
                image_bgr=image_bgr if use_ml else None,
                sail_boundary=sail_boundary if use_ml else None,
                use_ml_refiner=use_ml,
                keypoint_confidences=r.detection.keypoint_confidences,
            )
            if fused is not None:
                fused_luff, fused_leech, _conf_l, _conf_r = fused
                ep_pair = (fused_luff, fused_leech)
        elif method == "tangent_extension":
            ep_pair = compute_stripe_endpoints(
                r.detection.points,
                luff_polyline=sail_boundary.luff_polyline,
                leech_polyline=sail_boundary.leech_polyline,
                keypoint_confidences=r.detection.keypoint_confidences,
            )

        if ep_pair is not None:
            luff_ep_arr, leech_ep_arr = ep_pair
            luff_ep = np.asarray(luff_ep_arr, dtype=np.float32)
            leech_ep = np.asarray(leech_ep_arr, dtype=np.float32)
            luff_move = leech_move = 0.0
            luff_guarded = leech_guarded = False
        else:
            if use_ml:
                luff_ep, leech_ep, (luff_move, leech_move) = refine_endpoints_two_stage(
                    r.raw_luff_ep, r.raw_leech_ep, image_bgr, sail_boundary,
                    config, max_nudge_px=max_nudge,
                )
            else:
                luff_ep = r.raw_luff_ep.astype(np.float32)
                leech_ep = r.raw_leech_ep.astype(np.float32)
                luff_move = leech_move = 0.0
            luff_ep, luff_guarded = guard_endpoint(luff_ep, r.detection.points, guard_max_perp)
            leech_ep, leech_guarded = guard_endpoint(leech_ep, r.detection.points, guard_max_perp)

        r.endpoint_snapped_to_sam = (False, False)
        r.luff_ep = luff_ep.astype(np.float32)
        r.leech_ep = leech_ep.astype(np.float32)
        r.endpoint_nudge_px = (luff_move, leech_move)
        r.endpoint_guarded = (luff_guarded, leech_guarded)
        r.spline_points = None
        r.spline_meta = None
        out.append(r)

    as_triples = [(r.detection, r.luff_ep, r.leech_ep) for r in out]
    kept = dedup_stripe_endpoints(as_triples, min_separation_px=15.0)
    kept_ids = {id(d) for d, _, _ in kept}
    out = [r for r in out if id(r.detection) in kept_ids]
    # Conservative merge: only fuse when point clouds are essentially
    # inseparable on the perpendicular axis. Cape31 close-spaced stripes
    # (~20 px apart) need sigma ≤ 0.3; the original 2.5 collapsed real
    # distinct stripes (jib_2, main_3 demonstrated 3→1).
    out = merge_overlapping_stripes(out, min_separation_sigma=0.3)
    # Additional merge: fuse collinear-but-disjoint-bbox detections that
    # represent the same physical stripe split by the detector.
    out = merge_collinear_stripes(out, max_angle_deg=8.0,
                                  max_perp_dist_px=15.0, max_gap_frac=0.30)
    return out


def _bbox_max_overlap_frac(b1, b2) -> float:
    """Max of (intersection / area_b1, intersection / area_b2).

    Returns 1.0 when one bbox is fully contained in the other (partial-
    within-full-span), reasonable values for overlapping halves, and 0
    for clearly-distinct boxes.
    """
    x_inter = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    y_inter = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    inter = x_inter * y_inter
    if inter <= 0:
        return 0.0
    a1 = max(1e-6, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1e-6, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return max(inter / a1, inter / a2)


def _stripes_are_coincident(
    r1: V7StripeResult,
    r2: V7StripeResult,
    max_angle_deg: float = 12.0,
    min_separation_sigma: float = 2.5,
    max_x_gap_frac: float = 0.50,
    min_bbox_overlap_for_same_stripe: float = 0.30,
) -> bool:
    """Statistical separability test on detection points.

    Two stripes are coincident (the SAME physical stripe) only when:

      1. Their chord direction angles agree within ``max_angle_deg``.
      2. Projected onto a SHARED perpendicular-to-chord axis, the two
         clouds CANNOT be cleanly separated — i.e. the cluster gap is
         smaller than ``min_separation_sigma`` × (max intra-cluster std).
         If they CAN be separated (clean perpendicular gap), they are
         distinct stripes regardless of how parallel they are.
      3. Their detection-point x-ranges either overlap, or the gap
         between them is at most ``max_x_gap_frac`` of the combined
         extent (so we don't merge two stripes that happen to be
         collinear but cover totally disjoint halves of the sail).

    The shared perpendicular axis comes from the AVERAGE of the two
    chord directions, so the test works regardless of which side of the
    sail the stripes cover.
    """
    pts1 = r1.detection.points if r1.detection is not None else None
    pts2 = r2.detection.points if r2.detection is not None else None
    if pts1 is None or pts2 is None or len(pts1) < 3 or len(pts2) < 3:
        return False

    # HARD CONSTRAINT: two stripes whose source bboxes are clearly
    # distinct (no significant overlap in either direction) are NEVER
    # merged regardless of how parallel their points are. Using max
    # overlap fraction (instead of IoU) catches the partial-bbox-
    # contained-in-full-span case correctly.
    overlap = _bbox_max_overlap_frac(r1.bbox, r2.bbox)
    if overlap < min_bbox_overlap_for_same_stripe:
        return False

    s1 = pts1[np.argsort(pts1[:, 0])].astype(np.float64)
    s2 = pts2[np.argsort(pts2[:, 0])].astype(np.float64)
    c1 = s1[-1] - s1[0]
    c2 = s2[-1] - s2[0]
    L1 = float(np.linalg.norm(c1))
    L2 = float(np.linalg.norm(c2))
    if L1 < 1.0 or L2 < 1.0:
        return False

    a1 = np.degrees(np.arctan2(c1[1], c1[0]))
    a2 = np.degrees(np.arctan2(c2[1], c2[0]))
    angle_diff = abs(((a1 - a2 + 180.0) % 360.0) - 180.0)
    if angle_diff > max_angle_deg:
        return False

    # Shared perpendicular axis from the average chord direction so the
    # cluster separation test is symmetric.
    u_avg = (c1 / L1 + c2 / L2) / 2.0
    nrm = float(np.linalg.norm(u_avg))
    if nrm < 1e-6:
        return False
    u_avg = u_avg / nrm
    n_avg = np.array([-u_avg[1], u_avg[0]], dtype=np.float64)

    perp1 = s1 @ n_avg
    perp2 = s2 @ n_avg
    mean1 = float(np.mean(perp1))
    mean2 = float(np.mean(perp2))
    std1 = float(np.std(perp1))
    std2 = float(np.std(perp2))
    max_std = max(std1, std2, 0.5)
    separation = abs(mean1 - mean2) / max_std
    if separation >= min_separation_sigma:
        return False  # cleanly separable along the perpendicular axis

    # Even when not separable, refuse to merge if x-ranges are far
    # apart with a huge gap (collinear but disjoint parts of the sail).
    x1_lo, x1_hi = float(s1[0, 0]), float(s1[-1, 0])
    x2_lo, x2_hi = float(s2[0, 0]), float(s2[-1, 0])
    x_inter = max(0.0, min(x1_hi, x2_hi) - max(x1_lo, x2_lo))
    if x_inter <= 0:
        gap = max(x2_lo - x1_hi, x1_lo - x2_hi)
        combined_extent = max(x1_hi, x2_hi) - min(x1_lo, x2_lo)
        if combined_extent <= 0:
            return False
        if (gap / combined_extent) > max_x_gap_frac:
            return False
    return True


def merge_overlapping_stripes(
    results: List[V7StripeResult],
    max_angle_deg: float = 12.0,
    min_separation_sigma: float = 2.5,
    max_x_gap_frac: float = 0.50,
    min_bbox_overlap_for_same_stripe: float = 0.30,
) -> List[V7StripeResult]:
    """Group V7StripeResults that represent the same physical stripe and
    fuse their detection points into a single richer detection.

    Coincidence is judged by ``_stripes_are_coincident`` — strict chord-
    line geometry (perpendicular distance + angle + x-overlap), so
    adjacent stripes with overlapping detection-point bounding boxes do
    NOT get merged.

    Within a group the result with the longest chord becomes the base;
    all detection points (with their per-keypoint confidences) are
    concatenated into the merged detection.
    """
    import copy as _copy
    if len(results) <= 1:
        return list(results)

    n = len(results)
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if _stripes_are_coincident(
                results[i], results[j],
                max_angle_deg=max_angle_deg,
                min_separation_sigma=min_separation_sigma,
                max_x_gap_frac=max_x_gap_frac,
                min_bbox_overlap_for_same_stripe=min_bbox_overlap_for_same_stripe,
            ):
                _union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(_find(i), []).append(i)

    merged: List[V7StripeResult] = []
    for idxs in groups.values():
        if len(idxs) == 1:
            merged.append(results[idxs[0]])
            continue
        base_idx = max(
            idxs,
            key=lambda k: float(np.linalg.norm(
                results[k].leech_ep - results[k].luff_ep
            )),
        )
        base = results[base_idx]
        all_pts = []
        all_confs = []
        for k in idxs:
            d = results[k].detection
            if d is None or d.points is None or len(d.points) == 0:
                continue
            all_pts.append(np.asarray(d.points, dtype=np.float32))
            if d.keypoint_confidences is not None and len(d.keypoint_confidences) == len(d.points):
                all_confs.append(np.asarray(d.keypoint_confidences, dtype=np.float32))
            else:
                all_confs.append(np.ones(len(d.points), dtype=np.float32) * float(d.confidence))
        if not all_pts:
            merged.append(base)
            continue
        pts = np.vstack(all_pts)
        confs = np.concatenate(all_confs)
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        confs = confs[order]
        new_det = StripeDetection(
            points=pts,
            confidence=float(np.mean([results[k].detection.confidence for k in idxs])),
            orientation_deg=float(base.detection.orientation_deg),
            keypoint_confidences=confs,
        )
        merged_r = _copy.copy(base)
        merged_r.detection = new_det
        merged_r.source = "+".join(sorted({results[k].source for k in idxs}))
        merged.append(merged_r)
    return merged


def _stripes_are_collinear(
    r1: V7StripeResult,
    r2: V7StripeResult,
    max_angle_deg: float = 8.0,
    max_perp_dist_px: float = 15.0,
    max_gap_frac: float = 0.30,
) -> bool:
    """Return True when two stripes are the same physical stripe split by
    the detector into two disjoint detections.

    Criteria (ALL must hold):
    1. Chord angle difference < ``max_angle_deg``.
    2. Perpendicular distance from each stripe's midpoint to the other
       stripe's chord line < ``max_perp_dist_px``.
    3. The gap between their x-ranges is < ``max_gap_frac`` of their
       combined x-extent (so we don't merge stripes that happen to be
       parallel but at opposite ends of the sail).
    """
    pts1 = r1.detection.points if r1.detection is not None else None
    pts2 = r2.detection.points if r2.detection is not None else None
    if pts1 is None or pts2 is None or len(pts1) < 2 or len(pts2) < 2:
        return False

    s1 = pts1[np.argsort(pts1[:, 0])].astype(np.float64)
    s2 = pts2[np.argsort(pts2[:, 0])].astype(np.float64)
    c1 = s1[-1] - s1[0]
    c2 = s2[-1] - s2[0]
    L1 = float(np.linalg.norm(c1))
    L2 = float(np.linalg.norm(c2))
    if L1 < 1.0 or L2 < 1.0:
        return False

    # 1. Angle check
    a1 = np.degrees(np.arctan2(c1[1], c1[0]))
    a2 = np.degrees(np.arctan2(c2[1], c2[0]))
    angle_diff = abs(((a1 - a2 + 180.0) % 360.0) - 180.0)
    if angle_diff > max_angle_deg:
        return False

    # 2. Perpendicular distance check — both midpoints must lie close to
    #    the chord line of the other stripe.
    def _perp_dist_to_chord(pt: np.ndarray, chord_start: np.ndarray,
                             chord_unit: np.ndarray) -> float:
        rel = pt - chord_start
        along = float(rel @ chord_unit)
        foot = chord_start + along * chord_unit
        return float(np.linalg.norm(pt - foot))

    mid1 = float(np.mean(s1[:, 0])), float(np.mean(s1[:, 1]))
    mid2 = float(np.mean(s2[:, 0])), float(np.mean(s2[:, 1]))
    mid1_arr = np.array(mid1)
    mid2_arr = np.array(mid2)
    u1 = c1 / L1
    u2 = c2 / L2
    d12 = _perp_dist_to_chord(mid2_arr, s1[0], u1)
    d21 = _perp_dist_to_chord(mid1_arr, s2[0], u2)
    if d12 > max_perp_dist_px or d21 > max_perp_dist_px:
        return False

    # 3. Gap check — x-ranges must not be too far apart
    x1_lo, x1_hi = float(s1[0, 0]), float(s1[-1, 0])
    x2_lo, x2_hi = float(s2[0, 0]), float(s2[-1, 0])
    x_inter = max(0.0, min(x1_hi, x2_hi) - max(x1_lo, x2_lo))
    if x_inter <= 0:
        gap = max(x2_lo - x1_hi, x1_lo - x2_hi)
        combined = max(x1_hi, x2_hi) - min(x1_lo, x2_lo)
        if combined <= 0 or (gap / combined) > max_gap_frac:
            return False
    return True


def merge_collinear_stripes(
    results: List[V7StripeResult],
    max_angle_deg: float = 8.0,
    max_perp_dist_px: float = 15.0,
    max_gap_frac: float = 0.30,
) -> List[V7StripeResult]:
    """Merge stripes that are collinear even when their bboxes are disjoint.

    This handles the case where the detector splits a single physical stripe
    into two detections covering different halves of the sail. The bbox
    overlap guard in ``_stripes_are_coincident`` prevents their merge, but
    collinear geometry makes it clear they're the same stripe.
    """
    import copy as _copy
    if len(results) <= 1:
        return list(results)

    n = len(results)
    parent = list(range(n))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if _stripes_are_collinear(
                results[i], results[j],
                max_angle_deg=max_angle_deg,
                max_perp_dist_px=max_perp_dist_px,
                max_gap_frac=max_gap_frac,
            ):
                _union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(_find(i), []).append(i)

    merged: List[V7StripeResult] = []
    for idxs in groups.values():
        if len(idxs) == 1:
            merged.append(results[idxs[0]])
            continue
        # Pick base as the longest chord
        base_idx = max(
            idxs,
            key=lambda k: float(np.linalg.norm(
                results[k].leech_ep - results[k].luff_ep
            )),
        )
        base = results[base_idx]
        all_pts = []
        all_confs = []
        for k in idxs:
            d = results[k].detection
            if d is None or d.points is None or len(d.points) == 0:
                continue
            all_pts.append(np.asarray(d.points, dtype=np.float32))
            if d.keypoint_confidences is not None and len(d.keypoint_confidences) == len(d.points):
                all_confs.append(np.asarray(d.keypoint_confidences, dtype=np.float32))
            else:
                all_confs.append(np.ones(len(d.points), dtype=np.float32) * float(d.confidence))
        if not all_pts:
            merged.append(base)
            continue
        pts = np.vstack(all_pts)
        confs = np.concatenate(all_confs)
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        confs = confs[order]
        # Expand endpoints to cover all merged detection points
        new_raw_luff = min(
            [results[k].raw_luff_ep for k in idxs],
            key=lambda ep: float(ep[0]),
        ).astype(np.float32)
        new_raw_leech = max(
            [results[k].raw_leech_ep for k in idxs],
            key=lambda ep: float(ep[0]),
        ).astype(np.float32)
        new_det = StripeDetection(
            points=pts,
            confidence=float(np.mean([results[k].detection.confidence for k in idxs])),
            orientation_deg=float(base.detection.orientation_deg),
            keypoint_confidences=confs,
        )
        merged_r = _copy.copy(base)
        merged_r.detection = new_det
        merged_r.raw_luff_ep = new_raw_luff
        merged_r.raw_leech_ep = new_raw_leech
        merged_r.luff_ep = new_raw_luff.copy()
        merged_r.leech_ep = new_raw_leech.copy()
        merged_r.source = "+".join(sorted({results[k].source for k in idxs}))
        merged.append(merged_r)
    return merged


def _splines_cross(
    sp1: np.ndarray,
    sp2: np.ndarray,
) -> bool:
    """Return True if two splines (polyline representations) cross each other."""
    if sp1 is None or sp2 is None or len(sp1) < 2 or len(sp2) < 2:
        return False

    def _seg_intersect(p1, p2, p3, p4) -> bool:
        """Do segment p1-p2 and p3-p4 intersect (strictly)?"""
        d1 = p2 - p1; d2 = p4 - p3
        denom = float(d1[0] * d2[1] - d1[1] * d2[0])
        if abs(denom) < 1e-9:
            return False
        t = float(((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom)
        u = float(((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / denom)
        return 0.05 < t < 0.95 and 0.05 < u < 0.95

    for i in range(len(sp1) - 1):
        a, b = sp1[i].astype(np.float64), sp1[i + 1].astype(np.float64)
        for j in range(len(sp2) - 1):
            c, d = sp2[j].astype(np.float64), sp2[j + 1].astype(np.float64)
            if _seg_intersect(a, b, c, d):
                return True
    return False


def filter_crossing_stripes(
    results: List[V7StripeResult],
) -> List[V7StripeResult]:
    """Drop lower-confidence stripes that cross a higher-confidence one.

    Two stripe splines must not intersect. When they do, one is a spurious
    detection. Drop the one with the lower detection confidence.
    """
    if len(results) <= 1:
        return list(results)

    n = len(results)
    # Sort by confidence descending so we keep high-confidence first
    order = sorted(range(n), key=lambda i: -float(results[i].detection.confidence))
    kept = []
    dropped = set()
    for i in order:
        if i in dropped:
            continue
        r_i = results[i]
        sp_i = r_i.spline_points
        cross = False
        for j in kept:
            sp_j = results[j].spline_points
            if _splines_cross(sp_i, sp_j):
                cross = True
                break
        if not cross:
            kept.append(i)
        else:
            dropped.add(i)

    # Return in original order
    kept_set = set(kept)
    return [results[i] for i in range(n) if i in kept_set]


def filter_wrong_draft_direction(
    results: List[V7StripeResult],
    head_point: Optional[np.ndarray] = None,
) -> List[V7StripeResult]:
    """Drop stripes whose camber bowl points the wrong way.

    Convention: on a sail, all stripes' draft bowls point away from the
    head (toward the foot). We compute the majority sign of the median
    perpendicular offset of each stripe's detection points relative to
    its chord, and reject stripes that disagree with the majority.

    If ``head_point`` is supplied, we use it to define the "toward foot"
    direction as an additional sanity check (the bowl must lie on the
    same side as the foot relative to each chord).
    """
    if len(results) <= 1:
        return list(results)

    signs = []
    for r in results:
        if r.detection is None or r.detection.points is None:
            signs.append(0)
            continue
        luff = np.asarray(r.luff_ep, dtype=np.float64)
        leech = np.asarray(r.leech_ep, dtype=np.float64)
        chord = leech - luff
        L = float(np.linalg.norm(chord))
        if L < 1.0:
            signs.append(0)
            continue
        u = chord / L
        n_vec = np.array([-u[1], u[0]], dtype=np.float64)
        pts = np.asarray(r.detection.points, dtype=np.float64)
        perp = (pts - luff) @ n_vec
        med = float(np.median(perp))
        signs.append(1 if med > 1.0 else (-1 if med < -1.0 else 0))

    majority = sum(1 for s in signs if s > 0) - sum(1 for s in signs if s < 0)
    majority_sign = 1 if majority >= 0 else -1

    out = []
    for r, s in zip(results, signs):
        if s == 0 or s == majority_sign:
            out.append(r)
        # drop stripes with wrong-direction bowl
    return out if out else list(results)


def run_v7_detection_stage(
    image_bgr: np.ndarray,
    sail_boundary: SailBoundary,
    bboxes: List[np.ndarray],
    kp_dets: List[StripeDetection],
    seg_model_path: str,
    config: Optional[dict] = None,
) -> List[V7StripeResult]:
    """Back-compat wrapper: detection + endpoints in one call."""
    dets = run_v7_detection_only(
        image_bgr, sail_boundary, bboxes, kp_dets, seg_model_path,
    )
    return run_v7_endpoints(dets, image_bgr, sail_boundary, config or {})


def fit_splines(
    results: List[V7StripeResult],
) -> List[V7StripeResult]:
    """Stage I — chord-space smoothing spline.

    Primary: ``fit_chord_smoothing_spline`` — 1D UnivariateSpline on
    (t, d) where t is chord coordinate and d is perpendicular offset.
    Endpoints pinned (weight 10). This follows the raw data faithfully
    without the sideways drift the 2D splprep had.

    Fallback: Bernstein degree 5 with reg 0.005 if the smoothing spline
    fails (too few unique t, monotonicity issues).
    """
    from src.flexible_fit import fit_chord_smoothing_spline, fit_bernstein_flex

    out: List[V7StripeResult] = []
    for r in results:
        fitted = False
        try:
            meta, _, spline = fit_chord_smoothing_spline(
                r.detection.points, r.luff_ep, r.leech_ep,
                smoothness=0.4,
                keypoint_confidences=r.detection.keypoint_confidences,
            )
            r.spline_points = spline
            r.spline_meta = meta
            out.append(r)
            fitted = True
        except Exception:
            pass
        if fitted:
            continue
        try:
            meta, _, spline = fit_bernstein_flex(
                r.detection.points, r.luff_ep, r.leech_ep,
                degree=5, reg=0.005,
                keypoint_confidences=r.detection.keypoint_confidences,
            )
            r.spline_points = spline
            r.spline_meta = meta
            out.append(r)
        except Exception:
            continue
    return out


def intersect_spline_with_polyline(
    spline_points: np.ndarray,
    polyline: np.ndarray,
    fallback_endpoint: np.ndarray,
) -> np.ndarray:
    """Find where the spline crosses the SAM polyline.

    Walks along the spline segments, reports the first point that lies
    within 2 px of the polyline. If no crossing found, returns the
    polyline point closest to the fallback endpoint.
    """
    if len(spline_points) < 2 or len(polyline) < 2:
        return fallback_endpoint.astype(np.float32)

    def _seg_dist(p, a, b):
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-9:
            return float(np.linalg.norm(a - p)), a
        t = float((p - a) @ ab / L2)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(proj - p)), proj

    best_d = np.inf
    best_pt = None
    for sp in spline_points:
        for i in range(len(polyline) - 1):
            d, proj = _seg_dist(sp, polyline[i], polyline[i + 1])
            if d < best_d:
                best_d = d
                best_pt = proj

    if best_pt is not None:
        return best_pt.astype(np.float32)
    return fallback_endpoint.astype(np.float32)


def combine_ab_results(
    results_a: List[V7StripeResult],
    results_b: List[V7StripeResult],
) -> List[V7StripeResult]:
    """Per bbox, pick whichever of (A, B) fit has lower mean residual to
    the stripe detection points.

    Residual = mean perpendicular distance from the stripe's detection
    points to the fitted spline curve.

    Tie-breaker: shorter chord length (less reaching beyond the data).
    """
    # Index by bbox tuple so we match A and B even after dedup dropped
    # different stripes.
    idx_a = {r.bbox: r for r in results_a if r.spline_points is not None}
    idx_b = {r.bbox: r for r in results_b if r.spline_points is not None}
    all_bboxes = set(idx_a.keys()) | set(idx_b.keys())

    combined: List[V7StripeResult] = []
    for bb in sorted(all_bboxes, key=lambda b: (b[1], b[0])):
        a = idx_a.get(bb)
        b = idx_b.get(bb)
        if a is None:
            combined.append(b)
            continue
        if b is None:
            combined.append(a)
            continue
        r_a = _spline_residual(a)
        r_b = _spline_residual(b)
        if r_a < r_b:
            combined.append(a)
        elif r_b < r_a:
            combined.append(b)
        else:
            # tie — prefer the shorter chord (less extrapolation)
            len_a = float(np.linalg.norm(a.leech_ep - a.luff_ep))
            len_b = float(np.linalg.norm(b.leech_ep - b.luff_ep))
            combined.append(a if len_a <= len_b else b)
    return combined


def _spline_residual(r: V7StripeResult) -> float:
    """Mean perpendicular distance from detection points to spline."""
    if r.spline_points is None or r.detection.points is None:
        return np.inf
    sp = r.spline_points
    pts = r.detection.points
    if len(sp) < 2 or len(pts) < 1:
        return np.inf
    ds = []
    for p in pts:
        best = np.inf
        for i in range(len(sp) - 1):
            a, b = sp[i], sp[i + 1]
            ab = b - a
            L2 = float(ab @ ab)
            if L2 < 1e-9:
                d = float(np.linalg.norm(a - p))
            else:
                t = float((p - a) @ ab / L2)
                t = max(0.0, min(1.0, t))
                proj = a + t * ab
                d = float(np.linalg.norm(proj - p))
            if d < best:
                best = d
        ds.append(best)
    return float(np.mean(ds))


def _ray_polyline_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    polyline: np.ndarray,
    max_distance: float = 5000.0,
) -> Optional[np.ndarray]:
    """Find where a ray (origin + t*direction, t >= 0) first crosses a polyline.

    Returns the intersection point or None.
    """
    if len(polyline) < 2:
        return None
    best_t = np.inf
    best_pt = None
    for i in range(len(polyline) - 1):
        a = polyline[i].astype(np.float64)
        b = polyline[i + 1].astype(np.float64)
        # Solve origin + t * direction = a + s * (b - a)  for t>=0, s in [0, 1]
        ab = b - a
        M = np.column_stack([direction.astype(np.float64), -ab])
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        if abs(det) < 1e-9:
            continue
        rhs = a - origin.astype(np.float64)
        t = (rhs[0] * M[1, 1] - rhs[1] * M[0, 1]) / det
        s = (M[0, 0] * rhs[1] - M[1, 0] * rhs[0]) / det
        if t < 0 or t > max_distance:
            continue
        if s < -0.01 or s > 1.01:
            continue
        if t < best_t:
            best_t = t
            best_pt = a + s * ab
    if best_pt is None:
        return None
    return best_pt.astype(np.float32)


def extend_and_intersect_spline_with_sam(
    spline_points: np.ndarray,
    luff_polyline: np.ndarray,
    leech_polyline: np.ndarray,
    fallback_luff: np.ndarray,
    fallback_leech: np.ndarray,
    max_extrap_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project the spline's tangent directions outward at each end and
    intersect with the SAM luff / leech polylines.

    Each stripe is ALWAYS pulled all the way to the sail boundary:

    1. Cast a tangent ray from the spline endpoint toward the polyline.
       Hit within ``max_extrap_frac × chord_length`` → use that point.
    2. If no ray hit, perpendicular-project the stripe extreme onto the
       polyline (so the curve still reaches the boundary, just along the
       shortest path instead of along the tangent).

    This guarantees ``luff_ep`` lies on ``luff_polyline`` and ``leech_ep``
    lies on ``leech_polyline`` whenever the polylines are non-empty.
    """
    if spline_points is None or len(spline_points) < 4:
        luff_ep = (
            _project_to_polyline(fallback_luff, luff_polyline)
            if len(luff_polyline) else fallback_luff.astype(np.float32)
        )
        leech_ep = (
            _project_to_polyline(fallback_leech, leech_polyline)
            if len(leech_polyline) else fallback_leech.astype(np.float32)
        )
        return luff_ep.astype(np.float32), leech_ep.astype(np.float32)

    sp = spline_points.astype(np.float64)
    n = len(sp)
    window = min(5, n // 4)

    chord_len = float(np.linalg.norm(sp[-1] - sp[0]))
    max_extrap = max(chord_len * max_extrap_frac, 25.0)

    dir_luff = sp[0] - sp[window]
    if np.linalg.norm(dir_luff) < 1e-6:
        dir_luff = sp[0] - sp[1]
    dir_luff = dir_luff / (np.linalg.norm(dir_luff) + 1e-9)
    luff_hit = _ray_polyline_intersection(
        sp[0], dir_luff, luff_polyline, max_distance=max_extrap,
    )

    dir_leech = sp[-1] - sp[-window - 1]
    if np.linalg.norm(dir_leech) < 1e-6:
        dir_leech = sp[-1] - sp[-2]
    dir_leech = dir_leech / (np.linalg.norm(dir_leech) + 1e-9)
    leech_hit = _ray_polyline_intersection(
        sp[-1], dir_leech, leech_polyline, max_distance=max_extrap,
    )

    if luff_hit is None:
        luff_hit = (
            _project_to_polyline(fallback_luff, luff_polyline)
            if len(luff_polyline) else fallback_luff
        )
    if leech_hit is None:
        leech_hit = (
            _project_to_polyline(fallback_leech, leech_polyline)
            if len(leech_polyline) else fallback_leech
        )
    return luff_hit.astype(np.float32), leech_hit.astype(np.float32)


def _sample_intensity_profile(
    image_gray: np.ndarray,
    origin: np.ndarray,
    tangent_unit: np.ndarray,
    n_steps: int = 80,
    step_px: float = 1.5,
    half_width: float = 2.0,
) -> np.ndarray:
    """Return mean intensity along ``origin + s·tangent`` for s in [0,
    n_steps·step_px], averaged over a perpendicular half-width window.
    """
    H, W = image_gray.shape[:2]
    normal = np.array([-tangent_unit[1], tangent_unit[0]], dtype=np.float64)
    profile = np.full(n_steps, np.nan, dtype=np.float64)
    perp_offsets = np.arange(-half_width, half_width + 0.5, 1.0)
    for i in range(n_steps):
        s = step_px * i
        center = origin + s * tangent_unit
        samples = []
        for off in perp_offsets:
            p = center + off * normal
            x = int(round(p[0])); y = int(round(p[1]))
            if 0 <= x < W and 0 <= y < H:
                samples.append(float(image_gray[y, x]))
        if samples:
            profile[i] = float(np.mean(samples))
    return profile


def _photometric_endpoint(
    image_gray: np.ndarray,
    spline_points: np.ndarray,
    side: str,                    # "luff" or "leech"
    fallback_ep: np.ndarray,
    polyline: np.ndarray,
    bright_on_dark: bool = True,
    n_steps: int = 80,
    step_px: float = 1.5,
    sigma_thresh: float = 2.5,
    sam_clamp_px: float = 12.0,
) -> np.ndarray:
    """Find the visible end of a stripe by 1-D intensity profile.

    Cast a tangent ray *outward* from the spline endpoint along the
    spline's local tangent. Sample mean intensity in a thin perpendicular
    window. Walk outward until the signal drops below
    ``mean_inside ± sigma_thresh × std_inside`` — that's the photometric
    end. Optionally clamp the result to the SAM ``polyline`` (within
    ``sam_clamp_px``) as a sanity net.
    """
    if spline_points is None or len(spline_points) < 6:
        return fallback_ep.astype(np.float32)
    sp = spline_points.astype(np.float64)
    n = len(sp)
    win = max(3, n // 10)

    if side == "luff":
        # Tangent points outward at the start: from sp[win] toward sp[0].
        origin = sp[0]
        t = sp[0] - sp[win]
    else:
        origin = sp[-1]
        t = sp[-1] - sp[-1 - win]
    L = float(np.linalg.norm(t))
    if L < 1e-6:
        return fallback_ep.astype(np.float32)
    t /= L

    # Inside-stripe statistics: sample the spline interior
    inside = np.zeros(min(20, n - 2), dtype=np.float64)
    H, W = image_gray.shape[:2]
    for j, idx in enumerate(np.linspace(2, n - 3, len(inside)).astype(int)):
        x = int(round(sp[idx, 0])); y = int(round(sp[idx, 1]))
        if 0 <= x < W and 0 <= y < H:
            inside[j] = float(image_gray[y, x])
    if inside.std() < 1e-3:
        return fallback_ep.astype(np.float32)
    mu_in = float(np.mean(inside))
    sd_in = float(np.std(inside))

    profile = _sample_intensity_profile(
        image_gray, origin, t, n_steps=n_steps, step_px=step_px,
    )
    if np.all(np.isnan(profile)):
        return fallback_ep.astype(np.float32)

    # Background reference: the last quarter of the profile (far from stripe)
    tail = profile[3 * n_steps // 4:]
    tail_valid = tail[~np.isnan(tail)]
    if len(tail_valid) < 3:
        return fallback_ep.astype(np.float32)
    mu_bg = float(np.mean(tail_valid))

    # Stripe end: outermost s where profile is still on the
    # bright (or dark) side of (mu_bg + threshold).
    threshold_band = sigma_thresh * sd_in
    if bright_on_dark:
        in_stripe = profile > (mu_bg + threshold_band)
    else:
        in_stripe = profile < (mu_bg - threshold_band)
    in_stripe[np.isnan(profile)] = False
    if not in_stripe.any():
        return fallback_ep.astype(np.float32)
    last_in = int(np.where(in_stripe)[0].max())
    end_pt = origin + step_px * last_in * t

    # Sanity clamp: stay within sam_clamp_px of the SAM polyline
    if len(polyline) >= 2:
        proj = _project_to_polyline(end_pt.astype(np.float32), polyline)
        if float(np.linalg.norm(end_pt - proj)) > sam_clamp_px:
            end_pt = proj
    return end_pt.astype(np.float32)


# Module-level slot: callers (e.g. cape31 test rig) set this to the
# greyscale image before invoking ``fit_and_intersect`` so the
# photometric endpoint anchor can run without changing the function
# signature used elsewhere.
_CURRENT_IMAGE_GRAY: Optional[np.ndarray] = None


def fit_and_intersect(
    results: List[V7StripeResult],
    sail_boundary: SailBoundary,
    image_gray: Optional[np.ndarray] = None,
) -> List[V7StripeResult]:
    """Fit spline with stripe extremes → intersect spline with SAM → refit.

    This is the endpoint strategy the user asked for:
      a) find points (stripe detection)
      b) fit spline with stripe extremes as initial endpoints
      c) extrapolate spline tangentially, intersect with luff/leech
         polylines → those are the final endpoints
      d) refit spline with those endpoints
    """
    from src.stripe_fit import fit_consensus_spline
    from src.flexible_fit import fit_chord_smoothing_spline, fit_bernstein_flex

    out: List[V7StripeResult] = []
    for r in results:
        # Use the fused endpoints already computed by run_v7_endpoints.
        # User decision (2026-05-01): the fused endpoints + smoothed-
        # consensus spline IS the canonical pipeline. No SAM-extension,
        # no photometric drift, no second refit pass — those override the
        # carefully-fused endpoints with downstream heuristics.
        luff_ep = (
            r.luff_ep if r.luff_ep is not None and np.all(np.isfinite(r.luff_ep))
            else r.raw_luff_ep
        )
        leech_ep = (
            r.leech_ep if r.leech_ep is not None and np.all(np.isfinite(r.leech_ep))
            else r.raw_leech_ep
        )

        spline = fit_consensus_spline(
            r.detection.points,
            luff_ep,
            leech_ep,
            keypoint_confidences=r.detection.keypoint_confidences,
        )
        meta = {"method": "consensus"}
        if spline is None:
            # Fallback chain: chord smoothing → Bernstein → drop stripe
            try:
                meta, _, spline = fit_chord_smoothing_spline(
                    r.detection.points, luff_ep, leech_ep,
                    smoothness=0.4,
                    keypoint_confidences=r.detection.keypoint_confidences,
                )
            except Exception:
                try:
                    meta, _, spline = fit_bernstein_flex(
                        r.detection.points, luff_ep, leech_ep,
                        degree=5, reg=0.005,
                        keypoint_confidences=r.detection.keypoint_confidences,
                    )
                except Exception:
                    continue
        if spline is None:
            continue

        r.spline_points = spline
        r.spline_meta = meta
        r.luff_ep = np.asarray(luff_ep, dtype=np.float32)
        r.leech_ep = np.asarray(leech_ep, dtype=np.float32)
        out.append(r)

    # Issue 5: drop stripes whose bowl points the wrong way
    out = filter_wrong_draft_direction(out)
    # Issue 4: drop lower-confidence stripes that cross a higher-confidence one
    out = filter_crossing_stripes(out)
    return out
