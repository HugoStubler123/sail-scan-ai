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
    # Trimmed to the three variants that win most often on our test
    # fleet. Cuts ~60 % off per-bbox seg inference time on CPU deploys
    # without losing the edge-finding benefit of channel-min.
    ("raw",         lambda x: x),
    ("CLAHE-L",     lambda x: _clahe_l(x, 3.5)),
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
        # Keep only the channel-min legacy variant — the raw one
        # almost never wins and costs a full inference per bbox.
        legacy_variants = [
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


def run_v7_detection_only(
    image_bgr: np.ndarray,
    sail_boundary: SailBoundary,
    bboxes: List[np.ndarray],
    kp_dets: List[StripeDetection],
    seg_model_path: str,
    legacy_seg_model_path: Optional[str] = None,
) -> List[V7StripeResult]:
    """Run stages D-F (detection + fusion) for every bbox.

    Returns one V7StripeResult per bbox with empty endpoints. Caller
    must run ``run_v7_endpoints`` to populate stages G-H.
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

        raw_ep = endpoints_from_detection(det, bb_t, sail_boundary=sail_boundary)
        if raw_ep is None:
            continue
        raw_luff, raw_leech = raw_ep

        results.append(V7StripeResult(
            bbox=bb_t,
            detection=det,
            fusion=diag,
            source=source,
            raw_luff_ep=raw_luff.astype(np.float32),
            raw_leech_ep=raw_leech.astype(np.float32),
            luff_ep=raw_luff.astype(np.float32),   # placeholder
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

    Creates NEW V7StripeResult objects so the same detection list can
    be re-used with different endpoint configs (e.g. A/B ML-on vs ML-off).
    """
    import copy as _copy
    config = config or {}
    use_ml = bool(config.get("endpoint_use_ml_refiner", True))
    guard_max_perp = float(config.get("endpoint_guard_max_perp_px", 12.0))
    max_nudge = float(config.get("endpoint_max_nudge_px", 30.0))

    out: List[V7StripeResult] = []
    for r_in in detections:
        r = _copy.copy(r_in)           # shallow copy; arrays replaced below

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

        # SAM-snap REVERTED: the "project to polyline" step pulled
        # top-sail endpoints up to the head in pathological cases (e.g.
        # raving_jib.jpg top stripe), because the nearest luff-polyline
        # point to a stripe-extreme near the masthead is the head corner.
        # The guard alone handles the outlier cases we actually need to
        # fix.
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
    return [r for r in out if id(r.detection) in kept_ids]


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
    max_extrap_frac: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project the spline's tangent directions outward at each end,
    intersect with SAM luff / leech polylines.

    ``max_extrap_frac``: the ray cast is capped at ``frac × chord_length``
    from the spline endpoint. If the SAM intersection lies beyond that
    (the SAM edge is too far and the spline would have to stretch),
    keep the fallback (stripe extreme). This prevents the "yellow stripe
    fit doesn't match the points" pathology where the spline over-extends.
    """
    if spline_points is None or len(spline_points) < 4:
        return fallback_luff.astype(np.float32), fallback_leech.astype(np.float32)

    sp = spline_points.astype(np.float64)
    n = len(sp)
    window = min(5, n // 4)

    # Chord length from the spline itself
    chord_len = float(np.linalg.norm(sp[-1] - sp[0]))
    max_extrap = max(chord_len * max_extrap_frac, 25.0)

    # Luff side — use first points. Direction from spline[window] → spline[0].
    dir_luff = sp[0] - sp[window]
    if np.linalg.norm(dir_luff) < 1e-6:
        dir_luff = sp[0] - sp[1]
    dir_luff = dir_luff / (np.linalg.norm(dir_luff) + 1e-9)
    luff_hit = _ray_polyline_intersection(sp[0], dir_luff, luff_polyline,
                                            max_distance=max_extrap)

    # Leech side — last points, direction spline[-window-1] → spline[-1]
    dir_leech = sp[-1] - sp[-window - 1]
    if np.linalg.norm(dir_leech) < 1e-6:
        dir_leech = sp[-1] - sp[-2]
    dir_leech = dir_leech / (np.linalg.norm(dir_leech) + 1e-9)
    leech_hit = _ray_polyline_intersection(sp[-1], dir_leech, leech_polyline,
                                            max_distance=max_extrap)

    luff_ep = luff_hit if luff_hit is not None else fallback_luff.astype(np.float32)
    leech_ep = leech_hit if leech_hit is not None else fallback_leech.astype(np.float32)
    return luff_ep, leech_ep


def fit_and_intersect(
    results: List[V7StripeResult],
    sail_boundary: SailBoundary,
) -> List[V7StripeResult]:
    """Fit spline with stripe extremes → intersect spline with SAM → refit.

    This is the endpoint strategy the user asked for:
      a) find points (stripe detection)
      b) fit spline with stripe extremes as initial endpoints
      c) extrapolate spline tangentially, intersect with luff/leech
         polylines → those are the final endpoints
      d) refit spline with those endpoints
    """
    from src.flexible_fit import fit_chord_smoothing_spline, fit_bernstein_flex

    out: List[V7StripeResult] = []
    for r in results:
        # Step 1: initial fit using stripe extremes as endpoints
        try:
            meta, _, spline = fit_chord_smoothing_spline(
                r.detection.points, r.raw_luff_ep, r.raw_leech_ep,
                smoothness=0.4,
                keypoint_confidences=r.detection.keypoint_confidences,
            )
        except Exception:
            try:
                meta, _, spline = fit_bernstein_flex(
                    r.detection.points, r.raw_luff_ep, r.raw_leech_ep,
                    degree=5, reg=0.005,
                    keypoint_confidences=r.detection.keypoint_confidences,
                )
            except Exception:
                continue

        # Step 2: intersect spline extrapolation with SAM polylines
        luff_ep, leech_ep = extend_and_intersect_spline_with_sam(
            spline,
            sail_boundary.luff_polyline,
            sail_boundary.leech_polyline,
            r.raw_luff_ep,
            r.raw_leech_ep,
        )

        # Step 3: refit spline with the new SAM-intersection endpoints
        try:
            meta2, _, spline2 = fit_chord_smoothing_spline(
                r.detection.points, luff_ep, leech_ep,
                smoothness=0.4,
                keypoint_confidences=r.detection.keypoint_confidences,
            )
            r.spline_points = spline2
            r.spline_meta = meta2
        except Exception:
            # keep the initial fit
            r.spline_points = spline
            r.spline_meta = meta

        r.luff_ep = luff_ep.astype(np.float32)
        r.leech_ep = leech_ep.astype(np.float32)
        out.append(r)
    return out
