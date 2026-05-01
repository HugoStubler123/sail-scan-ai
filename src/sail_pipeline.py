"""Single-call sail analysis entry point.

Wraps the v7 pipeline (segment → detect → fit → intersect → CST) so both
``build_v7_report.py`` (3-sail diagnostic report) and ``streamlit_app.py``
(single-sail upload flow → PDF) can share the same logic.

Returns a plain dict with everything the downstream renderers need:

    {
      "image_rgb": HxWx3 uint8,
      "sail_boundary": SailBoundary,
      "luff_depth": EdgeDepth,
      "leech_depth": EdgeDepth,
      "v7_results": List[V7StripeResult],
      "cst_splines": List[Optional[np.ndarray]],
      "refined_stripes": List[RefinedStripe],
      "reading": SailReading,
      "sail_type": "main" | "jib",
      "luff_m": float | None,
      "foot_m": float | None,
    }
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Per-sail-type config overrides (mirrors build_cape31_v7_real.py).
# Mainsails need looser keypoint confidence to recover the foreshortened
# top stripe; the bbox imgsz bump gives the model more pixels.
_MAIN_OVERRIDES = {
    "min_confidence": 0.30,
    "min_keypoint_confidence": 0.15,
    "bbox_imgsz": 1280,
    "keypoint_imgsz": 1280,
    "seg_imgsz": 1280,
}


def _bboxes_from_dets(
    dets,
    image_shape,
    pad_y_frac: float = 0.20,
    pad_x_frac: float = 0.04,
) -> List[np.ndarray]:
    """Padded axis-aligned bboxes from kp / seg detections."""
    H, W = image_shape
    out: List[np.ndarray] = []
    for d in dets:
        if d.points is None or len(d.points) < 2:
            continue
        x1 = float(d.points[:, 0].min())
        y1 = float(d.points[:, 1].min())
        x2 = float(d.points[:, 0].max())
        y2 = float(d.points[:, 1].max())
        bh = y2 - y1
        bw = x2 - x1
        if bh < 1 or bw < 1:
            continue
        py = max(8.0, pad_y_frac * bh)
        px = max(4.0, pad_x_frac * bw)
        x1 = max(0.0, x1 - px)
        y1 = max(0.0, y1 - py)
        x2 = min(float(W - 1), x2 + px)
        y2 = min(float(H - 1), y2 + py)
        out.append(np.array([x1, y1, x2, y2], dtype=np.float32))
    return out


def _build_enriched_bboxes(
    image_bgr: np.ndarray,
    sail_mask: Optional[np.ndarray],
    det_cfg: dict,
    sail_type: str,
) -> List[np.ndarray]:
    """Build the enriched bbox set used by the cape31 harness.

    For mains: native YOLO bboxes + kp-synthesised bboxes +
               seg-decompose bboxes + cluster-synth bboxes,
               then reject oversized and dedup.
    For jibs: full-span anchors (or native) with oversized rejection only.
    """
    from src.detection import _get_yolo_bboxes, _detect_from_keypoints_model
    from src.polygon_fusion import seg_on_crop_all, dedup_bboxes
    from src.bbox_full_span import (
        filter_full_span_bboxes,
        collapse_overlapping_bboxes,
        reject_oversized_bboxes,
    )
    from src.top_stripe_recovery import synth_kp_from_bbox_cluster

    bbox_model = det_cfg.get("bbox_model_path", "stripe_bbox_v1.pt")
    min_conf = float(det_cfg.get("min_confidence", 0.3))
    bbox_imgsz = int(det_cfg.get("bbox_imgsz", 640))

    boxes_native = _get_yolo_bboxes(
        image_bgr, bbox_model, min_conf=min_conf, imgsz=bbox_imgsz,
    )

    # Full-span anchors (low-conf pass filtered to sail-width bboxes)
    full_span: List[np.ndarray] = []
    if sail_mask is not None:
        try:
            low_conf_boxes = _get_yolo_bboxes(
                image_bgr, bbox_model, min_conf=0.10, imgsz=bbox_imgsz,
            )
            full_span = filter_full_span_bboxes(
                list(low_conf_boxes), sail_mask,
                min_span_frac=0.75, edge_tol_frac=0.12,
            )
            full_span = collapse_overlapping_bboxes(full_span, iou_thresh=0.40)
            logger.info(
                "bbox enrichment: full-span anchors=%d (from %d low-conf candidates)",
                len(full_span), len(low_conf_boxes),
            )
        except Exception as exc:
            logger.warning("full-span filter failed: %s", exc)

    # Jib path: just use full-span (if ≥ 2) or native, then drop oversized.
    if sail_type == "jib":
        chosen = full_span if len(full_span) >= 2 else list(boxes_native)
        n_before = len(chosen)
        if sail_mask is not None:
            chosen = reject_oversized_bboxes(
                chosen, sail_mask, max_height_frac_of_sail=0.40,
            )
        logger.info(
            "bbox enrichment (jib): %d -> %d (dropped %d oversized)",
            n_before, len(chosen), n_before - len(chosen),
        )
        return chosen

    # ---- Main path ----

    # kp-synth bboxes
    kp_dets = _detect_from_keypoints_model(image_bgr, None, det_cfg)
    kp_synth = _bboxes_from_dets(kp_dets, image_bgr.shape[:2])

    # seg-decompose: crack suspicious (low-aspect) YOLO bboxes via seg model
    seg_decompose: List[np.ndarray] = []
    seg_path = det_cfg.get("seg_model_path", "../models/sail_seg_model.pt")
    if sail_mask is not None:
        for src_bbox in list(boxes_native):
            bw = float(src_bbox[2] - src_bbox[0])
            bh = float(src_bbox[3] - src_bbox[1])
            src_aspect = bw / max(bh, 1.0)
            if src_aspect >= 4.0:
                continue
            try:
                instances = seg_on_crop_all(
                    image_bgr, sail_mask, tuple(src_bbox),
                    seg_model_path=seg_path,
                    min_conf=0.08, imgsz=1280,
                )
            except Exception:
                instances = []
            stripe_instances = [
                (sub_bb, conf) for sub_bb, conf in instances
                if (sub_bb[3] - sub_bb[1]) >= 1
                and ((sub_bb[2] - sub_bb[0]) / max(sub_bb[3] - sub_bb[1], 1)) >= 4.0
            ]
            if len(stripe_instances) < 2:
                continue
            for sub_bb, _ in stripe_instances:
                seg_decompose.append(np.array(sub_bb, dtype=np.float32))

    # bbox-cluster → synth-kp recovery
    cluster_synth: List[np.ndarray] = []
    try:
        cluster_dets = synth_kp_from_bbox_cluster(
            image_bgr, sail_mask, list(boxes_native),
            head_frac=0.30, n_keypoints=8, min_cluster_size=3,
        )
        if cluster_dets:
            cluster_synth = _bboxes_from_dets(cluster_dets, image_bgr.shape[:2])
            logger.info("bbox enrichment: cluster-synth=%d", len(cluster_dets))
    except Exception as exc:
        logger.warning("bbox-cluster kp synth failed: %s", exc)

    all_boxes = (
        list(boxes_native) + list(kp_synth)
        + list(seg_decompose) + list(cluster_synth)
    )

    # Height cap
    H = image_bgr.shape[0]
    max_h = 0.6 * H
    kept = [b for b in all_boxes if 1 <= (b[3] - b[1]) <= max_h]

    # Reject absurdly tall bboxes
    n_before = len(kept)
    if sail_mask is not None:
        kept = reject_oversized_bboxes(
            kept, sail_mask, max_height_frac_of_sail=0.40,
        )
    deduped = dedup_bboxes(kept)
    logger.info(
        "bbox enrichment (main): %d native + %d kp + %d seg-decompose + "
        "%d cluster -> %d dedup (dropped %d oversized)",
        len(boxes_native), len(kp_synth), len(seg_decompose),
        len(cluster_synth), len(deduped), n_before - len(kept),
    )
    return deduped


def analyze_sail(
    image_bgr: np.ndarray,
    sail_meta: Optional[dict] = None,
    config: Optional[dict] = None,
    photo_name: str = "sail",
) -> Dict[str, Any]:
    """Run the full v7 pipeline on a single image.

    Args:
      image_bgr: HxWx3 BGR (as loaded by cv2.imread).
      sail_meta: optional {"type": "main"|"jib", "luff_m", "foot_m",
                 "inventory", "display_name"}.
      config: full config dict (detection.seg_model_path etc). If None,
              loads ``stripes_detector/config.yaml``.
      photo_name: used in display_name fallback.

    Returns a dict — see module docstring.
    """
    if config is None:
        import yaml
        root = Path(__file__).resolve().parent.parent
        with (root / "config.yaml").open() as f:
            config = yaml.safe_load(f)

    from src.calibration import calibrate_image, undistort_image
    from src.segmentation import segment_sail
    from src.sail_shape import (
        head_from_mask, analyze_sail_edges, resplit_luff_leech_at_head,
        detect_sail_orientation, cleanup_sail_mask, smooth_polyline,
    )
    from src.detection import _detect_from_keypoints_model, _get_yolo_bboxes
    from src.polygon_fusion import dedup_bboxes
    from src.pipeline_v7 import (
        run_v7_detection_only, run_v7_endpoints, fit_and_intersect,
        _CURRENT_IMAGE_GRAY,
    )
    from src.flexible_fit import fit_cst_airfoil
    from src.analysis import (
        build_fitted_stripes, compute_twist, extract_aero_params,
    )
    from src.sail_analysis import build_refined_stripes
    from src.trim_analyst import SailReading
    from src.top_stripe_recovery import detect_kp_on_top_crop
    import src.pipeline_v7 as _pipeline_v7

    # ---- Resize large inputs BEFORE anything else ------------------
    # Cloud tier has 1 GB RAM. A 4000x3000 RGBA photo held in 3-4
    # intermediate buffers by SAM2 + opencv can blow through that alone.
    # Downscale so the long side is <= max_side (default 1800 px). 1800
    # is empirically the sweet spot: SAM2 + YOLO still resolve stripes
    # cleanly; peak RAM drops by ~4x vs raw 4K input.
    max_side = int((config.get("pipeline") or {}).get("max_input_side", 1800))
    h_raw, w_raw = image_bgr.shape[:2]
    long_side = max(h_raw, w_raw)
    if long_side > max_side:
        scale = max_side / float(long_side)
        new_w = int(round(w_raw * scale))
        new_h = int(round(h_raw * scale))
        image_bgr = cv2.resize(image_bgr, (new_w, new_h),
                                interpolation=cv2.INTER_AREA)

    image_rgb_raw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    try:
        calib = calibrate_image(image_rgb_raw, method="opencv")
        image_rgb = undistort_image(image_rgb_raw, calib)
    except Exception:
        image_rgb = image_rgb_raw
    image_bgr_use = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    sail_type = (sail_meta or {}).get("type", "main")

    # Apply per-sail-type config overrides (mirroring cape31 harness).
    det_cfg_base = config.get("detection", {})
    if sail_type == "main":
        det_cfg = {**det_cfg_base, **_MAIN_OVERRIDES}
    else:
        det_cfg = dict(det_cfg_base)

    # ---- Auto-rotation: orient the photo so the sail head is at the top.
    # Same logic as build_cape31_v7_real.py — segment once, classify
    # orientation from the mask, rotate the image, then re-segment on the
    # corrected image so all downstream stages (head/tack/clew, stripe
    # detection) work on a head-up frame.
    sam2_cfg = config.get("sam2", {})
    pre_sail = segment_sail(
        image_rgb,
        model_path=sam2_cfg.get("model_path", "sam2.1_b.pt"),
        prompt_strategy=sam2_cfg.get("prompt_strategy", "multi_point"),
        grid_size=sam2_cfg.get("grid_size", 5),
        mask_cleanup=sam2_cfg.get("mask_cleanup", True),
    )
    rot_code = detect_sail_orientation(pre_sail.mask)
    rot_map = {
        1: cv2.ROTATE_90_COUNTERCLOCKWISE,
        2: cv2.ROTATE_180,
        3: cv2.ROTATE_90_CLOCKWISE,
    }
    if rot_code in rot_map:
        image_rgb = cv2.rotate(image_rgb, rot_map[rot_code])
        image_bgr_use = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Final segmentation on the head-up image
    sail = segment_sail(
        image_rgb,
        model_path=sam2_cfg.get("model_path", "sam2.1_b.pt"),
        prompt_strategy=sam2_cfg.get("prompt_strategy", "multi_point"),
        grid_size=sam2_cfg.get("grid_size", 5),
        mask_cleanup=sam2_cfg.get("mask_cleanup", True),
    )

    # Mask cleanup + polyline smoothing (cape31 harness does this via
    # monkey-patched segment_sail).
    try:
        sail.mask = cleanup_sail_mask(sail.mask)
    except Exception as exc:
        logger.warning("mask cleanup failed: %s", exc)
    try:
        if sail.luff_polyline is not None and len(sail.luff_polyline) >= 5:
            sail.luff_polyline = smooth_polyline(sail.luff_polyline)
        if sail.leech_polyline is not None and len(sail.leech_polyline) >= 5:
            sail.leech_polyline = smooth_polyline(sail.leech_polyline)
    except Exception as exc:
        logger.warning("polyline smoothing failed: %s", exc)

    head_top = head_from_mask(sail.mask)
    sail.head_point = head_top
    new_luff, new_leech, head_on_contour = resplit_luff_leech_at_head(sail, head_top)
    # Smooth resplit outputs (cape31 harness patches resplit_luff_leech_at_head)
    try:
        new_luff = smooth_polyline(np.asarray(new_luff))
        new_leech = smooth_polyline(np.asarray(new_leech))
    except Exception:
        pass
    sail.luff_polyline = new_luff
    sail.leech_polyline = new_leech
    sail.head_point = head_on_contour
    luff_depth, leech_depth = analyze_sail_edges(
        sail.luff_polyline, sail.leech_polyline
    )
    if (luff_depth and leech_depth
            and luff_depth.max_depth_pct > leech_depth.max_depth_pct):
        luff_depth, leech_depth = leech_depth, luff_depth
        sail.luff_polyline, sail.leech_polyline = (
            sail.leech_polyline, sail.luff_polyline,
        )

    # Set the gray image cache used by pipeline_v7 photometric anchors
    _pipeline_v7._CURRENT_IMAGE_GRAY = cv2.cvtColor(
        image_bgr_use, cv2.COLOR_BGR2GRAY,
    )

    # ---- Enriched bbox source (mirrors _enriched_get_yolo_bboxes) ----
    bboxes = _build_enriched_bboxes(
        image_bgr_use, sail.mask, det_cfg, sail_type,
    )

    # ---- Keypoint detections with top-crop recovery for mains --------
    kp_dets = _detect_from_keypoints_model(image_bgr_use, sail.mask, det_cfg)
    if sail_type == "main":
        try:
            top_dets = detect_kp_on_top_crop(
                image_bgr_use, sail.mask, det_cfg,
                _detect_from_keypoints_model,
                top_frac=0.40, target_size=1280, min_conf_override=0.10,
            )
            if top_dets:
                logger.info(
                    "top-crop kp recovered %d extra detections", len(top_dets),
                )
            kp_dets = list(kp_dets) + list(top_dets)
        except Exception as exc:
            logger.warning("top-crop kp pass failed: %s", exc)

    # Detection (D-F) — run once
    seg_path = det_cfg.get("seg_model_path", "../models/sail_seg_model.pt")
    legacy_path = det_cfg.get("legacy_seg_model_path", "stripe_seg_v1.pt")
    detections = run_v7_detection_only(
        image_bgr_use, sail, bboxes, kp_dets,
        seg_model_path=seg_path,
        legacy_seg_model_path=legacy_path,
    )

    # Endpoints (G-H) — geometric + guard
    detections_with_ep = run_v7_endpoints(
        detections, image_bgr_use, sail,
        {**det_cfg, "endpoint_use_ml_refiner": False},
    )
    # Spline fit + SAM intersection + refit (I)
    results = fit_and_intersect(detections_with_ep, sail)

    # Clear the gray cache
    _pipeline_v7._CURRENT_IMAGE_GRAY = None

    # The new canonical fit (consensus spline anchored at fused endpoints)
    # already lives in r.spline_points after fit_and_intersect. Pass it
    # through as cst_splines so the streamlit overlay renders the
    # consensus curve instead of an ad-hoc CST refit.
    cst_splines = [r.spline_points for r in results]

    # Aero params + refined stripes
    stripe_data = [
        (r.spline_points, r.luff_ep, r.leech_ep, None)
        for r in results if r.spline_points is not None
    ]
    aero_list = [extract_aero_params(sp, l, le) for sp, l, le, _ in stripe_data]
    aero_list = compute_twist(
        aero_list, chord_data=[(l, le) for _, l, le, _ in stripe_data],
    )
    fitted = build_fitted_stripes(stripe_data, aero_list)
    refined = build_refined_stripes(fitted, multi_color_results=None)

    # SailReading for trim analyst
    heights = [
        0.5 * (r.luff_endpoint[1] + r.leech_endpoint[1]) for r in refined
    ]
    ordered_idx = list(np.argsort(heights))
    stripes_tb = [
        {
            "camber_pct": float(refined[idx].aero.camber_depth_pct),
            "draft_pct": float(refined[idx].aero.draft_position_pct),
            "twist_deg": float(refined[idx].aero.twist_deg),
            "entry_deg": float(refined[idx].aero.entry_angle_deg),
            "exit_deg": float(refined[idx].aero.exit_angle_deg),
        }
        for idx in ordered_idx
    ]
    inventory = (sail_meta or {}).get("inventory", sail_type)
    display_name = (sail_meta or {}).get("display_name", photo_name)
    reading = SailReading(
        name=display_name,
        sail_type=inventory,
        stripes_top_to_bottom=stripes_tb,
        luff_max_bend_pct=float(luff_depth.max_depth_pct if luff_depth else 0.0),
        luff_max_bend_at_pct=float(
            luff_depth.max_depth_position_pct if luff_depth else 50.0
        ),
        leech_max_bend_pct=float(leech_depth.max_depth_pct if leech_depth else 0.0),
        chord_m_foot=(sail_meta or {}).get("foot_m"),
    )

    return {
        "image_rgb": image_rgb,
        "sail_boundary": sail,
        "luff_depth": luff_depth,
        "leech_depth": leech_depth,
        "v7_results": results,
        "cst_splines": cst_splines,
        "refined_stripes": refined,
        "reading": reading,
        "sail_type": sail_type,
        "luff_m": (sail_meta or {}).get("luff_m"),
        "foot_m": (sail_meta or {}).get("foot_m"),
        "display_name": display_name,
    }
