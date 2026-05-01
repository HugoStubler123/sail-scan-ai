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

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


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
        detect_sail_orientation,
    )
    from src.detection import _detect_from_keypoints_model, _get_yolo_bboxes
    from src.polygon_fusion import dedup_bboxes
    from src.pipeline_v7 import (
        run_v7_detection_only, run_v7_endpoints, fit_and_intersect,
    )
    from src.flexible_fit import fit_cst_airfoil
    from src.analysis import (
        build_fitted_stripes, compute_twist, extract_aero_params,
    )
    from src.sail_analysis import build_refined_stripes
    from src.trim_analyst import SailReading

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
    head_top = head_from_mask(sail.mask)
    sail.head_point = head_top
    new_luff, new_leech, head_on_contour = resplit_luff_leech_at_head(sail, head_top)
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

    # Bboxes + keypoints
    det_cfg = config.get("detection", {})
    bbox_model = det_cfg.get("bbox_model_path", "stripe_bbox_v1.pt")
    bboxes = dedup_bboxes(_get_yolo_bboxes(
        image_bgr_use, bbox_model,
        min_conf=float(det_cfg.get("min_confidence", 0.3)),
    ))
    kp_dets = _detect_from_keypoints_model(image_bgr_use, sail.mask, det_cfg)

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
    sail_type = (sail_meta or {}).get("type", "main")
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
