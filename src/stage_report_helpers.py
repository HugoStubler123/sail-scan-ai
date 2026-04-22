"""Plotting helpers for build_stage_report.py — keeps the runner lean."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

STRIPE_COLORS = [
    "#FF3B5C", "#00D4AA", "#FFB020", "#4DA6FF",
    "#C77DFF", "#FF6B9D", "#00E5FF",
]


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _new_ax(image_rgb: np.ndarray, title: str, figsize=(7, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image_rgb)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    return fig, ax


def plot_image(image_rgb: np.ndarray, title: str) -> str:
    fig, ax = _new_ax(image_rgb, title)
    return fig_to_b64(fig)


def plot_calibration(
    original_rgb: np.ndarray,
    undistorted_rgb: Optional[np.ndarray],
) -> List[Tuple[str, str]]:
    panels: List[Tuple[str, str]] = []
    panels.append(("Original photo", plot_image(original_rgb, "Original (pre-calibration)")))
    if undistorted_rgb is not None:
        panels.append(
            ("Undistorted image", plot_image(undistorted_rgb, "Post-calibration (lens corrected)"))
        )
        diff = cv2.absdiff(original_rgb, undistorted_rgb)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(diff_gray, cmap="inferno")
        ax.set_title("Distortion delta map (brighter = more correction)", fontsize=11)
        ax.axis("off")
        panels.append(("Δ map", fig_to_b64(fig)))
    else:
        panels.append(("Undistorted image", plot_image(original_rgb, "Calibration fell back to raw")))
    return panels


def _overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color=(255, 90, 200), alpha=0.35):
    overlay = image_rgb.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    m = mask.astype(bool)
    overlay[m] = overlay[m] * (1 - alpha) + color_arr * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_segmentation(
    image_rgb: np.ndarray, sail
) -> List[Tuple[str, str]]:
    panels: List[Tuple[str, str]] = []

    # Panel 1: translucent mask overlay
    shaded = _overlay_mask(image_rgb, sail.mask, color=(80, 200, 255), alpha=0.35)
    panels.append(("SAM2 mask (cyan)", plot_image(shaded, "Stage 1 — SAM2 sail mask")))

    # Panel 2: boundary polylines + corners
    fig, ax = _new_ax(image_rgb, "Luff (red) · Leech (blue) · corners")
    if sail.luff_polyline is not None and len(sail.luff_polyline) > 0:
        ax.plot(sail.luff_polyline[:, 0], sail.luff_polyline[:, 1],
                color="#FF3B5C", linewidth=2.5, label="luff")
    if sail.leech_polyline is not None and len(sail.leech_polyline) > 0:
        ax.plot(sail.leech_polyline[:, 0], sail.leech_polyline[:, 1],
                color="#4DA6FF", linewidth=2.5, label="leech")
    for name, pt, color in [
        ("head", sail.head_point, "#FFD400"),
        ("tack", sail.tack_point, "#00E676"),
        ("clew", sail.clew_point, "#FF6B9D"),
    ]:
        if pt is None:
            continue
        ax.plot(pt[0], pt[1], "o", color=color, markersize=10, markeredgecolor="white",
                markeredgewidth=2, label=name)
    ax.legend(loc="upper right", fontsize=9)
    panels.append(("Polylines & corners", fig_to_b64(fig)))

    # Panel 3: raw contour trace
    fig, ax = _new_ax(image_rgb, "Sail contour trace")
    if sail.contour is not None and len(sail.contour) > 0:
        ax.plot(sail.contour[:, 0], sail.contour[:, 1], color="#00E5FF", linewidth=1.5)
    panels.append(("Contour", fig_to_b64(fig)))

    return panels


def plot_sail_edge_depth(
    image_rgb: np.ndarray, sail, luff_depth, leech_depth,
) -> List[Tuple[str, str]]:
    """Overlay fitted splines on luff & leech, max-depth markers, bend chart."""
    panels: List[Tuple[str, str]] = []

    # ---- Panel 1: image overlay --------------------------------------------
    fig, ax = _new_ax(image_rgb, "Sail edge splines + max depth vs chord",
                      figsize=(8, 6))

    edges = (
        ("luff",  luff_depth,  "#FF3B5C"),
        ("leech", leech_depth, "#4DA6FF"),
    )
    for name, depth, color in edges:
        if depth is None:
            continue
        ax.plot(depth.spline[:, 0], depth.spline[:, 1],
                color=color, linewidth=2.2, label=f"{name} spline")
        ax.plot([depth.chord_start[0], depth.chord_end[0]],
                [depth.chord_start[1], depth.chord_end[1]],
                color=color, linestyle="--", linewidth=1.0, alpha=0.6)
        ax.plot([depth.max_depth_point[0], depth.max_depth_foot[0]],
                [depth.max_depth_point[1], depth.max_depth_foot[1]],
                color=color, linewidth=1.5, alpha=0.9)
        ax.plot(depth.max_depth_point[0], depth.max_depth_point[1], "o",
                color="white", markersize=9, markeredgecolor=color, markeredgewidth=2)
        ax.text(
            depth.max_depth_point[0] + 10, depth.max_depth_point[1],
            f"{name}: {depth.max_depth_px:.0f} px ({depth.max_depth_pct:.2f}% chord)\n"
            f"position: {depth.max_depth_position_pct:.1f}% from start",
            color=color, fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85,
                      edgecolor=color, linewidth=0.8),
        )

    # Head marker (topmost pixel)
    ax.plot(sail.head_point[0], sail.head_point[1], "v",
            color="#FFD400", markersize=14, markeredgecolor="black",
            markeredgewidth=1.5, label="head (topmost px)")
    ax.legend(loc="upper right", fontsize=9)
    panels.append(("Edge splines + max depth", fig_to_b64(fig)))

    # ---- Panel 2: bend chart with SPLINE smoothing --------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    try:
        from scipy.interpolate import UnivariateSpline
    except Exception:
        UnivariateSpline = None

    for name, depth, color in edges:
        if depth is None:
            continue
        spline = depth.spline.astype(np.float64)
        chord = depth.chord_end - depth.chord_start
        L = depth.chord_length_px
        if L < 1e-3:
            continue
        u = chord / L
        n = np.array([-u[1], u[0]])
        rel = spline - depth.chord_start
        proj = rel @ u
        perp = rel @ n
        t_pct = np.clip(proj / L * 100.0, 0, 100)
        pct = perp / L * 100.0

        # Raw polyline — dim
        ax.plot(t_pct, pct, color=color, linewidth=0.8, alpha=0.35)

        # Smooth spline fit through the bend profile
        if UnivariateSpline is not None and len(t_pct) >= 6:
            order = np.argsort(t_pct)
            try:
                s = UnivariateSpline(
                    t_pct[order], pct[order], k=4, s=len(t_pct) * 0.01,
                )
                t_fine = np.linspace(t_pct.min(), t_pct.max(), 200)
                ax.plot(t_fine, s(t_fine), color=color, linewidth=2.4, label=name)
            except Exception:
                ax.plot(t_pct, pct, color=color, linewidth=2.0, label=name)
        else:
            ax.plot(t_pct, pct, color=color, linewidth=2.0, label=name)

        ax.plot(depth.max_depth_position_pct,
                depth.max_depth_pct * np.sign(pct[np.argmax(np.abs(pct))]),
                "o", color=color, markersize=8,
                markeredgecolor="white", markeredgewidth=1.5)
    ax.axhline(0.0, color="#888888", linewidth=0.8, alpha=0.6)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("chord position (%)", fontsize=10)
    ax.set_ylabel("perpendicular deviation (% of chord)", fontsize=10)
    ax.set_title("Edge bend profile (spline-fitted)", fontsize=11)
    ax.legend(loc="best", fontsize=9)
    panels.append(("Bend profile", fig_to_b64(fig)))

    return panels


def plot_preprocessing(
    image_rgb: np.ndarray,
    clahe_rgb: np.ndarray,
    enhanced_gray: np.ndarray,
) -> List[Tuple[str, str]]:
    panels = [
        ("Original", plot_image(image_rgb, "Original RGB")),
        ("CLAHE corrected", plot_image(clahe_rgb, "CLAHE-corrected (sail region)")),
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(enhanced_gray, cmap="gray")
    ax.set_title("Stripe-enhanced (top-hat, for classical detector)", fontsize=11)
    ax.axis("off")
    panels.append(("Enhanced", fig_to_b64(fig)))
    return panels


def _draw_stripes(ax, stripes: Sequence, color_cycle=None, linewidth=2.0, show_points=False, label_prefix=None):
    """Draw a sequence of stripes. Each entry may be a StripeDetection or an
    ndarray of points."""
    if color_cycle is None:
        color_cycle = STRIPE_COLORS
    for i, s in enumerate(stripes):
        color = color_cycle[i % len(color_cycle)]
        if hasattr(s, "points"):
            pts = s.points
            conf = getattr(s, "confidence", None)
        else:
            pts = np.asarray(s)
            conf = None
        if pts is None or len(pts) < 2:
            continue
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth, alpha=0.95)
        if show_points:
            ax.plot(pts[:, 0], pts[:, 1], "o", color=color, markersize=4,
                    markeredgecolor="white", markeredgewidth=0.5)
        if label_prefix is not None and conf is not None:
            ax.text(pts[-1, 0] + 6, pts[-1, 1],
                    f"{label_prefix}{i+1} {conf:.2f}",
                    color=color, fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"))


def plot_detection_variants(
    image_rgb: np.ndarray,
    variants: List[dict],
) -> List[Tuple[str, str]]:
    """Each variant dict: name, image_rgb, kp_dets, rf_dets, kp_mean_conf, rf_mean_conf, total_score, winner(bool)."""
    panels: List[Tuple[str, str]] = []
    for v in variants:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(v["image_rgb"])
        axes[0].axis("off")
        axes[0].set_title(
            f"{v['name']} — kp detections (mean conf {v['kp_mean_conf']:.2f}, n={len(v['kp_dets'])})",
            fontsize=10)
        _draw_stripes(axes[0], v["kp_dets"], show_points=True, label_prefix="kp")

        axes[1].imshow(v["image_rgb"])
        axes[1].axis("off")
        axes[1].set_title(
            f"{v['name']} — Roboflow v7 polygons (mean conf {v['rf_mean_conf']:.2f}, n={len(v['rf_dets'])})",
            fontsize=10)
        _draw_stripes(axes[1], v["rf_dets"], show_points=False, label_prefix="rf")
        for j, rf in enumerate(v["rf_dets"]):
            poly = getattr(rf, "polygon", None)
            if poly is not None and len(poly) >= 3:
                color = STRIPE_COLORS[j % len(STRIPE_COLORS)]
                axes[1].fill(poly[:, 0], poly[:, 1], color=color, alpha=0.15)
                axes[1].plot(np.append(poly[:, 0], poly[0, 0]),
                             np.append(poly[:, 1], poly[0, 1]),
                             color=color, linewidth=1.2, alpha=0.8)

        suptitle = (f"{v['name']}  |  total score = {v['total_score']:.2f}"
                    + ("  ←  WINNER" if v.get("winner") else ""))
        fig.suptitle(suptitle, fontsize=11, fontweight="bold",
                     color=("#00A86B" if v.get("winner") else "#333"))
        panels.append((v["name"], fig_to_b64(fig)))
    return panels


def plot_ensemble(
    image_rgb: np.ndarray,
    kp_dets: list,
    seg_dets: list,
    bbox_boxes: list,
    rf_dets: list,
    final_dets: list,
) -> List[Tuple[str, str]]:
    panels: List[Tuple[str, str]] = []

    fig, ax = _new_ax(image_rgb, "All per-model candidates")
    # bbox (green rectangles)
    for b in bbox_boxes:
        x1, y1, x2, y2 = b
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1],
                color="#00D98A", linewidth=1.0, alpha=0.8)
    # kp (yellow)
    _draw_stripes(ax, kp_dets, color_cycle=["#FFD400"], linewidth=1.6)
    # seg (magenta)
    _draw_stripes(ax, seg_dets, color_cycle=["#FF00E6"], linewidth=1.6)
    # rf (cyan polygons)
    for rf in rf_dets:
        poly = getattr(rf, "polygon", None)
        if poly is not None and len(poly) >= 3:
            ax.fill(poly[:, 0], poly[:, 1], color="#00E5FF", alpha=0.15)
            ax.plot(np.append(poly[:, 0], poly[0, 0]),
                    np.append(poly[:, 1], poly[0, 1]),
                    color="#00E5FF", linewidth=1.2, alpha=0.9)
    # legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend = [
        Patch(color="#00D98A", label=f"bbox ({len(bbox_boxes)})"),
        Line2D([0], [0], color="#FFD400", lw=2, label=f"keypoint ({len(kp_dets)})"),
        Line2D([0], [0], color="#FF00E6", lw=2, label=f"seg ({len(seg_dets)})"),
        Patch(color="#00E5FF", label=f"Roboflow v7 ({len(rf_dets)})", alpha=0.5),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    panels.append(("Per-model candidates", fig_to_b64(fig)))

    fig, ax = _new_ax(image_rgb, f"Clustered ensemble result ({len(final_dets)} stripes)")
    _draw_stripes(ax, final_dets, show_points=True, label_prefix="S")
    panels.append(("Ensemble (merged)", fig_to_b64(fig)))
    return panels


def plot_polygon_fusion(
    image_rgb: np.ndarray, diagnostics: list
) -> List[Tuple[str, str]]:
    """Visualise polygon bottom edge + kp matches for each stripe."""
    panels: List[Tuple[str, str]] = []

    # Global overview: all polygons + bottoms + matches on the full image
    fig, ax = _new_ax(image_rgb, "Stage 3b — polygon bottom + kp matches", figsize=(10, 7))
    for i, diag in enumerate(diagnostics):
        color = STRIPE_COLORS[i % len(STRIPE_COLORS)]
        poly = diag.polygon
        if poly is not None and len(poly) >= 3:
            ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.08)
            ax.plot(np.append(poly[:, 0], poly[0, 0]),
                    np.append(poly[:, 1], poly[0, 1]),
                    color=color, linewidth=0.8, alpha=0.4, linestyle="--")
        if len(diag.bottom_edge) > 0:
            ax.plot(diag.bottom_edge[:, 0], diag.bottom_edge[:, 1],
                    color=color, linewidth=2.2, alpha=0.95)
        if len(diag.matched_kp) > 0:
            ax.plot(diag.matched_kp[:, 0], diag.matched_kp[:, 1], "o",
                    color="white", markersize=8, markeredgecolor=color, markeredgewidth=2)
        if len(diag.unmatched_kp) > 0:
            ax.plot(diag.unmatched_kp[:, 0], diag.unmatched_kp[:, 1], "x",
                    color="#FF3333", markersize=8, markeredgewidth=2)
        if len(diag.curvature_samples) > 0:
            ax.plot(diag.curvature_samples[:, 0], diag.curvature_samples[:, 1], ".",
                    color=color, markersize=4, alpha=0.5)
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color="#888888", lw=2, label="polygon bottom edge"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor="white",
               markersize=8, markeredgecolor="#888", markeredgewidth=2, lw=0, label="kp matched"),
        Line2D([0], [0], marker="x", color="#FF3333", markersize=8, lw=0, label="kp unmatched"),
        Line2D([0], [0], marker=".", color="#888888", markersize=6, lw=0, label="curvature samples"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    panels.append(("Bottom edge + matches (global)", fig_to_b64(fig)))

    # Per-stripe zoom panels
    for i, diag in enumerate(diagnostics):
        if diag.polygon is None or len(diag.polygon) < 3:
            continue
        x1, y1, x2, y2 = diag.bbox
        pad = max(25, int(0.06 * (x2 - x1)))
        x1p, x2p = max(0, x1 - pad), min(image_rgb.shape[1], x2 + pad)
        y1p, y2p = max(0, y1 - pad), min(image_rgb.shape[0], y2 + pad)
        crop = image_rgb[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.imshow(crop)
        ax.axis("off")
        off = np.array([x1p, y1p])

        def _loc(arr):
            return arr - off

        color = STRIPE_COLORS[i % len(STRIPE_COLORS)]
        poly = _loc(diag.polygon)
        ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.10)
        ax.plot(np.append(poly[:, 0], poly[0, 0]),
                np.append(poly[:, 1], poly[0, 1]),
                color=color, linewidth=1.1, alpha=0.6, linestyle="--")
        if len(diag.bottom_edge) > 0:
            be = _loc(diag.bottom_edge)
            ax.plot(be[:, 0], be[:, 1], color=color, linewidth=2.6)
        if len(diag.matched_kp) > 0:
            mk = _loc(diag.matched_kp)
            ax.plot(mk[:, 0], mk[:, 1], "o", color="white", markersize=9,
                    markeredgecolor=color, markeredgewidth=2)
        if len(diag.unmatched_kp) > 0:
            uk = _loc(diag.unmatched_kp)
            ax.plot(uk[:, 0], uk[:, 1], "x", color="#FF3333", markersize=9,
                    markeredgewidth=2)
        if len(diag.curvature_samples) > 0:
            cs = _loc(diag.curvature_samples)
            ax.plot(cs[:, 0], cs[:, 1], ".", color=color, markersize=5, alpha=0.7)
        title = (
            f"Stripe {i+1} — bbox {x2-x1}×{y2-y1}px · "
            f"bottom edge pts={len(diag.bottom_edge)} · "
            f"kp matched={len(diag.matched_kp)}/{len(diag.matched_kp)+len(diag.unmatched_kp)} · "
            f"curvature samples={len(diag.curvature_samples)}"
        )
        ax.set_title(title, fontsize=10)
        panels.append((f"Stripe {i+1}", fig_to_b64(fig)))
    return panels


def plot_endpoints(
    image_rgb: np.ndarray,
    stripe_data: list,
    sail,
) -> List[Tuple[str, str]]:
    """stripe_data: list of (spline_points, luff_ep, leech_ep, coeffs)."""
    panels: List[Tuple[str, str]] = []
    fig, ax = _new_ax(image_rgb, "Endpoints snapped to luff/leech polylines")
    if sail.luff_polyline is not None:
        ax.plot(sail.luff_polyline[:, 0], sail.luff_polyline[:, 1],
                color="#FF3B5C", linewidth=1.5, alpha=0.7)
    if sail.leech_polyline is not None:
        ax.plot(sail.leech_polyline[:, 0], sail.leech_polyline[:, 1],
                color="#4DA6FF", linewidth=1.5, alpha=0.7)
    for i, item in enumerate(stripe_data):
        spline, luff, leech = item[0], item[1], item[2]
        color = STRIPE_COLORS[i % len(STRIPE_COLORS)]
        ax.plot(spline[:, 0], spline[:, 1], color=color, linewidth=2.0)
        ax.plot([luff[0], leech[0]], [luff[1], leech[1]],
                color=color, linestyle="--", linewidth=1.0, alpha=0.5)
        ax.plot(luff[0], luff[1], "o", color="white", markersize=8,
                markeredgecolor=color, markeredgewidth=2)
        ax.plot(leech[0], leech[1], "o", color="white", markersize=8,
                markeredgecolor=color, markeredgewidth=2)
    panels.append(("Endpoints", fig_to_b64(fig)))
    return panels


def plot_bernstein_fit(
    image_rgb: np.ndarray,
    stripe_data: list,
    original_detections: list,
) -> List[Tuple[str, str]]:
    panels: List[Tuple[str, str]] = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(image_rgb)
    axes[0].axis("off")
    axes[0].set_title("Raw keypoints in", fontsize=11)
    _draw_stripes(axes[0], original_detections, show_points=True)

    axes[1].imshow(image_rgb)
    axes[1].axis("off")
    axes[1].set_title(f"4-param Bernstein spline ({len(stripe_data)} stripes)", fontsize=11)
    for i, item in enumerate(stripe_data):
        spline = item[0]
        color = STRIPE_COLORS[i % len(STRIPE_COLORS)]
        axes[1].plot(spline[:, 0], spline[:, 1], color=color, linewidth=2.2)
    panels.append(("Fit", fig_to_b64(fig)))
    return panels


def plot_aero(
    image_rgb: np.ndarray,
    fitted_stripes: list,
) -> List[Tuple[str, str]]:
    panels: List[Tuple[str, str]] = []
    fig, ax = _new_ax(image_rgb, "Aero parameter overlay", figsize=(9, 6))
    for i, s in enumerate(fitted_stripes):
        color = STRIPE_COLORS[i % len(STRIPE_COLORS)]
        pts = s.spline_points
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2.4)
        ax.plot([s.luff_endpoint[0], s.leech_endpoint[0]],
                [s.luff_endpoint[1], s.leech_endpoint[1]],
                color=color, linestyle="--", linewidth=1.0, alpha=0.5)
        ax.plot(s.luff_endpoint[0], s.luff_endpoint[1], "o",
                color="white", markersize=8, markeredgecolor=color, markeredgewidth=2)
        ax.plot(s.leech_endpoint[0], s.leech_endpoint[1], "o",
                color="white", markersize=8, markeredgecolor=color, markeredgewidth=2)
        mid = pts[len(pts) // 2]
        ax.text(
            mid[0], mid[1] - 14,
            f"c={s.aero_params.camber_depth_pct:.1f}% "
            f"d={s.aero_params.draft_position_pct:.0f}%\n"
            f"ent={s.aero_params.entry_angle_deg:.1f}° "
            f"ex={s.aero_params.exit_angle_deg:.1f}°",
            color=color, fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85,
                      edgecolor=color, linewidth=0.8),
        )
    panels.append(("Aero overlay", fig_to_b64(fig)))
    return panels


def aero_table_html(fitted_stripes: list) -> str:
    rows = []
    for i, s in enumerate(fitted_stripes):
        a = s.aero_params
        rows.append(
            f"<tr><td>stripe {i+1}</td>"
            f"<td>{a.camber_depth_pct:.2f}%</td>"
            f"<td>{a.draft_position_pct:.2f}%</td>"
            f"<td>{a.entry_angle_deg:.2f}°</td>"
            f"<td>{a.exit_angle_deg:.2f}°</td>"
            f"<td>{a.twist_deg:.2f}°</td>"
            f"<td>{s.chord_length:.0f} px</td></tr>"
        )
    return (
        "<table class='aero'>"
        "<tr><th>#</th><th>camber</th><th>draft</th>"
        "<th>entry</th><th>exit</th><th>twist</th><th>chord</th></tr>"
        + "".join(rows) + "</table>"
    )


def plot_color_refinement(
    image_rgb: np.ndarray,
    refinements: list,  # list of Optional[ColorRefineResult]
    fitted_stripes: list,
) -> List[Tuple[str, str]]:
    panels: List[Tuple[str, str]] = []

    # Global overlay: before vs after
    fig, ax = _new_ax(image_rgb, "Stage 7 — color-refined centerlines (orange=before, cyan=after)",
                      figsize=(9, 6))
    for i, (s, r) in enumerate(zip(fitted_stripes, refinements)):
        ax.plot(s.spline_points[:, 0], s.spline_points[:, 1],
                color="#FFB020", linewidth=2.2, alpha=0.9, linestyle=":")
        if r is not None:
            ax.plot(r.refined_points[:, 0], r.refined_points[:, 1],
                    color="#00E5FF", linewidth=2.2, alpha=0.95)
    panels.append(("Before vs After", fig_to_b64(fig)))

    # Per-stripe diagnostic panels
    for i, (s, r) in enumerate(zip(fitted_stripes, refinements)):
        if r is None:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
        # rotated crop with before/after centerlines
        rgb_crop = cv2.cvtColor(r.crop, cv2.COLOR_BGR2RGB) if r.crop.ndim == 3 else r.crop
        axes[0].imshow(rgb_crop)
        H, W = r.crop.shape[:2]
        x_axis = np.linspace(0, W - 1, len(r.chord_t))
        half = (H - 1) / 2.0
        axes[0].plot(x_axis, np.full_like(x_axis, half),
                     color="#FFB020", linewidth=1.5, linestyle=":", label="input center")
        # refined offsets relative to input center in rotated frame
        dx = r.refined_points - r.original_points
        # dx is in image coords; its magnitude along normal is what we need.
        offsets = np.linalg.norm(dx, axis=1) * np.sign(
            np.einsum("ij,j->i", dx, _rot_normal(s.luff_endpoint, s.leech_endpoint))
        )
        axes[0].plot(x_axis, half + offsets, color="#00E5FF", linewidth=1.8, label="refined")
        axes[0].set_title(f"Stripe {i+1} — rotated crop (H={H}px)", fontsize=10)
        axes[0].axis("off")
        axes[0].legend(loc="upper right", fontsize=8)

        # label map
        label_rgb = np.zeros((*r.label_map.shape, 3), dtype=np.uint8)
        label_rgb[r.label_map == 1] = (60, 160, 220)   # sail=blue
        label_rgb[r.label_map == 2] = (255, 90, 120)   # stripe=red
        axes[1].imshow(label_rgb)
        axes[1].set_title("Pixel classification (blue=sail, red=stripe)", fontsize=10)
        axes[1].axis("off")

        # color priors
        sail_patch = np.full((80, 80, 3), r.sail_mean_bgr[::-1].astype(np.uint8))  # BGR→RGB
        stripe_patch = np.full((80, 80, 3), r.stripe_mean_bgr[::-1].astype(np.uint8))
        combo = np.hstack([sail_patch, stripe_patch])
        axes[2].imshow(combo)
        axes[2].set_title(
            f"Color priors · Δ={np.linalg.norm(r.sail_mean_bgr - r.stripe_mean_bgr):.0f}\n"
            f"mean band width: {float(r.band_widths_px.mean()):.1f}px   "
            f"mean shift: {r.offset_px_mean:.1f}px",
            fontsize=9,
        )
        axes[2].axis("off")
        axes[2].text(20, 72, "SAIL", color="white", fontsize=10, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="#00000088", edgecolor="none"))
        axes[2].text(100, 72, "STRIPE", color="white", fontsize=10, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="#00000088", edgecolor="none"))

        panels.append((f"Stripe {i+1}", fig_to_b64(fig)))

    return panels


def plot_color_refinement_multi(
    image_rgb: np.ndarray,
    multi_results: list,   # list[dict[method -> ColorRefineResult|None]]
    fitted_stripes: list,
) -> List[Tuple[str, str]]:
    """Stage 7 report: global overlay + per-stripe 4-method side-by-side."""
    panels: List[Tuple[str, str]] = []

    METHODS = [
        ("mahalanobis", "#00E5FF"),
        ("kmeans",      "#FF6B9D"),
        ("grabcut",     "#FFD400"),
        ("canny",       "#00E676"),
    ]

    # Global overlay — input (orange dotted) vs each method
    fig, ax = _new_ax(image_rgb,
                      "Stage 7 — refined centerlines (input = orange dotted)",
                      figsize=(10, 7))
    for i, (s, d) in enumerate(zip(fitted_stripes, multi_results)):
        ax.plot(s.spline_points[:, 0], s.spline_points[:, 1],
                color="#FFB020", linewidth=2.0, alpha=0.9, linestyle=":")
        for mname, mcolor in METHODS:
            r = d.get(mname)
            if r is None:
                continue
            ax.plot(r.refined_points[:, 0], r.refined_points[:, 1],
                    color=mcolor, linewidth=1.8, alpha=0.85, label=mname if i == 0 else None)
    if any(d.get(m[0]) is not None for d in multi_results for m in METHODS):
        ax.legend(loc="upper right", fontsize=9)
    panels.append(("All methods overlay", fig_to_b64(fig)))

    # Per-stripe: 2x4 grid (row 1: cropped label map, row 2: colored result)
    for i, (s, d) in enumerate(zip(fitted_stripes, multi_results)):
        fig, axes = plt.subplots(2, 4, figsize=(16, 6))
        for col, (mname, mcolor) in enumerate(METHODS):
            r = d.get(mname)
            ax_top = axes[0, col]
            ax_bot = axes[1, col]
            if r is None:
                ax_top.text(0.5, 0.5, f"{mname}\nSKIPPED", ha="center", va="center",
                            color="#FF6B6B", fontsize=14, fontweight="bold",
                            transform=ax_top.transAxes)
                ax_top.axis("off")
                ax_bot.axis("off")
                continue
            # Top: label map
            label_rgb = np.zeros((*r.label_map.shape, 3), dtype=np.uint8)
            label_rgb[r.label_map == 1] = (60, 160, 220)
            label_rgb[r.label_map == 2] = (255, 90, 120)
            ax_top.imshow(label_rgb)
            ax_top.axis("off")
            ax_top.set_title(f"{mname} — pixel classification", fontsize=10)

            # Bottom: rotated crop with input vs refined line
            crop_rgb = cv2.cvtColor(r.crop, cv2.COLOR_BGR2RGB) if r.crop.ndim == 3 else r.crop
            ax_bot.imshow(crop_rgb)
            H, W = r.crop.shape[:2]
            x_axis = np.linspace(0, W - 1, len(r.chord_t))
            half = (H - 1) / 2.0
            ax_bot.plot(x_axis, np.full_like(x_axis, half),
                        color="#FFB020", linewidth=1.2, linestyle=":")
            # Compute per-sample perp offset (rotated frame) from refined_points
            dx = r.refined_points - r.original_points
            normal = _rot_normal(s.luff_endpoint, s.leech_endpoint)
            proj = np.einsum("ij,j->i", dx, normal)
            ax_bot.plot(x_axis, half + proj, color=mcolor, linewidth=1.8)
            ax_bot.axis("off")
            ax_bot.set_title(
                f"shift {r.offset_px_mean:.1f}px · band {float(r.band_widths_px.mean()):.1f}px",
                fontsize=9)
        fig.suptitle(f"Stripe {i+1}", fontsize=12, fontweight="bold")
        panels.append((f"Stripe {i+1}", fig_to_b64(fig)))

    return panels


def _rot_normal(luff, leech):
    v = leech - luff
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.array([0.0, 1.0])
    u = v / n
    return np.array([-u[1], u[0]])


def plot_full_sail_analysis(
    image_rgb: np.ndarray, refined_stripes: list
) -> List[Tuple[str, str]]:
    """Stage 8 — overlay all refined stripes + trend charts."""
    panels: List[Tuple[str, str]] = []

    # Panel 1: refined stripes overlay with per-stripe aero labels
    fig, ax = _new_ax(image_rgb, "Refined stripes — best method per stripe",
                      figsize=(9, 6))
    for s in refined_stripes:
        color = STRIPE_COLORS[s.index % len(STRIPE_COLORS)]
        pts = s.refined_points
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2.4)
        ax.plot([s.luff_endpoint[0], s.leech_endpoint[0]],
                [s.luff_endpoint[1], s.leech_endpoint[1]],
                color=color, linestyle="--", linewidth=1.0, alpha=0.5)
        ax.plot(s.luff_endpoint[0], s.luff_endpoint[1], "o",
                color="white", markersize=8, markeredgecolor=color, markeredgewidth=2)
        ax.plot(s.leech_endpoint[0], s.leech_endpoint[1], "o",
                color="white", markersize=8, markeredgecolor=color, markeredgewidth=2)
        mid = pts[len(pts) // 2]
        ax.text(
            mid[0], mid[1] - 14,
            f"#{s.index+1} ({s.method_used})\n"
            f"c={s.aero.camber_depth_pct:.1f}% d={s.aero.draft_position_pct:.0f}%\n"
            f"ent={s.aero.entry_angle_deg:.1f}° ex={s.aero.exit_angle_deg:.1f}° "
            f"tw={s.aero.twist_deg:.1f}°",
            color=color, fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85,
                      edgecolor=color, linewidth=0.8),
        )
    panels.append(("All stripes — labeled", fig_to_b64(fig)))

    # Panel 2: camber profiles
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for s in refined_stripes:
        color = STRIPE_COLORS[s.index % len(STRIPE_COLORS)]
        ax.plot(s.chord_t * 100.0, s.camber_profile_pct,
                color=color, linewidth=2.0, label=f"#{s.index+1}")
    ax.axhline(0.0, color="#888888", linewidth=0.8, alpha=0.6)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("chord position (%)", fontsize=10)
    ax.set_ylabel("camber (% of chord)", fontsize=10)
    ax.set_title("Stripe camber profiles", fontsize=11)
    ax.legend(loc="best", fontsize=9)
    panels.append(("Camber profiles", fig_to_b64(fig)))

    # Panel 3: trend plots (camber and twist vs stripe height)
    if refined_stripes:
        sorted_stripes = sorted(
            refined_stripes,
            key=lambda s: 0.5 * (s.luff_endpoint[1] + s.leech_endpoint[1]),
        )
        heights = np.arange(len(sorted_stripes))  # 0=top → N-1=bottom
        cambers = np.array([s.aero.camber_depth_pct for s in sorted_stripes])
        drafts = np.array([s.aero.draft_position_pct for s in sorted_stripes])
        twists = np.array([s.aero.twist_deg for s in sorted_stripes])
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax_, vals, title, color in (
            (axes[0], cambers, "camber (% chord)", "#00D4AA"),
            (axes[1], drafts,  "draft position (% chord)", "#FFB020"),
            (axes[2], twists,  "twist (°)", "#4DA6FF"),
        ):
            ax_.plot(vals, heights, "o-", color=color, linewidth=2.0, markersize=7)
            ax_.invert_yaxis()
            ax_.set_ylabel("stripe index (top ↓ bottom)", fontsize=10)
            ax_.set_title(title, fontsize=10)
            ax_.grid(True, alpha=0.25)
        panels.append(("Camber / draft / twist vs height", fig_to_b64(fig)))

    return panels


def render_html(
    title: str,
    stage_sections: List[dict],
    sail_names: List[str],
) -> str:
    """Compose the full HTML document.

    stage_sections: one per stage. Each dict has:
      - id: "stage-0"
      - label: "Stage 0 — Calibration"
      - blurb: str (markdown-ish simple HTML)
      - per_sail: list of dicts, each with {"name": sail_name, "panels": [(caption, b64)], "stats_html": str}
    """
    tab_buttons = []
    tab_bodies = []
    for i, stage in enumerate(stage_sections):
        active = " active" if i == 0 else ""
        tab_buttons.append(
            f"<button class='tab-btn{active}' data-tab='{stage['id']}'>"
            f"{stage['label']}</button>"
        )
        sail_cards = []
        for sail in stage["per_sail"]:
            img_html = "".join(
                f"<figure class='panel'><figcaption>{cap}</figcaption>"
                f"<img src='data:image/png;base64,{b64}'/></figure>"
                for cap, b64 in sail["panels"]
            )
            stats = sail.get("stats_html", "")
            embed = sail.get("embed_html", "")
            embed_wrap = (
                f"<div class='plotly-container'>{embed}</div>" if embed else ""
            )
            sail_cards.append(
                f"<section class='sail-card'>"
                f"<h3>{sail['name']}</h3>"
                f"{stats}"
                f"<div class='panel-grid'>{img_html}</div>"
                f"{embed_wrap}"
                f"</section>"
            )
        tab_bodies.append(
            f"<div class='tab-body{active}' id='{stage['id']}'>"
            f"<div class='stage-blurb'>{stage['blurb']}</div>"
            + "".join(sail_cards) + "</div>"
        )

    style = """
    body{font-family:-apple-system,Arial,sans-serif;margin:0;padding:0;background:#0e0f13;color:#e5e7eb}
    header{padding:22px 32px;border-bottom:1px solid #222;background:#14151a}
    h1{margin:0;font-size:22px;color:#fafafa}
    .sub{color:#9aa0a6;margin-top:4px;font-size:13px}
    .tabs{display:flex;flex-wrap:wrap;gap:4px;padding:10px 24px;background:#1a1c22;border-bottom:1px solid #262930;position:sticky;top:0;z-index:10}
    .tab-btn{background:#262930;color:#cfd3dc;border:1px solid #2f333c;padding:7px 14px;border-radius:6px;font-size:13px;cursor:pointer}
    .tab-btn:hover{background:#2f333c}
    .tab-btn.active{background:#0a84ff;color:white;border-color:#0a84ff}
    .tab-body{display:none;padding:22px 32px}
    .tab-body.active{display:block}
    .stage-blurb{background:#14151a;border:1px solid #23262e;padding:14px 18px;border-radius:10px;margin-bottom:20px;line-height:1.55;font-size:14px;color:#d0d3d9}
    .stage-blurb h2{margin:0 0 6px 0;font-size:16px;color:#fff}
    .sail-card{background:#14151a;border:1px solid #23262e;padding:14px 18px;border-radius:10px;margin-bottom:20px}
    .sail-card h3{margin:0 0 10px 0;font-size:15px;color:#fff;border-bottom:1px solid #23262e;padding-bottom:6px}
    .panel-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
    .panel{background:#0e0f13;border:1px solid #23262e;padding:8px;border-radius:8px;margin:0}
    .panel figcaption{font-size:12px;color:#9aa0a6;margin-bottom:6px;padding:2px 4px}
    .panel img{width:100%;display:block;border-radius:4px}
    .plotly-container{margin-top:12px;background:#0e0f13;border:1px solid #23262e;border-radius:8px;padding:6px}
    table.aero, table.stats {border-collapse:collapse;font-size:12px;margin:8px 0}
    table.aero th, table.aero td, table.stats th, table.stats td {border:1px solid #23262e;padding:4px 8px;text-align:right}
    table.aero th, table.stats th {background:#1a1c22;color:#fafafa}
    table.aero td:first-child, table.stats td:first-child {text-align:left;font-weight:600;color:#cfd3dc}
    .badge{display:inline-block;padding:3px 9px;border-radius:12px;font-size:11px;font-weight:600;margin-right:6px}
    .badge.ok{background:#0f4f3c;color:#6ee7b7}
    .badge.warn{background:#4a3b0e;color:#fbbf24}
    .badge.err{background:#4a1218;color:#f87171}
    """

    script = """
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-body').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.add('active');
      });
    });
    """

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{style}</style></head><body>"
        f"<header><h1>{title}</h1>"
        f"<div class='sub'>Sails: {' · '.join(sail_names)}</div></header>"
        f"<nav class='tabs'>{''.join(tab_buttons)}</nav>"
        f"{''.join(tab_bodies)}"
        f"<script>{script}</script>"
        "</body></html>"
    )
