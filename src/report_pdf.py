"""One-page PDF report — marine palette, gridspec layout (no overlaps).

A3 landscape. Layout:

  ┌────────────────────────────────────────────────────────────────────┐
  │  HEADER : logo · sail name · yacht · timestamp · telemetry chips   │
  ├──────────────────────────────────────┬─────────────────────────────┤
  │                                      │  GAUGE + HEADLINE TILES     │
  │                                      ├─────────────────────────────┤
  │       SAIL PHOTO + OVERLAY           │  CAMBER         │ TWIST     │
  │       (telemetry box inside)         ├─────────────────┴───────────┤
  │                                      │  LUFF / LEECH BEND           │
  │                                      ├─────────────────────────────┤
  │                                      │  STRIPE RECAP  ·  COMMENTS  │
  └──────────────────────────────────────┴─────────────────────────────┘
"""

from __future__ import annotations

import io
import textwrap
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Wedge


# ---- Marine palette ------------------------------------------------------

C = {
    "deep":   "#041526",
    "hull":   "#0a1f35",
    "steel":  "#0f2d4a",
    "stroke": "#1e3a5f",
    "sail":   "#f2ead3",
    "cyan":   "#2cc3ff",
    "orange": "#ff6b35",
    "green":  "#2fdd92",
    "amber":  "#ffb020",
    "red":    "#ff5e5e",
    "fg":     "#e8eef7",
    "fg2":    "#9aadc3",
}
STRIPE_COLORS = ["#FF3B5C", "#00D4AA", "#FFB020", "#4DA6FF",
                 "#C77DFF", "#FF6B9D", "#00E5FF"]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _style_axes(ax):
    ax.set_facecolor(C["hull"])
    ax.tick_params(colors=C["fg2"], labelsize=7)
    for s in ax.spines.values():
        s.set_color(C["stroke"])
    ax.grid(True, alpha=0.2, color=C["stroke"], linewidth=0.5)


def _panel(ax, title: str):
    """Turn an empty axis into a marine panel with a small title band.

    Returns a NEW inner axis (clipped inside the panel) that content
    should draw on.
    """
    ax.set_facecolor("none")
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.0,rounding_size=0.02",
        facecolor=C["hull"], edgecolor=C["stroke"], linewidth=0.8,
        transform=ax.transAxes, clip_on=False,
    ))
    ax.text(0.025, 0.955, title.upper(),
            color=C["fg2"], fontsize=8, weight="bold", family="monospace",
            transform=ax.transAxes, va="top")
    # Inner axis (within the panel box, leaving room for the title)
    fig = ax.figure
    bbox = ax.get_position()
    inset_left = bbox.x0 + bbox.width * 0.06
    inset_bottom = bbox.y0 + bbox.height * 0.08
    inset_w = bbox.width * 0.88
    inset_h = bbox.height * 0.82
    inner = fig.add_axes([inset_left, inset_bottom, inset_w, inset_h])
    return inner


def _tile(ax, label: str, value: str, unit: str = ""):
    ax.set_facecolor("none")
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        facecolor=C["steel"], edgecolor=C["stroke"], linewidth=0.6,
        transform=ax.transAxes, clip_on=False,
    ))
    ax.text(0.10, 0.72, label,
            fontsize=7.5, color=C["fg2"], family="monospace",
            weight="bold", transform=ax.transAxes)
    ax.text(0.10, 0.30, value,
            fontsize=17, color=C["fg"], family="monospace",
            weight="bold", transform=ax.transAxes)
    if unit:
        ax.text(0.10 + 0.03 + len(value) * 0.10, 0.32, unit,
                fontsize=10, color=C["fg2"], family="monospace",
                transform=ax.transAxes)


# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------

def _render_header(ax, analysis, telemetry, photo_time):
    ax.set_facecolor("none")
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Panel background
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.0,rounding_size=0.03",
        facecolor=C["hull"], edgecolor=C["stroke"], linewidth=0.8,
        transform=ax.transAxes, clip_on=False,
    ))
    # Orange accent stripe on the left
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.0), 0.006, 1.0,
        boxstyle="square,pad=0",
        facecolor=C["orange"], edgecolor="none",
        transform=ax.transAxes, clip_on=False,
    ))

    # Title + subtitle (left)
    ax.text(0.02, 0.74, "SAILSHAPE  ·  v7",
            fontsize=9, color=C["fg2"], weight="bold",
            family="monospace", transform=ax.transAxes)
    name = analysis.get("display_name", "Sail analysis")
    ax.text(0.02, 0.32, name,
            fontsize=18, color=C["fg"], weight="bold",
            transform=ax.transAxes)

    # Chips right-to-left
    chips = []
    yacht = analysis.get("yacht")
    if yacht:
        chips.append(("YACHT", str(yacht)))
    if photo_time is not None:
        chips.append(("CAPTURED", str(photo_time)))
    if telemetry:
        for k, unit in (("TWS", "kt"), ("TWA", "°"), ("AWS", "kt"),
                         ("AWA", "°"), ("BSP", "kt"), ("Heel", "°")):
            v = telemetry.get(k)
            if v is None:
                continue
            chips.append((k, f"{v:.1f} {unit}"))

    # Render chips right-aligned
    x_cursor = 0.99
    chip_h = 0.52
    chip_pad_x = 0.010
    for lbl, val in reversed(chips):
        txt = f"{lbl}  {val}"
        # Estimate width from char count
        chip_w = max(0.06, len(txt) * 0.0075)
        chip_left = x_cursor - chip_w
        ax.add_patch(FancyBboxPatch(
            (chip_left, 0.22), chip_w, chip_h,
            boxstyle="round,pad=0.0,rounding_size=0.2",
            facecolor=C["steel"], edgecolor=C["stroke"], linewidth=0.6,
            transform=ax.transAxes, clip_on=False,
        ))
        ax.text(
            chip_left + chip_pad_x, 0.22 + chip_h / 2.0, txt,
            ha="left", va="center",
            color=C["fg"], fontsize=8, family="monospace",
            transform=ax.transAxes,
        )
        x_cursor = chip_left - 0.01


# ---------------------------------------------------------------------------
# sail photo
# ---------------------------------------------------------------------------

def _render_photo(ax, analysis, telemetry):
    ax.set_facecolor(C["hull"])
    img = analysis["image_rgb"]
    ax.imshow(img)
    for i, (r, cst_xy) in enumerate(zip(analysis["v7_results"],
                                          analysis["cst_splines"])):
        clr = STRIPE_COLORS[i % len(STRIPE_COLORS)]
        if cst_xy is not None:
            ax.plot(cst_xy[:, 0], cst_xy[:, 1], "-", color=clr, lw=3.0,
                    solid_capstyle="round")
        elif r.spline_points is not None:
            ax.plot(r.spline_points[:, 0], r.spline_points[:, 1],
                    "-", color=clr, lw=3.0, solid_capstyle="round")
        ax.plot(r.luff_ep[0], r.luff_ep[1], "o", color=clr,
                markersize=11, markeredgecolor="white", markeredgewidth=1.4,
                zorder=5)
        ax.plot(r.leech_ep[0], r.leech_ep[1], "o", color=clr,
                markersize=11, markeredgecolor="white", markeredgewidth=1.4,
                zorder=5)
    ax.set_axis_off()

    if telemetry:
        h, w = img.shape[:2]
        lines = []
        for k in ("TWS", "TWA", "AWS", "AWA", "BSP", "Heel"):
            v = telemetry.get(k)
            if v is None:
                continue
            unit = {"TWS": "kt", "TWA": "°", "AWS": "kt",
                    "AWA": "°", "BSP": "kt", "Heel": "°"}[k]
            lines.append(f"{k:<4}  {v:>6.2f} {unit}")
        if telemetry.get("matched_time"):
            lines.insert(0, str(telemetry["matched_time"]))
        if lines:
            ax.text(
                w * 0.98, h * 0.98, "\n".join(lines),
                ha="right", va="bottom",
                fontsize=9, color=C["fg"], family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.55,rounding_size=0.3",
                    facecolor=f"{C['deep']}ee",
                    edgecolor=C["cyan"], linewidth=1.3,
                ),
            )


# ---------------------------------------------------------------------------
# quality gauge
# ---------------------------------------------------------------------------

def _render_gauge(ax, score: float):
    ax.set_facecolor("none")
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-0.35, 1.12)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Track — 4 coloured bands
    bands = [(180, 135, C["orange"]),
             (135, 90,  C["amber"]),
             (90, 45,   C["cyan"]),
             (45, 0,    C["green"])]
    for start, end, colour in bands:
        w = Wedge((0, 0), 1.0, end, start,
                  width=0.22, facecolor=colour, alpha=0.22,
                  edgecolor="none")
        ax.add_patch(w)

    # Active arc 180° → 0°
    s = max(0.0, min(100.0, float(score)))
    theta_end = 180.0 - (s / 100.0) * 180.0
    active = (C["green"] if s >= 75 else
              C["amber"] if s >= 50 else C["orange"])
    ax.add_patch(Wedge(
        (0, 0), 1.0, theta_end, 180,
        width=0.22, facecolor=active, edgecolor="none",
    ))

    # Needle + hub
    rad = np.deg2rad(theta_end)
    ax.plot([0, 0.92 * np.cos(rad)],
            [0, 0.92 * np.sin(rad)],
            color=C["fg"], lw=2.5, solid_capstyle="round")
    ax.plot(0, 0, "o", color=C["fg"], markersize=7, zorder=5)

    ax.text(0, -0.04, f"{s:.0f}",
            ha="center", va="center",
            fontsize=26, color=C["fg"], family="monospace", weight="bold")
    ax.text(0, -0.25, "QUALITY  SCORE",
            ha="center", va="center",
            fontsize=8, color=C["fg2"], family="monospace", weight="bold")


# ---------------------------------------------------------------------------
# camber overlay + twist view
# ---------------------------------------------------------------------------

def _cst_profile(cst_xy, r):
    if cst_xy is None:
        return None
    chord = r.leech_endpoint - r.luff_endpoint
    L = float(np.linalg.norm(chord))
    if L < 1e-3:
        return None
    u = chord / L
    n = np.array([-u[1], u[0]])
    rel = cst_xy.astype(np.float64) - r.luff_endpoint
    proj = (rel @ u) / L
    perp = rel @ n
    o = np.argsort(proj)
    return proj[o], perp[o] / L


def _render_camber(ax, refined, cst_splines):
    _style_axes(ax)
    heights = [0.5 * (r.luff_endpoint[1] + r.leech_endpoint[1]) for r in refined]
    order = list(np.argsort(heights))
    for pos, idx in enumerate(order):
        r = refined[idx]
        clr = STRIPE_COLORS[pos % len(STRIPE_COLORS)]
        prof = _cst_profile(
            cst_splines[idx] if idx < len(cst_splines) else None, r,
        )
        if prof is not None:
            t, d = prof
            ax.plot(t * 100, d * 100, lw=2.0, color=clr,
                    label=f"s{pos+1}  c{r.aero.camber_depth_pct:.1f}%")
        else:
            ax.plot(r.chord_t * 100, r.camber_profile_pct,
                    lw=2.0, color=clr, label=f"s{pos+1}")
    ax.axhline(0, color=C["stroke"], lw=0.6)
    ax.set_xlabel("chord %", color=C["fg"], fontsize=8)
    ax.set_ylabel("camber %", color=C["fg"], fontsize=8)
    ax.legend(facecolor=C["steel"], edgecolor=C["stroke"],
               labelcolor=C["fg"], fontsize=7, loc="best")


def _render_twist(ax, refined, cst_splines):
    _style_axes(ax)
    heights = [0.5 * (r.luff_endpoint[1] + r.leech_endpoint[1]) for r in refined]
    order = list(np.argsort(heights))
    cexag = 2.0
    for pos, idx in enumerate(order):
        r = refined[idx]
        clr = STRIPE_COLORS[pos % len(STRIPE_COLORS)]
        tw = np.radians(r.aero.twist_deg)
        leech = np.array([np.cos(tw), np.sin(tw)])
        prof = _cst_profile(
            cst_splines[idx] if idx < len(cst_splines) else None, r,
        )
        if prof is not None:
            t_norm, camber_rel = prof
            camber_pct = camber_rel * 100.0
        else:
            t_norm = r.chord_t.astype(np.float64)
            camber_pct = r.camber_profile_pct.astype(np.float64)
        u_vec = leech / (np.linalg.norm(leech) + 1e-9)
        n_vec = np.array([-u_vec[1], u_vec[0]])
        xs = t_norm * leech[0] + n_vec[0] * (camber_pct / 100.0) * cexag
        ys = t_norm * leech[1] + n_vec[1] * (camber_pct / 100.0) * cexag
        ax.plot(xs, ys, lw=2.0, color=clr)
        ax.plot([0, leech[0]], [0, leech[1]], ls="--", lw=0.6,
                color=clr, alpha=0.5)
        ax.plot(0, 0, "o", color=clr, markersize=4,
                markeredgecolor="white", markeredgewidth=0.4)
        ax.plot(leech[0], leech[1], "o", color=clr, markersize=4,
                markeredgecolor="white", markeredgewidth=0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("chord (norm)", color=C["fg"], fontsize=8)
    ax.set_ylabel("×2 camber", color=C["fg"], fontsize=8)


def _render_bend(ax, luff_depth, leech_depth):
    _style_axes(ax)
    if luff_depth is None or leech_depth is None:
        return

    def _profile(spline):
        pts = spline.astype(np.float64)
        start, end = pts[0], pts[-1]
        chord = end - start
        L = float(np.linalg.norm(chord))
        if L < 1e-3:
            return None, None
        u = chord / L
        n = np.array([-u[1], u[0]])
        rel = pts - start
        proj = (rel @ u) / L
        perp = rel @ n
        o = np.argsort(proj)
        return proj[o], perp[o]

    for depth, clr, label in (
        (luff_depth, C["cyan"], "luff"),
        (leech_depth, C["orange"], "leech"),
    ):
        proj, perp = _profile(depth.spline)
        if proj is None:
            continue
        y_pct = proj * 100.0
        ax.plot(perp, y_pct, "o", color=clr, markersize=2, alpha=0.35)
        if len(y_pct) > 4:
            coefs = np.polyfit(y_pct, perp, 3)
            x_f = np.linspace(0, 100, 200)
            y_f = np.polyval(coefs, x_f)
            ax.plot(y_f, x_f, lw=2.2, color=clr,
                    label=f"{label}  max {depth.max_depth_pct:.2f}%")
    ax.axvline(0, color=C["stroke"], lw=0.6)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("deviation (px)", color=C["fg"], fontsize=8)
    ax.set_ylabel("% tack → head", color=C["fg"], fontsize=8)
    ax.legend(facecolor=C["steel"], edgecolor=C["stroke"],
               labelcolor=C["fg"], fontsize=7, loc="lower right")


# ---------------------------------------------------------------------------
# recap table / comments
# ---------------------------------------------------------------------------

def _render_recap(ax, refined):
    ax.set_facecolor(C["hull"])
    ax.set_axis_off()
    if not refined:
        ax.text(0.5, 0.5, "No stripes detected",
                color=C["fg2"], ha="center", va="center")
        return
    heights = [0.5 * (r.luff_endpoint[1] + r.leech_endpoint[1]) for r in refined]
    order = list(np.argsort(heights))
    col_labels = ["#", "chord", "camber", "draft", "twist", "entry", "exit"]
    rows = []
    for pos, idx in enumerate(order):
        r = refined[idx]
        rows.append([
            f"s{pos + 1}",
            f"{r.chord_length_px:.0f}",
            f"{r.aero.camber_depth_pct:.1f}%",
            f"{r.aero.draft_position_pct:.0f}%",
            f"{r.aero.twist_deg:+.1f}°",
            f"{r.aero.entry_angle_deg:.0f}°",
            f"{r.aero.exit_angle_deg:.0f}°",
        ])
    tbl = ax.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r_i, c_i), cell in tbl.get_celld().items():
        cell.set_edgecolor(C["stroke"])
        cell.set_linewidth(0.5)
        if r_i == 0:
            cell.set_facecolor(C["steel"])
            cell.set_text_props(color=C["fg2"], weight="bold",
                                 family="monospace")
        else:
            cell.set_facecolor(C["hull"])
            cell.set_text_props(color=C["fg"], family="monospace")


def _render_comments(ax, comments: Optional[str]):
    ax.set_facecolor("none")
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if not comments:
        ax.text(0.5, 0.5, "No trim comments provided.",
                color=C["fg2"], ha="center", va="center",
                fontsize=9, style="italic", transform=ax.transAxes)
        return
    wrapped = textwrap.fill(comments.strip(), width=55)
    ax.text(0.03, 0.92, wrapped,
            color=C["fg"], fontsize=9, transform=ax.transAxes,
            va="top", family="sans-serif", linespacing=1.4)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def build_pdf(
    analysis: Dict[str, Any],
    telemetry: Optional[Dict[str, Any]] = None,
    quality: Optional[float] = None,
) -> bytes:
    """Render single-page A3 landscape PDF. Returns bytes."""
    photo_time = analysis.get("photo_time")
    comments = analysis.get("trim_comments")

    if quality is None:
        from src.yachts import analysis_quality_score
        quality = analysis_quality_score(analysis).score

    refined = analysis["refined_stripes"]
    n = len(refined)
    avg_c = (
        float(np.mean([r.aero.camber_depth_pct for r in refined])) if refined else 0.0
    )
    avg_draft = (
        float(np.mean([r.aero.draft_position_pct for r in refined])) if refined else 0.0
    )
    avg_twist = (
        float(np.mean([r.aero.twist_deg for r in refined])) if refined else 0.0
    )
    luff_bend = (
        analysis["luff_depth"].max_depth_pct if analysis["luff_depth"] else 0.0
    )
    leech_bend = (
        analysis["leech_depth"].max_depth_pct if analysis["leech_depth"] else 0.0
    )

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(16.5, 11.7))   # A3 landscape
        fig.patch.set_facecolor(C["deep"])

        # Master gridspec: header row + body row
        outer = fig.add_gridspec(
            nrows=2, ncols=1,
            height_ratios=[0.11, 0.89],
            hspace=0.04,
            left=0.018, right=0.982, top=0.975, bottom=0.02,
        )
        # Header
        ax_h = fig.add_subplot(outer[0])
        _render_header(ax_h, analysis, telemetry, photo_time)

        # Body grid: 2 columns
        body = outer[1].subgridspec(
            nrows=1, ncols=2,
            width_ratios=[0.58, 0.42],
            wspace=0.03,
        )

        # --- LEFT : sail photo panel ---
        ax_photo_panel = fig.add_subplot(body[0, 0])
        ax_photo_inner = _panel(ax_photo_panel, "sail overlay")
        _render_photo(ax_photo_inner, analysis, telemetry)

        # --- RIGHT : vertical stack of panels ---
        right = body[0, 1].subgridspec(
            nrows=4, ncols=1,
            height_ratios=[0.24, 0.28, 0.20, 0.28],
            hspace=0.08,
        )

        # 1) Quality gauge + tile grid
        top_split = right[0].subgridspec(
            nrows=1, ncols=2, width_ratios=[0.38, 0.62], wspace=0.05,
        )
        ax_gauge_panel = fig.add_subplot(top_split[0, 0])
        ax_gauge_inner = _panel(ax_gauge_panel, "analysis quality")
        _render_gauge(ax_gauge_inner, quality)

        ax_tiles_panel = fig.add_subplot(top_split[0, 1])
        ax_tiles_panel.set_facecolor("none")
        ax_tiles_panel.set_axis_off()
        # Build a 2x4 tile grid inside this panel
        ax_tiles_panel.add_patch(FancyBboxPatch(
            (0.0, 0.0), 1.0, 1.0,
            boxstyle="round,pad=0.0,rounding_size=0.02",
            facecolor=C["hull"], edgecolor=C["stroke"], linewidth=0.8,
            transform=ax_tiles_panel.transAxes, clip_on=False,
        ))
        ax_tiles_panel.text(0.03, 0.955, "HEADLINE",
                              color=C["fg2"], fontsize=8, weight="bold",
                              family="monospace",
                              transform=ax_tiles_panel.transAxes, va="top")

        tgrid = top_split[0, 1].subgridspec(
            nrows=2, ncols=4, wspace=0.08, hspace=0.14,
        )
        tiles = [
            ("STRIPES",  f"{n}",           ""),
            ("CAMBER",   f"{avg_c:.1f}",   "%"),
            ("DRAFT",    f"{avg_draft:.0f}", "%"),
            ("TWIST",    f"{avg_twist:+.1f}", "°"),
            ("LUFF",     f"{luff_bend:.2f}", "%"),
            ("LEECH",    f"{leech_bend:.2f}", "%"),
            ("TIME",     photo_time.strftime("%H:%M:%S") if hasattr(photo_time, "strftime") else "—", ""),
            ("CLASS",    (analysis.get("yacht") or "—")[:10], ""),
        ]
        for i, (lbl, val, unit) in enumerate(tiles):
            row, col = divmod(i, 4)
            # Shrink each tile into the cell with extra padding so it
            # doesn't touch the panel border.
            ax_t_outer = fig.add_subplot(tgrid[row, col])
            pos = ax_t_outer.get_position()
            pad = 0.006
            ax_t_outer.remove()
            ax_t = fig.add_axes([
                pos.x0 + pad, pos.y0 + pad,
                pos.width - 2 * pad, pos.height - 2 * pad,
            ])
            _tile(ax_t, lbl, val, unit)

        # 2) Camber overlay + twist view side-by-side
        mid = right[1].subgridspec(nrows=1, ncols=2, wspace=0.06)
        ax_cam_panel = fig.add_subplot(mid[0, 0])
        ax_cam_inner = _panel(ax_cam_panel, "camber profile (cst)")
        _render_camber(ax_cam_inner, refined, analysis["cst_splines"])

        ax_tw_panel = fig.add_subplot(mid[0, 1])
        ax_tw_inner = _panel(ax_tw_panel, "stripes 2D · twist")
        _render_twist(ax_tw_inner, refined, analysis["cst_splines"])

        # 3) Luff / leech bend
        ax_bend_panel = fig.add_subplot(right[2])
        ax_bend_inner = _panel(ax_bend_panel, "luff / leech bend")
        _render_bend(ax_bend_inner, analysis["luff_depth"],
                      analysis["leech_depth"])

        # 4) Recap + comments
        bot = right[3].subgridspec(nrows=1, ncols=2,
                                     width_ratios=[0.58, 0.42],
                                     wspace=0.05)
        ax_recap_panel = fig.add_subplot(bot[0, 0])
        ax_recap_inner = _panel(ax_recap_panel, "stripe recap")
        _render_recap(ax_recap_inner, refined)

        ax_com_panel = fig.add_subplot(bot[0, 1])
        ax_com_inner = _panel(ax_com_panel, "trim comments")
        _render_comments(ax_com_inner, comments)

        pdf.savefig(fig, facecolor=C["deep"])
        plt.close(fig)

    return buf.getvalue()
