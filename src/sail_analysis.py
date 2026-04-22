"""Stage 8 — full sail analysis and Stage 9 — 3D reconstruction.

Given the color-refined stripes from Stage 7 and the luff / leech splines
from Stage 1 we produce:

* A *fused* set of stripes (per-stripe best refinement) with aero
  parameters.
* Trend plots: camber vs stripe height, twist vs stripe height.
* A 3D plotly figure: stripes are horizontal sections of the sail;
  camber becomes out-of-plane depth; luff / leech provide the vertical
  envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.types import AeroParams


@dataclass
class RefinedStripe:
    """Best-per-method refinement result ready for sail-level analysis."""

    index: int
    refined_points: np.ndarray  # (N, 2) image coords
    luff_endpoint: np.ndarray   # (2,)
    leech_endpoint: np.ndarray  # (2,)
    chord_length_px: float
    aero: AeroParams
    method_used: str            # "mahalanobis" | "kmeans" | "grabcut" | "canny" | "original"
    camber_profile_pct: np.ndarray  # (100,) perpendicular deviation / chord * 100
    chord_t: np.ndarray             # (100,) normalized chord position


def _camber_profile(
    spline: np.ndarray, luff: np.ndarray, leech: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (chord_t, signed perpendicular deviation, chord_length)."""
    chord = leech - luff
    L = float(np.linalg.norm(chord))
    if L < 1e-3:
        n = len(spline)
        return np.linspace(0, 1, n), np.zeros(n, dtype=np.float32), 0.0
    u = chord / L
    n = np.array([-u[1], u[0]])
    rel = spline.astype(np.float64) - luff
    proj = (rel @ u) / L
    perp = rel @ n
    return proj.astype(np.float32), perp.astype(np.float32), L


def _aero_from_profile(
    proj: np.ndarray, perp: np.ndarray, L: float, chord_unit: np.ndarray
) -> AeroParams:
    """Derive camber depth, draft position and entry/exit angles from the
    signed camber profile alone."""
    if L < 1e-3 or len(perp) < 4:
        return AeroParams(0, 50, 0, 0, 0)

    idx = int(np.argmax(np.abs(perp)))
    camber_depth_pct = float(abs(perp[idx]) / L * 100.0)
    draft_pct = float(np.clip(proj[idx] * 100.0, 0, 100))

    # Entry / exit slopes from finite differences of perp vs proj (in %)
    if len(perp) >= 8:
        entry_slope = (perp[3] - perp[0]) / max(proj[3] - proj[0], 1e-6)
        exit_slope = (perp[-1] - perp[-4]) / max(proj[-1] - proj[-4], 1e-6)
    else:
        entry_slope = (perp[1] - perp[0]) / max(proj[1] - proj[0], 1e-6)
        exit_slope = (perp[-1] - perp[-2]) / max(proj[-1] - proj[-2], 1e-6)
    entry_angle = float(np.degrees(np.arctan(entry_slope)))
    exit_angle = float(np.degrees(-np.arctan(exit_slope)))

    return AeroParams(
        camber_depth_pct=camber_depth_pct,
        draft_position_pct=draft_pct,
        twist_deg=0.0,  # filled in at the sail level
        entry_angle_deg=float(np.clip(entry_angle, -45, 45)),
        exit_angle_deg=float(np.clip(exit_angle, -45, 45)),
    )


def _score_refinement(r) -> float:
    """Pick the method with the narrowest plausible band width (2–30 px)
    and smallest offset jitter."""
    if r is None:
        return -np.inf
    bw = float(r.band_widths_px.mean()) if len(r.band_widths_px) else 0.0
    if bw <= 0 or bw > 60:
        return -np.inf
    shift = r.offset_px_mean
    # Higher score = better: narrow band, small shift (stable refinement)
    return -bw - 0.3 * shift


def pick_best_refinement(methods_dict) -> Tuple[str, object]:
    """Return (method_name, result) — the best Stage-7 method for this stripe."""
    best_name = "original"
    best_res = None
    best_score = -np.inf
    for name, r in methods_dict.items():
        s = _score_refinement(r)
        if s > best_score:
            best_score = s
            best_name = name
            best_res = r
    return best_name, best_res


def build_refined_stripes(
    fitted_stripes: list,
    multi_color_results: Optional[list] = None,
) -> List[RefinedStripe]:
    """Aggregate per-stripe analysis. When ``multi_color_results`` is
    None (Stage 7 disabled) the fitted centerline is used directly.
    """
    out: List[RefinedStripe] = []
    for i, fs in enumerate(fitted_stripes):
        methods = multi_color_results[i] if multi_color_results else None
        if methods:
            method_name, best_r = pick_best_refinement(methods)
            spline = best_r.refined_points if best_r is not None else fs.spline_points
            if best_r is None:
                method_name = "original"
        else:
            spline = fs.spline_points
            method_name = "original"

        proj, perp, L = _camber_profile(spline, fs.luff_endpoint, fs.leech_endpoint)
        chord_unit = fs.leech_endpoint - fs.luff_endpoint
        if L > 1e-3:
            chord_unit = chord_unit / L
        aero = _aero_from_profile(proj, perp, L, chord_unit)

        # Resample camber profile at 100 fixed chord positions for plotting
        t_grid = np.linspace(0, 1, 100)
        order = np.argsort(proj)
        perp_ordered = perp[order]; proj_ordered = proj[order]
        camber_pct = np.interp(t_grid, proj_ordered, perp_ordered / max(L, 1e-3) * 100.0)

        out.append(RefinedStripe(
            index=i,
            refined_points=spline.astype(np.float32),
            luff_endpoint=fs.luff_endpoint.astype(np.float32),
            leech_endpoint=fs.leech_endpoint.astype(np.float32),
            chord_length_px=L,
            aero=aero,
            method_used=method_name,
            camber_profile_pct=camber_pct.astype(np.float32),
            chord_t=t_grid.astype(np.float32),
        ))

    # Fill twist: angle of each chord vs the lowest (bottom-most) chord
    if len(out) >= 2:
        # Sort by chord-mid y (top to bottom in image)
        sorted_idx = sorted(range(len(out)),
                            key=lambda i: 0.5 * (out[i].luff_endpoint[1] + out[i].leech_endpoint[1]))
        ref_i = sorted_idx[-1]  # bottom stripe
        ref_vec = out[ref_i].leech_endpoint - out[ref_i].luff_endpoint
        ref_ang = np.degrees(np.arctan2(ref_vec[1], ref_vec[0]))
        for k in range(len(out)):
            v = out[k].leech_endpoint - out[k].luff_endpoint
            ang = np.degrees(np.arctan2(v[1], v[0]))
            twist = ang - ref_ang
            # Wrap to [-180, 180]
            twist = (twist + 180) % 360 - 180
            out[k].aero = AeroParams(
                camber_depth_pct=out[k].aero.camber_depth_pct,
                draft_position_pct=out[k].aero.draft_position_pct,
                twist_deg=float(twist),
                entry_angle_deg=out[k].aero.entry_angle_deg,
                exit_angle_deg=out[k].aero.exit_angle_deg,
            )
    return out


def _edge_x_at_y(spline: np.ndarray, y_target: float) -> float:
    """Interpolate ``spline``'s x-coordinate at image row ``y_target``."""
    if spline is None or len(spline) < 2:
        return float("nan")
    order = np.argsort(spline[:, 1])
    ys = spline[order, 1]
    xs = spline[order, 0]
    if y_target <= ys[0]:
        return float(xs[0])
    if y_target >= ys[-1]:
        return float(xs[-1])
    return float(np.interp(y_target, ys, xs))


def _polyline_arc_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def build_3d_plotly_html(
    refined: List[RefinedStripe],
    luff_depth,       # EdgeDepth | None
    leech_depth,      # EdgeDepth | None
    head_point: np.ndarray,
    tack_point: np.ndarray,
    clew_point: np.ndarray,
    luff_length_m: Optional[float] = None,
    foot_length_m: Optional[float] = None,
    sail_type: str = "main",
) -> str:
    """3D sail reconstruction using the DETECTED luff / leech shapes.

    The sail outline is built from the luff_depth.spline and
    leech_depth.spline rather than a straight triangle. Each image-pixel
    edge point is mapped to 3D (x, 0, z) using:

        z_3d = (y_foot_img - y_img) * (luff_length_m / span_y_px)
        x_3d = (x_img - x_tack_img) * (foot_length_m / foot_width_px)

    so the 3D outline reflects the sail's actual shape (including roach
    on the leech, mast bend on the luff, etc.). Camber from the stripes
    is applied in the +y direction perpendicular to the local chord.
    """
    try:
        import plotly.graph_objects as go
        import plotly.offline as offline
    except Exception:
        return "<div style='color:#f66'>plotly not installed</div>"

    if not refined or luff_depth is None or leech_depth is None:
        return "<div style='color:#aaa'>Need luff, leech, and ≥ 1 stripe for 3D.</div>"

    luff_m = float(luff_length_m) if (luff_length_m and luff_length_m > 0) else 10.0
    foot_m = float(foot_length_m) if (foot_length_m and foot_length_m > 0) else 4.0

    y_head_img = float(head_point[1])
    y_foot_img = max(float(tack_point[1]), float(clew_point[1]))
    span_y_px = max(y_foot_img - y_head_img, 1.0)
    z_scale = luff_m / span_y_px

    # Horizontal scale: use the bbox spanned by (tack → clew) in image px.
    # This preserves the ratio between how curved the leech looks in the
    # photo and the absolute foot dimension.
    x_tack_img = float(tack_point[0])
    x_clew_img = float(clew_point[0])
    foot_px = max(abs(x_clew_img - x_tack_img), 1.0)
    x_scale = foot_m / foot_px
    # Sign convention: want x=0 at the luff (mast). If clew is to the RIGHT
    # of the tack in the image, +x points toward the leech; else flip.
    x_sign = 1.0 if x_clew_img >= x_tack_img else -1.0

    def _img_to_3d(x_img: float, y_img: float) -> Tuple[float, float]:
        x_3d = (x_img - x_tack_img) * x_scale * x_sign
        z_3d = (y_foot_img - y_img) * z_scale
        return x_3d, z_3d

    # --- 1. Luff & leech edges in 3D (curves in the xz plane, y=0) --------
    def _edge_3d(spline: np.ndarray):
        # Sort by image y (head first if y_head < y_foot)
        order = np.argsort(spline[:, 1])[::-1]  # descending y → foot→head
        pts = spline[order]
        xs_3d = np.empty(len(pts))
        zs_3d = np.empty(len(pts))
        for i, p in enumerate(pts):
            xs_3d[i], zs_3d[i] = _img_to_3d(p[0], p[1])
        return xs_3d, zs_3d

    luff_x3, luff_z3 = _edge_3d(luff_depth.spline)
    leech_x3, leech_z3 = _edge_3d(leech_depth.spline)

    # Sanity clamp — force the highest point to z = luff_m (head) and the
    # lowest luff point to z = 0 (tack). Scale Z accordingly.
    luff_z3 = np.clip(luff_z3, 0, luff_m)
    leech_z3 = np.clip(leech_z3, 0, luff_m)

    def _edge_at_z(xs: np.ndarray, zs: np.ndarray, z_target: float) -> float:
        order = np.argsort(zs)
        zs_s = zs[order]; xs_s = xs[order]
        if z_target <= zs_s[0]:
            return float(xs_s[0])
        if z_target >= zs_s[-1]:
            return float(xs_s[-1])
        return float(np.interp(z_target, zs_s, xs_s))

    # --- 2. Per-stripe v position and camber profile c(u) -----------------
    n_u = 60
    u_grid = np.linspace(0.0, 1.0, n_u)

    stripe_vs: List[float] = []
    stripe_cambers_pct: List[np.ndarray] = []
    for s in refined:
        y_mid = 0.5 * (float(s.luff_endpoint[1]) + float(s.leech_endpoint[1]))
        v_stripe = float(np.clip((y_foot_img - y_mid) / span_y_px, 0.0, 1.0))
        camber_u_pct = np.interp(u_grid, s.chord_t, s.camber_profile_pct)
        camber_u_pct[0] = 0.0
        camber_u_pct[-1] = 0.0
        stripe_vs.append(v_stripe)
        stripe_cambers_pct.append(camber_u_pct)

    order = np.argsort(stripe_vs)
    vs_arr = np.array([stripe_vs[i] for i in order])
    cambers_arr = np.stack([stripe_cambers_pct[i] for i in order], axis=0)

    def _camber_pct_at_v(v: float) -> np.ndarray:
        if v >= 1.0 - 1e-6 or len(vs_arr) == 0:
            return np.zeros(n_u)
        if v <= vs_arr[0]:
            return cambers_arr[0] * (v / max(vs_arr[0], 1e-3))
        if v >= vs_arr[-1]:
            return cambers_arr[-1] * ((1.0 - v) / max(1.0 - vs_arr[-1], 1e-3))
        idx = int(np.searchsorted(vs_arr, v))
        idx = max(1, min(idx, len(vs_arr) - 1))
        v_lo, v_hi = vs_arr[idx - 1], vs_arr[idx]
        t = (v - v_lo) / max(v_hi - v_lo, 1e-6)
        return (1 - t) * cambers_arr[idx - 1] + t * cambers_arr[idx]

    # --- 3. Build the sail surface using the detected edges --------------
    n_v = 90
    v_grid = np.linspace(0.0, 1.0, n_v)
    X = np.zeros((n_v, n_u))
    Y = np.zeros((n_v, n_u))
    Z = np.zeros((n_v, n_u))
    for i, v in enumerate(v_grid):
        z_3d = v * luff_m
        x_luff = _edge_at_z(luff_x3, luff_z3, z_3d)
        x_leech = _edge_at_z(leech_x3, leech_z3, z_3d)
        chord_m = x_leech - x_luff
        camber_pct = _camber_pct_at_v(v)
        camber_m = camber_pct / 100.0 * abs(chord_m)
        for j, u in enumerate(u_grid):
            X[i, j] = x_luff + u * chord_m
            Y[i, j] = camber_m[j]
            Z[i, j] = z_3d

    # --- 4. Build plotly traces ------------------------------------------
    traces = []

    traces.append(go.Surface(
        x=X, y=Y, z=Z,
        colorscale=[[0.0, "#1a1d24"], [1.0, "#3a414e"]],
        showscale=False, opacity=0.9,
        contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
        hoverinfo="skip", name="sail surface",
        lighting=dict(ambient=0.55, diffuse=0.75, roughness=0.9,
                      specular=0.2, fresnel=0.15),
    ))

    # Stripes as green 3D lines (camber visible)
    for pos, s in enumerate(refined):
        v_s = stripe_vs[pos]
        z_3d = v_s * luff_m
        x_luff = _edge_at_z(luff_x3, luff_z3, z_3d)
        x_leech = _edge_at_z(leech_x3, leech_z3, z_3d)
        chord_m = x_leech - x_luff
        camber_m = stripe_cambers_pct[pos] / 100.0 * abs(chord_m)
        xs = x_luff + u_grid * chord_m
        ys = camber_m
        zs = np.full(n_u, z_3d)
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="#3dff6e", width=6),
            name=f"stripe {s.index + 1}",
            hovertemplate=(
                f"stripe {s.index+1}<br>"
                f"height: {z_3d:.2f} m<br>"
                f"chord: {abs(chord_m):.2f} m<br>"
                f"camber: {s.aero.camber_depth_pct:.1f}%<br>"
                f"draft: {s.aero.draft_position_pct:.0f}%<br>"
                f"twist: {s.aero.twist_deg:.1f}°"
                "<extra></extra>"
            ),
        ))

    # Detected luff and leech edges as 3D lines (in y=0 plane)
    traces.append(go.Scatter3d(
        x=luff_x3, y=np.zeros_like(luff_x3), z=luff_z3,
        mode="lines", line=dict(color="#FF3B5C", width=7), name="luff (detected)",
    ))
    traces.append(go.Scatter3d(
        x=leech_x3, y=np.zeros_like(leech_x3), z=leech_z3,
        mode="lines", line=dict(color="#4DA6FF", width=7), name="leech (detected)",
    ))

    # Foot — straight line between tack (0, 0, 0) and clew (foot_m, 0, 0)
    traces.append(go.Scatter3d(
        x=[0, foot_m], y=[0, 0], z=[0, 0],
        mode="lines", line=dict(color="#FFD400", width=5, dash="dot"), name="foot",
    ))

    # Mast stub — vertical, slightly taller than head
    traces.append(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, luff_m * 1.08],
        mode="lines", line=dict(color="#999999", width=4, dash="dash"), name="mast",
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="chord (m)",
            yaxis_title="camber / depth (m)",
            zaxis_title="sail height (m)",
            aspectmode="data",  # keep real proportions
            camera=dict(eye=dict(x=1.6, y=1.4, z=0.8)),
            bgcolor="#0e0f13",
            xaxis=dict(gridcolor="#2a2d36", color="#cfd3dc"),
            yaxis=dict(gridcolor="#2a2d36", color="#cfd3dc"),
            zaxis=dict(gridcolor="#2a2d36", color="#cfd3dc", range=[0, luff_m * 1.1]),
        ),
        paper_bgcolor="#0e0f13",
        plot_bgcolor="#0e0f13",
        font=dict(color="#cfd3dc"),
        margin=dict(l=10, r=10, t=20, b=10),
        height=700,
        title=dict(
            text=f"{sail_type} — luff {luff_m:.2f} m · foot {foot_m:.2f} m",
            font=dict(size=12, color="#cfd3dc"), y=0.98,
        ),
        legend=dict(bgcolor="#14151a", bordercolor="#2a2d36"),
    )
    return offline.plot(
        fig, include_plotlyjs="cdn", output_type="div", auto_open=False
    )


def build_3d_plotly_html_v2(
    refined: List[RefinedStripe],
    luff_depth,
    leech_depth,
    head_point: np.ndarray,
    tack_point: np.ndarray,
    clew_point: np.ndarray,
    luff_length_m: Optional[float] = None,
    foot_length_m: Optional[float] = None,
    sail_type: str = "main",
    camber_exaggeration: float = 3.0,
) -> str:
    """Cleaner 3D reconstruction — triangular outline, sail-like shading.

    Differences from build_3d_plotly_html:
      * Uses the DETECTED luff & leech polyline shapes in the xz plane.
      * Camber applied in +y with a smooth "head taper" factor so the
        top of the sail tucks to a point rather than retaining full
        camber all the way up.
      * ``camber_exaggeration`` scales the depth for visualisation —
        real sail camber is only 5–15 % of chord so at true scale it's
        invisible from a distance. 3× is the default; the hover tips
        still report true values.
      * Better lighting, a chord-plane ghost for reference, sail-fabric
        colour palette (warm white), and a 3/4-view default camera so
        the bulge is actually visible.
    """
    try:
        import plotly.graph_objects as go
        import plotly.offline as offline
    except Exception:
        return "<div style='color:#f66'>plotly not installed</div>"

    if not refined or luff_depth is None or leech_depth is None:
        return "<div style='color:#aaa'>Need luff, leech, and ≥ 1 stripe for 3D.</div>"

    luff_m = float(luff_length_m) if (luff_length_m and luff_length_m > 0) else 10.0
    foot_m = float(foot_length_m) if (foot_length_m and foot_length_m > 0) else 4.0

    y_head_img = float(head_point[1])
    y_foot_img = max(float(tack_point[1]), float(clew_point[1]))
    span_y_px = max(y_foot_img - y_head_img, 1.0)
    z_scale = luff_m / span_y_px
    x_tack_img = float(tack_point[0])
    x_clew_img = float(clew_point[0])
    foot_px = max(abs(x_clew_img - x_tack_img), 1.0)
    x_scale = foot_m / foot_px
    x_sign = 1.0 if x_clew_img >= x_tack_img else -1.0

    def _img_to_3d(x_img: float, y_img: float) -> Tuple[float, float]:
        x_3d = (x_img - x_tack_img) * x_scale * x_sign
        z_3d = (y_foot_img - y_img) * z_scale
        return x_3d, z_3d

    # ---- Detected edges → (x, z) in metres -----------------------------
    def _edge_xz(spline: np.ndarray):
        order = np.argsort(spline[:, 1])[::-1]
        pts = spline[order]
        xs_3d = np.empty(len(pts)); zs_3d = np.empty(len(pts))
        for i, p in enumerate(pts):
            xs_3d[i], zs_3d[i] = _img_to_3d(p[0], p[1])
        zs_3d = np.clip(zs_3d, 0, luff_m)
        return xs_3d, zs_3d

    luff_x, luff_z = _edge_xz(luff_depth.spline)
    leech_x, leech_z = _edge_xz(leech_depth.spline)

    # Force edge endpoints at the corners
    luff_z[0], luff_x[0] = 0.0, luff_x[np.argmin(luff_z)]
    leech_z[0], leech_x[0] = 0.0, leech_x[np.argmin(leech_z)]

    def _x_at_z(xs, zs, z_target):
        order = np.argsort(zs)
        zs_s, xs_s = zs[order], xs[order]
        if z_target <= zs_s[0]:
            return float(xs_s[0])
        if z_target >= zs_s[-1]:
            return float(xs_s[-1])
        return float(np.interp(z_target, zs_s, xs_s))

    # ---- Stripe heights (v in [0,1]), cambers, and twist ----------------
    n_u = 80
    u_grid = np.linspace(0.0, 1.0, n_u)
    stripe_vs: List[float] = []
    cambers_pct: List[np.ndarray] = []
    stripe_twists: List[float] = []  # degrees vs foot chord
    for s in refined:
        y_mid = 0.5 * (float(s.luff_endpoint[1]) + float(s.leech_endpoint[1]))
        v = float(np.clip((y_foot_img - y_mid) / span_y_px, 0.0, 1.0))
        c = np.interp(u_grid, s.chord_t, s.camber_profile_pct).astype(np.float32)
        c[0] = 0.0; c[-1] = 0.0
        stripe_vs.append(v)
        cambers_pct.append(c)
        stripe_twists.append(float(s.aero.twist_deg))

    order = np.argsort(stripe_vs)
    vs_arr = np.array([stripe_vs[i] for i in order])
    cambers_arr = np.stack([cambers_pct[i] for i in order], axis=0)
    twists_arr = np.array([stripe_twists[i] for i in order])

    def _camber_at_v(v: float) -> np.ndarray:
        if len(vs_arr) == 0:
            return np.zeros(n_u)
        head_taper = np.clip((1.0 - v) / 0.15, 0.0, 1.0)  # tighter head taper
        foot_taper = np.clip(v / 0.05, 0.0, 1.0)
        taper = head_taper * foot_taper
        if v <= vs_arr[0]:
            return cambers_arr[0] * taper * (v / max(vs_arr[0], 1e-3))
        if v >= vs_arr[-1]:
            return cambers_arr[-1] * taper
        idx = int(np.searchsorted(vs_arr, v))
        idx = max(1, min(idx, len(vs_arr) - 1))
        v_lo, v_hi = vs_arr[idx - 1], vs_arr[idx]
        t = (v - v_lo) / max(v_hi - v_lo, 1e-6)
        return ((1 - t) * cambers_arr[idx - 1] + t * cambers_arr[idx]) * taper

    def _twist_at_v(v: float) -> float:
        """Twist (deg) relative to foot chord, interpolated over height."""
        if len(vs_arr) == 0:
            return 0.0
        if v <= vs_arr[0]:
            return float(twists_arr[0] * (v / max(vs_arr[0], 1e-3)))
        if v >= vs_arr[-1]:
            # Extrapolate last-slope for v > last stripe, so the head
            # keeps twisting open gently instead of snapping to 0.
            return float(twists_arr[-1])
        idx = int(np.searchsorted(vs_arr, v))
        idx = max(1, min(idx, len(vs_arr) - 1))
        v_lo, v_hi = vs_arr[idx - 1], vs_arr[idx]
        t = (v - v_lo) / max(v_hi - v_lo, 1e-6)
        return float((1 - t) * twists_arr[idx - 1] + t * twists_arr[idx])

    # ---- Build twisted sail surface ------------------------------------
    # At each height z we:
    #   1. get chord endpoints from detected luff/leech curves (xz plane)
    #   2. rotate the chord around the luff end by twist(v) so upper
    #      sections "open" off the wind (physical sail twist)
    #   3. apply camber perpendicular to the ROTATED chord (in 3D)
    n_v = 120
    v_grid = np.linspace(0.0, 1.0, n_v)
    X = np.zeros((n_v, n_u))
    Y = np.zeros((n_v, n_u))
    Z = np.zeros((n_v, n_u))
    cexag = float(camber_exaggeration)
    # Amplify twist visually: real twists are 5–15° and already look small
    # at sail-size; bump them 1.4× so they're actually visible.
    twist_visual = 1.4
    for i, v in enumerate(v_grid):
        z = v * luff_m
        x_luff = _x_at_z(luff_x, luff_z, z)
        x_leech = _x_at_z(leech_x, leech_z, z)
        chord_len = abs(x_leech - x_luff)
        twist_rad = np.radians(_twist_at_v(v) * twist_visual)
        # Rotated chord-end: luff stays fixed (mast-side); leech rotates
        # around the vertical through the luff.
        c_x = x_luff + np.cos(twist_rad) * (x_leech - x_luff)
        c_y = np.sin(twist_rad) * (x_leech - x_luff)
        camber_m = _camber_at_v(v) / 100.0 * chord_len * cexag
        # Perpendicular to rotated chord in the horizontal plane
        chord_vec = np.array([c_x - x_luff, c_y])
        L = np.linalg.norm(chord_vec) + 1e-9
        n_vec = np.array([-chord_vec[1], chord_vec[0]]) / L
        for j, u in enumerate(u_grid):
            cx = x_luff + u * (c_x - x_luff)
            cy = u * c_y
            X[i, j] = cx + n_vec[0] * camber_m[j]
            Y[i, j] = cy + n_vec[1] * camber_m[j]
            Z[i, j] = z

    # Gaussian-smooth the mesh along the vertical (v) direction to kill
    # stair-stepping between stripes. Smoothing along u would flatten
    # the chord, which we don't want.
    try:
        from scipy.ndimage import gaussian_filter1d
        X = gaussian_filter1d(X, sigma=2.0, axis=0, mode="nearest")
        Y = gaussian_filter1d(Y, sigma=2.0, axis=0, mode="nearest")
        # Z we keep strict (it's our height axis)
    except Exception:
        pass

    traces = []

    # Surface — dark cloth with subtle top highlight
    surface_color = (Z / max(luff_m, 1e-3))
    traces.append(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=surface_color,
        colorscale=[
            [0.0, "#161a26"],
            [0.5, "#2a3146"],
            [1.0, "#4c5874"],
        ],
        showscale=False,
        opacity=0.94,
        contours=dict(
            z=dict(show=True, usecolormap=False,
                   color="#8fa0c0",
                   start=0.0, end=luff_m,
                   size=max(0.5, luff_m / 12)),
        ),
        lighting=dict(
            ambient=0.35, diffuse=0.9, roughness=0.4,
            specular=0.6, fresnel=0.25,
        ),
        lightposition=dict(x=-200, y=-200, z=300),
        hoverinfo="skip", name="sail",
    ))

    # Stripes ON the surface (same twist + camber applied — they sit on the mesh)
    for pos, s in enumerate(refined):
        v_s = stripe_vs[pos]
        z = v_s * luff_m
        x_luff = _x_at_z(luff_x, luff_z, z)
        x_leech = _x_at_z(leech_x, leech_z, z)
        chord_len = abs(x_leech - x_luff)
        twist_rad = np.radians(float(s.aero.twist_deg) * twist_visual)
        c_x = x_luff + np.cos(twist_rad) * (x_leech - x_luff)
        c_y = np.sin(twist_rad) * (x_leech - x_luff)
        chord_vec = np.array([c_x - x_luff, c_y])
        L = np.linalg.norm(chord_vec) + 1e-9
        n_vec = np.array([-chord_vec[1], chord_vec[0]]) / L
        camber_m = cambers_pct[pos] / 100.0 * chord_len * cexag
        head_taper = np.clip((1.0 - v_s) / 0.15, 0.0, 1.0)
        xs = x_luff + u_grid * (c_x - x_luff) + n_vec[0] * camber_m * head_taper
        ys = u_grid * c_y + n_vec[1] * camber_m * head_taper
        zs = np.full(n_u, z)
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="#ff6b5a", width=5),
            name=f"stripe {s.index + 1}",
            hovertemplate=(
                f"stripe {s.index+1}<br>"
                f"height: {z:.2f} m<br>"
                f"chord: {chord_len:.2f} m<br>"
                f"camber: {s.aero.camber_depth_pct:.1f}% (×{cexag:.0f} visual)<br>"
                f"draft: {s.aero.draft_position_pct:.0f}%<br>"
                f"twist: {s.aero.twist_deg:.1f}°"
                "<extra></extra>"
            ),
        ))

    # Luff stays at y=0 (mast / forestay line — luff is the rotation axis)
    traces.append(go.Scatter3d(
        x=luff_x, y=np.zeros_like(luff_x), z=luff_z,
        mode="lines", line=dict(color="#2cc3ff", width=6),
        name="luff",
    ))

    # Leech — re-derived at each height with twist applied
    leech_xs = np.zeros(n_v); leech_ys = np.zeros(n_v); leech_zs = np.zeros(n_v)
    for i, v in enumerate(v_grid):
        z = v * luff_m
        x_l = _x_at_z(luff_x, luff_z, z)
        x_le = _x_at_z(leech_x, leech_z, z)
        t_rad = np.radians(_twist_at_v(v) * twist_visual)
        leech_xs[i] = x_l + np.cos(t_rad) * (x_le - x_l)
        leech_ys[i] = np.sin(t_rad) * (x_le - x_l)
        leech_zs[i] = z
    traces.append(go.Scatter3d(
        x=leech_xs, y=leech_ys, z=leech_zs,
        mode="lines", line=dict(color="#ff915a", width=6),
        name="leech (twisted)",
    ))

    # (Foot line intentionally omitted — the sail opens at the foot,
    # a closed orange line there made the 3D look like a closed
    # triangular kite rather than a sail.)

    # Head vertex
    head_x, head_z = _img_to_3d(float(head_point[0]), float(head_point[1]))
    traces.append(go.Scatter3d(
        x=[head_x], y=[0], z=[head_z],
        mode="markers", marker=dict(size=5, color="#ffd400"),
        name="head", showlegend=False, hoverinfo="skip",
    ))

    # Mast or forestay marker
    if sail_type.lower() == "main":
        traces.append(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, luff_m * 1.04],
            mode="lines", line=dict(color="#4b5563", width=6),
            name="mast",
        ))
    else:
        # Forestay = the detected luff curve itself, already drawn. Add
        # a label line from tack to head.
        tack_x3, _ = _img_to_3d(float(tack_point[0]), float(tack_point[1]))
        traces.append(go.Scatter3d(
            x=[tack_x3, head_x], y=[0, 0], z=[0, head_z],
            mode="lines", line=dict(color="#4b5563", width=3, dash="dash"),
            name="forestay axis",
        ))

    # ---- Scene & layout -------------------------------------------------
    x_min = min(float(X.min()), 0.0) - 0.3
    x_max = float(X.max()) + 0.3
    y_abs = max(float(np.abs(Y).max()), 0.4)
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="chord X (m)",
            yaxis_title="depth Y (m)",
            zaxis_title="height Z (m)",
            # Give Y ~40 % of cube (so bulge is obvious) and keep a fat,
            # non-pinchy look — no more tall-pyramid problem.
            aspectmode="manual",
            aspectratio=dict(x=1.3, y=0.85, z=1.4),
            camera=dict(
                eye=dict(x=1.5, y=1.9, z=0.4),   # looking slightly up from leeward
                up=dict(x=0, y=0, z=1),
                center=dict(x=0.2, y=0.2, z=0.45),
            ),
            bgcolor="#0b1020",
            xaxis=dict(gridcolor="#1b2640", color="#cfd3dc",
                       range=[x_min, x_max], backgroundcolor="#0b1020"),
            yaxis=dict(gridcolor="#1b2640", color="#cfd3dc",
                       range=[-y_abs * 1.4, y_abs * 1.4],
                       backgroundcolor="#0b1020"),
            zaxis=dict(gridcolor="#1b2640", color="#cfd3dc",
                       range=[0, luff_m * 1.08], backgroundcolor="#0b1020"),
        ),
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        font=dict(color="#cfd3dc"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=720,
        title=dict(
            text=(
                f"{sail_type} — luff {luff_m:.2f} m · foot {foot_m:.2f} m "
                f"· {len(refined)} stripes · camber ×{camber_exaggeration:.0f} "
                f"for visibility"
            ),
            font=dict(size=13, color="#e4eaf2"), y=0.98,
        ),
        legend=dict(bgcolor="#0f162b", bordercolor="#1b2640",
                    font=dict(color="#cfd3dc")),
    )
    return offline.plot(
        fig, include_plotlyjs="cdn", output_type="div", auto_open=False
    )


def build_3d_plotly_detected_only(
    refined: List[RefinedStripe],
    luff_depth,
    leech_depth,
    sail_boundary,           # SailBoundary with luff/leech polylines
    head_point: np.ndarray,
    tack_point: np.ndarray,
    clew_point: np.ndarray,
    luff_length_m: Optional[float] = None,
    foot_length_m: Optional[float] = None,
    sail_type: str = "main",
    camber_exaggeration: float = 2.5,
) -> str:
    """3D sail using ONLY the detected shape — no synthetic flat-foot
    extrapolation. The rendered surface is bounded:

      * Bottom: the detected foot / bottom-most stripe height (whichever
        is lower). The foot line follows the ACTUAL mask-contour curve
        between tack and clew, so it's not flat.
      * Top: head (tip of luff polyline).

    Between detected stripes, camber is linearly interpolated. Above the
    top-most stripe, camber decays smoothly to zero at the head. Below
    the bottom-most stripe, camber decays to 0 at the foot but the
    outline CURVATURE is preserved (foot hollow shown).

    Returns a plotly HTML div.
    """
    try:
        import plotly.graph_objects as go
        import plotly.offline as offline
    except Exception:
        return "<div style='color:#f66'>plotly not installed</div>"

    if not refined or luff_depth is None or leech_depth is None:
        return "<div style='color:#aaa'>Need stripes + edges to render.</div>"

    luff_m = float(luff_length_m) if (luff_length_m and luff_length_m > 0) else 10.0
    foot_m = float(foot_length_m) if (foot_length_m and foot_length_m > 0) else 4.0

    y_head_img = float(head_point[1])
    y_foot_img = max(float(tack_point[1]), float(clew_point[1]))
    span_y_px = max(y_foot_img - y_head_img, 1.0)
    z_scale = luff_m / span_y_px
    x_tack_img = float(tack_point[0])
    x_clew_img = float(clew_point[0])
    foot_px = max(abs(x_clew_img - x_tack_img), 1.0)
    x_scale = foot_m / foot_px
    x_sign = 1.0 if x_clew_img >= x_tack_img else -1.0

    def _img_to_3d(x_img: float, y_img: float) -> Tuple[float, float]:
        return ((x_img - x_tack_img) * x_scale * x_sign,
                (y_foot_img - y_img) * z_scale)

    def _edge_xz(spline: np.ndarray):
        order = np.argsort(spline[:, 1])[::-1]
        pts = spline[order]
        xs = np.empty(len(pts)); zs = np.empty(len(pts))
        for i, p in enumerate(pts):
            xs[i], zs[i] = _img_to_3d(p[0], p[1])
        return xs, zs

    luff_x, luff_z = _edge_xz(luff_depth.spline)
    leech_x, leech_z = _edge_xz(leech_depth.spline)

    def _x_at_z(xs, zs, z_t):
        order = np.argsort(zs)
        zs_s, xs_s = zs[order], xs[order]
        if z_t <= zs_s[0]:  return float(xs_s[0])
        if z_t >= zs_s[-1]: return float(xs_s[-1])
        return float(np.interp(z_t, zs_s, xs_s))

    # Detected foot curve: if the sail_boundary has a foot polyline use it,
    # else approximate by walking the detected contour from tack to clew
    # at the lowest z.
    foot_x = foot_z = None
    if sail_boundary is not None and getattr(sail_boundary, "foot_polyline", None) is not None and len(sail_boundary.foot_polyline) >= 3:
        fx = np.empty(len(sail_boundary.foot_polyline))
        fz = np.empty(len(sail_boundary.foot_polyline))
        for i, p in enumerate(sail_boundary.foot_polyline):
            fx[i], fz[i] = _img_to_3d(float(p[0]), float(p[1]))
        foot_x, foot_z = fx, fz
    else:
        # Fall back: straight line tack → clew in 3D (still not flat at
        # z=0 because clew and tack might be at slightly different z).
        tack_3d = _img_to_3d(x_tack_img, float(tack_point[1]))
        clew_3d = _img_to_3d(x_clew_img, float(clew_point[1]))
        foot_x = np.array([tack_3d[0], clew_3d[0]])
        foot_z = np.array([tack_3d[1], clew_3d[1]])

    # --- Stripe heights, cambers, twists ---
    n_u = 60
    u_grid = np.linspace(0.0, 1.0, n_u)
    stripe_vs: List[float] = []
    cambers_pct: List[np.ndarray] = []
    stripe_twists: List[float] = []
    for s in refined:
        y_mid = 0.5 * (float(s.luff_endpoint[1]) + float(s.leech_endpoint[1]))
        v = float(np.clip((y_foot_img - y_mid) / span_y_px, 0.0, 1.0))
        c = np.interp(u_grid, s.chord_t, s.camber_profile_pct).astype(np.float32)
        c[0] = 0.0; c[-1] = 0.0
        stripe_vs.append(v)
        cambers_pct.append(c)
        stripe_twists.append(float(s.aero.twist_deg))

    order = np.argsort(stripe_vs)
    vs_arr = np.array([stripe_vs[i] for i in order])
    cambers_arr = np.stack([cambers_pct[i] for i in order], axis=0)
    twists_arr = np.array([stripe_twists[i] for i in order])

    # Detected span: from foot z_min to head z_max, BUT only draw surface
    # between (foot_avg_z) and (head_z) — this is "detected shape only".
    foot_avg_z = float(np.mean(foot_z))
    head_z_3d = _img_to_3d(float(head_point[0]), float(head_point[1]))[1]
    v_min = max(0.0, foot_avg_z / max(luff_m, 1e-3))
    v_max = min(1.0, head_z_3d / max(luff_m, 1e-3))

    def _camber_at_v(v: float) -> np.ndarray:
        if len(vs_arr) == 0:
            return np.zeros(n_u)
        # Below bottom-most stripe: decay linearly toward 0 at foot
        # (v = v_min). Above top-most: decay to 0 at head (v = v_max).
        if v <= vs_arr[0]:
            frac = (v - v_min) / max(vs_arr[0] - v_min, 1e-3)
            return cambers_arr[0] * max(frac, 0.0)
        if v >= vs_arr[-1]:
            frac = (v_max - v) / max(v_max - vs_arr[-1], 1e-3)
            return cambers_arr[-1] * max(frac, 0.0)
        idx = int(np.searchsorted(vs_arr, v))
        idx = max(1, min(idx, len(vs_arr) - 1))
        v_lo, v_hi = vs_arr[idx - 1], vs_arr[idx]
        t = (v - v_lo) / max(v_hi - v_lo, 1e-6)
        return (1 - t) * cambers_arr[idx - 1] + t * cambers_arr[idx]

    def _twist_at_v(v: float) -> float:
        if len(vs_arr) == 0:
            return 0.0
        if v <= vs_arr[0]:  return float(twists_arr[0])
        if v >= vs_arr[-1]: return float(twists_arr[-1])
        idx = int(np.searchsorted(vs_arr, v))
        idx = max(1, min(idx, len(vs_arr) - 1))
        v_lo, v_hi = vs_arr[idx - 1], vs_arr[idx]
        t = (v - v_lo) / max(v_hi - v_lo, 1e-6)
        return float((1 - t) * twists_arr[idx - 1] + t * twists_arr[idx])

    # --- Surface mesh (v_min → v_max ONLY) ---
    n_v = 70
    v_grid = np.linspace(v_min, v_max, n_v)
    X = np.zeros((n_v, n_u)); Y = np.zeros((n_v, n_u)); Z = np.zeros((n_v, n_u))
    twist_visual = 1.4
    cexag = float(camber_exaggeration)
    for i, v in enumerate(v_grid):
        z = v * luff_m
        x_l = _x_at_z(luff_x, luff_z, z)
        x_le = _x_at_z(leech_x, leech_z, z)
        chord_len = abs(x_le - x_l)
        tw_rad = np.radians(_twist_at_v(v) * twist_visual)
        c_x = x_l + np.cos(tw_rad) * (x_le - x_l)
        c_y = np.sin(tw_rad) * (x_le - x_l)
        camber_m = _camber_at_v(v) / 100.0 * chord_len * cexag
        chord_vec = np.array([c_x - x_l, c_y])
        L = np.linalg.norm(chord_vec) + 1e-9
        n_vec = np.array([-chord_vec[1], chord_vec[0]]) / L
        for j, u in enumerate(u_grid):
            cx = x_l + u * (c_x - x_l); cy = u * c_y
            X[i, j] = cx + n_vec[0] * camber_m[j]
            Y[i, j] = cy + n_vec[1] * camber_m[j]
            Z[i, j] = z

    # Gaussian-smooth along v for a cleaner surface
    try:
        from scipy.ndimage import gaussian_filter1d
        X = gaussian_filter1d(X, sigma=1.5, axis=0, mode="nearest")
        Y = gaussian_filter1d(Y, sigma=1.5, axis=0, mode="nearest")
    except Exception:
        pass

    traces = []
    traces.append(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=(Z / max(luff_m, 1e-3)),
        colorscale=[
            [0.0, "#141a26"],
            [0.5, "#293246"],
            [1.0, "#4a5774"],
        ],
        showscale=False, opacity=0.96,
        lighting=dict(ambient=0.4, diffuse=0.9, roughness=0.35,
                      specular=0.65, fresnel=0.3),
        lightposition=dict(x=-150, y=-250, z=300),
        hoverinfo="skip", name="sail",
    ))

    # Detected stripes lying on the mesh
    for pos, s in enumerate(refined):
        v_s = stripe_vs[pos]
        z = v_s * luff_m
        x_l = _x_at_z(luff_x, luff_z, z)
        x_le = _x_at_z(leech_x, leech_z, z)
        chord_len = abs(x_le - x_l)
        tw_rad = np.radians(float(s.aero.twist_deg) * twist_visual)
        c_x = x_l + np.cos(tw_rad) * (x_le - x_l)
        c_y = np.sin(tw_rad) * (x_le - x_l)
        chord_vec = np.array([c_x - x_l, c_y])
        L = np.linalg.norm(chord_vec) + 1e-9
        n_vec = np.array([-chord_vec[1], chord_vec[0]]) / L
        camber_m = cambers_pct[pos] / 100.0 * chord_len * cexag
        xs = x_l + u_grid * (c_x - x_l) + n_vec[0] * camber_m
        ys = u_grid * c_y + n_vec[1] * camber_m
        zs = np.full(n_u, z)
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="#ff6b5a", width=5),
            name=f"stripe {s.index + 1}",
            hovertemplate=(
                f"stripe {s.index+1}<br>"
                f"height: {z:.2f} m<br>"
                f"camber: {s.aero.camber_depth_pct:.1f}% (×{cexag:.1f} visual)<br>"
                f"twist: {s.aero.twist_deg:.1f}°"
                "<extra></extra>"
            ),
        ))

    # Luff edge (blue) in 3D — actual detected shape
    traces.append(go.Scatter3d(
        x=luff_x, y=np.zeros_like(luff_x), z=np.clip(luff_z, 0, luff_m),
        mode="lines", line=dict(color="#2cc3ff", width=6), name="luff",
    ))
    # Leech (orange) — twist applied per-height
    leech_tw_x = np.zeros(n_v); leech_tw_y = np.zeros(n_v); leech_tw_z = np.zeros(n_v)
    for i, v in enumerate(v_grid):
        z = v * luff_m
        x_l = _x_at_z(luff_x, luff_z, z)
        x_le = _x_at_z(leech_x, leech_z, z)
        t_rad = np.radians(_twist_at_v(v) * twist_visual)
        leech_tw_x[i] = x_l + np.cos(t_rad) * (x_le - x_l)
        leech_tw_y[i] = np.sin(t_rad) * (x_le - x_l)
        leech_tw_z[i] = z
    traces.append(go.Scatter3d(
        x=leech_tw_x, y=leech_tw_y, z=leech_tw_z,
        mode="lines", line=dict(color="#ff915a", width=6), name="leech",
    ))
    # Foot curve — detected, not straight
    traces.append(go.Scatter3d(
        x=foot_x, y=np.zeros_like(foot_x), z=np.clip(foot_z, 0, luff_m),
        mode="lines", line=dict(color="#f5a623", width=4), name="foot",
    ))

    # Head vertex
    hx, hz = _img_to_3d(float(head_point[0]), float(head_point[1]))
    traces.append(go.Scatter3d(
        x=[hx], y=[0], z=[hz],
        mode="markers", marker=dict(size=5, color="#ffd400"),
        name="head", showlegend=False, hoverinfo="skip",
    ))

    fig = go.Figure(data=traces)
    x_min = min(float(X.min()), 0.0, float(foot_x.min())) - 0.3
    x_max = max(float(X.max()), float(foot_x.max())) + 0.3
    y_abs = max(float(np.abs(Y).max()), float(np.abs(leech_tw_y).max()), 0.4)
    fig.update_layout(
        scene=dict(
            xaxis_title="chord X (m)",
            yaxis_title="camber Y (m)",
            zaxis_title="height Z (m)",
            aspectmode="manual",
            aspectratio=dict(x=1.3, y=0.8, z=1.6),
            camera=dict(
                eye=dict(x=1.5, y=1.8, z=0.5),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0.2, y=0.2, z=0.4),
            ),
            bgcolor="#041526",
            xaxis=dict(gridcolor="#1e3a5f", color="#e8eef7",
                        range=[x_min, x_max], backgroundcolor="#041526"),
            yaxis=dict(gridcolor="#1e3a5f", color="#e8eef7",
                        range=[-y_abs * 1.4, y_abs * 1.4], backgroundcolor="#041526"),
            zaxis=dict(gridcolor="#1e3a5f", color="#e8eef7",
                        range=[max(0, foot_avg_z - 0.4), luff_m * 1.05],
                        backgroundcolor="#041526"),
        ),
        paper_bgcolor="#041526",
        plot_bgcolor="#041526",
        font=dict(color="#e8eef7"),
        margin=dict(l=0, r=0, t=20, b=0),
        height=600,
        title=dict(
            text=(f"3D sail scan — {sail_type}  ·  "
                  f"luff {luff_m:.1f} m / foot {foot_m:.1f} m  ·  "
                  f"camber ×{cexag:.1f}"),
            font=dict(size=12, color="#e8eef7"), y=0.98,
        ),
        legend=dict(bgcolor="#0a1f35", bordercolor="#1e3a5f",
                     font=dict(color="#e8eef7")),
    )
    return offline.plot(
        fig, include_plotlyjs="cdn", output_type="div", auto_open=False,
    )
