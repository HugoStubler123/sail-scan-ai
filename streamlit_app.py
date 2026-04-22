"""SailShape — Streamlit UI for the v7 sail-analysis pipeline.

Marine-palette dark theme. Upload single or batch, or snap with your
device camera. Per-photo PDF one-pager.

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import hashlib
import io
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from PIL import ExifTags, Image

# ------------------------------------------------------------------
# Repo imports + module hot-reload guard
# ------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _reload_stale_src_modules() -> None:
    """Force-reimport `src.*` modules when their source files change on
    disk. Streamlit only re-executes the top-level script, not nested
    imports — without this, edits to the pipeline would be ignored until
    the server is restarted.
    """
    for mod_name in list(sys.modules):
        if not mod_name.startswith("src.") and mod_name != "src":
            continue
        mod = sys.modules.get(mod_name)
        mod_file = getattr(mod, "__file__", None)
        if not mod_file:
            continue
        try:
            disk_mtime = Path(mod_file).stat().st_mtime
        except OSError:
            continue
        cached_mtime = getattr(mod, "__sail_mtime__", None)
        if cached_mtime is None or disk_mtime > cached_mtime:
            for stale in [m for m in sys.modules if m.startswith("src.") or m == "src"]:
                sys.modules.pop(stale, None)
            return


_reload_stale_src_modules()

from src.sail_pipeline import analyze_sail
from src.report_pdf import build_pdf
from src.yachts import YACHT_RIGS, dimensions_for, analysis_quality_score
from src.sail_analysis import build_3d_plotly_detected_only

for _m in ("src.sail_pipeline", "src.report_pdf", "src.pipeline_v7",
            "src.flexible_fit", "src.polygon_fusion", "src.endpoints",
            "src.sail_analysis", "src.trim_analyst", "src.yachts"):
    _obj = sys.modules.get(_m)
    _f = getattr(_obj, "__file__", None) if _obj else None
    if _f:
        try:
            _obj.__sail_mtime__ = Path(_f).stat().st_mtime  # type: ignore[attr-defined]
        except OSError:
            pass


# ------------------------------------------------------------------
# Theme / CSS
# ------------------------------------------------------------------

MARINE = {
    "deep": "#041526",      # background
    "hull": "#0a1f35",      # panels
    "steel": "#0f2d4a",     # card bg
    "stroke": "#1e3a5f",    # borders / grid
    "sail":   "#f2ead3",    # sail-white accent
    "cyan":   "#2cc3ff",    # luff colour / secondary
    "orange": "#ff6b35",    # signal accent / CTA
    "green":  "#2fdd92",    # success
    "amber":  "#ffb020",    # warning
    "fg":     "#e8eef7",    # primary text
    "fg2":    "#9aadc3",    # muted
}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root, [data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(ellipse at top left, {MARINE['steel']}55, transparent 50%),
        radial-gradient(ellipse at bottom right, {MARINE['hull']}88, transparent 60%),
        {MARINE['deep']} !important;
    color: {MARINE['fg']} !important;
    font-family: 'Barlow', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stSidebar"] {{
    background: {MARINE['hull']} !important;
    border-right: 1px solid {MARINE['stroke']} !important;
}}
[data-testid="stSidebar"] > div {{ padding-top: 1.4rem; }}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ color: {MARINE['fg']}; }}

/* Hero header bar */
.hero-bar {{
    background: linear-gradient(135deg, {MARINE['hull']} 0%, {MARINE['steel']} 100%);
    border: 1px solid {MARINE['stroke']};
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 18px;
    box-shadow: 0 8px 40px -12px rgba(0,0,0,0.55);
}}
.hero-logo {{
    width: 52px; height: 52px;
    border-radius: 14px;
    background: linear-gradient(135deg, {MARINE['cyan']}, {MARINE['orange']});
    display: flex; align-items: center; justify-content: center;
    font-size: 28px;
    box-shadow: 0 0 30px -6px {MARINE['cyan']}66;
}}
.hero-title {{
    font-size: 22px; font-weight: 700; color: {MARINE['fg']};
    letter-spacing: -0.5px; line-height: 1.1;
}}
.hero-subtitle {{
    font-size: 13px; color: {MARINE['fg2']}; margin-top: 4px;
}}

/* Sidebar form groups */
.form-group-label {{
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 600;
    color: {MARINE['fg2']};
    margin: 14px 0 4px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid {MARINE['stroke']};
}}

/* Metric tile */
.metric-tile {{
    background: {MARINE['steel']};
    border: 1px solid {MARINE['stroke']};
    border-radius: 12px;
    padding: 12px 14px;
}}
.metric-tile .m-label {{
    font-size: 10px; letter-spacing: 1.4px;
    text-transform: uppercase; color: {MARINE['fg2']};
}}
.metric-tile .m-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px; font-weight: 600; color: {MARINE['fg']};
    margin-top: 2px;
}}
.metric-tile .m-unit {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: {MARINE['fg2']}; margin-left: 4px;
}}

/* Stripe-row cards */
.stripe-card {{
    background: {MARINE['steel']};
    border: 1px solid {MARINE['stroke']};
    border-radius: 12px;
    padding: 10px 12px;
    display: grid;
    grid-template-columns: 28px 1fr 1fr 1fr 1fr 1fr;
    gap: 10px;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    margin-bottom: 6px;
}}
.stripe-card .stripe-num {{
    width: 28px; height: 28px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; color: white; font-size: 12px;
    font-family: 'Barlow', sans-serif;
}}
.stripe-card .s-lbl {{ font-size: 10px; color: {MARINE['fg2']}; text-transform: uppercase; letter-spacing: 1px; }}
.stripe-card .s-val {{ font-size: 14px; color: {MARINE['fg']}; font-weight: 600; }}

/* Primary button */
.stButton > button,
button[kind="primary"] {{
    background: linear-gradient(135deg, {MARINE['orange']} 0%, #e05028 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 6px 24px -6px {MARINE['orange']}99 !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 10px 30px -8px {MARINE['orange']} !important;
}}
.stDownloadButton > button {{
    background: {MARINE['steel']} !important;
    color: {MARINE['fg']} !important;
    border: 1px solid {MARINE['cyan']} !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
}}
.stDownloadButton > button:hover {{
    background: {MARINE['cyan']}15 !important;
    border-color: {MARINE['cyan']} !important;
}}

/* Inputs */
.stSelectbox [data-baseweb="select"] > div,
.stNumberInput input, .stTextInput input, .stTextArea textarea {{
    background: {MARINE['deep']} !important;
    border: 1px solid {MARINE['stroke']} !important;
    color: {MARINE['fg']} !important;
    border-radius: 8px !important;
}}
.stSelectbox [data-baseweb="select"] > div:focus-within,
.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {{
    border-color: {MARINE['cyan']} !important;
    box-shadow: 0 0 0 2px {MARINE['cyan']}22 !important;
}}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {{
    background: {MARINE['hull']} !important;
    border: 2px dashed {MARINE['stroke']} !important;
    border-radius: 14px !important;
    padding: 22px !important;
    transition: all 0.2s ease;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: {MARINE['cyan']} !important;
    background: {MARINE['steel']} !important;
}}

/* Divider */
hr {{ border-color: {MARINE['stroke']} !important; opacity: 0.6; }}

/* Dataframe */
.stDataFrame {{ background: {MARINE['steel']} !important; border-radius: 10px; }}

/* Expander */
[data-testid="stExpander"] {{
    background: {MARINE['steel']} !important;
    border: 1px solid {MARINE['stroke']} !important;
    border-radius: 12px !important;
}}

/* Tabs */
button[role="tab"] {{ color: {MARINE['fg2']} !important; }}
button[role="tab"][aria-selected="true"] {{
    color: {MARINE['fg']} !important;
    border-bottom: 2px solid {MARINE['orange']} !important;
}}
</style>
"""


# ------------------------------------------------------------------
# EXIF / CSV helpers
# ------------------------------------------------------------------

CSV_FIELDS = {
    "TWS": "TWS - True Wind Speed",
    "TWA": "TWA - True Wind Angle",
    "AWA": "AWA - App. Wind Angle",
    "AWS": "AWS - App. Wind Speed",
    "BSP": "BSP - Boat Speed Water",
    "Heel": "Heel",
}


def _exif_datetime(image_bytes: bytes) -> Optional[datetime]:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif = img.getexif()
        if not exif:
            return None
        tag_map = {v: k for k, v in ExifTags.TAGS.items()}
        for name in ("DateTimeOriginal", "DateTime"):
            tag_id = tag_map.get(name)
            if tag_id is None:
                continue
            raw = exif.get(tag_id)
            if raw:
                try:
                    return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
                except Exception:
                    continue
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False)
def _load_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    if "time" in df.columns:
        df["_t"] = pd.to_datetime(df["time"], errors="coerce")
    elif "servertime" in df.columns:
        df["_t"] = pd.to_datetime(df["servertime"], errors="coerce")
    else:
        df["_t"] = pd.NaT
    return df


def _match_row(df: pd.DataFrame, ts: datetime) -> Optional[pd.Series]:
    if df is None or df.empty or pd.isna(ts):
        return None
    ts_pd = pd.Timestamp(ts)
    dt = (df["_t"] - ts_pd).abs()
    if dt.isna().all():
        return None
    idx = int(dt.idxmin())
    return df.loc[idx]


def _telemetry_from_row(row: pd.Series, photo_time: Optional[datetime]) -> Dict[str, Any]:
    tel: Dict[str, Any] = {"photo_time": photo_time}
    tel["matched_time"] = row["_t"] if "_t" in row and not pd.isna(row["_t"]) else None
    for short, col in CSV_FIELDS.items():
        if col in row and pd.notna(row[col]):
            try:
                tel[short] = float(row[col])
            except Exception:
                continue
    return tel


# ------------------------------------------------------------------
# UI components
# ------------------------------------------------------------------

def _hero():
    st.markdown(
        f"""
        <div class="hero-bar">
          <div class="hero-logo">⛵</div>
          <div>
            <div class="hero-title">SailShape · v7</div>
            <div class="hero-subtitle">Photo-to-shape analyser for performance sailmakers &amp; trimmers</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _group_label(text: str):
    st.markdown(f'<div class="form-group-label">{text}</div>', unsafe_allow_html=True)


def _metric_tile(label: str, value: str, unit: str = ""):
    st.markdown(
        f"""
        <div class="metric-tile">
          <div class="m-label">{label}</div>
          <div class="m-value">{value}<span class="m-unit">{unit}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hex_to_rgba(hex_colour: str, alpha: float = 1.0) -> str:
    """Convert '#rrggbb' + alpha → 'rgba(r,g,b,a)' for plotly."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _quality_gauge_html(score: float) -> str:
    """Plotly radial gauge with marine styling."""
    import plotly.graph_objects as go
    colour = (
        MARINE["green"] if score >= 75
        else MARINE["amber"] if score >= 50
        else MARINE["orange"]
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(float(score), 1),
        number={"suffix": "%", "font": {"size": 36, "color": MARINE["fg"],
                                           "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": MARINE["fg2"],
                     "tickwidth": 1, "tickfont": {"color": MARINE["fg2"], "size": 10}},
            "bar": {"color": colour, "thickness": 0.25},
            "bgcolor": MARINE["steel"],
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50],  "color": _hex_to_rgba(MARINE["orange"], 0.12)},
                {"range": [50, 75], "color": _hex_to_rgba(MARINE["amber"], 0.12)},
                {"range": [75, 100],"color": _hex_to_rgba(MARINE["green"], 0.12)},
            ],
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=240,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color=MARINE["fg"]),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _render_overlay(analysis: Dict[str, Any]):
    """Big matplotlib overlay in the main column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#FF3B5C", "#00D4AA", "#FFB020", "#4DA6FF",
              "#C77DFF", "#FF6B9D", "#00E5FF"]
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(MARINE["deep"])
    ax.imshow(analysis["image_rgb"])
    for i, (r, cst_xy) in enumerate(zip(analysis["v7_results"],
                                         analysis["cst_splines"])):
        clr = colors[i % len(colors)]
        if cst_xy is not None:
            ax.plot(cst_xy[:, 0], cst_xy[:, 1], "-", color=clr, lw=3.0)
        elif r.spline_points is not None:
            ax.plot(r.spline_points[:, 0], r.spline_points[:, 1],
                    "-", color=clr, lw=3.0)
        ax.plot(r.luff_ep[0], r.luff_ep[1], "o", color=clr,
                markersize=12, markeredgecolor="white", markeredgewidth=1.5)
        ax.plot(r.leech_ep[0], r.leech_ep[1], "o", color=clr,
                markersize=12, markeredgecolor="white", markeredgewidth=1.5)
    ax.set_axis_off()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _stripe_cards(refined):
    colors = ["#FF3B5C", "#00D4AA", "#FFB020", "#4DA6FF",
              "#C77DFF", "#FF6B9D", "#00E5FF"]
    heights = [0.5 * (r.luff_endpoint[1] + r.leech_endpoint[1]) for r in refined]
    order = list(np.argsort(heights))
    for pos, idx in enumerate(order):
        r = refined[idx]
        clr = colors[pos % len(colors)]
        st.markdown(
            f"""
            <div class="stripe-card">
              <div class="stripe-num" style="background: {clr}">{pos + 1}</div>
              <div>
                <div class="s-lbl">chord</div>
                <div class="s-val">{r.chord_length_px:.0f} px</div>
              </div>
              <div>
                <div class="s-lbl">camber</div>
                <div class="s-val">{r.aero.camber_depth_pct:.1f}%</div>
              </div>
              <div>
                <div class="s-lbl">draft</div>
                <div class="s-val">{r.aero.draft_position_pct:.0f}%</div>
              </div>
              <div>
                <div class="s-lbl">twist</div>
                <div class="s-val">{r.aero.twist_deg:+.1f}°</div>
              </div>
              <div>
                <div class="s-lbl">entry/exit</div>
                <div class="s-val">{r.aero.entry_angle_deg:.0f}° / {r.aero.exit_angle_deg:.0f}°</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

st.set_page_config(
    page_title="SailShape · v7",
    page_icon="⛵",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- SIDEBAR ------------------------------------------------------

with st.sidebar:
    st.markdown("### Setup")

    _group_label("Yacht class")
    yacht_names = list(YACHT_RIGS.keys())
    yacht = st.selectbox(
        "Select class",
        yacht_names,
        index=yacht_names.index("Swan 42 (IRC)") if "Swan 42 (IRC)" in yacht_names else 0,
        label_visibility="collapsed",
    )

    _group_label("Sail")
    sail_type = st.selectbox(
        "Sail type",
        ["main", "jib"], index=0,
        label_visibility="collapsed",
    )

    dims = dimensions_for(yacht, sail_type)
    st.caption(
        f"Class defaults: luff **{dims['luff_m']:.2f} m** · "
        f"foot **{dims['foot_m']:.2f} m**"
    )
    override_dims = st.toggle("Override class dimensions", value=False)
    if override_dims:
        luff_m = st.number_input("Luff (m)", min_value=1.0, max_value=40.0,
                                   value=float(dims["luff_m"]), step=0.01)
        foot_m = st.number_input("Foot (m)", min_value=1.0, max_value=15.0,
                                   value=float(dims["foot_m"]), step=0.01)
    else:
        luff_m = dims["luff_m"]
        foot_m = dims["foot_m"]

    _group_label("Inventory")
    default_inventory = "main" if sail_type == "main" else "J2"
    inventory = st.text_input("Inventory label",
                                value=default_inventory,
                                label_visibility="collapsed")

    _group_label("Trim comments")
    trim_comments = st.text_area(
        "Mast tune, forestay, conditions etc.",
        placeholder=(
            "e.g. 8 kt TWS, backstay 2.1 t, mast prebend 50 mm, "
            "runner on 60 %, inner track."
        ),
        height=120,
        label_visibility="collapsed",
    )

    st.divider()
    _group_label("Telemetry (optional)")
    csv_file = st.file_uploader(
        "Boat data CSV (NKE / Expedition)",
        type=["csv"],
        label_visibility="collapsed",
    )


# --- MAIN ---------------------------------------------------------

_hero()

st.markdown("### Upload sail photos")
st.caption(
    "Drag-and-drop one or more sail photos — JPG or PNG. EXIF timestamps "
    "are extracted automatically and matched to the telemetry CSV if "
    "provided."
)
uploaded_files = st.file_uploader(
    "Drop photos here",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not uploaded_files:
    st.info("⬆️  Upload one or more sail photos to begin.")
    st.stop()

# Photo previews
st.markdown(f"**{len(uploaded_files)}** photo(s) queued")
thumb_cols = st.columns(min(len(uploaded_files), 4))
for i, f in enumerate(uploaded_files):
    with thumb_cols[i % len(thumb_cols)]:
        st.image(f.getvalue(), caption=f.name, use_container_width=True)

st.divider()

run_clicked = st.button(
    f"🚀  Analyse  {len(uploaded_files)}  photo{'s' if len(uploaded_files) > 1 else ''}",
    type="primary", use_container_width=True,
)
if not run_clicked:
    st.stop()

# --- RUN BATCH ----------------------------------------------------

cfg_path = _HERE / "config.yaml"
with cfg_path.open() as f:
    config = yaml.safe_load(f)

csv_df = _load_csv(csv_file.getvalue()) if csv_file is not None else None

for idx, uploaded in enumerate(uploaded_files):
    image_bytes = uploaded.getvalue()
    _h = hashlib.sha256(image_bytes).hexdigest()[:10]

    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_rgb = np.array(pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    photo_time = _exif_datetime(image_bytes)
    matched_row = _match_row(csv_df, photo_time) if (csv_df is not None and photo_time) else None
    telemetry = _telemetry_from_row(matched_row, photo_time) if matched_row is not None else None

    # Friendly display name: {inventory} — {photo stem} — {time}
    stem = Path(uploaded.name).stem
    display_name = f"{inventory} · {stem}"

    sail_meta = {
        "type": sail_type,
        "luff_m": float(luff_m),
        "foot_m": float(foot_m),
        "inventory": inventory,
        "display_name": display_name,
        "trim_comments": trim_comments.strip() or None,
        "yacht": yacht,
    }

    st.markdown(
        f"### {display_name} "
        f"<span style='color:{MARINE['fg2']}; font-weight:400; font-size:14px;'>"
        f"— {uploaded.name}  ·  {pil.size[0]}×{pil.size[1]}  ·  sha {_h}</span>",
        unsafe_allow_html=True,
    )

    with st.status("Running v7 pipeline…", expanded=False) as status:
        t0 = time.time()
        analysis = analyze_sail(
            image_bgr=image_bgr, sail_meta=sail_meta, config=config,
            photo_name=display_name,
        )
        status.update(label=f"Done in {time.time() - t0:.1f}s", state="complete")

    # Attach trim comments + yacht + telemetry to analysis for PDF
    analysis["trim_comments"] = trim_comments.strip() or None
    analysis["yacht"] = yacht
    analysis["telemetry"] = telemetry
    analysis["photo_time"] = photo_time

    # Layout: overlay left (wide), panel right (narrow)
    col_img, col_panel = st.columns([7, 4])

    with col_img:
        _render_overlay(analysis)

    with col_panel:
        quality = analysis_quality_score(analysis)
        st.components.v1.html(_quality_gauge_html(quality.score), height=250)

        n = len(analysis["refined_stripes"])
        lb = analysis["luff_depth"].max_depth_pct if analysis["luff_depth"] else 0
        lc = analysis["leech_depth"].max_depth_pct if analysis["leech_depth"] else 0

        m_cols = st.columns(3)
        with m_cols[0]:
            _metric_tile("Stripes", f"{n}", "")
        with m_cols[1]:
            _metric_tile("Luff bend", f"{lb:.2f}", "%")
        with m_cols[2]:
            _metric_tile("Leech", f"{lc:.2f}", "%")

        if telemetry:
            st.markdown(
                f'<div class="form-group-label" style="margin-top:14px">Telemetry match</div>',
                unsafe_allow_html=True,
            )
            tel_cols = st.columns(3)
            slots = [("TWS", "kt"), ("TWA", "°"), ("AWS", "kt"),
                     ("AWA", "°"), ("BSP", "kt"), ("Heel", "°")]
            for i, (k, u) in enumerate(slots):
                with tel_cols[i % 3]:
                    v = telemetry.get(k)
                    if v is not None:
                        _metric_tile(k, f"{v:.1f}", u)

    st.markdown(
        f'<div class="form-group-label">Stripes</div>',
        unsafe_allow_html=True,
    )
    if analysis["refined_stripes"]:
        _stripe_cards(analysis["refined_stripes"])
    else:
        st.warning("No stripes detected.")

    # 3D scan
    with st.expander("🌐  3D scan (detected shape only)", expanded=False):
        if analysis["refined_stripes"]:
            try:
                html_3d = build_3d_plotly_detected_only(
                    analysis["refined_stripes"],
                    analysis["luff_depth"],
                    analysis["leech_depth"],
                    analysis["sail_boundary"],
                    analysis["sail_boundary"].head_point,
                    analysis["sail_boundary"].tack_point,
                    analysis["sail_boundary"].clew_point,
                    luff_length_m=analysis["luff_m"],
                    foot_length_m=analysis["foot_m"],
                    sail_type=analysis["sail_type"],
                    camber_exaggeration=2.5,
                )
                st.components.v1.html(html_3d, height=620, scrolling=False)
            except Exception as exc:
                st.error(f"3D scan failed: {exc}")
        else:
            st.info("No stripes detected — 3D unavailable.")

    # Quality notes
    if quality.notes:
        with st.expander("ℹ️  Quality notes"):
            for note in quality.notes:
                st.write(f"• {note}")

    # PDF download
    with st.spinner("Generating one-page PDF…"):
        pdf_bytes = build_pdf(analysis, telemetry=telemetry,
                               quality=quality.score)

    ts_str = (photo_time or datetime.now()).strftime("%Y%m%d_%H%M%S")
    sail_slug = display_name.replace(" · ", "_").replace(" ", "_")
    pdf_name = f"{sail_slug}_analysis_{ts_str}.pdf"

    st.download_button(
        label=f"⬇  Download {pdf_name}",
        data=pdf_bytes,
        file_name=pdf_name,
        mime="application/pdf",
        key=f"dl_{idx}_{_h}",
        use_container_width=True,
    )

    st.divider()

    # Release intermediate tensors / figures before the next batch photo
    # so we don't accumulate RAM on the cloud 1 GB tier.
    try:
        from src._model_cache import trim_memory
        del analysis
        trim_memory()
    except Exception:
        pass
