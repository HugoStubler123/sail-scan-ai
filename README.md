# SailShape · v7

Photo-to-shape analyser for performance sailmakers and trimmers. Upload a
sail photo — get stripe geometry (camber, draft, twist, entry/exit angles),
luff/leech bend profile, CST airfoil fit and a 3D reconstruction. Exports
a one-page PDF report.

![SailShape](docs/hero.png)

## Stack

- **Streamlit** UI with a custom marine dark theme
- **SAM 2.1 (base)** for sail segmentation
- **YOLO11** custom models for stripe bbox / keypoints / segmentation
- **Kulfan CST** airfoil fit + chord-smoothing spline with single-peak enforcement
- **Plotly** 3D + radial quality gauge, matplotlib PDF renderer

## Local run

```bash
git clone https://github.com/HugoStubler123/sail-scan-ai.git
cd sail-scan-ai
# If cloning without LFS, fetch the model weights:
git lfs install && git lfs pull

pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at http://localhost:8501.

## Deploy on Streamlit Community Cloud

1. Push this repo (it already contains `streamlit_app.py`, `requirements.txt`,
   `packages.txt` and the model weights via Git LFS).
2. Go to https://share.streamlit.io → **New app**.
3. Select `HugoStubler123/sail-scan-ai`, branch `main`, main file
   `streamlit_app.py`.
4. Under **Advanced settings**, Python version 3.11.
5. Deploy. First boot will take ~3-4 min (SAM2 + YOLO11 + torch download).

Streamlit Cloud honours `.streamlit/config.toml` for the dark marine theme
and `packages.txt` for `libgl1` (OpenCV runtime dep).

## Yacht library

Pre-filled rig dimensions for ClubSwan 36/42/50, RC44, TP52, Cape 31,
J/70, Melges 24, Class 40 and Swan 42 (IRC). Add more by editing
`src/yachts.py::YACHT_RIGS`.

## Optional: boat-telemetry overlay

Drop an NKE / Expedition CSV in the sidebar. The app matches the photo's
EXIF `DateTimeOriginal` to the nearest CSV row and prints TWS, TWA, AWS,
AWA, BSP and Heel on the bottom-right of the overlay and in the PDF.

## Project layout

```
.
├── streamlit_app.py           # UI entrypoint
├── config.yaml                # pipeline tuneables
├── requirements.txt
├── packages.txt               # apt packages for Streamlit Cloud
├── .streamlit/config.toml     # marine theme
├── src/
│   ├── sail_pipeline.py       # single-call analyse_sail()
│   ├── pipeline_v7.py         # detection + fusion + spline-SAM intersect
│   ├── flexible_fit.py        # CST airfoil + chord-smoothing spline
│   ├── sail_analysis.py       # 2D → 3D sail reconstruction
│   ├── yachts.py              # rig dimensions + quality scorer
│   ├── report_pdf.py          # marine one-pager PDF
│   ├── trim_analyst.py        # rule-based trim commentary
│   └── …
├── models/
│   └── sail_seg_model.pt      # new YOLO11-seg stripe model (63 MB)
├── sam2.1_b.pt                # SAM 2.1 base (154 MB, LFS)
├── stripe_bbox_v1.pt          # YOLO11n bbox (5 MB, LFS)
├── stripe_keypoints_v1.pt     # YOLO11n-pose 8 kpts (5 MB, LFS)
├── stripe_seg_v1.pt           # legacy fallback seg (17 MB, LFS)
├── stripe_endpoints_v2.pt     # endpoint refiner (5 MB, LFS)
└── test_photo/
    └── raving_jib.jpg
```

## Pipeline overview

1. **Segmentation** — SAM2.1b multi-point prompted; head = topmost
   mask pixel; luff/leech split by curvature, swap by max-depth convention.
2. **Bbox** — YOLO11n stripe detector → dedup (IoU 0.35).
3. **Keypoints** — YOLO11n-pose 8 keypoints per stripe, pooled per bbox.
4. **Seg on crop** — new `sail_seg_model.pt` swept across 7 color variants
   (raw, CLAHE-L, γ0.7, γ1.4, sat+50 %, unsharp, channel-min) + legacy
   model on raw and channel-min crops. Composite scorer picks best:
   `0.35·conf + 0.20·length + 0.15·edge_luff + 0.15·edge_leech + 0.15·thinness`.
5. **Fusion** — polygon bottom × matched kp anchors + curvature samples;
   channel-min 2nd-shape fill-in on luff/leech sides where kp is sparse.
6. **Endpoints** — stripe extremes → guard (12 px perp) → fit spline →
   tangentially intersect with SAM luff/leech polylines (15 % chord cap) →
   refit.
7. **Spline** — 1D `UnivariateSpline` in chord coords, single-peak
   enforced (bumped smoothness + monotonic-from-peak projection).
8. **Aero** — CST (`c(t) = t^0.5·(1-t)^1.0·Σ A_i B_i,4(t)`) preserves
   sharp leading-edge slope that plain NACA `t^α(1-t)^β` can't.
9. **3D** — metric-scaled loft between detected stripes; twist rotation
   around luff axis; Gaussian-smoothed mesh.

## License

MIT. See `LICENSE`.
