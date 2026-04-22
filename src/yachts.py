"""Yacht rig dimensions + analysis-quality scoring.

YACHT_RIGS — one-design / grand-prix class rig dimensions in metres.
Values reflect class-rule maxima or standard OD sail dimensions. Use the
owner's measurement certificate for quantitative analysis where available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


YACHT_RIGS: Dict[str, Dict[str, float]] = {
    "ClubSwan 42": {
        "main_luff_m": 16.60, "main_foot_m": 5.70,
        "jib_luff_m": 17.50,  "jib_foot_m": 5.80,
    },
    "ClubSwan 50": {
        "main_luff_m": 21.00, "main_foot_m": 7.00,
        "jib_luff_m": 21.50,  "jib_foot_m": 6.80,
    },
    "RC44": {
        "main_luff_m": 17.00, "main_foot_m": 6.20,
        "jib_luff_m": 17.60,  "jib_foot_m": 5.60,
    },
    "TP52": {
        "main_luff_m": 20.50, "main_foot_m": 7.20,
        "jib_luff_m": 21.40,  "jib_foot_m": 6.70,
    },
    "Cape 31": {
        "main_luff_m": 13.80, "main_foot_m": 4.90,
        "jib_luff_m": 13.00,  "jib_foot_m": 4.20,
    },
    "J/70": {
        "main_luff_m":  9.30, "main_foot_m": 3.30,
        "jib_luff_m":   8.80, "jib_foot_m":  3.05,
    },
    "ClubSwan 36": {
        "main_luff_m": 15.20, "main_foot_m": 5.40,
        "jib_luff_m":  14.80, "jib_foot_m":  4.80,   # est.
    },
    "Melges 24": {
        "main_luff_m":  8.90, "main_foot_m": 3.35,
        "jib_luff_m":   8.50, "jib_foot_m":  2.95,
    },
    "Class 40": {
        "main_luff_m": 17.50, "main_foot_m": 6.20,
        "jib_luff_m":  16.80, "jib_foot_m":  5.50,   # est. mid-fleet Mach 40
    },
    # Swan 42 (older IRC cruiser-racer, retained for back-compat with the
    # user's earlier test photos). Not the ClubSwan 42 one-design.
    "Swan 42 (IRC)": {
        "main_luff_m": 17.24, "main_foot_m": 5.78,
        "jib_luff_m":  18.68, "jib_foot_m":  4.98,
    },
    "Custom": {
        "main_luff_m": 15.00, "main_foot_m": 5.00,
        "jib_luff_m":  16.00, "jib_foot_m":  4.50,
    },
}


def dimensions_for(yacht: str, sail_type: str) -> Dict[str, float]:
    """Return (luff_m, foot_m) as a dict for a given yacht + sail type."""
    rig = YACHT_RIGS.get(yacht, YACHT_RIGS["Custom"])
    if sail_type.lower() == "main":
        return {"luff_m": rig["main_luff_m"], "foot_m": rig["main_foot_m"]}
    return {"luff_m": rig["jib_luff_m"], "foot_m": rig["jib_foot_m"]}


# ---------------------------------------------------------------------------
# Analysis-quality scorer
# ---------------------------------------------------------------------------

@dataclass
class QualityBreakdown:
    score: float                # 0–100
    stripe_count_score: float   # did we find a reasonable number of stripes?
    fit_residual_score: float   # how tight is the spline to the data?
    endpoint_guard_score: float # how often did the guard fire?
    confidence_score: float     # mean detection confidence
    sail_coverage_score: float  # how much of the sail width the stripes span
    notes: List[str]


def analysis_quality_score(analysis: Dict[str, Any]) -> QualityBreakdown:
    """Composite 0-100 score describing how trustworthy the analysis is.

    Components (each 0-1, then weighted):
      * ``stripe_count``      — at least 2 valid stripes, cap at 4 for the max.
      * ``fit_residual``      — mean perpendicular distance (px) from each
                                stripe's detection points to its spline, normalised
                                by the stripe's chord length. Smaller = better.
      * ``endpoint_guard``    — 1 - (fraction of endpoints that had to be pulled
                                back by the guard). If everything was clean, full
                                marks.
      * ``confidence``        — mean detection.confidence across stripes.
      * ``sail_coverage``     — how much of the bounding box between luff and
                                leech polylines the stripes span (a short stripe
                                on a wide sail is a bad sign).
    """
    import numpy as np
    results = analysis.get("v7_results") or []
    refined = analysis.get("refined_stripes") or []
    notes: List[str] = []

    # 1. Stripe count
    n = len(refined)
    if n == 0:
        return QualityBreakdown(
            score=0.0, stripe_count_score=0.0, fit_residual_score=0.0,
            endpoint_guard_score=0.0, confidence_score=0.0,
            sail_coverage_score=0.0,
            notes=["No stripes detected."],
        )
    stripe_count_score = min(1.0, n / 4.0)
    if n < 2:
        notes.append("Only 1 stripe — analysis is unreliable.")

    # 2. Fit residuals
    residuals = []
    for r in results:
        if r.spline_points is None or r.detection.points is None:
            continue
        sp = r.spline_points
        pts = r.detection.points
        chord = float(np.linalg.norm(r.leech_ep - r.luff_ep))
        if chord < 1e-3 or len(sp) < 2:
            continue
        # Nearest distance per point
        per_pt = []
        for p in pts:
            best = np.inf
            for i in range(len(sp) - 1):
                a = sp[i]; b = sp[i + 1]
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
            per_pt.append(best)
        if per_pt:
            residuals.append(float(np.mean(per_pt)) / chord)
    if residuals:
        mean_rel_residual = float(np.mean(residuals))
        # Map 0.0% → 1.0, 2% of chord → 0.0
        fit_residual_score = max(0.0, 1.0 - mean_rel_residual / 0.02)
    else:
        fit_residual_score = 0.0

    # 3. Endpoint guard
    guarded = 0
    total_eps = 0
    for r in results:
        total_eps += 2
        if r.endpoint_guarded[0]:
            guarded += 1
        if r.endpoint_guarded[1]:
            guarded += 1
    endpoint_guard_score = 1.0 - (guarded / max(total_eps, 1))

    # 4. Confidence
    confs = [r.detection.confidence for r in results if r.detection]
    confidence_score = float(np.mean(confs)) if confs else 0.0

    # 5. Sail coverage — how wide do the stripes span vs sail width at that height
    coverages = []
    sail = analysis.get("sail_boundary")
    if sail is not None and sail.luff_polyline is not None:
        for r in refined:
            mid_y = 0.5 * (r.luff_endpoint[1] + r.leech_endpoint[1])
            luff_x = _x_at_y(sail.luff_polyline, mid_y)
            leech_x = _x_at_y(sail.leech_polyline, mid_y)
            if luff_x is None or leech_x is None:
                continue
            sail_width = abs(leech_x - luff_x)
            if sail_width < 1e-3:
                continue
            stripe_width = r.chord_length_px
            coverages.append(min(1.0, stripe_width / sail_width))
    sail_coverage_score = float(np.mean(coverages)) if coverages else 0.8

    # Weights
    composite = (
        0.15 * stripe_count_score
        + 0.30 * fit_residual_score
        + 0.15 * endpoint_guard_score
        + 0.20 * confidence_score
        + 0.20 * sail_coverage_score
    )
    composite = float(max(0.0, min(1.0, composite)))

    if guarded:
        notes.append(f"{guarded} endpoint(s) were guarded (pulled back to stripe data).")
    if residuals and max(residuals) > 0.015:
        notes.append("One or more splines drift > 1.5 % of chord from detection points.")
    if coverages and min(coverages) < 0.6:
        notes.append("At least one stripe covers < 60 % of the sail width at its height.")

    return QualityBreakdown(
        score=composite * 100.0,
        stripe_count_score=stripe_count_score,
        fit_residual_score=fit_residual_score,
        endpoint_guard_score=endpoint_guard_score,
        confidence_score=confidence_score,
        sail_coverage_score=sail_coverage_score,
        notes=notes,
    )


def _x_at_y(polyline, y_target: float) -> Optional[float]:
    import numpy as np
    if polyline is None or len(polyline) < 2:
        return None
    ys = polyline[:, 1]
    xs = polyline[:, 0]
    order = np.argsort(ys)
    ys_s, xs_s = ys[order], xs[order]
    if y_target <= ys_s[0] or y_target >= ys_s[-1]:
        return None
    return float(np.interp(y_target, ys_s, xs_s))
