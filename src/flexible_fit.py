"""Flexible spline fits that actually follow the detected keypoints.

The legacy 4-parameter Bernstein model
(:func:`src.physics.constrained_bspline_fit`) has only 4 shape degrees of
freedom. On curved, asymmetric stripes it produces visibly shallower
curves than the input keypoints (the fit trails the data). This module
adds two replacements:

* :func:`fit_bernstein_flex` — Bernstein polynomials of configurable
  degree (default 6 → 5 interior basis functions). Still parametrised in
  chord coordinates so entry / exit angles are recoverable by finite
  differences.
* :func:`fit_spline_through_points` — confidence-weighted
  ``scipy.interpolate.splprep`` fit through endpoints + interior points,
  with a soft smoothness penalty. Used as a fallback / high-flex choice
  when Bernstein is still too stiff.

Both return ``(meta, t_fine, spline_points_xy)`` where ``meta`` is a
dict capturing whatever shape parameters the fit produced.
"""

from __future__ import annotations

from math import factorial
from typing import Dict, Optional, Tuple

import numpy as np


def _binom(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return factorial(n) // (factorial(k) * factorial(n - k))


def _chord_frame(luff: np.ndarray, leech: np.ndarray):
    chord = leech - luff
    length = float(np.linalg.norm(chord))
    if length < 1e-6:
        return None
    u = chord / length
    n = np.array([-u[1], u[0]], dtype=np.float64)
    return u, n, length


def fit_bernstein_flex(
    points: np.ndarray,
    luff_endpoint: np.ndarray,
    leech_endpoint: np.ndarray,
    degree: int = 6,
    reg: float = 0.0005,
    keypoint_confidences: Optional[np.ndarray] = None,
    boundary_weight: float = 0.5,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Fit a Bernstein polynomial of given degree to the camber profile.

    Degree ``N`` gives ``N-1`` basis functions that vanish at the chord
    endpoints (we skip ``k=0`` and ``k=N``). Default ``N=6`` ⇒ 5 shape
    DOF — enough flexibility to follow an asymmetric curved stripe
    without overfitting.

    Args:
        points: (P, 2) interior keypoints (luff-to-leech order).
        luff_endpoint: (2,) luff chord end.
        leech_endpoint: (2,) leech chord end.
        degree: Bernstein polynomial degree (>= 3).
        reg: Tikhonov ridge coefficient.
        keypoint_confidences: (P,) weights in [0, 1]; scales per-point fit weight.
        boundary_weight: weight applied to the pinned (0, 0) and (1, 0)
            camber conditions at the chord ends.

    Returns:
        (meta, t_fine, spline_points) where:
            meta has keys: ``coefficients`` (tuple),
            ``basis_indices`` (tuple, ``1..N-1``), ``degree``,
            ``chord_length_px``, ``entry_slope``, ``exit_slope``.
            t_fine: (100,) parameter values in [0, 1].
            spline_points: (100, 2) image-space xy samples.
    """
    frame = _chord_frame(luff_endpoint, leech_endpoint)
    if frame is None or degree < 3:
        t_fine = np.linspace(0.0, 1.0, 100)
        spline = np.outer(1 - t_fine, luff_endpoint) + np.outer(t_fine, leech_endpoint)
        return ({"coefficients": (), "degree": degree, "chord_length_px": 0.0}, t_fine, spline)

    chord_unit, normal_unit, L = frame

    rel = points.astype(np.float64) - luff_endpoint
    t = rel @ chord_unit / L
    d = rel @ normal_unit

    # Sort by t so the fit sees increasing chord coordinate.
    order = np.argsort(t)
    t = t[order]
    d = d[order]
    if keypoint_confidences is not None and len(keypoint_confidences) == len(order):
        kp_w = np.asarray(keypoint_confidences, dtype=np.float64)[order]
    else:
        kp_w = np.ones_like(t)

    # Add pinned boundary conditions.
    t_all = np.concatenate([[0.0], t, [1.0]])
    d_all = np.concatenate([[0.0], d, [0.0]])
    w_all = np.concatenate([[boundary_weight], np.clip(kp_w, 0.2, 1.0), [boundary_weight]])

    # Bernstein basis of degree N, indices k=1..N-1 (vanish at endpoints).
    basis_idx = tuple(range(1, degree))
    m = len(basis_idx)
    A = np.zeros((len(t_all), m), dtype=np.float64)
    for col, k in enumerate(basis_idx):
        coef = _binom(degree, k)
        A[:, col] = coef * (t_all ** k) * ((1.0 - t_all) ** (degree - k))

    W = np.sqrt(w_all)
    A_w = A * W[:, None]
    b_w = d_all * W

    # Regularization acts directly on the coefficients (not scaled by
    # chord length, which previously over-penalised large-chord stripes).
    A_full = np.vstack([A_w, np.sqrt(reg) * np.eye(m)])
    b_full = np.concatenate([b_w, np.zeros(m)])

    coeffs, *_ = np.linalg.lstsq(A_full, b_full, rcond=None)

    t_fine = np.linspace(0.0, 1.0, 100)
    d_fine = np.zeros_like(t_fine)
    for col, k in enumerate(basis_idx):
        coef = _binom(degree, k)
        d_fine += coeffs[col] * coef * (t_fine ** k) * ((1.0 - t_fine) ** (degree - k))

    # Physical-safety clamp: |camber| ≤ 22 % of chord
    d_fine = np.clip(d_fine, -0.22 * L, 0.22 * L)

    spline = (
        np.outer(1 - t_fine, luff_endpoint)
        + np.outer(t_fine, leech_endpoint)
        + np.outer(d_fine, normal_unit)
    )

    # Entry / exit slopes by finite difference on the camber curve
    if len(t_fine) >= 6:
        entry_slope = float((d_fine[3] - d_fine[0]) / (t_fine[3] - t_fine[0] + 1e-9))
        exit_slope = float((d_fine[-1] - d_fine[-4]) / (t_fine[-1] - t_fine[-4] + 1e-9))
    else:
        entry_slope = exit_slope = 0.0

    meta = {
        "coefficients": tuple(float(c) for c in coeffs),
        "basis_indices": basis_idx,
        "degree": degree,
        "chord_length_px": L,
        "entry_slope": entry_slope,
        "exit_slope": exit_slope,
    }
    return meta, t_fine, spline.astype(np.float32)


def fit_cst_airfoil(
    points: np.ndarray,
    luff_endpoint: np.ndarray,
    leech_endpoint: np.ndarray,
    n_shape: int = 4,
    reg: float = 0.001,
    keypoint_confidences: Optional[np.ndarray] = None,
    n_samples: int = 100,
    N1: float = 0.5,
    N2: float = 1.0,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Kulfan Class-Shape Transformation (CST) airfoil fit.

    ``c(t) = C(t) * S(t)`` where
      * ``C(t) = t^N1 * (1-t)^N2`` — class function.
        With ``N1=0.5, N2=1.0`` (standard airfoil), C has a √t slope
        singularity at t=0 — so the curve can capture a SHARP entry
        that plain polynomial NACA ``t^α(1-t)^β`` cannot (it's always
        C∞ smooth).
      * ``S(t) = Σ A_i B_i,n(t)`` — shape function, Bernstein basis
        order ``n_shape``. Default n=4 → 5 coefficients, the Kulfan
        sweet spot for single-sided camber lines.

    Endpoints are pinned to zero FOR FREE because the class function
    vanishes at t=0 and t=1.

    Single-peak enforcement:
      1. Fit at n=n_shape.
      2. Count sign changes of c'(t) at 500 sample points.
      3. If > 1, drop to n-1 and refit; repeat down to n=2.
      4. If STILL multi-peak (rare at low n), apply monotonic-from-peak
         projection.

    Regularization: Tikhonov on DIFFERENCES of A_i (smoothness prior),
    not magnitudes (avoids pulling fit toward zero on undersampled data).

    Returns ``(meta, t_fine, spline_points_xy)`` just like the other fits.
    """
    frame = _chord_frame(luff_endpoint, leech_endpoint)
    if frame is None or len(points) < 3:
        t_fine = np.linspace(0.0, 1.0, n_samples)
        spline = np.outer(1 - t_fine, luff_endpoint) + np.outer(t_fine, leech_endpoint)
        return (
            {"backend": "cst", "coefficients": (), "chord_length_px": 0.0,
             "N1": N1, "N2": N2, "n_shape": n_shape},
            t_fine, spline,
        )
    chord_unit, normal_unit, L = frame

    rel = np.asarray(points, dtype=np.float64) - luff_endpoint
    t = rel @ chord_unit / L
    d = rel @ normal_unit

    # Clamp to (0, 1) and sort
    valid = (t > 0.005) & (t < 0.995)
    t = t[valid]
    d = d[valid]
    if keypoint_confidences is not None and len(keypoint_confidences) == len(valid):
        w = np.clip(np.asarray(keypoint_confidences)[valid], 0.2, 1.0)
    else:
        w = np.ones_like(t)
    order = np.argsort(t)
    t = t[order]; d = d[order]; w = w[order]

    if len(t) < 3:
        t_fine = np.linspace(0.0, 1.0, n_samples)
        spline = np.outer(1 - t_fine, luff_endpoint) + np.outer(t_fine, leech_endpoint)
        return (
            {"backend": "cst", "coefficients": (), "chord_length_px": L,
             "N1": N1, "N2": N2, "n_shape": n_shape},
            t_fine, spline.astype(np.float32),
        )

    def _build_design_matrix(t_arr: np.ndarray, n: int) -> np.ndarray:
        """Class × Bernstein-shape matrix: (len(t), n+1)."""
        class_fn = t_arr ** N1 * (1.0 - t_arr) ** N2
        A = np.zeros((len(t_arr), n + 1), dtype=np.float64)
        for i in range(n + 1):
            binom = _binom(n, i)
            A[:, i] = class_fn * binom * t_arr ** i * (1.0 - t_arr) ** (n - i)
        return A

    def _fit_with_n(n: int) -> np.ndarray:
        """Weighted LSQ + Tikhonov on A_{i+1}-A_i differences."""
        A = _build_design_matrix(t, n)
        W = np.sqrt(w)
        A_w = A * W[:, None]
        b_w = d * W
        # Difference matrix: penalises (A_{i+1} - A_i) for smoothness
        D = np.zeros((n, n + 1), dtype=np.float64)
        for i in range(n):
            D[i, i] = -1.0
            D[i, i + 1] = 1.0
        A_full = np.vstack([A_w, np.sqrt(reg) * D])
        b_full = np.concatenate([b_w, np.zeros(n)])
        coefs, *_ = np.linalg.lstsq(A_full, b_full, rcond=None)
        return coefs

    def _eval(coefs: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
        A = _build_design_matrix(t_arr, len(coefs) - 1)
        return A @ coefs

    t_dense = np.linspace(0.001, 0.999, 500)
    n_try = n_shape
    coefs = _fit_with_n(n_try)
    d_dense = _eval(coefs, t_dense)

    def _count_peaks(arr: np.ndarray, tol_frac: float = 0.02) -> int:
        peak = float(np.max(np.abs(arr)))
        if peak < 1e-6:
            return 0
        sign = np.sign(arr[int(np.argmax(np.abs(arr)))])
        eff = sign * arr
        thresh = tol_frac * peak
        n = 0
        for i in range(1, len(eff) - 1):
            if eff[i] > eff[i - 1] + thresh and eff[i] > eff[i + 1] + thresh:
                n += 1
        return n

    while _count_peaks(d_dense) > 1 and n_try > 2:
        n_try -= 1
        coefs = _fit_with_n(n_try)
        d_dense = _eval(coefs, t_dense)

    t_fine = np.linspace(0.0, 1.0, n_samples)
    d_fine = _eval(coefs, t_fine)

    # Final single-peak safety net
    if _count_peaks(d_fine) > 1:
        peak_idx = int(np.argmax(np.abs(d_fine)))
        sign = np.sign(d_fine[peak_idx]) or 1.0
        out = d_fine.copy()
        for i in range(peak_idx - 1, -1, -1):
            if sign * out[i] > sign * out[i + 1]:
                out[i] = out[i + 1]
        for i in range(peak_idx + 1, len(out)):
            if sign * out[i] > sign * out[i - 1]:
                out[i] = out[i - 1]
        d_fine = out

    d_fine = np.clip(d_fine, -0.25 * L, 0.25 * L)

    spline = (
        np.outer(1 - t_fine, luff_endpoint)
        + np.outer(t_fine, leech_endpoint)
        + np.outer(d_fine, normal_unit)
    ).astype(np.float32)

    # Derived aero-relevant values
    peak_idx = int(np.argmax(np.abs(d_fine)))
    camber_peak_pct = float(abs(d_fine[peak_idx]) / max(L, 1e-3) * 100.0)
    draft_pos_pct = float(t_fine[peak_idx] * 100.0)
    # Entry/exit tangent from first/last segments of d_fine
    entry_slope = float((d_fine[3] - d_fine[0]) / (t_fine[3] - t_fine[0] + 1e-9))
    exit_slope = float((d_fine[-1] - d_fine[-4]) / (t_fine[-1] - t_fine[-4] + 1e-9))

    meta = {
        "backend": "cst",
        "coefficients": tuple(float(c) for c in coefs),
        "N1": N1, "N2": N2, "n_shape": len(coefs) - 1,
        "chord_length_px": L,
        "camber_peak_pct": camber_peak_pct,
        "draft_position_pct": draft_pos_pct,
        "entry_slope": entry_slope,
        "exit_slope": exit_slope,
    }
    return meta, t_fine, spline


def fit_naca_style(
    points: np.ndarray,
    luff_endpoint: np.ndarray,
    leech_endpoint: np.ndarray,
    keypoint_confidences: Optional[np.ndarray] = None,
    n_samples: int = 100,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Fit a 3-parameter NACA-like camber profile ``c(t) = A·t^α·(1-t)^β``.

    Guarantees a single-peak, smooth, airfoil-like shape — unlike free
    Bernstein polynomials which can produce wobbly camber when the input
    keypoints are noisy.

    The three shape parameters are:
        A — peak amplitude (signed; positive = sail bulges to one side)
        α — entry (luff-side) sharpness (> 0, typically 0.5…2)
        β — exit (leech-side) sharpness (> 0, typically 0.5…2)

    Peak position on the chord = α / (α + β). For typical sail stripes
    this sits between 30 % (α≈1, β≈1.8) and 50 % (α=β).

    Args:
        points: (P, 2) interior keypoints (image coords).
        luff_endpoint, leech_endpoint: (2,) chord endpoints.
        keypoint_confidences: per-point weights, clipped to [0.2, 1.0].
        n_samples: output curve resolution.

    Returns:
        ``(meta, t_fine, spline_points_xy)``:
            meta = {"A": float, "alpha": float, "beta": float,
                    "peak_position_pct": float, "chord_length_px": float}
    """
    frame = _chord_frame(luff_endpoint, leech_endpoint)
    if frame is None or len(points) < 3:
        t_fine = np.linspace(0.0, 1.0, n_samples)
        spline = np.outer(1 - t_fine, luff_endpoint) + np.outer(t_fine, leech_endpoint)
        return ({"A": 0.0, "alpha": 1.0, "beta": 1.0, "peak_position_pct": 50.0,
                 "chord_length_px": 0.0}, t_fine, spline)

    chord_unit, normal_unit, L = frame
    rel = points.astype(np.float64) - luff_endpoint
    t = np.clip(rel @ chord_unit / L, 1e-3, 1 - 1e-3)
    d = rel @ normal_unit

    if keypoint_confidences is not None and len(keypoint_confidences) == len(t):
        w = np.clip(np.asarray(keypoint_confidences, dtype=np.float64), 0.2, 1.0)
    else:
        w = np.ones_like(t)

    # Seed: A from the largest |d|, α = β = 1 (symmetric bell)
    idx = int(np.argmax(np.abs(d)))
    A0 = float(d[idx]) / max(t[idx] * (1 - t[idx]), 1e-3)
    A0 = float(np.clip(A0, -0.3 * L, 0.3 * L))

    def _residual(params):
        A, alpha, beta = params
        pred = A * (t ** alpha) * ((1.0 - t) ** beta)
        return (pred - d) * np.sqrt(w)

    try:
        from scipy.optimize import least_squares
        result = least_squares(
            _residual,
            x0=[A0, 1.0, 1.0],
            bounds=([-0.30 * L, 0.4, 0.4], [0.30 * L, 3.0, 3.0]),
            max_nfev=200,
        )
        A, alpha, beta = result.x
    except Exception:
        # Closed-form approximation: symmetric bell with median-fit amplitude
        A = A0
        alpha = beta = 1.0

    t_fine = np.linspace(0.0, 1.0, n_samples)
    d_fine = A * (t_fine ** alpha) * ((1.0 - t_fine) ** beta)

    # Post-fit clamp at 22% of chord (physical realism for sails)
    d_fine = np.clip(d_fine, -0.22 * L, 0.22 * L)

    spline = (
        np.outer(1 - t_fine, luff_endpoint)
        + np.outer(t_fine, leech_endpoint)
        + np.outer(d_fine, normal_unit)
    )

    peak_pos = float(alpha / (alpha + beta) * 100.0)
    meta = {
        "A": float(A),
        "alpha": float(alpha),
        "beta": float(beta),
        "peak_position_pct": peak_pos,
        "chord_length_px": L,
        "backend": "naca",
    }
    return meta, t_fine, spline.astype(np.float32)


def fit_chord_smoothing_spline(
    points: np.ndarray,
    luff_endpoint: np.ndarray,
    leech_endpoint: np.ndarray,
    smoothness: float = 0.4,
    n_samples: int = 100,
    keypoint_confidences: Optional[np.ndarray] = None,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """1D weighted smoothing spline in chord coordinates.

    This fits d(t) where t ∈ [0, 1] is position along the chord and d is
    perpendicular deviation (signed camber). Chord coordinates force the
    curve to monotonically traverse the chord without the 2D-splprep
    "sideways drift" we saw with bad fits.

    * Endpoints pinned heavily (weight 10) at (0, 0) and (1, 0).
    * Interior points weighted by keypoint_confidences (clipped [0.2, 1]).
    * ``smoothness`` scales the SciPy ``UnivariateSpline`` ``s`` parameter.
      Larger = smoother; smaller = follows data more closely.

    Falls back to :func:`fit_bernstein_flex` (degree 5, reg 0.005) if the
    smoothing spline fails (too few unique t values, fit divergent, etc.).
    """
    try:
        from scipy.interpolate import UnivariateSpline
    except Exception:
        return fit_bernstein_flex(
            points, luff_endpoint, leech_endpoint, degree=5, reg=0.005,
            keypoint_confidences=keypoint_confidences,
        )

    frame = _chord_frame(luff_endpoint, leech_endpoint)
    if frame is None or len(points) < 3:
        return fit_bernstein_flex(
            points, luff_endpoint, leech_endpoint, degree=5, reg=0.005,
            keypoint_confidences=keypoint_confidences,
        )
    chord_unit, normal_unit, L = frame

    rel = np.asarray(points, dtype=np.float64) - luff_endpoint
    t = rel @ chord_unit / L
    d = rel @ normal_unit

    # Clamp t into (0, 1) — interior points only
    valid = (t > 0.005) & (t < 0.995)
    t = t[valid]
    d = d[valid]
    if keypoint_confidences is not None and len(keypoint_confidences) == len(valid):
        kp_w = np.asarray(keypoint_confidences, dtype=np.float64)[valid]
    else:
        kp_w = np.ones_like(t)

    # Sort by t
    order = np.argsort(t)
    t = t[order]; d = d[order]; kp_w = kp_w[order]

    # Add endpoints (pinned heavily)
    t_all = np.concatenate([[0.0], t, [1.0]])
    d_all = np.concatenate([[0.0], d, [0.0]])
    w_all = np.concatenate([[10.0], np.clip(kp_w, 0.2, 1.0), [10.0]])

    # Dedup strictly increasing (UnivariateSpline requirement)
    keep = [0]
    for i in range(1, len(t_all)):
        if t_all[i] - t_all[keep[-1]] > 1e-3:
            keep.append(i)
    if len(keep) < 4:
        return fit_bernstein_flex(
            points, luff_endpoint, leech_endpoint, degree=5, reg=0.005,
            keypoint_confidences=keypoint_confidences,
        )
    t_all = t_all[keep]
    d_all = d_all[keep]
    w_all = w_all[keep]

    def _fit_with_s(s_val: float):
        k = min(4, len(t_all) - 1)
        return UnivariateSpline(t_all, d_all, w=w_all, s=s_val, k=k)

    try:
        s = smoothness * len(t_all)
        spline_fn = _fit_with_s(s)
    except Exception:
        return fit_bernstein_flex(
            points, luff_endpoint, leech_endpoint, degree=5, reg=0.005,
            keypoint_confidences=keypoint_confidences,
        )

    t_fine = np.linspace(0.0, 1.0, n_samples)
    d_fine = spline_fn(t_fine)

    # --- Enforce single-peak (NACA-like shape) ---------------------------
    # A sail stripe's camber profile physically has a SINGLE global
    # maximum along the chord — multiple local maxima are fit artefacts.
    # Strategy:
    #   1. Count local maxima in |d_fine|. Ignore tiny ripples below
    #      2 % of peak amplitude.
    #   2. If > 1, bump smoothness and refit. Try up to 3 times.
    #   3. If still multi-peak, post-process: make |d| monotonically
    #      non-increasing from the global peak in both directions.
    def _count_local_maxima(arr: np.ndarray, tol_frac: float = 0.02) -> int:
        peak = float(np.max(np.abs(arr)))
        if peak < 1e-6:
            return 0
        sign = np.sign(arr[int(np.argmax(np.abs(arr)))])
        eff = sign * arr                # align sign so peak is positive
        thresh = tol_frac * peak
        n = 0
        for i in range(1, len(eff) - 1):
            if eff[i] > eff[i - 1] + thresh and eff[i] > eff[i + 1] + thresh:
                n += 1
        return n

    retries = 0
    while _count_local_maxima(d_fine) > 1 and retries < 3:
        retries += 1
        s *= 3.0
        try:
            spline_fn = _fit_with_s(s)
            d_fine = spline_fn(t_fine)
        except Exception:
            break

    # Hard single-peak enforcement: monotonic non-increase from the
    # global-|d| peak in both directions.
    if _count_local_maxima(d_fine) > 1:
        peak_idx = int(np.argmax(np.abs(d_fine)))
        sign = np.sign(d_fine[peak_idx]) or 1.0
        out = d_fine.copy()
        # Left side: walking from peak to 0, |d| must not exceed running max going inward
        for i in range(peak_idx - 1, -1, -1):
            if sign * out[i] > sign * out[i + 1]:
                out[i] = out[i + 1]
        # Right side: same going right
        for i in range(peak_idx + 1, len(out)):
            if sign * out[i] > sign * out[i - 1]:
                out[i] = out[i - 1]
        d_fine = out

    # Physical-safety clamp on camber (±25 %)
    d_fine = np.clip(d_fine, -0.25 * L, 0.25 * L)

    spline_xy = (
        np.outer(1 - t_fine, luff_endpoint)
        + np.outer(t_fine, leech_endpoint)
        + np.outer(d_fine, normal_unit)
    ).astype(np.float32)

    # Entry / exit slopes by finite difference
    entry_slope = float((d_fine[3] - d_fine[0]) / (t_fine[3] - t_fine[0] + 1e-9))
    exit_slope = float((d_fine[-1] - d_fine[-4]) / (t_fine[-1] - t_fine[-4] + 1e-9))

    meta = {
        "coefficients": (),
        "degree": 3,
        "chord_length_px": L,
        "entry_slope": entry_slope,
        "exit_slope": exit_slope,
        "backend": "chord_smoothing_spline",
    }
    return meta, t_fine, spline_xy


def fit_spline_through_points(
    points: np.ndarray,
    luff_endpoint: np.ndarray,
    leech_endpoint: np.ndarray,
    smoothing_factor: float = 1.5,
    n_samples: int = 100,
    keypoint_confidences: Optional[np.ndarray] = None,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Cubic B-spline through (luff, interior, leech) with mild smoothing.

    Falls back to :func:`fit_bernstein_flex` (degree 6) if splprep fails
    (too few unique points, collinear etc.).
    """
    try:
        from scipy.interpolate import splprep, splev
    except Exception:
        return fit_bernstein_flex(points, luff_endpoint, leech_endpoint, degree=6)

    frame = _chord_frame(luff_endpoint, leech_endpoint)
    if frame is None:
        return fit_bernstein_flex(points, luff_endpoint, leech_endpoint, degree=6)
    chord_unit, _, L = frame

    pts = np.vstack([luff_endpoint[None], np.asarray(points, dtype=np.float64), leech_endpoint[None]])

    # Sort by chord coord
    t = (pts - luff_endpoint) @ chord_unit
    order = np.argsort(t)
    pts = pts[order]

    # Dedup near-identical points
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1.5:
            keep.append(i)
    pts = pts[keep]
    if len(pts) < 4:
        return fit_bernstein_flex(points, luff_endpoint, leech_endpoint, degree=6)

    # Weights (endpoints pinned)
    w = np.full(len(pts), 1.0)
    w[0] = 5.0
    w[-1] = 5.0
    if keypoint_confidences is not None:
        # Interior weights inherit from kp confidences (sorted)
        kp = np.asarray(keypoint_confidences, dtype=np.float64)
        if len(kp) == len(points):
            # insert 1.0 at endpoints, reorder like pts
            full = np.concatenate([[1.0], kp, [1.0]])
            full = full[order]
            full = full[keep]
            w[:len(full)] = np.clip(full, 0.2, 1.0)
            w[0] = 5.0; w[-1] = 5.0

    try:
        s = max(len(pts) * smoothing_factor, 2.0)
        tck, _ = splprep([pts[:, 0], pts[:, 1]], w=w, s=s, k=min(3, len(pts) - 1))
        t_grid = np.linspace(0.0, 1.0, n_samples)
        xs, ys = splev(t_grid, tck)
        spline = np.column_stack([xs, ys]).astype(np.float32)
    except Exception:
        return fit_bernstein_flex(points, luff_endpoint, leech_endpoint, degree=6)

    meta = {
        "coefficients": (),
        "degree": 3,
        "chord_length_px": L,
        "entry_slope": 0.0,
        "exit_slope": 0.0,
        "backend": "splprep",
    }
    return meta, np.linspace(0.0, 1.0, n_samples), spline
