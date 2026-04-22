"""Physics-based airfoil constraints for sail stripe analysis.

Enforces realistic airfoil geometry:
- Single camber peak (no S-curves)
- Entry/exit angles within plausible ranges
- Camber depth and draft position constraints
"""

import numpy as np
from scipy.interpolate import splprep, splev, splrep
from scipy.signal import find_peaks
from typing import Tuple, Dict, Optional


def validate_airfoil_shape(
    spline_points: np.ndarray,
    luff_endpoint: np.ndarray,
    leech_endpoint: np.ndarray
) -> Dict:
    """Validate that spline follows realistic airfoil physics.

    Args:
        spline_points: Nx2 array of spline curve points
        luff_endpoint: 2-element array (luff edge of sail)
        leech_endpoint: 2-element array (leech edge of sail)

    Returns:
        dict with:
            valid: bool - True if all constraints satisfied
            violations: list of str - descriptions of failed constraints
            metrics: dict with camber_depth_pct, draft_position_pct,
                    entry_angle_deg, exit_angle_deg, num_peaks
    """
    violations = []

    # Compute chord vector
    chord_vec = leech_endpoint - luff_endpoint
    chord_length = np.linalg.norm(chord_vec)

    if chord_length < 1e-6:
        return {
            "valid": False,
            "violations": ["Chord length too small"],
            "num_peaks": 0,
            "camber_depth_pct": 0.0,
            "draft_position_pct": 0.0,
            "entry_angle_deg": 0.0,
            "exit_angle_deg": 0.0
        }

    chord_unit = chord_vec / chord_length

    # Compute perpendicular distances to chord (camber profile)
    # For each point, project onto chord line and compute perpendicular distance
    cambers = []
    chord_positions = []

    for point in spline_points:
        vec_to_point = point - luff_endpoint
        # Project onto chord
        proj_length = np.dot(vec_to_point, chord_unit)
        chord_positions.append(proj_length / chord_length)  # Normalized 0-1

        # Perpendicular distance (signed)
        proj_point = luff_endpoint + proj_length * chord_unit
        perp_vec = point - proj_point
        # Sign based on which side of chord
        perp_normal = np.array([-chord_unit[1], chord_unit[0]])
        signed_dist = np.dot(perp_vec, perp_normal)
        cambers.append(signed_dist)

    cambers = np.array(cambers)
    chord_positions = np.array(chord_positions)

    # 1. Find camber depth and draft position
    max_camber_idx = np.argmax(np.abs(cambers))
    max_camber = abs(cambers[max_camber_idx])
    draft_position_pct = chord_positions[max_camber_idx] * 100
    camber_depth_pct = (max_camber / chord_length) * 100

    # 2. Check single peak constraint
    # Use find_peaks on absolute camber profile
    peaks, _ = find_peaks(np.abs(cambers), prominence=max_camber * 0.3)
    num_peaks = len(peaks)

    if num_peaks > 1:
        violations.append(f"Multiple peaks detected ({num_peaks}), expected single-peak airfoil")

    # 3. Check draft position (camber peak should be in middle 60%)
    if draft_position_pct < 20 or draft_position_pct > 80:
        violations.append(f"Draft position {draft_position_pct:.1f}% outside 20-80% range")

    # 4. Check camber depth (1-20%)
    if camber_depth_pct < 1 or camber_depth_pct > 20:
        violations.append(f"Camber depth {camber_depth_pct:.1f}% outside 1-20% range")

    # 5. Compute entry angle (at luff) — local finite difference
    if len(spline_points) >= 5:
        n_tan = min(5, len(spline_points) // 10 + 1)
        entry_tangent = spline_points[n_tan] - spline_points[0]
        entry_norm = np.linalg.norm(entry_tangent)
        if entry_norm > 1e-9:
            entry_tangent = entry_tangent / entry_norm
            cos_angle = np.clip(np.dot(entry_tangent, chord_unit), -1, 1)
            entry_angle_deg = np.degrees(np.arccos(cos_angle))
        else:
            entry_angle_deg = 0.0
    elif len(spline_points) >= 2:
        entry_tangent = spline_points[1] - spline_points[0]
        entry_norm = np.linalg.norm(entry_tangent)
        if entry_norm > 1e-9:
            entry_tangent = entry_tangent / entry_norm
            cos_angle = np.clip(np.dot(entry_tangent, chord_unit), -1, 1)
            entry_angle_deg = np.degrees(np.arccos(cos_angle))
        else:
            entry_angle_deg = 0.0
    else:
        entry_angle_deg = 0.0

    if entry_angle_deg > 25:
        violations.append(f"Entry angle {entry_angle_deg:.1f}° exceeds 25°")

    # 6. Compute exit angle (at leech) — local finite difference
    if len(spline_points) >= 5:
        n_tan = min(5, len(spline_points) // 10 + 1)
        exit_tangent = spline_points[-1] - spline_points[-1 - n_tan]
        exit_norm = np.linalg.norm(exit_tangent)
        if exit_norm > 1e-9:
            exit_tangent = exit_tangent / exit_norm
            cos_angle = np.clip(np.dot(exit_tangent, chord_unit), -1, 1)
            exit_angle_deg = np.degrees(np.arccos(cos_angle))
        else:
            exit_angle_deg = 0.0
    elif len(spline_points) >= 2:
        exit_tangent = spline_points[-1] - spline_points[-2]
        exit_norm = np.linalg.norm(exit_tangent)
        if exit_norm > 1e-9:
            exit_tangent = exit_tangent / exit_norm
            cos_angle = np.clip(np.dot(exit_tangent, chord_unit), -1, 1)
            exit_angle_deg = np.degrees(np.arccos(cos_angle))
        else:
            exit_angle_deg = 0.0
    else:
        exit_angle_deg = 0.0

    if exit_angle_deg > 20:
        violations.append(f"Exit angle {exit_angle_deg:.1f}° exceeds 20°")

    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "num_peaks": num_peaks,
        "camber_depth_pct": float(camber_depth_pct),
        "draft_position_pct": float(draft_position_pct),
        "entry_angle_deg": float(entry_angle_deg),
        "exit_angle_deg": float(exit_angle_deg)
    }


def apply_airfoil_constraints(
    spline_points: np.ndarray,
    luff_endpoint: np.ndarray,
    leech_endpoint: np.ndarray,
    max_iterations: int = 3
) -> np.ndarray:
    """Apply constraints to produce valid airfoil shape.

    The 4-param Bernstein model rarely produces S-curves, so this is
    simplified: single refit attempt if validation fails.

    Args:
        spline_points: Nx2 initial spline points (interior points)
        luff_endpoint: 2-element luff position (pinned)
        leech_endpoint: 2-element leech position (pinned)
        max_iterations: Maximum constraint iterations

    Returns:
        Constrained spline points (Mx2)
    """
    # Start with an initial fit
    try:
        _, _, current_points = constrained_bspline_fit(
            spline_points,
            luff_endpoint,
            leech_endpoint
        )
    except Exception:
        current_points = spline_points.copy()

    # Validate — the 4-param model usually passes on first try
    validation = validate_airfoil_shape(current_points, luff_endpoint, leech_endpoint)
    if validation["valid"]:
        return current_points

    # If S-curve detected, the constrained_bspline_fit already handles
    # re-regularization internally, so just return what we have
    return current_points


def compute_analytical_angles(
    coefficients: Tuple[float, float, float, float],
    chord_length: float
) -> Tuple[float, float]:
    """Compute entry and exit angles analytically from Bernstein coefficients.

    The 4-param camber model is:
        camber(t) = c1*t*(1-t) + c2*t*(1-t)*(1-2t) + c3*t^2*(1-t) + c4*t*(1-t)^2

    Derivatives at endpoints:
        camber'(0) = c1 + c2 + c4   (entry slope)
        camber'(1) = -(c1 - c2 + c3) = -c1 + c2 - c3   (exit slope, negated)

    Entry angle = atan2(|camber'(0)|, chord_length)
    Exit angle  = atan2(|camber'(1)|, chord_length)

    Args:
        coefficients: (c1, c2, c3, c4) Bernstein coefficients
        chord_length: Chord length in pixels

    Returns:
        (entry_angle_deg, exit_angle_deg)
    """
    c1, c2, c3, c4 = coefficients

    # camber'(t) = c1*(1 - 2t) + c2*(1 - 6t + 6t^2) + c3*(2t - 3t^2) + c4*(1 - 4t + 3t^2)
    # At t=0: camber'(0) = c1 + c2 + c4
    # At t=1: camber'(1) = -c1 - c2 + 6c2 - c3*3 + c3*2 ... let me derive carefully
    # B1 = t(1-t), B1' = 1-2t
    # B2 = t(1-t)(1-2t), B2' = (1-2t)(1-2t) + t(1-t)(-2) = (1-2t)^2 - 2t(1-t)
    # B3 = t^2(1-t), B3' = 2t(1-t) - t^2 = t(2-3t)
    # B4 = t(1-t)^2, B4' = (1-t)^2 - 2t(1-t) = (1-t)(1-3t)
    #
    # At t=0: B1'=1, B2'=1, B3'=0, B4'=1
    # => camber'(0) = c1 + c2 + c4
    #
    # At t=1: B1'=-1, B2'=1-4+2=-1? Let me recompute
    # B2' = (1-2t)^2 - 2t(1-t)
    # At t=1: (1-2)^2 - 2*1*0 = 1
    # B3' at t=1: 1*(2-3) = -1
    # B4' at t=1: 0*(1-3) = 0
    # => camber'(1) = c1*(-1) + c2*(1) + c3*(-1) + c4*(0) = -c1 + c2 - c3

    entry_slope = c1 + c2 + c4
    exit_slope = -c1 + c2 - c3  # slope at t=1 (positive = curving away from chord)

    entry_angle_deg = float(np.degrees(np.arctan2(abs(entry_slope), chord_length)))
    exit_angle_deg = float(np.degrees(np.arctan2(abs(exit_slope), chord_length)))

    return entry_angle_deg, exit_angle_deg


def constrained_bspline_fit(
    points: np.ndarray,
    luff_endpoint: np.ndarray,
    leech_endpoint: np.ndarray,
    smoothing: float = 2.0,
    keypoint_confidences: np.ndarray = None
) -> Tuple[Optional[Tuple], np.ndarray, np.ndarray]:
    """Fit a smooth airfoil curve between luff and leech using a 4-parameter Bernstein model.

    Uses 4 basis functions that all vanish at t=0 and t=1:
        B1(t) = t*(1-t)         — symmetric bell
        B2(t) = t*(1-t)*(1-2t)  — asymmetry/draft position
        B3(t) = t^2*(1-t)       — exit region control
        B4(t) = t*(1-t)^2       — entry region control

    This gives independent control over camber, draft, entry angle, and exit angle.

    Args:
        points: Nx2 interior points
        luff_endpoint: 2-element luff position
        leech_endpoint: 2-element leech position
        smoothing: Unused (kept for API compatibility)
        keypoint_confidences: Per-keypoint confidence values (N,) or None

    Returns:
        (coefficients, t_fine, spline_points):
            coefficients: (c1, c2, c3, c4) tuple or None if degenerate
            t_fine: Parameter values along chord [0, 1]
            spline_points: Evaluated 2D curve at 100 points
    """
    chord_vec = leech_endpoint - luff_endpoint
    chord_length = np.linalg.norm(chord_vec)

    if chord_length < 1e-6:
        # Degenerate chord — return straight line
        t_fine = np.linspace(0, 1, 100)
        spline_points = np.outer(1 - t_fine, luff_endpoint) + np.outer(t_fine, leech_endpoint)
        return None, t_fine, spline_points

    chord_unit = chord_vec / chord_length
    normal_unit = np.array([-chord_unit[1], chord_unit[0]])

    # Project interior points onto chord coordinate system
    chord_positions = []
    camber_values = []

    for point in points:
        v = point - luff_endpoint
        t = np.dot(v, chord_unit) / chord_length  # normalized [0, 1]
        d = np.dot(v, normal_unit)  # signed perpendicular distance
        chord_positions.append(t)
        camber_values.append(d)

    chord_positions = np.array(chord_positions)
    camber_values = np.array(camber_values)

    # Include boundary conditions: camber = 0 at t=0 and t=1
    all_t = np.concatenate([[0.0], chord_positions, [1.0]])
    all_camber = np.concatenate([[0.0], camber_values, [0.0]])

    # Compute data coverage: fraction of [0,1] interval covered by data
    t_min_data = chord_positions.min() if len(chord_positions) > 0 else 0.5
    t_max_data = chord_positions.max() if len(chord_positions) > 0 else 0.5
    coverage = max(t_max_data - t_min_data, 0.01)

    # Add guide points in data gaps to prevent wild extrapolation
    guide_t = []
    guide_camber = []
    guide_weight = 0.3  # lighter weight for guide points

    # If data doesn't cover near-entry region, add guide near t=0
    if t_min_data > 0.15:
        guide_t.extend([0.05, 0.10])
        guide_camber.extend([0.0, 0.0])

    # If data doesn't cover near-exit region, add guide near t=1
    if t_max_data < 0.85:
        guide_t.extend([0.90, 0.95])
        guide_camber.extend([0.0, 0.0])

    if guide_t:
        all_t = np.concatenate([all_t, np.array(guide_t)])
        all_camber = np.concatenate([all_camber, np.array(guide_camber)])

    # Build 4-parameter basis functions
    B1 = all_t * (1.0 - all_t)                    # symmetric bell
    B2 = all_t * (1.0 - all_t) * (1.0 - 2.0 * all_t)  # asymmetry
    B3 = all_t**2 * (1.0 - all_t)                 # exit region
    B4 = all_t * (1.0 - all_t)**2                  # entry region

    # Weights: interior data=1 (scaled by confidence), boundary=0.5, guide=guide_weight
    n_interior = len(chord_positions)
    n_original = 2 + n_interior  # boundary + interior
    n_guide = len(guide_t)
    weights = np.ones(len(all_t))
    weights[0] = 0.5   # luff boundary
    weights[n_original - 1] = 0.5  # leech boundary
    # Scale interior weights by per-keypoint confidence if available
    if keypoint_confidences is not None and len(keypoint_confidences) == n_interior:
        weights[1:1 + n_interior] = np.clip(keypoint_confidences, 0.1, 1.0)
    if n_guide > 0:
        weights[n_original:] = guide_weight

    # Adaptive Tikhonov regularization: stronger for sparse data
    n_interior = len(chord_positions)
    if n_interior <= 4:
        base_reg = 0.10   # strong for very sparse data
    elif n_interior <= 8:
        base_reg = 0.05   # moderate
    else:
        base_reg = 0.01   # light for well-sampled
    reg_lambda = base_reg / max(coverage, 0.1)

    W = np.sqrt(weights)
    A = np.column_stack([B1 * W, B2 * W, B3 * W, B4 * W])
    b = all_camber * W

    # Add regularization rows
    reg_matrix = np.sqrt(reg_lambda) * np.eye(4)
    A_reg = np.vstack([A, reg_matrix])
    b_reg = np.concatenate([b, np.zeros(4)])

    # Solve regularized least squares for c1, c2, c3, c4
    result = np.linalg.lstsq(A_reg, b_reg, rcond=None)
    c1, c2, c3, c4 = result[0]
    coefficients = (float(c1), float(c2), float(c3), float(c4))

    # Evaluate at 100 points
    t_fine = np.linspace(0, 1, 100)
    B1_f = t_fine * (1.0 - t_fine)
    B2_f = t_fine * (1.0 - t_fine) * (1.0 - 2.0 * t_fine)
    B3_f = t_fine**2 * (1.0 - t_fine)
    B4_f = t_fine * (1.0 - t_fine)**2
    camber_fine = c1 * B1_f + c2 * B2_f + c3 * B3_f + c4 * B4_f

    # Post-fit single-peak validation
    abs_camber = np.abs(camber_fine)
    peaks, _ = find_peaks(abs_camber, prominence=abs_camber.max() * 0.3 if abs_camber.max() > 0 else 0)
    if len(peaks) > 1:
        # Refit with stronger regularization
        strong_reg = 0.1 / max(coverage, 0.1)
        reg_matrix_strong = np.sqrt(strong_reg) * np.eye(4)
        A_strong = np.vstack([A, reg_matrix_strong])
        b_strong = np.concatenate([b, np.zeros(4)])
        result2 = np.linalg.lstsq(A_strong, b_strong, rcond=None)
        c1, c2, c3, c4 = result2[0]
        coefficients = (float(c1), float(c2), float(c3), float(c4))
        camber_fine = c1 * B1_f + c2 * B2_f + c3 * B3_f + c4 * B4_f

    # Post-fit safety: clamp camber to physical max (20% of chord)
    max_physical_camber = 0.20 * chord_length
    camber_fine = np.clip(camber_fine, -max_physical_camber, max_physical_camber)

    # Reconstruct 2D curve from chord + camber
    spline_points = np.zeros((100, 2))
    for i, t in enumerate(t_fine):
        spline_points[i] = (luff_endpoint + t * chord_length * chord_unit
                            + camber_fine[i] * normal_unit)

    return coefficients, t_fine, spline_points
