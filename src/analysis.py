"""Aerodynamic parameter extraction from fitted splines.

Computes camber depth, draft position, twist, entry/exit angles
from refined B-spline curves.
"""

import logging

import numpy as np
from typing import List, Tuple

from src.types import AeroParams, FittedStripe
from src.physics import compute_analytical_angles

logger = logging.getLogger(__name__)


def extract_aero_params(
    spline_points: np.ndarray,
    luff_endpoint: np.ndarray,
    leech_endpoint: np.ndarray,
    coefficients: tuple = None
) -> AeroParams:
    """Extract aerodynamic parameters from fitted spline.

    Args:
        spline_points: Nx2 spline curve points
        luff_endpoint: 2-element luff position
        leech_endpoint: 2-element leech position

    Returns:
        AeroParams with camber, draft, twist, entry/exit angles
    """
    # Compute chord vector
    chord_vec = leech_endpoint - luff_endpoint
    chord_length = np.linalg.norm(chord_vec)

    # Reject degenerate chords
    if chord_length < 1.0:
        logger.warning(f"Chord length {chord_length:.3f}px too small — returning default params")
        return AeroParams(
            camber_depth_pct=0.0,
            draft_position_pct=50.0,
            twist_deg=0.0,
            entry_angle_deg=0.0,
            exit_angle_deg=0.0
        )

    chord_unit = chord_vec / chord_length

    # Compute perpendicular distances (camber profile)
    cambers = []
    chord_positions = []

    for point in spline_points:
        vec_to_point = point - luff_endpoint
        # Project onto chord
        proj_length = np.dot(vec_to_point, chord_unit)
        chord_positions.append(proj_length)

        # Perpendicular distance (signed)
        proj_point = luff_endpoint + proj_length * chord_unit
        perp_vec = point - proj_point
        # Sign based on which side of chord
        perp_normal = np.array([-chord_unit[1], chord_unit[0]])
        signed_dist = np.dot(perp_vec, perp_normal)
        cambers.append(signed_dist)

    cambers = np.array(cambers)
    chord_positions = np.array(chord_positions)

    # Camber depth: max absolute perpendicular distance
    max_camber_idx = np.argmax(np.abs(cambers))
    max_camber = abs(cambers[max_camber_idx])
    camber_depth_pct = (max_camber / chord_length) * 100

    # Draft position: where max camber occurs, as % of chord
    draft_position = chord_positions[max_camber_idx]
    draft_position_pct = (draft_position / chord_length) * 100

    # Entry/exit angles: use analytical if coefficients available, else finite-difference
    if coefficients is not None and len(coefficients) == 4:
        entry_angle_deg, exit_angle_deg = compute_analytical_angles(coefficients, chord_length)
    else:
        # Finite-difference fallback (Splender path or no coefficients)
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

    # Flag values outside physical bounds
    if camber_depth_pct > 20.0:
        logger.warning(f"Camber depth {camber_depth_pct:.1f}% exceeds 20% — clamped")
    if draft_position_pct < 10.0 or draft_position_pct > 90.0:
        logger.warning(f"Draft position {draft_position_pct:.1f}% outside 10-90%")
    if entry_angle_deg > 25.0:
        logger.warning(f"Entry angle {entry_angle_deg:.1f}° exceeds 25° — clamped")
    if exit_angle_deg > 15.0:
        logger.warning(f"Exit angle {exit_angle_deg:.1f}° exceeds 15° — clamped")

    # Safety-net clamps aligned with physics.py
    camber_depth_pct = float(np.clip(camber_depth_pct, 0.0, 20.0))
    draft_position_pct = float(np.clip(draft_position_pct, 5.0, 95.0))
    entry_angle_deg = float(np.clip(entry_angle_deg, 0.0, 25.0))
    exit_angle_deg = float(np.clip(exit_angle_deg, 0.0, 15.0))

    # Twist: set to 0.0 here, will be computed relative to reference in compute_twist
    twist_deg = 0.0

    return AeroParams(
        camber_depth_pct=camber_depth_pct,
        draft_position_pct=draft_position_pct,
        twist_deg=twist_deg,
        entry_angle_deg=entry_angle_deg,
        exit_angle_deg=exit_angle_deg
    )


def compute_twist(
    all_aero_params: List[AeroParams],
    chord_data: List[Tuple[np.ndarray, np.ndarray]] = None
) -> List[AeroParams]:
    """Compute twist angles relative to reference (bottom) stripe.

    Args:
        all_aero_params: List of AeroParams (top to bottom)
        chord_data: List of (luff_endpoint, leech_endpoint) tuples

    Returns:
        Updated list with twist_deg computed relative to bottom stripe
    """
    if len(all_aero_params) == 0:
        return []

    if chord_data is None or len(chord_data) != len(all_aero_params):
        return all_aero_params

    # Compute chord angle for each stripe
    chord_angles = []
    for luff, leech in chord_data:
        chord_vec = leech - luff
        angle = np.degrees(np.arctan2(chord_vec[1], chord_vec[0]))
        chord_angles.append(angle)

    # Reference: bottom stripe (last in top-to-bottom sorted list)
    ref_angle = chord_angles[-1]

    for i, params in enumerate(all_aero_params):
        params.twist_deg = float(chord_angles[i] - ref_angle)

    return all_aero_params


def build_fitted_stripes(
    stripe_data: List[Tuple],
    all_aero_params: List[AeroParams]
) -> List[FittedStripe]:
    """Build FittedStripe objects from refined data and aero params.

    Args:
        stripe_data: List of (spline_points, luff_endpoint, leech_endpoint)
        all_aero_params: List of AeroParams

    Returns:
        List of FittedStripe objects
    """
    fitted_stripes = []

    for stripe_tuple, params in zip(stripe_data, all_aero_params):
        # Handle both 3-tuple and 4-tuple formats
        if len(stripe_tuple) == 4:
            spline_points, luff, leech, _ = stripe_tuple
        else:
            spline_points, luff, leech = stripe_tuple
        chord_length = float(np.linalg.norm(leech - luff))

        fitted = FittedStripe(
            spline_points=spline_points,
            luff_endpoint=luff,
            leech_endpoint=leech,
            chord_length=chord_length,
            aero_params=params
        )
        fitted_stripes.append(fitted)

    return fitted_stripes
