"""Shared type definitions for the sail stripe analysis pipeline.

All stages of the pipeline use these dataclasses to pass data between modules.
This prevents the "scavenger hunt" anti-pattern where each module defines its own types.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class CalibrationResult:
    """Camera calibration output from camera calibration module.

    Attributes:
        camera_matrix: 3x3 intrinsic camera matrix K
        dist_coeffs: Distortion coefficients (k1, k2, p1, p2, k3)
        focal_length: (fx, fy) focal lengths in pixels
        principal_point: (cx, cy) principal point coordinates
        undistorted_image: Calibrated image with distortion removed
    """
    camera_matrix: np.ndarray  # 3x3
    dist_coeffs: np.ndarray    # (5,) or (4,) depending on model
    focal_length: tuple[float, float]
    principal_point: tuple[float, float]
    undistorted_image: np.ndarray


@dataclass
class SailBoundary:
    """Sail segmentation and boundary extraction output.

    Attributes:
        mask: Binary mask (H x W) where True = sail pixels
        contour: Ordered contour points (N x 2) tracing sail boundary
        luff_polyline: Points from tack to head (M x 2)
        leech_polyline: Points from head to clew (M x 2)
        head_point: Top corner of sail (1 x 2)
        tack_point: Bottom-left corner (1 x 2)
        clew_point: Bottom-right corner (1 x 2)
        foot_polyline: Points from clew to tack (M x 2), or None
    """
    mask: np.ndarray           # bool, H x W
    contour: np.ndarray        # N x 2
    luff_polyline: np.ndarray  # M x 2
    leech_polyline: np.ndarray # M x 2
    head_point: np.ndarray     # 2,
    tack_point: np.ndarray     # 2,
    clew_point: np.ndarray     # 2,
    foot_polyline: np.ndarray = None  # M x 2, optional


@dataclass
class StripeDetection:
    """Raw stripe detection output from keypoint detector.

    These are interior points only - endpoints come from intersection
    with sail boundary.

    Attributes:
        points: Interior keypoint coordinates (P x 2)
        confidence: Detection confidence score [0, 1]
        orientation_deg: Stripe orientation angle from horizontal
        keypoint_confidences: Per-keypoint confidence scores (P,) or None
    """
    points: np.ndarray         # P x 2, interior keypoints
    confidence: float
    orientation_deg: float
    keypoint_confidences: np.ndarray = None  # P, per-keypoint confidence
    # Closed 2D polygon (N x 2) outlining the stripe, when available from
    # a segmentation backend (e.g. Roboflow fleurs-bot). Used by the
    # endpoint stage to intersect with the sail luff/leech polylines.
    polygon: np.ndarray = None


@dataclass
class AeroParams:
    """Aerodynamic parameters extracted from fitted stripe curve.

    These parameters describe the airfoil shape and are used for
    sailing performance analysis.

    Attributes:
        camber_depth_pct: Max perpendicular distance from chord as % of chord length
        draft_position_pct: Position of max camber along chord as % from luff
        twist_deg: Angle difference between top and bottom of stripe
        entry_angle_deg: Angle at luff entry point
        exit_angle_deg: Angle at leech exit point
    """
    camber_depth_pct: float
    draft_position_pct: float
    twist_deg: float
    entry_angle_deg: float
    exit_angle_deg: float


@dataclass
class FittedStripe:
    """Final fitted stripe curve with endpoints and aerodynamic parameters.

    This is the output of the complete pipeline for a single stripe.

    Attributes:
        spline_points: Densely sampled points along fitted B-spline (Q x 2)
        luff_endpoint: Intersection point with luff (2,)
        leech_endpoint: Intersection point with leech (2,)
        chord_length: Distance between endpoints in pixels
        aero_params: Computed aerodynamic parameters
    """
    spline_points: np.ndarray  # Q x 2
    luff_endpoint: np.ndarray  # 2,
    leech_endpoint: np.ndarray # 2,
    chord_length: float
    aero_params: AeroParams
    fit_confidence: float = 1.0  # composite confidence score [0, 1]
