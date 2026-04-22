"""Per-stripe confidence scoring.

Combines detection confidence, data coverage, and fit quality into
a single composite score for each stripe.
"""

import numpy as np
from src.types import FittedStripe
from src.physics import validate_airfoil_shape


def compute_stripe_confidence(
    fitted_stripe: FittedStripe,
    detection_confidence: float = 0.5,
    n_valid_keypoints: int = 8,
    total_keypoints: int = 8
) -> float:
    """Compute composite confidence score for a fitted stripe.

    confidence = 0.4 * detection_conf + 0.3 * coverage_ratio + 0.3 * fit_quality

    Args:
        fitted_stripe: Fitted stripe with spline and aero params
        detection_confidence: YOLO detection confidence (0.5 for classical)
        n_valid_keypoints: Number of valid keypoints used
        total_keypoints: Total possible keypoints (8 for YOLO model)

    Returns:
        Confidence score in [0, 1]
    """
    # Detection confidence component
    det_conf = np.clip(detection_confidence, 0.0, 1.0)

    # Coverage ratio: fraction of keypoints that were valid
    coverage = n_valid_keypoints / max(total_keypoints, 1)
    coverage = np.clip(coverage, 0.0, 1.0)

    # Fit quality: based on airfoil validation
    validation = validate_airfoil_shape(
        fitted_stripe.spline_points,
        fitted_stripe.luff_endpoint,
        fitted_stripe.leech_endpoint
    )

    # Fit quality: 1.0 if valid, penalize for each violation
    n_violations = len(validation.get("violations", []))
    fit_quality = max(0.0, 1.0 - 0.25 * n_violations)

    confidence = 0.4 * det_conf + 0.3 * coverage + 0.3 * fit_quality
    return float(np.clip(confidence, 0.0, 1.0))
