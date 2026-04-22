"""Post-processing filters for stripe detection results.

Filters out false positives such as:
- Headstay (near-vertical line near luff)
- Battens (straight horizontal lines)
- Seams (features at sail boundary)
- Excessively steep or shallow stripes

All filters operate on lists of StripeDetection objects.
"""

import numpy as np
from typing import List
from src.types import StripeDetection, SailBoundary


def filter_by_orientation(
    detections: List[StripeDetection],
    max_angle_deg: float = 45.0
) -> List[StripeDetection]:
    """Filter stripe detections by orientation angle from horizontal.

    Rejects detections that are too vertical (headstay, rigging) or too steep.
    Real stripes on tilted sails can be 30-40 degrees from horizontal.

    Args:
        detections: List of StripeDetection objects to filter
        max_angle_deg: Maximum absolute angle from horizontal (degrees)

    Returns:
        Filtered list containing only detections within ±max_angle_deg of horizontal
    """
    filtered = []

    for detection in detections:
        # Check if orientation is within acceptable range
        if abs(detection.orientation_deg) <= max_angle_deg:
            filtered.append(detection)

    return filtered


def filter_by_curvature(
    detections: List[StripeDetection],
    min_curvature: float = 0.001,
    max_curvature: float = 0.05
) -> List[StripeDetection]:
    """Filter stripe detections by mean curvature.

    Rejects:
    - Straight lines (curvature too low) = battens
    - Highly curved lines (curvature too high) = noise, artifacts

    Stripes have gentle curvature due to airfoil shape.

    Args:
        detections: List of StripeDetection objects to filter
        min_curvature: Minimum mean curvature threshold
        max_curvature: Maximum mean curvature threshold

    Returns:
        Filtered list containing only detections with appropriate curvature
    """
    filtered = []

    for detection in detections:
        points = detection.points

        if len(points) < 3:
            # Need at least 3 points to compute curvature
            continue

        # Compute curvature using finite differences
        # Curvature at each point: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        # Simplified: use second derivative approximation

        # First derivatives (velocity)
        dx = np.diff(points[:, 0])
        dy = np.diff(points[:, 1])

        # Second derivatives (acceleration)
        if len(dx) < 2:
            continue

        ddx = np.diff(dx)
        ddy = np.diff(dy)

        # Curvature at each interior point using proper formula:
        # κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        # Use midpoint velocities aligned with accelerations
        dx_mid = dx[:-1]
        dy_mid = dy[:-1]
        numerator = np.abs(dx_mid * ddy - dy_mid * ddx)
        denominator = (dx_mid**2 + dy_mid**2) ** 1.5
        denominator = np.maximum(denominator, 1e-10)  # avoid division by zero
        curvatures = numerator / denominator

        # Mean curvature
        mean_curvature = np.mean(curvatures)

        # Keep if curvature is in acceptable range
        if min_curvature <= mean_curvature <= max_curvature:
            filtered.append(detection)

    return filtered


def filter_by_location(
    detections: List[StripeDetection],
    mask: np.ndarray,
    boundary_margin_pct: float = 5.0
) -> List[StripeDetection]:
    """Filter out detections near the sail boundary.

    Rejects detections whose center point is within boundary_margin_pct
    of the mask edge. This removes foot seams and edge artifacts.

    Args:
        detections: List of StripeDetection objects to filter
        mask: Binary sail mask (H, W) bool array
        boundary_margin_pct: Margin as percentage of mask dimensions

    Returns:
        Filtered list containing only interior detections
    """
    filtered = []

    # Compute eroded mask (exclude boundary region)
    # Erosion size based on percentage of image dimensions
    h, w = mask.shape
    margin_px = int(min(h, w) * boundary_margin_pct / 100.0)

    if margin_px < 1:
        # No erosion needed, return all detections
        return detections

    # Erode mask using morphological operation
    import cv2
    kernel = np.ones((margin_px * 2, margin_px * 2), dtype=np.uint8)
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    for detection in detections:
        # Check that majority of points (>60%) lie inside the eroded mask
        points = detection.points.astype(int)
        inside_count = 0
        total_valid = 0

        for pt in points:
            if 0 <= pt[1] < h and 0 <= pt[0] < w:
                total_valid += 1
                if eroded_mask[pt[1], pt[0]]:
                    inside_count += 1

        if total_valid > 0 and (inside_count / total_valid) >= 0.6:
            filtered.append(detection)

    return filtered


def reject_headstay(
    detections: List[StripeDetection],
    sail_boundary: SailBoundary
) -> List[StripeDetection]:
    """Reject headstay (near-vertical line near the luff).

    The headstay is a structural line running from tack to head on the
    leading edge. It appears as a near-vertical white line clustered
    near the luff polyline.

    Rejection criteria:
    1. Most points are within 20px of luff polyline
    2. Orientation is steep (>40 degrees from horizontal)

    Args:
        detections: List of StripeDetection objects to filter
        sail_boundary: SailBoundary with luff_polyline

    Returns:
        Filtered list with headstay removed
    """
    filtered = []

    luff_polyline = sail_boundary.luff_polyline

    for detection in detections:
        # Check orientation first (fast rejection)
        if abs(detection.orientation_deg) < 40.0:
            # Not steep enough to be headstay
            filtered.append(detection)
            continue

        # Compute distance of each point to luff polyline
        points = detection.points
        distances = []

        for point in points:
            # Find minimum distance to any segment of luff polyline
            min_dist = float('inf')

            for i in range(len(luff_polyline) - 1):
                p1 = luff_polyline[i]
                p2 = luff_polyline[i + 1]

                # Distance from point to line segment p1-p2
                # Vector from p1 to p2
                segment = p2 - p1
                segment_length = np.linalg.norm(segment)

                if segment_length < 1e-6:
                    # Degenerate segment
                    dist = np.linalg.norm(point - p1)
                else:
                    # Normalized direction
                    segment_unit = segment / segment_length

                    # Project point onto line
                    t = np.dot(point - p1, segment_unit)
                    t = np.clip(t, 0, segment_length)

                    # Closest point on segment
                    closest = p1 + t * segment_unit

                    # Distance
                    dist = np.linalg.norm(point - closest)

                min_dist = min(min_dist, dist)

            distances.append(min_dist)

        # Check if most points (>70%) are within 40px of luff
        close_to_luff = sum(d < 40.0 for d in distances)
        close_ratio = close_to_luff / len(distances)

        if close_ratio > 0.7:
            # This is likely the headstay, reject it
            continue
        else:
            # Not headstay
            filtered.append(detection)

    return filtered


def apply_all_filters(
    detections: List[StripeDetection],
    mask: np.ndarray,
    sail_boundary: SailBoundary,
    config: dict = None
) -> List[StripeDetection]:
    """Apply all post-processing filters in sequence.

    Filter order:
    1. Orientation (reject vertical lines)
    2. Curvature (reject straight battens and noisy curves)
    3. Location (reject boundary features)
    4. Headstay (reject near-luff vertical features)
    5. Sort top-to-bottom by mean y-coordinate
    6. Limit to max_stripes

    Args:
        detections: Raw stripe detections from detector
        mask: Binary sail mask
        sail_boundary: SailBoundary with luff/leech polylines
        config: Configuration dict with filter parameters (optional)

    Returns:
        Filtered, sorted, limited list of StripeDetection objects
    """
    if config is None:
        config = {}

    # Extract filter parameters from config
    max_angle = config.get('max_orientation_deg', 45.0)
    min_curv = config.get('min_curvature', 0.001)
    max_curv = config.get('max_curvature', 0.05)
    boundary_margin = config.get('boundary_margin_pct', 5.0)
    max_stripes = config.get('max_stripes', 5)

    # Apply filters in sequence
    result = detections
    detection_method = config.get('method', 'classical')

    # Skip orientation filter for ML-based methods (stripes on tilted sails
    # can exceed 45 degrees and should not be rejected)
    if detection_method not in ('keypoint', 'hybrid', 'ensemble', 'segmentation', 'roboflow'):
        result = filter_by_orientation(result, max_angle_deg=max_angle)

    # Skip curvature and location filters for ML keypoint detections:
    # - Curvature: YOLO already validates stripe shape via confidence
    # - Location: ML keypoints span luff-to-leech, so points near edges are expected
    if detection_method not in ('keypoint', 'segmentation', 'hybrid', 'ensemble', 'roboflow'):
        result = filter_by_curvature(result, min_curvature=min_curv, max_curvature=max_curv)
        result = filter_by_location(result, mask, boundary_margin_pct=boundary_margin)
    # Skip orientation and headstay filters for ML-based methods
    if detection_method not in ('keypoint', 'hybrid', 'ensemble', 'segmentation', 'roboflow'):
        result = reject_headstay(result, sail_boundary)

    # Sort by vertical position (top to bottom)
    # Mean y-coordinate (smaller y = higher up)
    result = sorted(result, key=lambda det: np.mean(det.points[:, 1]))

    # Limit to max_stripes
    result = result[:max_stripes]

    return result
