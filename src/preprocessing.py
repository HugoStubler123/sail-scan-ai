"""
Preprocessing module for sail stripe detection.

Provides color/lighting correction and stripe enhancement for sail regions.
"""

import numpy as np
import cv2


def correct_lighting(
    image: np.ndarray,
    mask: np.ndarray,
    clip_limit: float = 2.5,
    tile_size: int = 8
) -> np.ndarray:
    """
    Apply CLAHE-based lighting correction to the sail region.

    Args:
        image: Input BGR image (H, W, 3) uint8
        mask: Binary mask (H, W) bool indicating sail region
        clip_limit: CLAHE clip limit for contrast enhancement
        tile_size: Tile grid size for local adaptation

    Returns:
        Corrected BGR image with same shape and dtype as input
    """
    # Make a copy to avoid modifying the original
    result = image.copy()

    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Create CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )

    # Apply CLAHE to L channel
    l_enhanced = clahe.apply(l_channel)

    # Merge back
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])

    # Convert back to BGR
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Apply mask: only replace pixels where mask is True
    result[mask] = bgr_enhanced[mask]

    return result


def enhance_stripes(
    image: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Enhance stripe structures in the sail region.

    Creates a single-channel image emphasizing horizontal linear structures
    (stripes) while reducing noise and non-stripe features.

    Args:
        image: Input BGR image (H, W, 3) uint8
        mask: Binary mask (H, W) bool indicating sail region

    Returns:
        Enhanced single-channel image (H, W) uint8
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Compute gradient magnitude using Scharr operators (better for fine structures)
    grad_x = cv2.Scharr(filtered, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(filtered, cv2.CV_64F, 0, 1)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize gradient to 0-255
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

    # Apply morphological top-hat transform with horizontally-elongated kernel
    # This enhances thin horizontal structures (stripes) vs background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    tophat = cv2.morphologyEx(gradient_magnitude, cv2.MORPH_TOPHAT, kernel)

    # Apply mask to sail region only
    result = np.zeros(mask.shape, dtype=np.uint8)
    result[mask] = tophat[mask]

    # Normalize to full 0-255 range for better visibility
    if result.max() > 0:
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

    return result


def _adaptive_clip_limit(
    image: np.ndarray,
    mask: np.ndarray
) -> float:
    """
    Compute adaptive CLAHE clip limit based on image contrast.

    Args:
        image: Input BGR image (H, W, 3) uint8
        mask: Binary mask (H, W) bool indicating region of interest

    Returns:
        Recommended clip limit (1.5-4.0)
    """
    # Convert to LAB and extract L channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # Get masked region
    masked_l = l_channel[mask]

    # Compute standard deviation of L channel in masked region
    if len(masked_l) == 0:
        return 2.5  # Default for empty mask

    std = np.std(masked_l)

    # Adaptive logic:
    # - Low contrast (std < 20): use higher clip limit (3.0-4.0)
    # - Medium contrast (20 <= std <= 50): use moderate clip limit (2.0-3.0)
    # - High contrast (std > 50): use lower clip limit (1.5-2.0)

    if std < 20:
        # Low contrast (overcast, dim lighting)
        return 3.5
    elif std > 50:
        # High contrast (sunny with reflections)
        return 1.8
    else:
        # Medium contrast
        return 2.5
