"""Sail segmentation and boundary extraction module.

Provides sail detection from photos using SAM2 (Segment Anything Model 2)
with classical segmentation fallback.

Functions:
    segment_sail: Detect and segment sail from image
    extract_boundary: Extract sail boundary polylines from mask
"""

import numpy as np
import cv2
import warnings
from typing import Optional
from skimage import measure
from src.types import SailBoundary
from src.utils.geometry import split_contour_luff_leech


# Lazy model loading to avoid loading heavy models when not needed
_sam2_model = None


def _get_sam2_model(model_path: str = "sam2.1_b.pt"):
    """Lazy load SAM2 model (singleton pattern).

    Args:
        model_path: Path to SAM2 model file

    Returns:
        SAM model or None if unavailable
    """
    global _sam2_model

    if _sam2_model is not None:
        return _sam2_model

    try:
        from ultralytics import SAM

        # Load model
        model = SAM(model_path)
        _sam2_model = model
        return model

    except ImportError:
        warnings.warn(
            "ultralytics SAM not found. Install with: pip install ultralytics>=8.3.0"
        )
        return None
    except Exception as e:
        warnings.warn(f"Failed to load SAM2 model: {e}")
        return None


def _find_sail_prompt(image: np.ndarray):
    """Find sail location using classical segmentation for SAM2 prompting.

    Uses Otsu threshold + largest connected component to find the sail's
    centroid and bounding box, which are then used to prompt SAM2 for
    more focused segmentation.

    Args:
        image: Input RGB image (H, W, 3) uint8

    Returns:
        (centroid, bbox) where:
            centroid: (x, y) center of sail region
            bbox: (x1, y1, x2, y2) bounding box of sail region
        Returns (None, None) if no sail-like region found.
    """
    try:
        mask = _classical_segment(image)

        # Check that mask covers a reasonable portion of image (>5% area)
        area_ratio = np.sum(mask) / mask.size
        if area_ratio < 0.05 or area_ratio > 0.95:
            return None, None

        # Find bounding box of mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None, None

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Centroid
        ys, xs = np.where(mask)
        centroid = (float(np.mean(xs)), float(np.mean(ys)))

        # Bounding box with small margin (5% of dimension)
        h, w = image.shape[:2]
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        bbox = (
            max(0, int(x_min) - margin_x),
            max(0, int(y_min) - margin_y),
            min(w, int(x_max) + margin_x),
            min(h, int(y_max) + margin_y)
        )

        return centroid, bbox
    except Exception:
        return None, None


def _extract_masks_from_results(results, image_shape):
    """Extract masks as numpy arrays from SAM2 results.

    Args:
        results: SAM2 ultralytics results
        image_shape: (H, W) of original image

    Returns:
        List of bool masks, or empty list if none found
    """
    if len(results) == 0 or not hasattr(results[0], 'masks'):
        return []

    masks_data = results[0].masks
    if masks_data is None or len(masks_data) == 0:
        return []

    masks = []
    if hasattr(masks_data, 'data'):
        import torch
        masks_tensor = masks_data.data.cpu().numpy()
        for i in range(len(masks_tensor)):
            mask = masks_tensor[i]
            if mask.shape != image_shape:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image_shape[1], image_shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                mask = mask.astype(bool)
            masks.append(mask)
    else:
        for mask in masks_data:
            if isinstance(mask, np.ndarray):
                masks.append(mask.astype(bool))

    return masks


def _score_sail_mask(mask: np.ndarray, gray: np.ndarray) -> float:
    """Score a candidate sail mask by area, brightness, and compactness.

    Args:
        mask: Binary mask (H, W) bool
        gray: Grayscale image (H, W) uint8

    Returns:
        Score (higher = better sail candidate), 0 if rejected
    """
    area = np.sum(mask)
    total = mask.size
    area_pct = area / total

    # Reject too small or too large
    if area_pct < 0.08 or area_pct > 0.80:
        return 0.0

    # Reject if median brightness is too high (sky)
    masked_pixels = gray[mask]
    if len(masked_pixels) == 0:
        return 0.0
    median_brightness = np.median(masked_pixels)
    if median_brightness > 175:
        return 0.0

    brightness_score = 1.0 - (median_brightness / 255.0)

    # Compactness: 4*pi*area / perimeter^2
    contours = measure.find_contours(mask.astype(np.uint8), 0.5)
    if not contours:
        return 0.0
    longest = max(contours, key=len)
    perimeter = np.sum(np.linalg.norm(np.diff(longest, axis=0), axis=1))
    if perimeter < 1e-6:
        return 0.0
    compactness = 4 * np.pi * area / (perimeter ** 2)

    return area_pct * compactness * brightness_score


def _find_best_mask_multipoint(
    model,
    image: np.ndarray,
    gray: np.ndarray,
    grid_size: int = 5
) -> Optional[np.ndarray]:
    """Find best sail mask using grid of point prompts.

    Generates a grid of point prompts across the image, runs SAM2 on each,
    and returns the best-scoring mask.

    Args:
        model: SAM2 model
        image: RGB image (H, W, 3) uint8
        gray: Grayscale image (H, W) uint8
        grid_size: Grid dimension (grid_size x grid_size points)

    Returns:
        Best sail mask or None
    """
    h, w = image.shape[:2]
    image_shape = (h, w)

    best_mask = None
    best_score = 0.0

    # Generate grid points at 20/35/50/65/80% of dimensions
    fractions = np.linspace(0.2, 0.8, grid_size)
    for fx in fractions:
        for fy in fractions:
            px, py = int(w * fx), int(h * fy)
            try:
                results = model(image, points=[[px, py]], labels=[1])
                masks = _extract_masks_from_results(results, image_shape)
                for mask in masks:
                    score = _score_sail_mask(mask, gray)
                    if score > best_score:
                        best_score = score
                        best_mask = mask
            except Exception:
                continue

    return best_mask


def _refine_sail_mask(
    mask: np.ndarray,
    cleanup_kernel_pct: float = 2.5,
    fill_kernel_pct: float = 12.0,
    fill_top_concavities: bool = False,
    top_band_frac: float = 0.08,
) -> np.ndarray:
    """Clean up sail mask with morphological operations.

    Removes rigging/stays with opening, keeps largest component, fills
    inward notches with an aggressive closing pass. The close kernel is
    sized independently so it can fill larger holes (e.g. SAM artefacts
    where it grabs sky/headboard hardware in the upper sail) without
    affecting the open kernel that strips rigging.

    Args:
        mask: Binary mask (H, W) bool
        cleanup_kernel_pct: Open kernel size as % of image diagonal
            (controls how thin a protrusion is removed; bigger = strips
            more rigging but also nibbles the sail edge).
        fill_kernel_pct: Close kernel size as % of image diagonal
            (controls how wide a notch can be filled; bigger = fills
            larger SAM notches but may bridge gaps to neighbouring sails).

    Returns:
        Refined binary mask (H, W) bool
    """
    h, w = mask.shape
    diagonal = np.sqrt(h**2 + w**2)

    def _odd_kernel(pct: float) -> int:
        k = max(3, int(diagonal * pct / 100.0))
        return k + 1 if k % 2 == 0 else k

    open_size = _odd_kernel(cleanup_kernel_pct)
    close_size = _odd_kernel(fill_kernel_pct)

    mask_uint8 = mask.astype(np.uint8)
    original_area = np.sum(mask_uint8)

    # Morphological opening (removes thin protrusions like rigging)
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_size, open_size)
    )
    opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, open_kernel)

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if num_labels <= 1:
        # Opening removed everything — skip cleanup
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    cleaned = (labels == largest_label).astype(np.uint8)

    # Skip cleanup if area decreased too much (>30%)
    cleaned_area = np.sum(cleaned)
    if cleaned_area < 0.7 * original_area:
        return mask

    # Morphological closing (fill inward notches in the sail boundary)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    # Convex-hull pass on the top band to fill notches the close kernel
    # cannot reach. Sails are convex above the tack-clew foot line — any
    # inward concavity in this band is a SAM artefact (sky bleed,
    # headboard hardware, mast/boom shadow). The bottom 70% is left
    # untouched so concave foot curves on mainsails survive.
    if fill_top_concavities:
        cleaned = _fill_concavities_in_top_band(cleaned, top_band_frac)

    return cleaned.astype(bool)


def _fill_concavities_in_top_band(mask: np.ndarray, top_band_frac: float) -> np.ndarray:
    """Replace the upper portion of the mask with its convex hull, leave
    the lower portion (foot) untouched.

    Args:
        mask: uint8 binary mask (H, W)
        top_band_frac: fraction of the mask's vertical extent (from top)
            to convex-hullify. 0.30 means top 30 % of the sail's bounding
            rows.
    """
    if mask is None or mask.size == 0:
        return mask
    rows = np.where(mask.any(axis=1))[0]
    if len(rows) < 2:
        return mask
    top_row, bot_row = int(rows[0]), int(rows[-1])
    extent = bot_row - top_row
    if extent < 10:
        return mask
    cutoff_row = top_row + int(top_band_frac * extent)

    # Carve out the upper band, take its convex hull, paste back
    upper = mask.copy()
    upper[cutoff_row + 1 :, :] = 0
    if not upper.any():
        return mask
    ys, xs = np.where(upper)
    pts = np.column_stack([xs, ys]).astype(np.int32)
    if len(pts) < 3:
        return mask
    try:
        hull = cv2.convexHull(pts)
    except cv2.error:
        return mask
    hull_mask = np.zeros_like(mask)
    cv2.fillPoly(hull_mask, [hull], 1)
    # Restrict the hull fill to the upper band only — anything below the
    # cutoff comes from the original mask, untouched.
    out = mask.copy()
    out[: cutoff_row + 1, :] = np.maximum(
        mask[: cutoff_row + 1, :], hull_mask[: cutoff_row + 1, :]
    )
    return out


def segment_sail(
    image: np.ndarray,
    model_path: str = "sam2.1_b.pt",
    prompt_strategy: str = "auto",
    grid_size: int = 5,
    mask_cleanup: bool = True
) -> SailBoundary:
    """Segment sail from image using SAM2 with prompted mode and classical fallback.

    Prompt strategy "auto" (default):
    1. Multi-point grid (best mask across grid of prompts)
    2. Bbox-prompted SAM2
    3. Point-prompted SAM2
    4. Classical segmentation only

    All strategies pass through mask refinement before boundary extraction.

    Args:
        image: Input RGB image (H, W, 3) uint8
        model_path: Path to SAM2 model file
        prompt_strategy: "auto" or "multi_point" or "unprompted"
        grid_size: Grid size for multi-point prompting
        mask_cleanup: Whether to apply morphological cleanup

    Returns:
        SailBoundary with mask, contour, luff/leech polylines, and corner points
    """
    model = _get_sam2_model(model_path)
    image_shape = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

    def _finalize_mask(mask):
        if mask_cleanup:
            mask = _refine_sail_mask(mask)
        return extract_boundary(mask)

    if model is not None:
        # Find sail prompt from classical segmentation
        centroid, bbox = None, None
        if prompt_strategy in ("auto", "multi_point"):
            centroid, bbox = _find_sail_prompt(image)

        # Strategy 1: multi-point grid (primary)
        if prompt_strategy in ("auto", "multi_point"):
            try:
                best_mask = _find_best_mask_multipoint(model, image, gray, grid_size)
                if best_mask is not None:
                    return _finalize_mask(best_mask)
            except Exception as e:
                warnings.warn(f"SAM2 multi-point failed: {e}")

        # Strategy 2: bbox-prompted SAM2
        if bbox is not None:
            try:
                results = model(image, bboxes=[list(bbox)])
                masks = _extract_masks_from_results(results, image_shape)
                if masks:
                    largest_mask = masks[np.argmax([np.sum(m) for m in masks])]
                    return _finalize_mask(largest_mask)
            except Exception as e:
                warnings.warn(f"SAM2 bbox-prompted failed: {e}")

        # Strategy 3: point-prompted SAM2
        if centroid is not None:
            try:
                results = model(image, points=[list(centroid)], labels=[1])
                masks = _extract_masks_from_results(results, image_shape)
                if masks:
                    largest_mask = masks[np.argmax([np.sum(m) for m in masks])]
                    return _finalize_mask(largest_mask)
            except Exception as e:
                warnings.warn(f"SAM2 point-prompted failed: {e}")

        # Strategy 4: unprompted SAM2
        try:
            results = model(image)
            masks = _extract_masks_from_results(results, image_shape)
            if masks:
                largest_mask = masks[np.argmax([np.sum(m) for m in masks])]
                return _finalize_mask(largest_mask)
        except Exception as e:
            warnings.warn(f"SAM2 unprompted failed: {e}. Falling back to classical.")

    # Strategy 5: classical fallback
    mask = _classical_segment(image)
    if mask_cleanup:
        mask = _refine_sail_mask(mask)
    return extract_boundary(mask)


def _classical_segment(image: np.ndarray) -> np.ndarray:
    """Classical segmentation fallback using Otsu thresholding.

    Args:
        image: Input RGB image (H, W, 3) uint8

    Returns:
        Binary mask (H, W) bool where True = sail pixels
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # Find largest component (excluding background label 0)
    if num_labels <= 1:
        # No components found, return the binary mask as-is
        return binary.astype(bool)

    areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
    largest_label = np.argmax(areas) + 1  # +1 to account for skipping background

    # Create mask of largest component
    mask = (labels == largest_label)

    return mask


def extract_boundary(mask: np.ndarray) -> SailBoundary:
    """Extract sail boundary polylines from binary mask.

    Finds the contour of the mask, identifies corner points (tack, head, clew),
    and splits the boundary into luff and leech edges.

    Args:
        mask: Binary mask (H, W) bool where True = sail pixels

    Returns:
        SailBoundary with all boundary components
    """
    # Pad the mask by one pixel of background so contours stay closed even
    # when the sail touches an image edge. The longest closed contour then
    # traces the full sail outline; we drop any segment lying on the
    # (synthetic) image border so luff/leech classification only operates
    # on real visible sail boundary.
    h_mask, w_mask = mask.shape
    padded = np.zeros((h_mask + 2, w_mask + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask.astype(np.uint8)
    contours = measure.find_contours(padded, 0.5)

    if len(contours) == 0:
        raise ValueError("No contours found in mask")

    longest_idx = np.argmax([len(c) for c in contours])
    contour = contours[longest_idx]

    # (row, col) -> (x, y), undo the 1-pixel pad shift
    contour = np.fliplr(contour) - np.array([1.0, 1.0])

    # Flag image-edge points (within 2 px of any image border) and drop
    # them from the contour. Keep only the "visible sail" segments.
    edge_tol = 2.0
    on_edge = (
        (contour[:, 0] <= edge_tol)
        | (contour[:, 0] >= w_mask - 1 - edge_tol)
        | (contour[:, 1] <= edge_tol)
        | (contour[:, 1] >= h_mask - 1 - edge_tol)
    )
    if on_edge.all():
        # Degenerate: mask fills image. Fall back to raw contour.
        contour[:, 0] = np.clip(contour[:, 0], 0, w_mask - 1)
        contour[:, 1] = np.clip(contour[:, 1], 0, h_mask - 1)
    else:
        # The closed contour visits image edges in one or more runs.
        # Roll the array so it starts at the first NON-edge point, then
        # split by runs of on-edge values. Keep the longest non-edge run.
        first_interior = int(np.argmax(~on_edge))
        contour = np.roll(contour, -first_interior, axis=0)
        on_edge = np.roll(on_edge, -first_interior)

        # Build runs of non-edge segments
        segments: list[np.ndarray] = []
        i = 0
        n = len(contour)
        while i < n:
            if on_edge[i]:
                i += 1
                continue
            j = i
            while j < n and not on_edge[j]:
                j += 1
            segments.append(contour[i:j])
            i = j

        if segments:
            longest_seg = max(segments, key=len)
            contour = longest_seg
        contour[:, 0] = np.clip(contour[:, 0], 0, w_mask - 1)
        contour[:, 1] = np.clip(contour[:, 1], 0, h_mask - 1)

        # Mask-fills-frame fix: when the sail extends below the visible
        # head ridge (i.e. the bottom of the mask is far below the kept
        # contour), the cropped head-ridge segment alone gives degenerate
        # polylines. Extend the contour by tracing the leftmost/rightmost
        # mask pixel in each row from the head-ridge endpoints down to the
        # bottom of the mask. This produces an OPEN contour spanning
        # bottom-left → head → bottom-right, so split_contour_luff_leech
        # places tack/clew at the bottom of the visible sail and the luff/
        # leech polylines run down the sides of the mask.
        ys_mask, xs_mask = np.where(mask)
        if len(ys_mask) > 0:
            y_max_mask = int(ys_mask.max())
            ridge_y_max = float(contour[:, 1].max())
            if (y_max_mask - ridge_y_max) > 0.15 * h_mask:
                order = np.argsort(contour[:, 0])
                head_ridge = contour[order]
                y_left = float(head_ridge[0, 1])
                y_right = float(head_ridge[-1, 1])

                left_pts = []
                for yi in range(int(y_left) + 1, y_max_mask + 1):
                    row = mask[yi]
                    if row.any():
                        left_pts.append([int(np.argmax(row)), yi])
                right_pts = []
                for yi in range(int(y_right) + 1, y_max_mask + 1):
                    row = mask[yi]
                    if row.any():
                        right_pts.append([
                            int(len(row) - 1 - np.argmax(row[::-1])),
                            yi,
                        ])

                left_arr = (np.array(left_pts, dtype=np.float64)
                            if left_pts else np.empty((0, 2)))
                right_arr = (np.array(right_pts, dtype=np.float64)
                             if right_pts else np.empty((0, 2)))

                parts = []
                if len(left_arr):
                    parts.append(left_arr[::-1])  # bottom -> top
                parts.append(head_ridge)           # left -> right
                if len(right_arr):
                    parts.append(right_arr)        # top -> bottom
                contour = np.vstack(parts)
                contour[:, 0] = np.clip(contour[:, 0], 0, w_mask - 1)
                contour[:, 1] = np.clip(contour[:, 1], 0, h_mask - 1)

    # Split into luff, leech, and foot
    luff_polyline, leech_polyline, foot_polyline, head_point, tack_point, clew_point = \
        split_contour_luff_leech(contour)

    return SailBoundary(
        mask=mask,
        contour=contour,
        luff_polyline=luff_polyline,
        leech_polyline=leech_polyline,
        head_point=head_point,
        tack_point=tack_point,
        clew_point=clew_point,
        foot_polyline=foot_polyline
    )
