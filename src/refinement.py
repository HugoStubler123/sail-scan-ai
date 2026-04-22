"""Splender-based spline refinement for sail stripes.

Uses differentiable rendering to refine B-spline detections to sub-pixel accuracy.
Gracefully falls back to original spline if Splender/JAX unavailable.
"""

import numpy as np
import logging
from typing import Optional, Tuple, List

from src.physics import apply_airfoil_constraints, constrained_bspline_fit

logger = logging.getLogger(__name__)


def refine_stripe_splender(
    image: np.ndarray,
    spline_points: np.ndarray,
    luff_endpoint: np.ndarray,
    leech_endpoint: np.ndarray,
    config: Optional[dict] = None
) -> np.ndarray:
    """Refine spline using Splender differentiable rendering.

    Args:
        image: HxWx3 or HxW image array
        spline_points: Nx2 interior spline points
        luff_endpoint: 2-element luff position
        leech_endpoint: 2-element leech position
        config: Optional config with iterations, learning_rate, global_scale

    Returns:
        Refined spline points (Mx2), or original if Splender unavailable
    """
    if config is None:
        config = {}

    iterations = config.get("iterations", 200)
    learning_rate = config.get("learning_rate", 0.01)
    global_scale = config.get("global_scale", 0.4)

    # Try to import Splender and JAX
    try:
        import splender
        import splender.knot_init
        import splender.image
        import splender.optim
        import jax
        import jax.numpy as jnp
    except ImportError as e:
        logger.warning(f"Splender or JAX not available: {e}. Using original spline.")
        return spline_points

    try:
        # Build full spline: luff -> interior -> leech
        full_points = np.vstack([luff_endpoint, spline_points, leech_endpoint])

        # Get image dimensions
        if len(image.shape) == 3:
            h, w = image.shape[:2]
        else:
            h, w = image.shape

        # Initialize Splender knots from spline points
        # Downsample to reasonable number of control points
        num_knots = min(6, len(full_points) // 3 + 1)
        init_knots = splender.knot_init.downsample_points(full_points, num_knots=num_knots)

        # Normalize knots to [0, 1]
        init_knots_norm = init_knots.copy()
        init_knots_norm[:, 0] /= w
        init_knots_norm[:, 1] /= h

        # Create Splender model
        key = jax.random.PRNGKey(42)
        model = splender.image.SplenderImage(
            key=key,
            init_knots=init_knots_norm,
            res=max(h, w),
            global_scale=global_scale
        )

        # Normalize image to [0, 1] float32
        if image.dtype == np.uint8:
            image_norm = image.astype(np.float32) / 255.0
        else:
            image_norm = image.astype(np.float32)

        # Convert to grayscale if needed
        if len(image_norm.shape) == 3:
            image_norm = np.mean(image_norm, axis=2)

        # Convert to JAX array
        image_jax = jnp.array(image_norm)

        # Run Splender optimization
        optimized_model = splender.optim.fit(
            model,
            image_jax,
            video=False,
            # Can add iterations, learning_rate if Splender supports
        )

        # Extract refined spline from optimized model
        # Evaluate at many points for smooth curve
        u_fine = np.linspace(0, 1, 500)
        refined_spline = optimized_model.fit_spline

        # Get spline points (convert from JAX if needed)
        if hasattr(refined_spline, 'points'):
            refined_points_norm = np.array(refined_spline.points)
        elif hasattr(optimized_model, 'knots'):
            # Fallback: use knots as control points
            refined_points_norm = np.array(optimized_model.knots)
        else:
            # Can't extract, return original
            logger.warning("Cannot extract refined spline from Splender model")
            return spline_points

        # Convert back to pixel coordinates
        refined_points = refined_points_norm.copy()
        refined_points[:, 0] *= w
        refined_points[:, 1] *= h

        # Constraint: ensure endpoints stay within 5px of sail boundary
        # Check if first/last points drifted too far
        if len(refined_points) > 2:
            if np.linalg.norm(refined_points[0] - luff_endpoint) > 5:
                refined_points[0] = luff_endpoint
            if np.linalg.norm(refined_points[-1] - leech_endpoint) > 5:
                refined_points[-1] = leech_endpoint

        return refined_points

    except Exception as e:
        logger.warning(f"Splender refinement failed: {e}. Using original spline.")
        return spline_points


def refine_all_stripes(
    image: np.ndarray,
    stripe_data: List[Tuple],
    config: Optional[dict] = None
) -> List[Tuple]:
    """Refine all detected stripes with Splender and physics constraints.

    Args:
        image: HxWx3 or HxW image
        stripe_data: List of (detection, luff_endpoint, leech_endpoint) tuples
        config: Optional refinement config

    Returns:
        List of (refined_spline_points, luff_endpoint, leech_endpoint) tuples
    """
    refined_stripes = []

    for item in stripe_data:
        # Handle both 3-tuple and 4-tuple (with coefficients) formats
        if len(item) == 4:
            detection, luff_endpoint, leech_endpoint, _coeffs = item
        else:
            detection, luff_endpoint, leech_endpoint = item
        # Extract spline points from detection
        if hasattr(detection, 'points'):
            spline_points = detection.points
        else:
            # Assume detection is already points array
            spline_points = detection

        # Refine with Splender
        refined_spline = refine_stripe_splender(
            image, spline_points, luff_endpoint, leech_endpoint, config
        )

        # Apply physics constraints
        constrained_spline = apply_airfoil_constraints(
            refined_spline, luff_endpoint, leech_endpoint
        )

        # Refit Bernstein to get new coefficients after refinement
        try:
            new_coeffs, _, _ = constrained_bspline_fit(
                constrained_spline, luff_endpoint, leech_endpoint
            )
        except Exception:
            new_coeffs = None

        refined_stripes.append((constrained_spline, luff_endpoint, leech_endpoint, new_coeffs))

    return refined_stripes
