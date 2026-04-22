"""Camera calibration module for sail photos.

Provides automatic camera calibration using AnyCalib (deep learning-based)
with OpenCV fallback for environments where AnyCalib is unavailable.

Functions:
    calibrate_image: Estimate camera intrinsics from a single image
    undistort_image: Remove lens distortion using calibration results
"""

import numpy as np
import cv2
import warnings
from typing import Optional
from src.types import CalibrationResult


# Lazy model loading to avoid loading heavy models when not needed
_anycalib_model = None
_anycalib_device = None


def _get_anycalib_model(model_id: str = "anycalib_pinhole"):
    """Lazy load AnyCalib model (singleton pattern).

    Args:
        model_id: AnyCalib model identifier

    Returns:
        Tuple of (model, device) or (None, None) if unavailable
    """
    global _anycalib_model, _anycalib_device

    if _anycalib_model is not None:
        return _anycalib_model, _anycalib_device

    try:
        import torch
        from anycalib import AnyCalib

        # Select device: cuda > mps > cpu
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Load model
        model = AnyCalib(model_id=model_id).to(device)
        _anycalib_model = model
        _anycalib_device = device

        return model, device

    except ImportError:
        warnings.warn(
            "AnyCalib not found. Install with: "
            "pip install -e '/Users/hugostubler/Documents/Quantum Sport Analytics/"
            "Tennis/AI-pipeline/Camera_calibration/AnyCalib-main'"
        )
        return None, None
    except Exception as e:
        warnings.warn(f"Failed to load AnyCalib: {e}")
        return None, None


def calibrate_image(
    image: np.ndarray,
    method: str = "anycalib",
    model_id: str = "anycalib_pinhole",
    camera_model: str = "radial:2"
) -> CalibrationResult:
    """Calibrate camera from a single image using deep learning or classical methods.

    Estimates camera intrinsic parameters (focal length, principal point, distortion)
    and returns an undistorted version of the input image.

    Args:
        image: Input RGB image (H, W, 3) uint8
        method: Calibration method - "anycalib" or "opencv"
        model_id: AnyCalib model identifier (if method="anycalib")
        camera_model: Camera distortion model - "radial:2" or "pinhole"

    Returns:
        CalibrationResult with camera_matrix, dist_coeffs, focal_length,
        principal_point, and undistorted_image
    """
    h, w = image.shape[:2]

    if method == "anycalib":
        # Try AnyCalib first
        model, device = _get_anycalib_model(model_id)

        if model is not None and device is not None:
            try:
                import torch

                # Convert image to torch tensor
                image_torch = torch.tensor(
                    image, dtype=torch.float32, device=device
                ).permute(2, 0, 1) / 255.0

                # Add batch dimension
                image_torch = image_torch.unsqueeze(0)

                # Predict intrinsics
                with torch.no_grad():
                    output = model.predict(image_torch, cam_id=camera_model)

                # Extract intrinsics
                intrinsics = output["intrinsics"].detach().cpu().numpy().squeeze()

                # Parse based on camera model
                if camera_model == "radial:2":
                    # Format: [fx, fy, cx, cy, k1, k2]
                    fx, fy, cx, cy, k1, k2 = intrinsics
                    dist_coeffs = np.array([k1, k2, 0.0, 0.0, 0.0])
                elif camera_model == "pinhole":
                    # Format: [fx, fy, cx, cy]
                    fx, fy, cx, cy = intrinsics
                    dist_coeffs = np.zeros(5)
                else:
                    raise ValueError(f"Unsupported camera_model: {camera_model}")

                # Build camera matrix
                camera_matrix = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float64)

                focal_length = (float(fx), float(fy))
                principal_point = (float(cx), float(cy))

                # Undistort image
                undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)

                return CalibrationResult(
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    focal_length=focal_length,
                    principal_point=principal_point,
                    undistorted_image=undistorted
                )

            except Exception as e:
                warnings.warn(f"AnyCalib inference failed: {e}. Falling back to OpenCV.")
                # Fall through to OpenCV fallback

    # OpenCV fallback (or method == "opencv")
    # Estimate reasonable default parameters for modern phone cameras
    focal_length_px = max(h, w)  # Typical for phone cameras
    cx = w / 2.0
    cy = h / 2.0

    camera_matrix = np.array([
        [focal_length_px, 0, cx],
        [0, focal_length_px, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    # Assume no distortion for fallback
    dist_coeffs = np.zeros(5, dtype=np.float64)

    focal_length = (float(focal_length_px), float(focal_length_px))
    principal_point = (float(cx), float(cy))

    # Undistort (will be no-op with zero distortion, but kept for consistency)
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)

    return CalibrationResult(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        focal_length=focal_length,
        principal_point=principal_point,
        undistorted_image=undistorted
    )


def undistort_image(
    image: np.ndarray,
    calib: CalibrationResult
) -> np.ndarray:
    """Remove lens distortion from an image using calibration parameters.

    Args:
        image: Input image (H, W, 3) uint8
        calib: CalibrationResult from calibrate_image

    Returns:
        Undistorted image of same shape as input
    """
    return cv2.undistort(image, calib.camera_matrix, calib.dist_coeffs)
