"""Stripe detection module for sail analysis.

Provides multiple detection backends:
- Classical: Multi-scale Hessian ridge detection (no ML required)
- Keypoint model: YOLOv11-pose trained on annotated keypoints dataset
- Bbox local: YOLO11n bbox detection + multi-method local refinement
- Segmentation model: YOLOv11-seg trained on thin-line segmentation dataset
- Roboflow API: Existing trained model via API

All backends return list of StripeDetection objects with interior keypoints.
Post-processing filters reject headstay, battens, and boundary false positives.
"""

import numpy as np
import cv2
from typing import List, Optional
from scipy.spatial import cKDTree
from src.types import StripeDetection, SailBoundary
from src.utils.filtering import apply_all_filters


class StripeDetector:
    """Stripe detector with configurable backend and filtering.

    Supports multiple detection methods with automatic fallback to classical
    approach if ML models unavailable.
    """

    def __init__(self, config: dict = None):
        """Initialize stripe detector with configuration.

        Args:
            config: Configuration dictionary with keys:
                - method: 'classical', 'keypoint', 'bbox_local', 'segmentation', 'roboflow'
                - max_stripes: Maximum number of stripes to return (default 5)
                - min_confidence: Minimum detection confidence (default 0.3)
                - filter parameters for post-processing
        """
        if config is None:
            config = {}

        self.config = config
        self.method = config.get('method', 'classical')

    def detect(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        sail_boundary: SailBoundary
    ) -> List[StripeDetection]:
        """Detect stripes in sail image.

        Args:
            image: Input BGR image (H, W, 3) uint8
            mask: Binary sail mask (H, W) bool
            sail_boundary: SailBoundary with luff/leech polylines

        Returns:
            List of StripeDetection objects sorted top-to-bottom
        """
        # Dispatch to appropriate backend
        if self.method == 'classical':
            raw_detections = _detect_classical(image, mask)
        elif self.method == 'keypoint':
            raw_detections = _detect_from_keypoints_model(image, mask, self.config)
        elif self.method == 'bbox_local':
            raw_detections = _detect_bbox_local(image, mask, sail_boundary, self.config)
        elif self.method == 'hybrid':
            raw_detections = _detect_hybrid(image, mask, sail_boundary, self.config)
        elif self.method == 'ensemble':
            raw_detections = _detect_ensemble(image, mask, sail_boundary, self.config)
        elif self.method == 'segmentation':
            raw_detections = _detect_from_segmentation_model(image, mask, self.config)
        elif self.method == 'roboflow':
            raw_detections = _detect_roboflow(image, self.config)
        else:
            # Unknown method, fallback to classical
            raw_detections = _detect_classical(image, mask)

        # Apply post-processing filters
        filtered_detections = apply_all_filters(
            raw_detections,
            mask,
            sail_boundary,
            self.config
        )

        return filtered_detections


def detect_stripes(
    image: np.ndarray,
    mask: np.ndarray,
    sail_boundary: SailBoundary,
    method: str = "classical",
    config: dict = None
) -> List[StripeDetection]:
    """Convenience function for stripe detection.

    Args:
        image: Input BGR image (H, W, 3) uint8
        mask: Binary sail mask (H, W) bool
        sail_boundary: SailBoundary with luff/leech polylines
        method: Detection method ('classical', 'keypoint', 'bbox_local', 'segmentation', 'roboflow')
        config: Optional configuration dict

    Returns:
        List of StripeDetection objects sorted top-to-bottom
    """
    if config is None:
        config = {}

    config['method'] = method

    detector = StripeDetector(config)
    return detector.detect(image, mask, sail_boundary)


def _detect_classical(
    image: np.ndarray,
    mask: np.ndarray
) -> List[StripeDetection]:
    """Classical stripe detection using Meijering neuriteness filter.

    Uses the Meijering filter (designed for continuous curvilinear structures)
    which is more selective than Hessian for ridge-like features.
    Adapts to sail brightness: bright ridges on dark sails, dark ridges on light sails.

    Args:
        image: Input BGR image (H, W, 3) uint8
        mask: Binary sail mask (H, W) bool

    Returns:
        List of StripeDetection objects with interior keypoints
    """
    from skimage.filters import meijering
    from skimage.morphology import skeletonize
    from skimage.measure import label, regionprops

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Erode mask to exclude boundary edge responses
    kernel_size = max(5, int(min(image.shape[:2]) * 0.03))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded_mask = cv2.erode(mask.astype(np.uint8), erode_kernel).astype(bool)

    # Determine sail color: dark sail → detect bright ridges, light sail → detect dark
    sail_median = np.median(gray[mask])
    black_ridges = sail_median > 128  # Light sail → look for dark stripes

    # Normalize to float [0, 1] for skimage
    gray_f = gray.astype(np.float64) / 255.0

    # Meijering neuriteness filter: tuned for stripe widths 2-6px
    sigmas = range(1, 4)
    ridge_response = meijering(gray_f, sigmas=sigmas, black_ridges=black_ridges)

    # Apply eroded mask
    ridge_response[~eroded_mask] = 0

    # Adaptive threshold within masked region
    masked_values = ridge_response[eroded_mask]
    if len(masked_values) == 0 or masked_values.max() == 0:
        return []

    # Use top 15% of ridge responses
    nonzero = masked_values[masked_values > 0]
    if len(nonzero) == 0:
        return []
    threshold = np.percentile(nonzero, 85)
    ridge_binary = (ridge_response > threshold).astype(np.uint8)

    # Bridge small horizontal gaps
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    ridge_binary = cv2.morphologyEx(ridge_binary, cv2.MORPH_CLOSE, close_kernel)

    # Skeletonize to get 1-pixel wide centerlines
    skeleton = skeletonize(ridge_binary > 0)

    # Trace connected components
    labeled = label(skeleton)
    regions = regionprops(labeled)

    # Minimum length scales with image size
    min_length = max(30, int(min(image.shape[:2]) * 0.05))

    detections = []

    for region in regions:
        coords = region.coords  # (row, col) = (y, x)

        if len(coords) < min_length:
            continue

        # Convert to (x, y) format
        points = np.column_stack([coords[:, 1], coords[:, 0]])

        # Order points along skeleton path
        points = _trace_skeleton_path(points)

        # Subsample to ~10-15 evenly-spaced keypoints
        num_points = min(15, len(points))
        if num_points < 5:
            continue

        indices = np.linspace(0, len(points) - 1, num_points).astype(int)
        sampled_points = points[indices]

        # Compute orientation (angle of fitted line)
        if len(sampled_points) >= 2:
            x = sampled_points[:, 0]
            y = sampled_points[:, 1]

            if np.std(x) > 1e-6:
                coeffs = np.polyfit(x, y, 1)
                slope = coeffs[0]
                orientation_deg = np.arctan(slope) * 180.0 / np.pi
            else:
                orientation_deg = 90.0
        else:
            orientation_deg = 0.0

        detection = StripeDetection(
            points=sampled_points,
            confidence=0.5,
            orientation_deg=orientation_deg
        )

        detections.append(detection)

    return detections


def _trace_skeleton_path(points: np.ndarray) -> np.ndarray:
    """Order skeleton points along a smooth path using greedy nearest-neighbor walk.

    Uses adjacency graph with greedy walk that prefers straight-ahead
    neighbors to avoid zigzag artifacts at junction points.
    Falls back to x-sort for features without clear endpoints.

    Args:
        points: Unordered skeleton points (N x 2) in (x, y) format

    Returns:
        Points ordered along the skeleton path (N x 2)
    """
    if len(points) < 3:
        return points

    # Build adjacency graph using cKDTree with 8-connectivity radius
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=1.5)

    # Build adjacency list
    adj = [[] for _ in range(len(points))]
    for i, j in pairs:
        adj[i].append(j)
        adj[j].append(i)

    # Find degree-1 nodes (endpoints of the skeleton)
    degrees = [len(neighbors) for neighbors in adj]
    endpoints = [i for i, d in enumerate(degrees) if d == 1]

    if len(endpoints) < 2:
        # No clear endpoints — fall back to x-sort
        sorted_indices = np.argsort(points[:, 0])
        return points[sorted_indices]

    # Greedy walk from leftmost endpoint, preferring straight-ahead direction
    # Pick the endpoint with smallest x as start
    start = min(endpoints, key=lambda i: points[i, 0])
    visited = set()
    path = [start]
    visited.add(start)

    current = start
    while True:
        neighbors = [n for n in adj[current] if n not in visited]
        if not neighbors:
            break

        if len(path) < 2:
            # No direction yet — pick neighbor with largest x (go rightward)
            next_node = max(neighbors, key=lambda n: points[n, 0])
        else:
            # Prefer the neighbor most aligned with current direction
            direction = points[current] - points[path[-2]]
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-9:
                direction = direction / dir_norm

            best = None
            best_score = -2.0
            for n in neighbors:
                vec = points[n] - points[current]
                vec_norm = np.linalg.norm(vec)
                if vec_norm > 1e-9:
                    score = np.dot(vec / vec_norm, direction)
                else:
                    score = 0.0
                if score > best_score:
                    best_score = score
                    best = n
            next_node = best

        visited.add(next_node)
        path.append(next_node)
        current = next_node

    return points[np.array(path)]


def _detect_from_keypoints_model(
    image: np.ndarray,
    mask: np.ndarray,
    config: dict
) -> List[StripeDetection]:
    """Detect stripes using YOLOv11-pose keypoint detection model.

    Loads a trained YOLOv11-pose model and runs inference to detect
    stripe keypoints (8 per stripe, ordered luff-to-leech).

    Args:
        image: Input BGR image (H, W, 3) uint8
        mask: Binary sail mask (H, W) bool
        config: Configuration dict with keys:
            - keypoint_model_path: Path to .pt model file
            - min_confidence: Minimum detection confidence (default 0.3)
            - min_keypoint_confidence: Minimum per-keypoint confidence (default 0.3)

    Returns:
        List of StripeDetection objects, or falls back to classical if model unavailable
    """
    from pathlib import Path

    model_path = config.get('keypoint_model_path', 'stripe_keypoints_v1.pt')

    if not Path(model_path).exists():
        print(f"Warning: Keypoint model not found at {model_path}, using classical fallback")
        return _detect_classical(image, mask)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Warning: ultralytics not installed, using classical fallback")
        return _detect_classical(image, mask)

    min_conf = config.get('min_confidence', 0.3)
    min_kp_conf = config.get('min_keypoint_confidence', 0.3)

    model = YOLO(model_path)
    results = model(image, conf=min_conf, verbose=False)
    r = results[0]

    if r.keypoints is None or len(r.keypoints.data) == 0:
        return []

    detections = []
    h, w = image.shape[:2]

    for i in range(len(r.boxes)):
        det_conf = float(r.boxes.conf[i])
        kp_data = r.keypoints.data[i].cpu().numpy()  # (8, 3): x, y, conf

        kp_xy = kp_data[:, :2]  # (8, 2)
        kp_conf = kp_data[:, 2]  # (8,)

        # Filter out low-confidence keypoints
        valid_mask = kp_conf >= min_kp_conf
        if valid_mask.sum() < 3:
            continue

        valid_points = kp_xy[valid_mask]
        valid_confidences = kp_conf[valid_mask]

        # Clip to image bounds
        valid_points[:, 0] = np.clip(valid_points[:, 0], 0, w - 1)
        valid_points[:, 1] = np.clip(valid_points[:, 1], 0, h - 1)

        # Compute orientation from first to last valid keypoint
        if len(valid_points) >= 2:
            dx = valid_points[-1, 0] - valid_points[0, 0]
            dy = valid_points[-1, 1] - valid_points[0, 1]
            if abs(dx) > 1e-6:
                orientation_deg = np.arctan(dy / dx) * 180.0 / np.pi
            else:
                orientation_deg = 90.0
        else:
            orientation_deg = 0.0

        detection = StripeDetection(
            points=valid_points,
            confidence=det_conf,
            orientation_deg=orientation_deg,
            keypoint_confidences=valid_confidences
        )
        detections.append(detection)

    return detections


def _detect_from_segmentation_model(
    image: np.ndarray,
    mask: np.ndarray,
    config: dict
) -> List[StripeDetection]:
    """Detect stripes using YOLOv11-seg instance segmentation model.

    Runs the trained seg model, skeletonises each instance mask, orders
    the skeleton points along the stripe, and returns StripeDetection
    objects with ~10 sampled keypoints each.

    Args:
        image: Input BGR image (H, W, 3) uint8
        mask: Binary sail mask (H, W) bool
        config: Configuration dict with keys:
            - seg_model_path: Path to YOLO-seg .pt file
            - min_confidence: Detection confidence threshold (default 0.25)

    Returns:
        List of StripeDetection objects, or classical fallback on failure.
    """
    from pathlib import Path
    from skimage.morphology import skeletonize

    model_path = config.get('seg_model_path', 'stripe_seg_v1.pt')
    if not Path(model_path).exists():
        print(f"Warning: Seg model not found at {model_path}, using classical fallback")
        return _detect_classical(image, mask)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Warning: ultralytics not installed, using classical fallback")
        return _detect_classical(image, mask)

    min_conf = config.get('min_confidence', 0.25)
    model = YOLO(model_path)
    results = model(image, conf=min_conf, verbose=False)
    r = results[0]

    if r.masks is None or len(r.masks.data) == 0:
        return []

    h, w = image.shape[:2]
    sail_mask = mask.astype(bool)

    detections: List[StripeDetection] = []
    for i in range(len(r.masks.data)):
        det_conf = float(r.boxes.conf[i]) if r.boxes is not None else 1.0

        inst = r.masks.data[i].cpu().numpy().astype(np.uint8)
        if inst.shape != (h, w):
            inst = cv2.resize(inst, (w, h), interpolation=cv2.INTER_NEAREST)
        inst = inst > 0
        inst &= sail_mask
        if inst.sum() < 20:
            continue

        skeleton = skeletonize(inst)
        ys, xs = np.where(skeleton)
        if len(xs) < 8:
            continue

        pts = np.column_stack([xs, ys]).astype(np.float32)
        ordered = _trace_skeleton_path(pts)
        n_sample = min(12, len(ordered))
        if n_sample < 5:
            continue
        idx = np.linspace(0, len(ordered) - 1, n_sample).astype(int)
        sampled = ordered[idx]

        sampled[:, 0] = np.clip(sampled[:, 0], 0, w - 1)
        sampled[:, 1] = np.clip(sampled[:, 1], 0, h - 1)

        dx = sampled[-1, 0] - sampled[0, 0]
        dy = sampled[-1, 1] - sampled[0, 1]
        orientation_deg = (
            np.arctan(dy / dx) * 180.0 / np.pi if abs(dx) > 1e-6 else 90.0
        )

        detections.append(
            StripeDetection(
                points=sampled,
                confidence=det_conf,
                orientation_deg=orientation_deg,
            )
        )

    return detections


_RF_MODEL_CACHE: dict = {}


def _get_roboflow_model(api_key: str, workspace: str, project: str, version: int):
    key = (api_key, workspace, project, int(version))
    if key in _RF_MODEL_CACHE:
        return _RF_MODEL_CACHE[key]
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Warning: roboflow SDK not installed")
        return None
    rf = Roboflow(api_key=api_key)
    mdl = rf.workspace(workspace).project(project).version(int(version)).model
    _RF_MODEL_CACHE[key] = mdl
    return mdl


def _polygon_to_mask(points: list[dict], h: int, w: int) -> np.ndarray:
    poly = np.array(
        [[int(round(p['x'])), int(round(p['y']))] for p in points],
        dtype=np.int32,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(poly) >= 3:
        cv2.fillPoly(mask, [poly], 255)
    return mask.astype(bool)


def _detect_roboflow(
    image: np.ndarray,
    config: dict,
) -> List[StripeDetection]:
    """Detect stripes using a hosted Roboflow segmentation model.

    Reads config keys:
      - roboflow_api_key, roboflow_workspace, roboflow_project,
        roboflow_version, roboflow_confidence_pct

    The model produces polygon masks per stripe; we skeletonise each mask
    and sample ~12 points to build StripeDetection objects.
    """
    from skimage.morphology import skeletonize
    import tempfile

    api_key = config.get('roboflow_api_key')
    workspace = config.get('roboflow_workspace', 'sailing-project')
    project = config.get('roboflow_project', 'fleurs-bot')
    version = int(config.get('roboflow_version', 7))
    conf_pct = float(config.get('roboflow_confidence_pct', 30))

    if not api_key:
        print("Warning: no roboflow_api_key in config")
        return []

    model = _get_roboflow_model(api_key, workspace, project, version)
    if model is None:
        return []

    h, w = image.shape[:2]
    # Roboflow SDK takes a file path; write to a temp file.
    # (image may be RGB; convert back to BGR for disk.)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tf:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image
        cv2.imwrite(tf.name, bgr)
        tmp_path = tf.name

    try:
        result = model.predict(tmp_path, confidence=conf_pct).json()
    except Exception as exc:
        print(f"Warning: Roboflow inference failed: {exc}")
        return []
    finally:
        try:
            import os
            os.unlink(tmp_path)
        except OSError:
            pass

    detections: List[StripeDetection] = []
    for pred in result.get('predictions', []):
        pts = pred.get('points') or []
        if len(pts) < 3:
            continue
        polygon = np.array(
            [[float(p['x']), float(p['y'])] for p in pts], dtype=np.float32
        )
        mask = _polygon_to_mask(pts, h, w)
        if mask.sum() < 20:
            continue
        skeleton = skeletonize(mask)
        ys, xs = np.where(skeleton)
        if len(xs) < 8:
            continue
        skel_pts = np.column_stack([xs, ys]).astype(np.float32)
        ordered = _trace_skeleton_path(skel_pts)
        n_sample = min(12, len(ordered))
        idx = np.linspace(0, len(ordered) - 1, n_sample).astype(int)
        sampled = ordered[idx]
        sampled[:, 0] = np.clip(sampled[:, 0], 0, w - 1)
        sampled[:, 1] = np.clip(sampled[:, 1], 0, h - 1)

        dx = sampled[-1, 0] - sampled[0, 0]
        dy = sampled[-1, 1] - sampled[0, 1]
        orient = np.arctan(dy / dx) * 180.0 / np.pi if abs(dx) > 1e-6 else 90.0

        detections.append(
            StripeDetection(
                points=sampled,
                confidence=float(pred.get('confidence', 0.5)),
                orientation_deg=orient,
                polygon=polygon,
            )
        )
    return detections


def _detect_bbox_local(
    image: np.ndarray,
    mask: np.ndarray,
    sail_boundary: SailBoundary,
    config: dict
) -> List[StripeDetection]:
    """Detect stripes using YOLO bbox detection + local multi-method refinement.

    Pipeline:
    1. YOLO11n bbox model detects stripe bounding boxes
    2. Each bbox is cropped with padding and masked by SAM2 sail mask
    3. Multi-method local detection (ridge/color/profile) extracts centerline
    4. Results converted to StripeDetection objects

    Falls back to classical detection if bbox model is unavailable.

    Args:
        image: Input BGR image (H, W, 3) uint8
        mask: Binary sail mask (H, W) bool
        sail_boundary: SailBoundary with luff/leech polylines
        config: Configuration dict with keys:
            - bbox_model_path: Path to YOLO bbox .pt file
            - min_confidence: Detection confidence threshold
            - bbox_padding: Fractional padding around bbox (default 0.2)

    Returns:
        List of StripeDetection objects sorted top-to-bottom
    """
    from pathlib import Path

    bbox_model_path = config.get('bbox_model_path', 'stripe_bbox_v1.pt')
    min_conf = config.get('min_confidence', 0.3)
    bbox_padding = config.get('bbox_padding', 0.2)

    # Step 1: Get bounding boxes from YOLO
    bboxes = _get_yolo_bboxes(image, bbox_model_path, min_conf)

    if not bboxes:
        print("Warning: No bbox detections, falling back to classical")
        return _detect_classical(image, mask)

    # Step 2: Run local detection within each bbox
    from src.local_stripe_detect import detect_all_stripes_local

    local_results = detect_all_stripes_local(
        image, bboxes, mask,
        padding=bbox_padding,
        min_confidence=0.15
    )

    if not local_results:
        print("Warning: Local detection found no stripes, falling back to classical")
        return _detect_classical(image, mask)

    # Step 3: Convert to StripeDetection objects
    detections = []
    for result in local_results:
        points = result.points
        if len(points) < 3:
            continue

        # Compute orientation
        dx = points[-1, 0] - points[0, 0]
        dy = points[-1, 1] - points[0, 1]
        if abs(dx) > 1e-6:
            orientation_deg = np.arctan(dy / dx) * 180.0 / np.pi
        else:
            orientation_deg = 90.0

        detection = StripeDetection(
            points=points,
            confidence=result.confidence,
            orientation_deg=orientation_deg
        )
        detections.append(detection)

    return detections


def _get_yolo_bboxes(
    image: np.ndarray,
    model_path: str,
    min_conf: float = 0.3
) -> List[np.ndarray]:
    """Run YOLO bbox detection and return sorted bounding boxes.

    Args:
        image: Input BGR image
        model_path: Path to YOLO .pt model
        min_conf: Minimum confidence threshold

    Returns:
        List of [x1, y1, x2, y2] arrays, sorted top-to-bottom by y-center
    """
    from pathlib import Path

    if not Path(model_path).exists():
        print(f"Warning: Bbox model not found at {model_path}")
        return []

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Warning: ultralytics not installed")
        return []

    model = YOLO(model_path)
    results = model(image, conf=min_conf, verbose=False)
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return []

    bboxes = []
    for i in range(len(r.boxes)):
        box = r.boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
        bboxes.append(box)

    # Sort by y-center (top to bottom)
    bboxes.sort(key=lambda b: (b[1] + b[3]) / 2)

    return bboxes


def _detect_ensemble(
    image: np.ndarray,
    mask: np.ndarray,
    sail_boundary: SailBoundary,
    config: dict,
) -> List[StripeDetection]:
    """Ensemble: fuse bbox / keypoint / segmentation model outputs.

    Strategy:
      * Get candidate bboxes from bbox + keypoint + seg detectors (they all
        produce boxes).
      * Cluster them by IoU + center proximity (one cluster == one stripe).
      * For each cluster pick the "richest" shape representation, preferring
        (seg skeleton) > (keypoint model) > (bbox horizontal fallback).
      * Keypoint-model detections have per-keypoint confidence that we keep.

    The ensemble is robust: if any single model misses a stripe the others
    can still produce a detection, and when multiple fire on the same stripe
    the best shape wins.
    """
    """Polygon-primary ensemble.

    Strategy (stripe detection only; endpoint snapping happens in
    ``src/endpoints.py``):
      1. Get candidate regions from: bbox detector (count/location),
         keypoint detector (fallback shape), segmentation, Roboflow v7
         (polygon shape).
      2. Cluster candidates; each cluster is one stripe.
      3. For each cluster, **prefer the Roboflow polygon skeleton** as the
         stripe centerline when available (far cleaner than kp's 8 points
         on curved stripes). kp acts as a fallback when no polygon fires.
      4. Carry the polygon through so the endpoint stage can intersect it
         with the SAM2 luff/leech polylines.
      5. Post-pass: if a stripe's shape (camber, orientation) is an outlier
         vs neighbouring stripes, swap in a neighbour-interpolated shape.
    """
    kp_dets = _detect_from_keypoints_model(image, mask, config)
    seg_dets = _detect_from_segmentation_model(image, mask, config)
    bbox_model = config.get('bbox_model_path', 'stripe_bbox_v1.pt')
    min_conf = config.get('min_confidence', 0.3)
    bboxes = _get_yolo_bboxes(image, bbox_model, min_conf)

    rf_dets: List[StripeDetection] = []
    if config.get('roboflow_enabled') and config.get('roboflow_api_key'):
        rf_dets = _detect_roboflow(image, config)

    candidates: List[dict] = []
    for det in kp_dets:
        candidates.append(
            {'source': 'kp', 'det': det, 'bbox': _bbox_from_points(det.points)}
        )
    for det in seg_dets:
        candidates.append(
            {'source': 'seg', 'det': det, 'bbox': _bbox_from_points(det.points)}
        )
    for box in bboxes:
        candidates.append({'source': 'bbox', 'det': None, 'bbox': box})

    clusters = _cluster_candidates_by_bbox(
        candidates,
        iou_thresh=0.2,
        vertical_overlap_thresh=0.5,
        horizontal_overlap_thresh=0.3,
    )

    h, w = image.shape[:2]
    ensemble: List[StripeDetection] = []
    priority_rank = {'kp': 3, 'seg': 2, 'bbox': 1}

    def _sort_key(item):
        det = item.get('det')
        conf = det.confidence if det is not None else 0.0
        return (priority_rank[item['source']], conf)

    for cluster in clusters:
        best = max(cluster, key=_sort_key)

        if best['source'] == 'kp' and rf_dets:
            kp_det = best['det']
            kp_y = float(kp_det.points[:, 1].mean())
            kp_bbox = _bbox_from_points(kp_det.points)
            best_rf, best_overlap = None, 0.0
            for rf in rf_dets:
                rf_bbox = _bbox_from_points(rf.points)
                overlap = _horizontal_overlap_ratio(kp_bbox, rf_bbox)
                rf_y = float(rf.points[:, 1].mean())
                if overlap >= 0.3 and abs(rf_y - kp_y) < 0.5 * max(
                    kp_bbox[3] - kp_bbox[1], 1.0
                ):
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_rf = rf
            if best_rf is not None:
                merged = _extend_kp_with_rf_endpoints(kp_det, best_rf)
                merged.polygon = best_rf.polygon
                best = {
                    'source': 'kp',
                    'det': merged,
                    'bbox': _bbox_from_points(kp_det.points),
                }

        if best['det'] is not None:
            # Classical fallback: if a wider bbox exists in this cluster
            # and the chosen detection spans <70% of it, try pixel-based
            # ridge tracing inside bbox × sail-mask to recover the full
            # stripe extent.
            bbox_peers = [
                c for c in cluster
                if c['source'] == 'bbox' and c['det'] is None
            ]
            if bbox_peers and not _has_polygon(best['det']):
                det_pts = best['det'].points
                det_width = float(det_pts[:, 0].max() - det_pts[:, 0].min())
                for peer in bbox_peers:
                    bx1, by1, bx2, by2 = peer['bbox']
                    bbox_width = float(bx2 - bx1)
                    if bbox_width < 20.0 or det_width >= 0.7 * bbox_width:
                        continue
                    try:
                        from src.stripe_in_bbox import (
                            detect_stripe_centerline_in_bbox,
                        )
                        centerline = detect_stripe_centerline_in_bbox(
                            image, mask, peer['bbox']
                        )
                    except Exception:
                        centerline = None
                    if centerline is None or len(centerline) < 5:
                        continue
                    cx_center = float(centerline[:, 0].mean())
                    cy_center = float(centerline[:, 1].mean())
                    det_y = float(det_pts[:, 1].mean())
                    # Accept only if the traced centerline is at the
                    # right y-level (prevents picking a neighbour stripe).
                    if abs(cy_center - det_y) > 0.6 * max(by2 - by1, 1.0):
                        continue
                    dx = centerline[-1, 0] - centerline[0, 0]
                    dy = centerline[-1, 1] - centerline[0, 1]
                    orient = (
                        np.arctan(dy / dx) * 180.0 / np.pi
                        if abs(dx) > 1e-6
                        else 90.0
                    )
                    # Attach a synthetic polygon — the centerline itself,
                    # slightly thickened — so find_endpoints treats the
                    # traced shape as a trustworthy polygon-bearing
                    # detection (skipping pathological polyline snaps).
                    synth_poly = np.vstack([
                        centerline + [0.0, -4.0],
                        centerline[::-1] + [0.0, 4.0],
                    ]).astype(np.float32)
                    best = {
                        'source': 'pixel',
                        'det': StripeDetection(
                            points=centerline,
                            confidence=0.5,
                            orientation_deg=orient,
                            polygon=synth_poly,
                        ),
                        'bbox': peer['bbox'],
                    }
                    break
            ensemble.append(best['det'])
            continue

        # Bbox-only fallback: horizontal line through bbox center
        x1, y1, x2, y2 = best['bbox']
        cy = (y1 + y2) / 2.0
        n_pts = 8
        xs = np.linspace(x1, x2, n_pts)
        pts = np.column_stack([xs, np.full(n_pts, cy)])
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        ensemble.append(
            StripeDetection(points=pts, confidence=0.4, orientation_deg=0.0)
        )

    ensemble = _apply_shape_prior(ensemble)
    return ensemble


def _camber_ratio(det: StripeDetection) -> float:
    """Return (max perpendicular chord deviation) / chord_length."""
    pts = det.points
    if len(pts) < 3:
        return 0.0
    p0, p1 = pts[0], pts[-1]
    chord = float(np.linalg.norm(p1 - p0))
    if chord < 1.0:
        return 0.0
    t = p1 - p0
    t_unit = t / chord
    normal = np.array([-t_unit[1], t_unit[0]])
    deviations = np.abs((pts[1:-1] - p0) @ normal)
    return float(deviations.max() / chord)


def _apply_shape_prior(
    stripes: List[StripeDetection],
    max_abs_camber_ratio: float = 0.22,
) -> List[StripeDetection]:
    """Clamp stripe shapes to neighbour-consistent camber.

    Physical sail stripes rarely exceed 20% camber ratio (perpendicular
    deviation divided by chord length). When a detection reports ≥22%
    it's almost always wrong — the kp model traced through a different
    stripe or confused a batten for the centerline. This is most common
    on the top stripe where training coverage is thinnest.

    When a stripe's camber is an outlier:
      1. Sort stripes top-to-bottom.
      2. Find the nearest neighbour(s) with camber <= max.
      3. Copy the neighbour's interior shape, scaled to the outlier's
         chord vector. Endpoints of the outlier are preserved.
    """
    if len(stripes) < 2:
        return stripes

    # Sort by y-center so "neighbour" is geometrically meaningful.
    order = sorted(
        range(len(stripes)), key=lambda i: float(stripes[i].points[:, 1].mean())
    )
    sorted_stripes = [stripes[i] for i in order]
    ratios = [_camber_ratio(s) for s in sorted_stripes]

    def _shape_from_neighbour(
        outlier: StripeDetection, neighbour: StripeDetection
    ) -> StripeDetection:
        p0, p1 = outlier.points[0], outlier.points[-1]
        chord_vec = p1 - p0
        chord_len = float(np.linalg.norm(chord_vec))
        if chord_len < 1.0:
            return outlier
        t_unit = chord_vec / chord_len
        normal = np.array([-t_unit[1], t_unit[0]])

        # Project neighbour interior points into its chord-normal frame,
        # then re-project into outlier's chord frame (so the camber shape
        # is preserved, scaled to the outlier's chord length).
        n_pts = neighbour.points
        np0, np1 = n_pts[0], n_pts[-1]
        n_chord = float(np.linalg.norm(np1 - np0))
        if n_chord < 1.0:
            return outlier
        n_t = (np1 - np0) / n_chord
        n_n = np.array([-n_t[1], n_t[0]])

        new_points = [p0]
        for q in n_pts[1:-1]:
            rel = q - np0
            along = float(rel @ n_t) / n_chord  # 0..1
            cross = float(rel @ n_n) / n_chord  # normalised camber
            new_p = (
                p0 + along * chord_len * t_unit + cross * chord_len * normal
            )
            new_points.append(new_p)
        new_points.append(p1)

        pts_arr = np.asarray(new_points, dtype=np.float32)
        dx, dy = p1 - p0
        orient = (
            np.arctan(dy / dx) * 180.0 / np.pi if abs(dx) > 1e-6 else 90.0
        )
        return StripeDetection(
            points=pts_arr,
            confidence=outlier.confidence * 0.8,
            orientation_deg=orient,
            keypoint_confidences=None,
            polygon=outlier.polygon,
        )

    corrected = list(sorted_stripes)
    for i, r in enumerate(ratios):
        if r == 0 or r <= max_abs_camber_ratio:
            continue
        # Find nearest neighbour(s) with acceptable camber.
        neighbour: Optional[StripeDetection] = None
        for off in range(1, len(sorted_stripes)):
            for sign in (-1, 1):
                j = i + sign * off
                if 0 <= j < len(sorted_stripes) and ratios[j] > 0 and ratios[j] <= max_abs_camber_ratio:
                    neighbour = sorted_stripes[j]
                    break
            if neighbour is not None:
                break
        if neighbour is None:
            continue
        corrected[i] = _shape_from_neighbour(sorted_stripes[i], neighbour)

    # Reorder back to the caller's original order.
    result = [None] * len(stripes)
    for orig_idx, sort_idx in enumerate(order):
        result[sort_idx] = corrected[orig_idx]
    return result


def _has_polygon(det: Optional[StripeDetection]) -> bool:
    if det is None:
        return False
    poly = getattr(det, 'polygon', None)
    return poly is not None and len(poly) >= 3


def _bbox_from_points(points: np.ndarray) -> np.ndarray:
    xs = points[:, 0]
    ys = points[:, 1]
    return np.array(
        [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    )


def _extend_kp_with_rf_endpoints(
    kp_det: StripeDetection, rf_det: StripeDetection
) -> StripeDetection:
    """Validate kp with rf polygon, then extend endpoints.

    Two steps:
      1. **Validate** — drop kp interior points that fall outside the
         rf polygon (with a small tolerance ring). When the kp model
         hallucinates a point outside the real stripe (e.g. on J2 where
         the leftmost kp lands on sail-cloth between stripes), the rf
         polygon rejects it.
      2. **Extend** — replace the kp's outermost points with the rf
         polygon's leftmost/rightmost extremes (when further out).

    If fewer than 3 kp points survive validation, fall back to the rf
    polygon's own skeleton for the stripe shape.
    """
    kp_pts = kp_det.points
    rf_pts = rf_det.points
    rf_polygon = rf_det.polygon
    if len(rf_pts) < 2 or len(kp_pts) < 3:
        return kp_det

    kp_order = np.argsort(kp_pts[:, 0])
    kp_sorted = kp_pts[kp_order]
    kp_conf = None
    if kp_det.keypoint_confidences is not None:
        kp_conf = np.asarray(kp_det.keypoint_confidences)[kp_order]

    if rf_polygon is not None and len(rf_polygon) >= 3:
        poly_int = rf_polygon.astype(np.int32)
        # Validate kpts against the polygon. Use a generous 20 px tolerance
        # since the polygon is ~15-20 px thick (draft side of the stripe).
        in_poly = np.array([
            cv2.pointPolygonTest(poly_int, (float(p[0]), float(p[1])), True)
            > -20.0
            for p in kp_sorted
        ])
    else:
        in_poly = np.ones(len(kp_sorted), dtype=bool)

    valid_kp = kp_sorted[in_poly]
    valid_conf = kp_conf[in_poly] if kp_conf is not None else None

    rf_sorted = rf_pts[np.argsort(rf_pts[:, 0])]
    rf_left = rf_sorted[0]
    rf_right = rf_sorted[-1]

    # If validation kept < 3 kpts, augment with rf skeleton points to
    # keep the Bernstein fit well-conditioned. The kpts we trust go in
    # first, then rf points (not overlapping in x-range).
    if len(valid_kp) < 3:
        if len(valid_kp) == 0:
            combined = rf_sorted
        else:
            # Merge: valid kp + rf points outside kp x-span
            kp_x_min = float(valid_kp[:, 0].min())
            kp_x_max = float(valid_kp[:, 0].max())
            rf_mask = (
                (rf_sorted[:, 0] < kp_x_min) | (rf_sorted[:, 0] > kp_x_max)
            )
            combined = np.vstack([valid_kp, rf_sorted[rf_mask]])
            combined = combined[np.argsort(combined[:, 0])]
        dx = combined[-1, 0] - combined[0, 0]
        dy = combined[-1, 1] - combined[0, 1]
        orient = (
            np.arctan(dy / dx) * 180.0 / np.pi if abs(dx) > 1e-6 else 90.0
        )
        return StripeDetection(
            points=combined.astype(np.float32),
            confidence=float(max(kp_det.confidence, rf_det.confidence)),
            orientation_deg=orient,
            polygon=rf_polygon,
        )

    interior = valid_kp[1:-1]
    new_left = rf_left if rf_left[0] < valid_kp[0, 0] else valid_kp[0]
    new_right = (
        rf_right if rf_right[0] > valid_kp[-1, 0] else valid_kp[-1]
    )

    new_pts = np.vstack([new_left, interior, new_right])
    if valid_conf is not None:
        new_conf = np.concatenate([
            [min(1.0, float(valid_conf[0]))],
            valid_conf[1:-1],
            [min(1.0, float(valid_conf[-1]))],
        ])
    else:
        new_conf = None

    dx = new_pts[-1, 0] - new_pts[0, 0]
    dy = new_pts[-1, 1] - new_pts[0, 1]
    orient = np.arctan(dy / dx) * 180.0 / np.pi if abs(dx) > 1e-6 else 90.0

    return StripeDetection(
        points=new_pts.astype(np.float32),
        confidence=float(kp_det.confidence),
        orientation_deg=orient,
        keypoint_confidences=new_conf,
        polygon=rf_polygon,
    )


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def _vertical_overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Share-of-smaller-height vertical overlap. Handles thin horizontal stripes."""
    y1 = max(a[1], b[1])
    y2 = min(a[3], b[3])
    inter = max(0.0, y2 - y1)
    smaller = max(1e-6, min(a[3] - a[1], b[3] - b[1]))
    return float(inter / smaller)


def _horizontal_overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    x2 = min(a[2], b[2])
    inter = max(0.0, x2 - x1)
    smaller = max(1e-6, min(a[2] - a[0], b[2] - b[0]))
    return float(inter / smaller)


def _cluster_candidates_by_bbox(
    candidates: List[dict],
    iou_thresh: float = 0.2,
    vertical_overlap_thresh: float = 0.5,
    horizontal_overlap_thresh: float = 0.3,
) -> List[List[dict]]:
    """Cluster thin horizontal stripe candidates.

    Two candidates belong to the same stripe when either:
      * their bboxes have IoU >= iou_thresh, or
      * they overlap both vertically (>=vertical_overlap_thresh of the
        smaller height) AND horizontally (>=horizontal_overlap_thresh).

    The vertical+horizontal check keeps stripes stacked above each other
    separate while merging detections that came from different models on the
    same stripe.
    """
    n = len(candidates)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            a, b = candidates[i]['bbox'], candidates[j]['bbox']
            ca_y = 0.5 * (a[1] + a[3])
            cb_y = 0.5 * (b[1] + b[3])
            ha = max(1e-6, a[3] - a[1])
            hb = max(1e-6, b[3] - b[1])
            smaller_h = min(ha, hb)
            y_center_gap = abs(ca_y - cb_y)
            # Require the vertical centres to be within half of the smaller height.
            if y_center_gap > smaller_h * 0.45:
                continue
            iou = _bbox_iou(a, b)
            h_overlap = _horizontal_overlap_ratio(a, b)
            if iou >= iou_thresh or h_overlap >= horizontal_overlap_thresh:
                union(i, j)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(candidates[i])
    return list(groups.values())


def _detect_hybrid(
    image: np.ndarray,
    mask: np.ndarray,
    sail_boundary: SailBoundary,
    config: dict
) -> List[StripeDetection]:
    """Hybrid detection: YOLO bbox for count/location + keypoints for shape.

    Pipeline:
    1. Run YOLO bbox model → get bounding boxes (stripe locations)
    2. Run YOLO keypoint model → get keypoint detections (stripe shapes)
    3. Match each bbox to closest keypoint detection by center proximity
    4. Matched: use keypoint coordinates as stripe shape
    5. Unmatched: generate points along bbox center line as fallback

    This combines the reliable counting of bbox detection with the shape
    information from keypoint detection.

    Args:
        image: Input BGR image (H, W, 3) uint8
        mask: Binary sail mask (H, W) bool
        sail_boundary: SailBoundary with luff/leech polylines
        config: Configuration dict

    Returns:
        List of StripeDetection objects sorted top-to-bottom
    """
    bbox_model_path = config.get('bbox_model_path', 'stripe_bbox_v1.pt')
    min_conf = config.get('min_confidence', 0.3)

    # Step 1: Get bounding boxes
    bboxes = _get_yolo_bboxes(image, bbox_model_path, min_conf)
    if not bboxes:
        # Fallback to keypoint-only detection
        return _detect_from_keypoints_model(image, mask, config)

    # Step 2: Get keypoint detections
    kp_detections = _detect_from_keypoints_model(image, mask, config)

    # Step 3: Match bboxes to keypoint detections by center proximity
    detections = []
    used_kp = set()

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        bbox_cx = (x1 + x2) / 2
        bbox_cy = (y1 + y2) / 2
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # Find best matching keypoint detection
        best_kp_idx = None
        best_dist = float('inf')

        for ki, kp_det in enumerate(kp_detections):
            if ki in used_kp:
                continue

            kp_center = np.mean(kp_det.points, axis=0)
            # Check if keypoint center is within (or near) the bbox
            if (x1 - bbox_w * 0.3 <= kp_center[0] <= x2 + bbox_w * 0.3 and
                y1 - bbox_h * 0.5 <= kp_center[1] <= y2 + bbox_h * 0.5):
                dist = np.sqrt((kp_center[0] - bbox_cx)**2 +
                             (kp_center[1] - bbox_cy)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_kp_idx = ki

        if best_kp_idx is not None:
            # Use keypoint detection (has shape information)
            kp_det = kp_detections[best_kp_idx]
            used_kp.add(best_kp_idx)
            detections.append(kp_det)
        else:
            # Fallback: generate points along bbox center line
            n_points = 8
            xs = np.linspace(x1, x2, n_points)
            ys = np.full(n_points, bbox_cy)
            points = np.column_stack([xs, ys])

            # Clip to image bounds
            h, w = image.shape[:2]
            points[:, 0] = np.clip(points[:, 0], 0, w - 1)
            points[:, 1] = np.clip(points[:, 1], 0, h - 1)

            dx = points[-1, 0] - points[0, 0]
            dy = points[-1, 1] - points[0, 1]
            orientation = np.degrees(np.arctan2(dy, dx)) if abs(dx) > 1e-6 else 0.0

            detection = StripeDetection(
                points=points,
                confidence=0.5,  # lower confidence for bbox-only
                orientation_deg=orientation
            )
            detections.append(detection)

    return detections
