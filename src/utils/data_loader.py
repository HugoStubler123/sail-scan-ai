"""Data loading utilities for COCO-format stripe annotations.

Provides functions to load and convert annotated datasets:
- Keypoints dataset: data/stripe_keypoints/ (8 keypoints per stripe)
- Segmentation dataset: data/stripe_segmentation/ (thin-line instance masks)
- COCO to YOLO format conversion for ultralytics training
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def load_coco_keypoints(json_path: str) -> List[Dict]:
    """Load COCO keypoints annotations.

    Args:
        json_path: Path to COCO JSON file (e.g., train/_annotations.coco.json)

    Returns:
        List of annotation dicts with keys:
            - image_id: int
            - image_path: str (relative to json_path parent)
            - image_width: int
            - image_height: int
            - keypoints: numpy array (N, 3) with [x, y, visibility]
            - bbox: numpy array (4,) with [x, y, w, h]
            - category_id: int
    """
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Build image id to image info mapping
    images = {img['id']: img for img in coco_data['images']}

    annotations = []

    for ann in coco_data['annotations']:
        if 'keypoints' not in ann:
            continue

        image_info = images[ann['image_id']]

        # Parse keypoints (COCO format: [x1, y1, v1, x2, y2, v2, ...])
        kp_flat = ann['keypoints']
        num_keypoints = len(kp_flat) // 3
        keypoints = np.array(kp_flat).reshape(num_keypoints, 3)

        # Parse bbox [x, y, width, height]
        bbox = np.array(ann['bbox'])

        annotation = {
            'image_id': ann['image_id'],
            'image_path': image_info['file_name'],
            'image_width': image_info['width'],
            'image_height': image_info['height'],
            'keypoints': keypoints,
            'bbox': bbox,
            'category_id': ann['category_id'],
        }

        annotations.append(annotation)

    return annotations


def load_coco_segmentation(json_path: str) -> List[Dict]:
    """Load COCO segmentation annotations.

    Args:
        json_path: Path to COCO JSON file

    Returns:
        List of annotation dicts with keys:
            - image_id: int
            - image_path: str
            - image_width: int
            - image_height: int
            - segmentation: list of polygon arrays (for RLE format)
            - area: float
            - bbox: numpy array (4,) with [x, y, w, h]
            - category_id: int
    """
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Build image id to image info mapping
    images = {img['id']: img for img in coco_data['images']}

    annotations = []

    for ann in coco_data['annotations']:
        if 'segmentation' not in ann:
            continue

        image_info = images[ann['image_id']]

        # Parse segmentation (list of polygons or RLE)
        segmentation = ann['segmentation']

        # Parse bbox [x, y, width, height]
        bbox = np.array(ann['bbox'])

        annotation = {
            'image_id': ann['image_id'],
            'image_path': image_info['file_name'],
            'image_width': image_info['width'],
            'image_height': image_info['height'],
            'segmentation': segmentation,
            'area': ann.get('area', 0.0),
            'bbox': bbox,
            'category_id': ann['category_id'],
        }

        annotations.append(annotation)

    return annotations


def coco_to_yolo_keypoints(coco_dir: str, output_dir: str):
    """Convert COCO keypoints dataset to YOLO-pose format for ultralytics.

    YOLO keypoint format per line:
    <class_id> <x_center> <y_center> <width> <height> <kp1_x> <kp1_y> <kp1_vis> ...

    Coordinates are normalized to [0, 1].

    Args:
        coco_dir: Directory containing train/, valid/, test/ subdirs with _annotations.coco.json
        output_dir: Output directory for YOLO format dataset
    """
    coco_path = Path(coco_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in ['train', 'valid', 'test']:
        split_dir = coco_path / split
        json_file = split_dir / '_annotations.coco.json'

        if not json_file.exists():
            print(f"Warning: {json_file} not found, skipping {split}")
            continue

        # Load annotations
        annotations = load_coco_keypoints(str(json_file))

        # Create output split directory
        output_split_dir = output_path / split
        output_split_dir.mkdir(parents=True, exist_ok=True)

        # Group annotations by image
        from collections import defaultdict
        image_annotations = defaultdict(list)

        for ann in annotations:
            image_annotations[ann['image_path']].append(ann)

        # Write YOLO format labels
        for image_path, anns in image_annotations.items():
            # Label file has same name as image but .txt extension
            label_file = output_split_dir / (Path(image_path).stem + '.txt')

            with open(label_file, 'w') as f:
                for ann in anns:
                    w = ann['image_width']
                    h = ann['image_height']

                    # Normalize bbox
                    bbox = ann['bbox']
                    x_center = (bbox[0] + bbox[2] / 2) / w
                    y_center = (bbox[1] + bbox[3] / 2) / h
                    width = bbox[2] / w
                    height = bbox[3] / h

                    # Normalize keypoints
                    keypoints = ann['keypoints']
                    kp_normalized = []

                    for kp in keypoints:
                        kp_x = kp[0] / w
                        kp_y = kp[1] / h
                        kp_vis = int(kp[2])  # 0=not labeled, 1=labeled but not visible, 2=labeled and visible
                        kp_normalized.extend([kp_x, kp_y, kp_vis])

                    # Write line: class_id bbox keypoints
                    class_id = ann['category_id']
                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

                    for val in kp_normalized:
                        line += f" {val:.6f}"

                    f.write(line + '\n')

        print(f"Converted {split}: {len(image_annotations)} images")

    print(f"YOLO keypoints dataset written to {output_path}")


def coco_to_yolo_seg(coco_dir: str, output_dir: str):
    """Convert COCO segmentation dataset to YOLO-seg format for ultralytics.

    YOLO segmentation format per line:
    <class_id> <x1> <y1> <x2> <y2> ... (polygon vertices, normalized)

    Args:
        coco_dir: Directory containing train/, valid/, test/ subdirs with _annotations.coco.json
        output_dir: Output directory for YOLO format dataset
    """
    coco_path = Path(coco_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in ['train', 'valid', 'test']:
        split_dir = coco_path / split
        json_file = split_dir / '_annotations.coco.json'

        if not json_file.exists():
            print(f"Warning: {json_file} not found, skipping {split}")
            continue

        # Load annotations
        annotations = load_coco_segmentation(str(json_file))

        # Create output split directory
        output_split_dir = output_path / split
        output_split_dir.mkdir(parents=True, exist_ok=True)

        # Group annotations by image
        from collections import defaultdict
        image_annotations = defaultdict(list)

        for ann in annotations:
            image_annotations[ann['image_path']].append(ann)

        # Write YOLO format labels
        for image_path, anns in image_annotations.items():
            label_file = output_split_dir / (Path(image_path).stem + '.txt')

            with open(label_file, 'w') as f:
                for ann in anns:
                    w = ann['image_width']
                    h = ann['image_height']

                    # Parse segmentation polygons
                    segmentation = ann['segmentation']

                    if isinstance(segmentation, list) and len(segmentation) > 0:
                        # Polygon format: list of [x1, y1, x2, y2, ...]
                        for polygon in segmentation:
                            if len(polygon) < 6:  # Need at least 3 points
                                continue

                            # Normalize polygon coordinates
                            normalized = []
                            for i in range(0, len(polygon), 2):
                                x_norm = polygon[i] / w
                                y_norm = polygon[i + 1] / h
                                normalized.extend([x_norm, y_norm])

                            # Write line: class_id + normalized polygon
                            class_id = ann['category_id']
                            line = f"{class_id}"

                            for val in normalized:
                                line += f" {val:.6f}"

                            f.write(line + '\n')

        print(f"Converted {split}: {len(image_annotations)} images")

    print(f"YOLO segmentation dataset written to {output_path}")
