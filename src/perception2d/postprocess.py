"""
Post-processing utilities for 2D object detections.

This module provides functions for filtering, NMS, and processing detection
results from the ObjectDetector2D.

Non-Maximum Suppression (NMS):
==============================
NMS removes redundant overlapping detections. The algorithm:

1. Sort detections by confidence score (descending)
2. Select the highest-scoring detection, add to output
3. Remove all detections with IoU > threshold with the selected detection
4. Repeat until no detections remain

IoU (Intersection over Union):
==============================
           area of overlap
IoU = --------------------------
           area of union

       intersection(A, B)
    = ---------------------
      area(A) + area(B) - intersection(A, B)

IoU ranges from 0 (no overlap) to 1 (perfect overlap).
Typical NMS threshold: 0.5 (50% overlap triggers suppression)
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .detector import Detection


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.

    Args:
        box1: First box [x1, y1, x2, y2].
        box2: Second box [x1, y1, x2, y2].

    Returns:
        IoU value in range [0, 1].

    Example:
        >>> box1 = np.array([0, 0, 10, 10])
        >>> box2 = np.array([5, 5, 15, 15])
        >>> iou = compute_iou(box1, box2)
        >>> print(f"IoU: {iou:.2f}")  # ~0.14
    """
    # Compute intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute intersection area (0 if boxes don't overlap)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute individual box areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute union area
    union = area1 + area2 - intersection

    # Return IoU (handle edge case of zero union)
    return intersection / union if union > 0 else 0.0


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between all pairs of boxes (vectorized).

    Args:
        boxes1: (N, 4) array of boxes.
        boxes2: (M, 4) array of boxes.

    Returns:
        (N, M) array of IoU values.
    """
    # Expand dimensions for broadcasting
    # boxes1: (N, 1, 4), boxes2: (1, M, 4)
    boxes1 = boxes1[:, np.newaxis, :]
    boxes2 = boxes2[np.newaxis, :, :]

    # Compute intersection
    x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Compute areas
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # Compute union and IoU
    union = area1 + area2 - intersection
    iou = np.where(union > 0, intersection / union, 0.0)

    return iou


def apply_nms(
    detections: List[Detection],
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> List[Detection]:
    """
    Apply Non-Maximum Suppression to detections.

    NMS removes redundant overlapping detections, keeping only the highest
    confidence detection among overlapping boxes.

    Args:
        detections: List of Detection objects.
        iou_threshold: IoU threshold above which boxes are suppressed.
            - 0.5: Standard threshold (boxes with >50% overlap are suppressed)
            - Lower: More aggressive suppression
            - Higher: Less suppression, may keep more overlapping boxes
        class_agnostic: If True, apply NMS across all classes together.
            If False, apply NMS separately per class.

    Returns:
        Filtered list of detections after NMS.

    Algorithm:
        1. Sort detections by confidence (highest first)
        2. Take highest confidence detection as output
        3. Remove all detections with IoU > threshold
        4. Repeat until no detections remain
    """
    if len(detections) == 0:
        return []

    # Convert to numpy arrays for efficiency
    boxes = np.array([det.bbox for det in detections])
    scores = np.array([det.confidence for det in detections])
    classes = np.array([det.class_id for det in detections])

    if class_agnostic:
        # Apply NMS across all classes together
        keep_indices = _nms_numpy(boxes, scores, iou_threshold)
    else:
        # Apply NMS separately for each class
        keep_indices = []
        for class_id in np.unique(classes):
            class_mask = classes == class_id
            class_indices = np.where(class_mask)[0]

            if len(class_indices) == 0:
                continue

            class_keep = _nms_numpy(
                boxes[class_mask],
                scores[class_mask],
                iou_threshold,
            )
            keep_indices.extend(class_indices[class_keep])

    return [detections[i] for i in keep_indices]


def _nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> List[int]:
    """
    Pure numpy NMS implementation.

    This is a standard greedy NMS algorithm.

    Args:
        boxes: (N, 4) array of boxes [x1, y1, x2, y2].
        scores: (N,) array of confidence scores.
        threshold: IoU threshold for suppression.

    Returns:
        List of indices to keep.
    """
    if len(boxes) == 0:
        return []

    # Extract coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute box areas
    areas = (x2 - x1) * (y2 - y1)

    # Sort by score (descending) -> get sorted indices
    order = scores.argsort()[::-1]

    keep = []

    while len(order) > 0:
        # Take the highest scoring box
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        # Compute IoU with remaining boxes
        # Intersection coordinates
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Intersection area
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h

        # IoU
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        # Keep boxes with IoU <= threshold
        mask = iou <= threshold
        order = order[1:][mask]

    return keep


def filter_by_class(
    detections: List[Detection],
    allowed_classes: List[str],
) -> List[Detection]:
    """
    Filter detections to keep only specified classes.

    Args:
        detections: List of detections.
        allowed_classes: List of class names to keep (e.g., ["Car", "Pedestrian"]).

    Returns:
        Filtered detections containing only allowed classes.

    Example:
        >>> filtered = filter_by_class(detections, ["Car", "Pedestrian"])
    """
    return [det for det in detections if det.class_name in allowed_classes]


def filter_by_confidence(
    detections: List[Detection],
    threshold: float,
) -> List[Detection]:
    """
    Filter detections by minimum confidence threshold.

    Args:
        detections: List of detections.
        threshold: Minimum confidence score to keep.

    Returns:
        Detections with confidence >= threshold.
    """
    return [det for det in detections if det.confidence >= threshold]


def filter_by_size(
    detections: List[Detection],
    min_width: int = 0,
    min_height: int = 0,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
) -> List[Detection]:
    """
    Filter detections by bounding box size.

    Useful for removing:
    - Very small detections (likely false positives)
    - Very large detections (likely background/errors)

    Args:
        detections: List of detections.
        min_width: Minimum box width in pixels.
        min_height: Minimum box height in pixels.
        max_width: Maximum box width (None = no limit).
        max_height: Maximum box height (None = no limit).

    Returns:
        Filtered detections within size constraints.
    """
    filtered = []

    for det in detections:
        width = det.width
        height = det.height

        if width < min_width or height < min_height:
            continue

        if max_width is not None and width > max_width:
            continue

        if max_height is not None and height > max_height:
            continue

        filtered.append(det)

    return filtered


def filter_by_region(
    detections: List[Detection],
    region: Tuple[float, float, float, float],
    mode: str = "center",
) -> List[Detection]:
    """
    Filter detections by image region.

    Args:
        detections: List of detections.
        region: Region of interest (x1, y1, x2, y2).
        mode: How to check if detection is in region:
            - 'center': Detection center must be inside
            - 'any': Any part of detection must overlap
            - 'full': Entire detection must be inside

    Returns:
        Detections within the specified region.
    """
    x1, y1, x2, y2 = region
    filtered = []

    for det in detections:
        if mode == "center":
            cx, cy = det.center
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                filtered.append(det)

        elif mode == "any":
            # Check for any overlap
            if not (det.x2 < x1 or det.x1 > x2 or det.y2 < y1 or det.y1 > y2):
                filtered.append(det)

        elif mode == "full":
            # Check if entire box is inside region
            if det.x1 >= x1 and det.x2 <= x2 and det.y1 >= y1 and det.y2 <= y2:
                filtered.append(det)

    return filtered


def filter_by_aspect_ratio(
    detections: List[Detection],
    min_ratio: float = 0.0,
    max_ratio: float = float("inf"),
) -> List[Detection]:
    """
    Filter detections by aspect ratio (width/height).

    Useful for class-specific filtering:
    - Pedestrians: typically tall (ratio < 1)
    - Cars: typically wide (ratio > 1)

    Args:
        detections: List of detections.
        min_ratio: Minimum width/height ratio.
        max_ratio: Maximum width/height ratio.

    Returns:
        Detections within aspect ratio range.
    """
    return [
        det for det in detections
        if min_ratio <= det.aspect_ratio <= max_ratio
    ]


def sort_detections(
    detections: List[Detection],
    key: str = "confidence",
    reverse: bool = True,
) -> List[Detection]:
    """
    Sort detections by specified key.

    Args:
        detections: List of detections.
        key: Sort key ('confidence', 'area', 'x1', 'y1').
        reverse: Sort descending if True.

    Returns:
        Sorted list of detections.
    """
    key_funcs = {
        "confidence": lambda d: d.confidence,
        "area": lambda d: d.area,
        "x1": lambda d: d.x1,
        "y1": lambda d: d.y1,
        "width": lambda d: d.width,
        "height": lambda d: d.height,
    }

    if key not in key_funcs:
        raise ValueError(f"Unknown sort key: {key}. Valid: {list(key_funcs.keys())}")

    return sorted(detections, key=key_funcs[key], reverse=reverse)


class DetectionPostProcessor:
    """
    Configurable post-processor for 2D detections.

    Applies a pipeline of filtering and processing steps to clean up
    raw detection results.

    Usage:
        processor = DetectionPostProcessor(
            min_confidence=0.3,
            min_box_size=(20, 20),
            nms_threshold=0.5,
            allowed_classes=["Car", "Pedestrian"],
        )
        clean_detections = processor.process(raw_detections)
    """

    def __init__(
        self,
        min_confidence: float = 0.0,
        min_box_size: Tuple[int, int] = (10, 10),
        max_box_size: Optional[Tuple[int, int]] = None,
        nms_threshold: float = 0.5,
        nms_class_agnostic: bool = False,
        allowed_classes: Optional[List[str]] = None,
        region: Optional[Tuple[float, float, float, float]] = None,
    ):
        """
        Initialize post-processor.

        Args:
            min_confidence: Minimum confidence threshold.
            min_box_size: Minimum (width, height) for boxes.
            max_box_size: Maximum (width, height) for boxes (None = no limit).
            nms_threshold: IoU threshold for NMS.
            nms_class_agnostic: Apply NMS across all classes.
            allowed_classes: Classes to keep (None = all classes).
            region: Region of interest (x1, y1, x2, y2) to filter.
        """
        self.min_confidence = min_confidence
        self.min_width, self.min_height = min_box_size
        self.max_width = max_box_size[0] if max_box_size else None
        self.max_height = max_box_size[1] if max_box_size else None
        self.nms_threshold = nms_threshold
        self.nms_class_agnostic = nms_class_agnostic
        self.allowed_classes = allowed_classes
        self.region = region

    def process(self, detections: List[Detection]) -> List[Detection]:
        """
        Apply all post-processing steps.

        Processing order:
        1. Filter by confidence
        2. Filter by size
        3. Filter by class
        4. Filter by region
        5. Apply NMS

        Args:
            detections: Raw detections from detector.

        Returns:
            Cleaned up detections.
        """
        if len(detections) == 0:
            return []

        # Step 1: Filter by confidence
        if self.min_confidence > 0:
            detections = filter_by_confidence(detections, self.min_confidence)

        # Step 2: Filter by size
        detections = filter_by_size(
            detections,
            min_width=self.min_width,
            min_height=self.min_height,
            max_width=self.max_width,
            max_height=self.max_height,
        )

        # Step 3: Filter by class
        if self.allowed_classes:
            detections = filter_by_class(detections, self.allowed_classes)

        # Step 4: Filter by region
        if self.region:
            detections = filter_by_region(detections, self.region)

        # Step 5: Apply NMS
        if self.nms_threshold < 1.0:
            detections = apply_nms(
                detections,
                iou_threshold=self.nms_threshold,
                class_agnostic=self.nms_class_agnostic,
            )

        return detections

    def __repr__(self) -> str:
        return (
            f"DetectionPostProcessor("
            f"conf>={self.min_confidence}, "
            f"size>=({self.min_width}, {self.min_height}), "
            f"nms={self.nms_threshold})"
        )
