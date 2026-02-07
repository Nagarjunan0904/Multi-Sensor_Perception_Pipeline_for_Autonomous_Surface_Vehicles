"""
2D Perception module for object detection.

This module provides 2D object detection using YOLOv8 with post-processing
utilities for filtering and NMS.

Classes:
    ObjectDetector2D: YOLOv8-based object detector
    Detection: Standardized detection result format
    DetectionPostProcessor: Configurable post-processing pipeline

Functions:
    apply_nms: Non-Maximum Suppression
    filter_by_class: Filter detections by class
    filter_by_confidence: Filter by confidence threshold
    compute_iou: Calculate IoU between boxes

Example:
    >>> from perception2d import ObjectDetector2D, DetectionPostProcessor
    >>>
    >>> # Initialize detector
    >>> detector = ObjectDetector2D(model_name="yolov8n", confidence_threshold=0.3)
    >>>
    >>> # Detect objects
    >>> detections = detector.detect(image)
    >>>
    >>> # Post-process
    >>> processor = DetectionPostProcessor(nms_threshold=0.5)
    >>> clean_detections = processor.process(detections)
"""

from .detector import (
    Detection,
    ObjectDetector2D,
    YOLOv8Detector,  # Alias for backward compatibility
)
from .postprocess import (
    DetectionPostProcessor,
    apply_nms,
    filter_by_class,
    filter_by_confidence,
    filter_by_size,
    filter_by_region,
    filter_by_aspect_ratio,
    sort_detections,
    compute_iou,
    compute_iou_matrix,
)

__all__ = [
    # Detector
    "ObjectDetector2D",
    "YOLOv8Detector",
    "Detection",
    # Post-processing
    "DetectionPostProcessor",
    "apply_nms",
    "filter_by_class",
    "filter_by_confidence",
    "filter_by_size",
    "filter_by_region",
    "filter_by_aspect_ratio",
    "sort_detections",
    "compute_iou",
    "compute_iou_matrix",
]
