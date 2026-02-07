"""
Comprehensive Evaluation Metrics for Multi-Sensor Perception Pipeline.

This module provides metrics for evaluating:
1. 2D Object Detection: Precision, Recall, mAP
2. 3D Object Detection: Distance error, Orientation error, 3D IoU, Dimension error

Metric Definitions:
-------------------

2D Metrics:
- **Precision**: TP / (TP + FP) - How many predictions are correct
- **Recall**: TP / (TP + FN) - How many ground truths are detected
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **AP (Average Precision)**: Area under Precision-Recall curve

3D Metrics:
- **Center Distance Error**: Euclidean distance between predicted and GT centers
- **Orientation Error**: Angular difference in yaw (rotation_y), wrapped to [-pi, pi]
- **3D IoU**: Volume of intersection / Volume of union for 3D boxes
- **Dimension Error**: |predicted_dims - GT_dims| for [length, width, height]

Distance-based Analysis:
- Near (0-20m): Objects close to ego vehicle
- Medium (20-40m): Mid-range objects
- Far (40m+): Distant objects (harder to detect)

Author: Multi-Sensor Perception Pipeline
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Scipy for rotation utilities (optional)
try:
    from scipy.spatial.transform import Rotation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Detection2D:
    """2D detection result."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    class_name: str
    class_id: int = 0


@dataclass
class GroundTruth2D:
    """2D ground truth annotation."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    class_name: str
    difficult: bool = False
    truncated: float = 0.0
    occluded: int = 0


@dataclass
class Detection3D:
    """3D detection result."""
    center: np.ndarray  # [x, y, z] in camera coordinates
    dimensions: np.ndarray  # [length, width, height]
    rotation_y: float  # Rotation around Y-axis
    score: float
    class_name: str


@dataclass
class GroundTruth3D:
    """3D ground truth annotation (KITTI format)."""
    center: np.ndarray  # [x, y, z] in camera coordinates
    dimensions: np.ndarray  # [length, width, height] or [h, w, l] depending on convention
    rotation_y: float
    class_name: str
    truncated: float = 0.0
    occluded: int = 0
    alpha: float = 0.0  # Observation angle
    bbox_2d: Optional[np.ndarray] = None


@dataclass
class MatchResult:
    """Result of matching detections to ground truths."""
    det_idx: int
    gt_idx: int
    iou: float
    distance: float = 0.0


@dataclass
class MetricsResult:
    """Container for all computed metrics."""
    # 2D Metrics
    precision_2d: float = 0.0
    recall_2d: float = 0.0
    f1_2d: float = 0.0
    ap_2d: float = 0.0

    # Per-class 2D metrics
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_ap: Dict[str, float] = field(default_factory=dict)

    # 3D Metrics
    mean_center_error: float = 0.0
    median_center_error: float = 0.0
    mean_orientation_error: float = 0.0
    mean_dimension_error: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mean_iou_3d: float = 0.0

    # Distance-binned metrics
    metrics_by_distance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Raw data for plotting
    center_errors: List[float] = field(default_factory=list)
    orientation_errors: List[float] = field(default_factory=list)
    dimension_errors: List[np.ndarray] = field(default_factory=list)
    iou_3d_values: List[float] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)

    # Counts
    num_detections: int = 0
    num_ground_truths: int = 0
    num_true_positives: int = 0
    num_false_positives: int = 0
    num_false_negatives: int = 0


# =============================================================================
# 2D Metrics
# =============================================================================

def compute_iou_2d(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute 2D Intersection over Union.

    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format

    Returns:
        IoU value in [0, 1]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision using 11-point interpolation (PASCAL VOC style).

    Args:
        recalls: Array of recall values (sorted ascending)
        precisions: Array of precision values

    Returns:
        Average Precision value
    """
    if len(recalls) == 0:
        return 0.0

    # Add sentinel values
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])

    # Compute precision envelope (monotonically decreasing)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max() / 11

    return ap


def compute_ap_voc(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision using all-point interpolation (VOC 2010+ style).

    Args:
        recalls: Array of recall values
        precisions: Array of precision values

    Returns:
        Average Precision value
    """
    if len(recalls) == 0:
        return 0.0

    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]

    # Add sentinel values
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])

    # Compute precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Find where recall changes
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1

    # Sum (r[i+1] - r[i]) * p[i+1]
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

    return ap


class Metrics2DCalculator:
    """Calculate 2D detection metrics."""

    def __init__(self, iou_threshold: float = 0.5, classes: List[str] = None):
        """
        Initialize 2D metrics calculator.

        Args:
            iou_threshold: IoU threshold for matching (default 0.5 for mAP@0.5)
            classes: List of class names to evaluate
        """
        self.iou_threshold = iou_threshold
        self.classes = classes or ["Car", "Pedestrian", "Cyclist"]

        # Accumulators
        self.all_detections: Dict[str, List[Tuple[float, bool]]] = {c: [] for c in self.classes}
        self.num_gt: Dict[str, int] = {c: 0 for c in self.classes}

    def reset(self):
        """Reset accumulators."""
        self.all_detections = {c: [] for c in self.classes}
        self.num_gt = {c: 0 for c in self.classes}

    def add_frame(
        self,
        detections: List[Detection2D],
        ground_truths: List[GroundTruth2D],
    ):
        """
        Add detections and ground truths from one frame.

        Args:
            detections: List of 2D detections
            ground_truths: List of 2D ground truths
        """
        for cls in self.classes:
            # Filter by class
            cls_dets = [d for d in detections if d.class_name == cls]
            cls_gts = [g for g in ground_truths if g.class_name == cls]

            # Count ground truths (excluding difficult)
            self.num_gt[cls] += sum(1 for g in cls_gts if not g.difficult)

            # Sort detections by score (descending)
            cls_dets = sorted(cls_dets, key=lambda x: x.score, reverse=True)

            # Track which GTs are matched
            gt_matched = [False] * len(cls_gts)

            for det in cls_dets:
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(cls_gts):
                    if gt_matched[gt_idx]:
                        continue
                    if gt.difficult:
                        continue

                    iou = compute_iou_2d(det.bbox, gt.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                # Check if match
                if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                    gt_matched[best_gt_idx] = True
                    self.all_detections[cls].append((det.score, True))  # TP
                else:
                    self.all_detections[cls].append((det.score, False))  # FP

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute final metrics.

        Returns:
            Dictionary with per-class and overall metrics.
        """
        results = {}

        all_ap = []
        total_tp = 0
        total_fp = 0
        total_gt = 0

        for cls in self.classes:
            dets = self.all_detections[cls]
            num_gt = self.num_gt[cls]

            if num_gt == 0:
                results[cls] = {"precision": 0.0, "recall": 0.0, "ap": 0.0, "f1": 0.0}
                continue

            # Sort by score
            dets = sorted(dets, key=lambda x: x[0], reverse=True)

            # Compute precision/recall at each threshold
            tp_cumsum = 0
            fp_cumsum = 0
            precisions = []
            recalls = []

            for score, is_tp in dets:
                if is_tp:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1

                precision = tp_cumsum / (tp_cumsum + fp_cumsum)
                recall = tp_cumsum / num_gt

                precisions.append(precision)
                recalls.append(recall)

            precisions = np.array(precisions)
            recalls = np.array(recalls)

            # Compute AP
            ap = compute_ap_voc(recalls, precisions)
            all_ap.append(ap)

            # Final precision/recall
            final_precision = precisions[-1] if len(precisions) > 0 else 0.0
            final_recall = recalls[-1] if len(recalls) > 0 else 0.0
            f1 = 2 * final_precision * final_recall / (final_precision + final_recall + 1e-6)

            results[cls] = {
                "precision": final_precision,
                "recall": final_recall,
                "ap": ap,
                "f1": f1,
                "num_gt": num_gt,
                "num_det": len(dets),
                "num_tp": tp_cumsum,
                "num_fp": fp_cumsum,
            }

            total_tp += tp_cumsum
            total_fp += fp_cumsum
            total_gt += num_gt

        # Overall metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / total_gt if total_gt > 0 else 0.0
        mAP = np.mean(all_ap) if all_ap else 0.0

        results["overall"] = {
            "precision": overall_precision,
            "recall": overall_recall,
            "mAP": mAP,
            "f1": 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-6),
        }

        return results


# =============================================================================
# 3D Metrics
# =============================================================================

def compute_center_distance(center1: np.ndarray, center2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two 3D centers.

    Args:
        center1: [x, y, z] predicted center
        center2: [x, y, z] ground truth center

    Returns:
        Euclidean distance in meters
    """
    return np.linalg.norm(np.array(center1) - np.array(center2))


def compute_orientation_error(rot1: float, rot2: float) -> float:
    """
    Compute angular difference between two rotations.

    Args:
        rot1: Predicted rotation_y in radians
        rot2: Ground truth rotation_y in radians

    Returns:
        Angular error in radians, wrapped to [0, pi]
    """
    diff = rot1 - rot2

    # Wrap to [-pi, pi]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi

    return abs(diff)


def compute_dimension_error(dims1: np.ndarray, dims2: np.ndarray) -> np.ndarray:
    """
    Compute dimension errors.

    Args:
        dims1: [l, w, h] predicted dimensions
        dims2: [l, w, h] ground truth dimensions

    Returns:
        Absolute errors for each dimension
    """
    return np.abs(np.array(dims1) - np.array(dims2))


def get_box_corners_3d(center: np.ndarray, dims: np.ndarray, rotation_y: float) -> np.ndarray:
    """
    Get 8 corners of a 3D bounding box.

    Args:
        center: [x, y, z] box center
        dims: [l, w, h] or [length, width, height]
        rotation_y: Rotation around Y-axis

    Returns:
        (8, 3) array of corner coordinates
    """
    l, w, h = dims

    # 8 corners in object frame (centered at origin)
    # Order: front-left-bottom, front-right-bottom, back-right-bottom, back-left-bottom,
    #        front-left-top, front-right-top, back-right-top, back-left-top
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]  # Y points down in camera coords
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = np.array([x_corners, y_corners, z_corners])  # (3, 8)

    # Rotation matrix around Y-axis
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

    # Rotate and translate
    corners = R @ corners
    corners = corners.T + center  # (8, 3)

    return corners


def compute_iou_3d(
    center1: np.ndarray, dims1: np.ndarray, rot1: float,
    center2: np.ndarray, dims2: np.ndarray, rot2: float,
) -> float:
    """
    Compute 3D Intersection over Union.

    Uses a simplified axis-aligned approximation for efficiency.
    For exact computation, use convex hull intersection.

    Args:
        center1, dims1, rot1: Predicted box parameters
        center2, dims2, rot2: Ground truth box parameters

    Returns:
        3D IoU value in [0, 1]
    """
    # Get corners
    corners1 = get_box_corners_3d(center1, dims1, rot1)
    corners2 = get_box_corners_3d(center2, dims2, rot2)

    # Axis-aligned bounding boxes of the rotated boxes
    min1 = corners1.min(axis=0)
    max1 = corners1.max(axis=0)
    min2 = corners2.min(axis=0)
    max2 = corners2.max(axis=0)

    # Intersection
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(0, inter_max - inter_min)
    intersection = np.prod(inter_dims)

    # Volumes
    vol1 = np.prod(dims1)
    vol2 = np.prod(dims2)

    # Union
    union = vol1 + vol2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def compute_iou_3d_exact(
    center1: np.ndarray, dims1: np.ndarray, rot1: float,
    center2: np.ndarray, dims2: np.ndarray, rot2: float,
) -> float:
    """
    Compute exact 3D IoU using convex hull intersection.

    Requires scipy for ConvexHull.
    Falls back to approximate method if scipy unavailable.
    """
    try:
        from scipy.spatial import ConvexHull, Delaunay
        from scipy.spatial.qhull import QhullError
    except ImportError:
        return compute_iou_3d(center1, dims1, rot1, center2, dims2, rot2)

    corners1 = get_box_corners_3d(center1, dims1, rot1)
    corners2 = get_box_corners_3d(center2, dims2, rot2)

    # Combine all corners
    all_corners = np.vstack([corners1, corners2])

    try:
        # Check if boxes intersect using Delaunay triangulation
        hull1 = Delaunay(corners1)
        hull2 = Delaunay(corners2)

        # Sample points inside each box and check intersection
        # This is an approximation but more accurate than AABB
        intersection_volume = 0.0

        # Use Monte Carlo sampling for intersection volume
        n_samples = 1000
        vol1 = np.prod(dims1)
        vol2 = np.prod(dims2)

        # Sample in box 1
        samples1 = np.random.uniform(-0.5, 0.5, (n_samples, 3))
        samples1 *= dims1

        # Rotate and translate
        c, s = np.cos(rot1), np.sin(rot1)
        R1 = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        samples1 = (R1 @ samples1.T).T + center1

        # Check which samples are inside box 2
        in_box2 = hull2.find_simplex(samples1) >= 0
        intersection_volume = vol1 * np.mean(in_box2)

        union = vol1 + vol2 - intersection_volume

        if union <= 0:
            return 0.0

        return intersection_volume / union

    except (QhullError, ValueError):
        return compute_iou_3d(center1, dims1, rot1, center2, dims2, rot2)


class Metrics3DCalculator:
    """Calculate 3D detection metrics."""

    def __init__(
        self,
        iou_threshold: float = 0.5,
        distance_threshold: float = 2.0,
        classes: List[str] = None,
        distance_bins: List[Tuple[float, float]] = None,
    ):
        """
        Initialize 3D metrics calculator.

        Args:
            iou_threshold: 3D IoU threshold for matching
            distance_threshold: Maximum center distance for matching (meters)
            classes: List of class names to evaluate
            distance_bins: List of (min, max) distance ranges for analysis
        """
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.classes = classes or ["Car", "Pedestrian", "Cyclist"]
        self.distance_bins = distance_bins or [
            (0, 20),    # Near
            (20, 40),   # Medium
            (40, 80),   # Far
        ]

        # Accumulators
        self.results = MetricsResult()
        self.reset()

    def reset(self):
        """Reset accumulators."""
        self.results = MetricsResult()
        self.per_class_errors: Dict[str, Dict[str, List]] = {
            cls: {
                "center_errors": [],
                "orientation_errors": [],
                "dimension_errors": [],
                "iou_3d": [],
                "distances": [],
            }
            for cls in self.classes
        }

    def match_detections(
        self,
        detections: List[Detection3D],
        ground_truths: List[GroundTruth3D],
    ) -> Tuple[List[MatchResult], List[int], List[int]]:
        """
        Match detections to ground truths using Hungarian algorithm.

        Args:
            detections: List of 3D detections
            ground_truths: List of 3D ground truths

        Returns:
            Tuple of (matches, unmatched_det_indices, unmatched_gt_indices)
        """
        if len(detections) == 0 or len(ground_truths) == 0:
            return [], list(range(len(detections))), list(range(len(ground_truths)))

        # Compute distance matrix
        n_det = len(detections)
        n_gt = len(ground_truths)
        dist_matrix = np.full((n_det, n_gt), np.inf)
        iou_matrix = np.zeros((n_det, n_gt))

        for i, det in enumerate(detections):
            for j, gt in enumerate(ground_truths):
                # Only match same class
                if det.class_name != gt.class_name:
                    continue

                dist = compute_center_distance(det.center, gt.center)
                iou = compute_iou_3d(
                    det.center, det.dimensions, det.rotation_y,
                    gt.center, gt.dimensions, gt.rotation_y
                )

                dist_matrix[i, j] = dist
                iou_matrix[i, j] = iou

        # Greedy matching (can be replaced with Hungarian algorithm)
        matches = []
        matched_det = set()
        matched_gt = set()

        # Sort by IoU descending
        flat_indices = np.argsort(iou_matrix.flatten())[::-1]

        for idx in flat_indices:
            i = idx // n_gt
            j = idx % n_gt

            if i in matched_det or j in matched_gt:
                continue

            iou = iou_matrix[i, j]
            dist = dist_matrix[i, j]

            # Check thresholds
            if iou >= self.iou_threshold or dist <= self.distance_threshold:
                matches.append(MatchResult(
                    det_idx=i,
                    gt_idx=j,
                    iou=iou,
                    distance=dist,
                ))
                matched_det.add(i)
                matched_gt.add(j)

        unmatched_det = [i for i in range(n_det) if i not in matched_det]
        unmatched_gt = [j for j in range(n_gt) if j not in matched_gt]

        return matches, unmatched_det, unmatched_gt

    def add_frame(
        self,
        detections: List[Detection3D],
        ground_truths: List[GroundTruth3D],
    ):
        """
        Add detections and ground truths from one frame.

        Args:
            detections: List of 3D detections
            ground_truths: List of 3D ground truths
        """
        matches, unmatched_det, unmatched_gt = self.match_detections(
            detections, ground_truths
        )

        self.results.num_detections += len(detections)
        self.results.num_ground_truths += len(ground_truths)
        self.results.num_true_positives += len(matches)
        self.results.num_false_positives += len(unmatched_det)
        self.results.num_false_negatives += len(unmatched_gt)

        # Compute errors for matched pairs
        for match in matches:
            det = detections[match.det_idx]
            gt = ground_truths[match.gt_idx]

            # Center error
            center_error = compute_center_distance(det.center, gt.center)
            self.results.center_errors.append(center_error)

            # Orientation error
            orient_error = compute_orientation_error(det.rotation_y, gt.rotation_y)
            self.results.orientation_errors.append(orient_error)

            # Dimension error
            dim_error = compute_dimension_error(det.dimensions, gt.dimensions)
            self.results.dimension_errors.append(dim_error)

            # 3D IoU
            iou_3d = compute_iou_3d(
                det.center, det.dimensions, det.rotation_y,
                gt.center, gt.dimensions, gt.rotation_y
            )
            self.results.iou_3d_values.append(iou_3d)

            # Distance from ego
            distance = np.linalg.norm(gt.center)
            self.results.distances.append(distance)

            # Per-class accumulation
            cls = det.class_name
            if cls in self.per_class_errors:
                self.per_class_errors[cls]["center_errors"].append(center_error)
                self.per_class_errors[cls]["orientation_errors"].append(orient_error)
                self.per_class_errors[cls]["dimension_errors"].append(dim_error)
                self.per_class_errors[cls]["iou_3d"].append(iou_3d)
                self.per_class_errors[cls]["distances"].append(distance)

    def compute_metrics(self) -> MetricsResult:
        """
        Compute final metrics.

        Returns:
            MetricsResult with all computed metrics.
        """
        r = self.results

        # Overall precision/recall
        if (r.num_true_positives + r.num_false_positives) > 0:
            r.precision_2d = r.num_true_positives / (r.num_true_positives + r.num_false_positives)
        if (r.num_true_positives + r.num_false_negatives) > 0:
            r.recall_2d = r.num_true_positives / (r.num_true_positives + r.num_false_negatives)
        if (r.precision_2d + r.recall_2d) > 0:
            r.f1_2d = 2 * r.precision_2d * r.recall_2d / (r.precision_2d + r.recall_2d)

        # 3D error metrics
        if r.center_errors:
            r.mean_center_error = np.mean(r.center_errors)
            r.median_center_error = np.median(r.center_errors)

        if r.orientation_errors:
            r.mean_orientation_error = np.mean(r.orientation_errors)

        if r.dimension_errors:
            r.mean_dimension_error = np.mean(r.dimension_errors, axis=0)

        if r.iou_3d_values:
            r.mean_iou_3d = np.mean(r.iou_3d_values)

        # Distance-binned metrics
        for bin_min, bin_max in self.distance_bins:
            bin_name = f"{bin_min}-{bin_max}m"

            # Filter by distance
            mask = [(bin_min <= d < bin_max) for d in r.distances]

            bin_center_errors = [e for e, m in zip(r.center_errors, mask) if m]
            bin_ious = [i for i, m in zip(r.iou_3d_values, mask) if m]

            r.metrics_by_distance[bin_name] = {
                "count": sum(mask),
                "mean_center_error": np.mean(bin_center_errors) if bin_center_errors else 0.0,
                "mean_iou_3d": np.mean(bin_ious) if bin_ious else 0.0,
            }

        # Per-class metrics
        for cls in self.classes:
            errors = self.per_class_errors[cls]
            if errors["center_errors"]:
                r.per_class_precision[cls] = len(errors["center_errors"]) / max(1, r.num_ground_truths)
                r.per_class_recall[cls] = len(errors["center_errors"]) / max(1, r.num_ground_truths)

        return r


# =============================================================================
# Combined Metrics Calculator
# =============================================================================

class MetricsCalculator:
    """
    Combined 2D and 3D metrics calculator.

    Provides a unified interface for computing all evaluation metrics.
    """

    def __init__(
        self,
        iou_threshold_2d: float = 0.5,
        iou_threshold_3d: float = 0.5,
        distance_threshold: float = 2.0,
        classes: List[str] = None,
    ):
        """
        Initialize combined metrics calculator.

        Args:
            iou_threshold_2d: 2D IoU threshold for mAP
            iou_threshold_3d: 3D IoU threshold for matching
            distance_threshold: Max center distance for 3D matching
            classes: List of class names
        """
        self.classes = classes or ["Car", "Pedestrian", "Cyclist"]

        self.calc_2d = Metrics2DCalculator(
            iou_threshold=iou_threshold_2d,
            classes=self.classes,
        )

        self.calc_3d = Metrics3DCalculator(
            iou_threshold=iou_threshold_3d,
            distance_threshold=distance_threshold,
            classes=self.classes,
        )

    def reset(self):
        """Reset all accumulators."""
        self.calc_2d.reset()
        self.calc_3d.reset()

    def add_frame_2d(
        self,
        detections: List[Detection2D],
        ground_truths: List[GroundTruth2D],
    ):
        """Add 2D detections and ground truths."""
        self.calc_2d.add_frame(detections, ground_truths)

    def add_frame_3d(
        self,
        detections: List[Detection3D],
        ground_truths: List[GroundTruth3D],
    ):
        """Add 3D detections and ground truths."""
        self.calc_3d.add_frame(detections, ground_truths)

    def compute_all(self) -> Dict[str, Any]:
        """
        Compute all metrics.

        Returns:
            Dictionary with 2D and 3D metrics.
        """
        metrics_2d = self.calc_2d.compute_metrics()
        metrics_3d = self.calc_3d.compute_metrics()

        return {
            "2d": metrics_2d,
            "3d": metrics_3d,
        }

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of key metrics.

        Returns:
            Dictionary with key metric values.
        """
        all_metrics = self.compute_all()

        m2d = all_metrics["2d"]
        m3d = all_metrics["3d"]

        return {
            "mAP@0.5": m2d["overall"]["mAP"],
            "precision_2d": m2d["overall"]["precision"],
            "recall_2d": m2d["overall"]["recall"],
            "precision_3d": m3d.precision_2d,
            "recall_3d": m3d.recall_2d,
            "mean_center_error": m3d.mean_center_error,
            "mean_orientation_error_deg": np.degrees(m3d.mean_orientation_error),
            "mean_iou_3d": m3d.mean_iou_3d,
        }


# =============================================================================
# KITTI Label Parser
# =============================================================================

def parse_kitti_label(label_path: str) -> Tuple[List[GroundTruth2D], List[GroundTruth3D]]:
    """
    Parse KITTI label file.

    KITTI label format:
    type truncated occluded alpha bbox_2d(4) dimensions(3) location(3) rotation_y [score]

    Args:
        label_path: Path to label file

    Returns:
        Tuple of (2D ground truths, 3D ground truths)
    """
    gts_2d = []
    gts_3d = []

    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue

                class_name = parts[0]

                # Skip DontCare and other classes
                if class_name in ["DontCare", "Misc"]:
                    continue

                truncated = float(parts[1])
                occluded = int(parts[2])
                alpha = float(parts[3])

                # 2D bbox: left, top, right, bottom
                bbox_2d = np.array([float(parts[4]), float(parts[5]),
                                    float(parts[6]), float(parts[7])])

                # Dimensions: height, width, length
                h, w, l = float(parts[8]), float(parts[9]), float(parts[10])

                # Location: x, y, z in camera coordinates
                x, y, z = float(parts[11]), float(parts[12]), float(parts[13])

                # Rotation around Y-axis
                rotation_y = float(parts[14])

                # Determine if difficult (heavily truncated or occluded)
                difficult = truncated > 0.5 or occluded > 1

                # 2D ground truth
                gts_2d.append(GroundTruth2D(
                    bbox=bbox_2d,
                    class_name=class_name,
                    difficult=difficult,
                    truncated=truncated,
                    occluded=occluded,
                ))

                # 3D ground truth
                gts_3d.append(GroundTruth3D(
                    center=np.array([x, y, z]),
                    dimensions=np.array([l, w, h]),  # length, width, height
                    rotation_y=rotation_y,
                    class_name=class_name,
                    truncated=truncated,
                    occluded=occluded,
                    alpha=alpha,
                    bbox_2d=bbox_2d,
                ))

    except FileNotFoundError:
        pass

    return gts_2d, gts_3d
