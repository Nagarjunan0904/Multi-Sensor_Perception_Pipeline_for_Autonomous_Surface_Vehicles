"""
Outlier Filtering for 3D Detections.

This module provides comprehensive filtering for 3D object detections:
- Geometric constraints (depth, height, dimensions)
- Confidence-based filtering
- Physical plausibility checks
- Temporal consistency (for sequences)
- Visualization utilities for debugging

Author: Multi-Sensor Perception Pipeline
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np

from .bbox3d_generator import BBox3D


# =============================================================================
# Enums and Data Classes
# =============================================================================

class FilterReason(Enum):
    """Reasons why a detection was filtered."""
    VALID = "valid"
    LOW_SCORE = "low_score"
    DEPTH_OUT_OF_RANGE = "depth_out_of_range"
    HEIGHT_OUT_OF_RANGE = "height_out_of_range"
    INVALID_DIMENSIONS = "invalid_dimensions"
    LOW_POINT_DENSITY = "low_point_density"
    DEPTH_INCONSISTENT = "depth_inconsistent"
    PHYSICALLY_IMPLAUSIBLE = "physically_implausible"
    SUPPRESSED_BY_NMS = "suppressed_by_nms"


@dataclass
class FilterResult:
    """
    Result of filtering a single detection.

    Attributes:
        is_valid: Whether the detection passed all filters.
        reason: Reason for filtering (if filtered).
        details: Additional details about the filter decision.
        confidence_adjustment: Suggested confidence adjustment factor.
    """
    is_valid: bool = True
    reason: FilterReason = FilterReason.VALID
    details: str = ""
    confidence_adjustment: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "reason": self.reason.value,
            "details": self.details,
            "confidence_adjustment": self.confidence_adjustment,
        }


@dataclass
class FilterStats:
    """
    Statistics from filtering operation.

    Attributes:
        total_input: Number of input detections.
        total_valid: Number of valid detections after filtering.
        filtered_by_reason: Count of filtered detections by reason.
    """
    total_input: int = 0
    total_valid: int = 0
    filtered_by_reason: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_input": self.total_input,
            "total_valid": self.total_valid,
            "filtered_by_reason": self.filtered_by_reason,
            "filter_rate": 1 - (self.total_valid / self.total_input) if self.total_input > 0 else 0,
        }


# =============================================================================
# Dimension Constraints
# =============================================================================

# Expected dimension ranges per class (min, max) for (length, width, height)
CLASS_DIMENSION_BOUNDS = {
    "Car": {
        "length": (2.5, 6.0),
        "width": (1.2, 2.5),
        "height": (1.0, 2.2),
    },
    "Van": {
        "length": (3.5, 7.0),
        "width": (1.5, 2.8),
        "height": (1.5, 3.0),
    },
    "Truck": {
        "length": (5.0, 20.0),
        "width": (1.8, 3.5),
        "height": (2.0, 4.5),
    },
    "Pedestrian": {
        "length": (0.2, 1.2),
        "width": (0.2, 1.0),
        "height": (1.0, 2.2),
    },
    "Person_sitting": {
        "length": (0.3, 1.0),
        "width": (0.3, 0.8),
        "height": (0.8, 1.5),
    },
    "Cyclist": {
        "length": (1.0, 2.5),
        "width": (0.3, 1.2),
        "height": (1.2, 2.2),
    },
    "Tram": {
        "length": (10.0, 30.0),
        "width": (2.0, 3.5),
        "height": (2.5, 4.5),
    },
    "Misc": {
        "length": (0.3, 15.0),
        "width": (0.3, 5.0),
        "height": (0.3, 5.0),
    },
}

# Generic bounds for unknown classes
DEFAULT_DIMENSION_BOUNDS = {
    "length": (0.3, 20.0),
    "width": (0.2, 5.0),
    "height": (0.3, 5.0),
}


# =============================================================================
# Main Filter Class
# =============================================================================

class OutlierFilter:
    """
    Filter outlier 3D detections based on multiple criteria.

    This filter applies the following checks:
    1. Confidence threshold
    2. Depth range (valid sensing range)
    3. Height range (ground plane constraints)
    4. Dimension validity (class-specific bounds)
    5. Point density requirements
    6. Depth consistency check
    7. Physical plausibility

    Example:
        >>> filter = OutlierFilter(depth_range=(1.0, 50.0))
        >>> filtered_boxes = filter.filter(boxes)
    """

    def __init__(
        self,
        # Basic thresholds
        score_threshold: float = 0.3,
        depth_range: Tuple[float, float] = (0.5, 80.0),
        height_range: Tuple[float, float] = (-3.0, 3.0),
        # Dimension constraints
        dimension_tolerance: float = 1.5,
        custom_dimension_bounds: Optional[Dict[str, Dict]] = None,
        # Point-based filtering
        min_points: int = 3,
        min_depth_consistency: float = 0.3,
        # Physical constraints
        max_aspect_ratio: float = 10.0,
        min_volume: float = 0.01,
        max_volume: float = 500.0,
        # NMS settings
        nms_iou_threshold: float = 0.5,
        apply_nms: bool = True,
    ):
        """
        Initialize outlier filter.

        Args:
            score_threshold: Minimum detection confidence score.
            depth_range: Valid depth range (min, max) in meters.
            height_range: Valid height range (y in camera frame).
            dimension_tolerance: Tolerance factor for dimension bounds.
            custom_dimension_bounds: Override default dimension bounds.
            min_points: Minimum LiDAR points required.
            min_depth_consistency: Minimum depth consistency score.
            max_aspect_ratio: Maximum dimension aspect ratio.
            min_volume: Minimum box volume in cubic meters.
            max_volume: Maximum box volume in cubic meters.
            nms_iou_threshold: IoU threshold for 3D NMS.
            apply_nms: Whether to apply NMS.
        """
        self.score_threshold = score_threshold
        self.depth_range = depth_range
        self.height_range = height_range
        self.dimension_tolerance = dimension_tolerance
        self.min_points = min_points
        self.min_depth_consistency = min_depth_consistency
        self.max_aspect_ratio = max_aspect_ratio
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.nms_iou_threshold = nms_iou_threshold
        self.apply_nms = apply_nms

        # Dimension bounds
        self.dimension_bounds = CLASS_DIMENSION_BOUNDS.copy()
        if custom_dimension_bounds:
            self.dimension_bounds.update(custom_dimension_bounds)

        # Statistics
        self._last_stats = FilterStats()

    # -------------------------------------------------------------------------
    # Main Filtering Methods
    # -------------------------------------------------------------------------

    def filter(
        self,
        boxes: List[Optional[BBox3D]],
        return_results: bool = False,
    ) -> List[Optional[BBox3D]]:
        """
        Filter outlier 3D boxes.

        Args:
            boxes: List of 3D boxes (may contain None).
            return_results: Whether to return detailed filter results.

        Returns:
            Filtered list with outliers set to None.
            If return_results=True, returns (filtered_boxes, filter_results).
        """
        self._last_stats = FilterStats(total_input=len(boxes))

        results = []
        filter_results = []

        for box in boxes:
            if box is None:
                results.append(None)
                filter_results.append(FilterResult(is_valid=False, reason=FilterReason.VALID))
                continue

            result = self._check_box(box)
            filter_results.append(result)

            if result.is_valid:
                results.append(box)
                self._last_stats.total_valid += 1
            else:
                results.append(None)
                reason_str = result.reason.value
                self._last_stats.filtered_by_reason[reason_str] = \
                    self._last_stats.filtered_by_reason.get(reason_str, 0) + 1

        # Apply NMS if enabled
        if self.apply_nms:
            results = self._apply_nms(results)

        if return_results:
            return results, filter_results
        return results

    def filter_with_adjustments(
        self,
        boxes: List[Optional[BBox3D]],
    ) -> Tuple[List[Optional[BBox3D]], List[float]]:
        """
        Filter boxes and return confidence adjustments.

        Boxes that pass filtering may have their confidence adjusted
        based on how close they are to filter thresholds.

        Args:
            boxes: List of 3D boxes.

        Returns:
            Tuple of (filtered_boxes, confidence_adjustments).
        """
        filtered, results = self.filter(boxes, return_results=True)

        adjustments = []
        for i, (box, result) in enumerate(zip(boxes, results)):
            if box is None or filtered[i] is None:
                adjustments.append(0.0)
            else:
                adjustments.append(result.confidence_adjustment)

        return filtered, adjustments

    # -------------------------------------------------------------------------
    # Individual Checks
    # -------------------------------------------------------------------------

    def _check_box(self, box: BBox3D) -> FilterResult:
        """
        Check if a single box is valid.

        Applies all filter checks in order of computational cost.
        """
        # 1. Score threshold (cheapest check first)
        if box.score < self.score_threshold:
            return FilterResult(
                is_valid=False,
                reason=FilterReason.LOW_SCORE,
                details=f"Score {box.score:.3f} < threshold {self.score_threshold}",
            )

        # 2. Depth range
        depth = box.center[2]
        if not (self.depth_range[0] <= depth <= self.depth_range[1]):
            return FilterResult(
                is_valid=False,
                reason=FilterReason.DEPTH_OUT_OF_RANGE,
                details=f"Depth {depth:.2f}m outside range {self.depth_range}",
            )

        # 3. Height range (y coordinate)
        height_y = box.center[1]
        if not (self.height_range[0] <= height_y <= self.height_range[1]):
            return FilterResult(
                is_valid=False,
                reason=FilterReason.HEIGHT_OUT_OF_RANGE,
                details=f"Height {height_y:.2f}m outside range {self.height_range}",
            )

        # 4. Dimension validity
        dim_result = self._check_dimensions(box)
        if not dim_result.is_valid:
            return dim_result

        # 5. Point density (if stats available)
        if hasattr(box, 'stats') and box.stats.num_points < self.min_points:
            return FilterResult(
                is_valid=False,
                reason=FilterReason.LOW_POINT_DENSITY,
                details=f"Only {box.stats.num_points} points (min: {self.min_points})",
            )

        # 6. Depth consistency (if confidence available)
        if hasattr(box, 'confidence'):
            if box.confidence.depth_consistency < self.min_depth_consistency:
                return FilterResult(
                    is_valid=False,
                    reason=FilterReason.DEPTH_INCONSISTENT,
                    details=f"Depth consistency {box.confidence.depth_consistency:.2f} < {self.min_depth_consistency}",
                )

        # 7. Physical plausibility
        plaus_result = self._check_physical_plausibility(box)
        if not plaus_result.is_valid:
            return plaus_result

        # All checks passed - compute confidence adjustment
        adjustment = self._compute_confidence_adjustment(box)

        return FilterResult(
            is_valid=True,
            reason=FilterReason.VALID,
            confidence_adjustment=adjustment,
        )

    def _check_dimensions(self, box: BBox3D) -> FilterResult:
        """Check if box dimensions are within valid bounds."""
        l, w, h = box.dimensions

        # Get class-specific bounds
        if box.class_name in self.dimension_bounds:
            bounds = self.dimension_bounds[box.class_name]
        else:
            bounds = DEFAULT_DIMENSION_BOUNDS

        tol = self.dimension_tolerance

        # Check length
        l_min, l_max = bounds["length"]
        if not (l_min / tol <= l <= l_max * tol):
            return FilterResult(
                is_valid=False,
                reason=FilterReason.INVALID_DIMENSIONS,
                details=f"Length {l:.2f}m outside bounds [{l_min/tol:.2f}, {l_max*tol:.2f}]",
            )

        # Check width
        w_min, w_max = bounds["width"]
        if not (w_min / tol <= w <= w_max * tol):
            return FilterResult(
                is_valid=False,
                reason=FilterReason.INVALID_DIMENSIONS,
                details=f"Width {w:.2f}m outside bounds [{w_min/tol:.2f}, {w_max*tol:.2f}]",
            )

        # Check height
        h_min, h_max = bounds["height"]
        if not (h_min / tol <= h <= h_max * tol):
            return FilterResult(
                is_valid=False,
                reason=FilterReason.INVALID_DIMENSIONS,
                details=f"Height {h:.2f}m outside bounds [{h_min/tol:.2f}, {h_max*tol:.2f}]",
            )

        return FilterResult(is_valid=True)

    def _check_physical_plausibility(self, box: BBox3D) -> FilterResult:
        """Check physical plausibility of the box."""
        l, w, h = box.dimensions

        # Aspect ratio check
        max_dim = max(l, w, h)
        min_dim = max(min(l, w, h), 0.01)  # Avoid division by zero
        aspect_ratio = max_dim / min_dim

        if aspect_ratio > self.max_aspect_ratio:
            return FilterResult(
                is_valid=False,
                reason=FilterReason.PHYSICALLY_IMPLAUSIBLE,
                details=f"Aspect ratio {aspect_ratio:.1f} > max {self.max_aspect_ratio}",
            )

        # Volume check
        volume = l * w * h
        if volume < self.min_volume:
            return FilterResult(
                is_valid=False,
                reason=FilterReason.PHYSICALLY_IMPLAUSIBLE,
                details=f"Volume {volume:.3f}m³ < min {self.min_volume}m³",
            )

        if volume > self.max_volume:
            return FilterResult(
                is_valid=False,
                reason=FilterReason.PHYSICALLY_IMPLAUSIBLE,
                details=f"Volume {volume:.1f}m³ > max {self.max_volume}m³",
            )

        return FilterResult(is_valid=True)

    def _compute_confidence_adjustment(self, box: BBox3D) -> float:
        """
        Compute confidence adjustment based on how close box is to thresholds.

        Returns a factor in [0.5, 1.0] where lower values indicate
        the box is closer to filter thresholds.
        """
        adjustment = 1.0

        # Depth penalty (objects at extreme ranges)
        depth = box.center[2]
        depth_mid = (self.depth_range[0] + self.depth_range[1]) / 2
        depth_range = (self.depth_range[1] - self.depth_range[0]) / 2
        depth_normalized = abs(depth - depth_mid) / depth_range
        depth_penalty = max(0.0, 1.0 - depth_normalized * 0.3)
        adjustment *= depth_penalty

        # Score margin penalty
        score_margin = box.score - self.score_threshold
        score_range = 1.0 - self.score_threshold
        if score_range > 0:
            score_factor = min(1.0, 0.5 + 0.5 * (score_margin / score_range))
            adjustment *= score_factor

        return max(0.5, adjustment)

    # -------------------------------------------------------------------------
    # NMS (Non-Maximum Suppression)
    # -------------------------------------------------------------------------

    def _apply_nms(self, boxes: List[Optional[BBox3D]]) -> List[Optional[BBox3D]]:
        """Apply 3D NMS to remove overlapping boxes."""
        valid_indices = [i for i, box in enumerate(boxes) if box is not None]

        if len(valid_indices) <= 1:
            return boxes

        # Sort by score
        valid_boxes = [(i, boxes[i]) for i in valid_indices]
        valid_boxes.sort(key=lambda x: x[1].score, reverse=True)

        keep_indices = set()
        suppressed = set()

        for i, (idx, box) in enumerate(valid_boxes):
            if idx in suppressed:
                continue

            keep_indices.add(idx)

            # Check overlap with remaining boxes
            for j in range(i + 1, len(valid_boxes)):
                other_idx, other_box = valid_boxes[j]

                if other_idx in suppressed:
                    continue

                iou = self._compute_3d_iou(box, other_box)

                if iou > self.nms_iou_threshold:
                    suppressed.add(other_idx)

        # Update statistics
        for idx in suppressed:
            self._last_stats.filtered_by_reason["suppressed_by_nms"] = \
                self._last_stats.filtered_by_reason.get("suppressed_by_nms", 0) + 1
            self._last_stats.total_valid -= 1

        # Build result
        result = []
        for i, box in enumerate(boxes):
            if box is None or i in suppressed:
                result.append(None)
            else:
                result.append(box)

        return result

    def _compute_3d_iou(self, box1: BBox3D, box2: BBox3D) -> float:
        """
        Compute 3D IoU between two boxes (axis-aligned approximation).

        For more accurate IoU with rotations, a proper rotated box
        intersection algorithm would be needed.
        """
        c1, d1 = box1.center, box1.dimensions
        c2, d2 = box2.center, box2.dimensions

        # Compute overlap in each dimension
        # Mapping: dims = (length, width, height), center = (x, y, z)
        # length -> x, width -> z, height -> y
        overlap = np.ones(3)

        for i in range(3):
            # Map dimension index to center coordinate
            if i == 0:  # X dimension (length)
                half1, half2 = d1[0] / 2, d2[0] / 2
            elif i == 1:  # Y dimension (height)
                half1, half2 = d1[2] / 2, d2[2] / 2
            else:  # Z dimension (width)
                half1, half2 = d1[1] / 2, d2[1] / 2

            min1, max1 = c1[i] - half1, c1[i] + half1
            min2, max2 = c2[i] - half2, c2[i] + half2

            overlap[i] = max(0, min(max1, max2) - max(min1, min2))

        intersection = np.prod(overlap)
        vol1 = np.prod(d1)
        vol2 = np.prod(d2)
        union = vol1 + vol2 - intersection

        return intersection / union if union > 0 else 0

    # -------------------------------------------------------------------------
    # Statistics and Debugging
    # -------------------------------------------------------------------------

    @property
    def last_stats(self) -> FilterStats:
        """Get statistics from last filter operation."""
        return self._last_stats

    def get_filter_summary(self) -> str:
        """Get human-readable summary of last filter operation."""
        stats = self._last_stats
        lines = [
            f"Filter Summary:",
            f"  Input: {stats.total_input} boxes",
            f"  Valid: {stats.total_valid} boxes ({100*stats.total_valid/max(1,stats.total_input):.1f}%)",
            f"  Filtered by reason:",
        ]

        for reason, count in sorted(stats.filtered_by_reason.items()):
            lines.append(f"    - {reason}: {count}")

        return "\n".join(lines)


# =============================================================================
# Specialized Filters
# =============================================================================

class TemporalFilter:
    """
    Filter based on temporal consistency across frames.

    Tracks object positions over time and filters detections
    that don't match expected motion patterns.
    """

    def __init__(
        self,
        max_velocity: float = 30.0,  # m/s
        max_acceleration: float = 10.0,  # m/s²
        history_length: int = 5,
        position_threshold: float = 2.0,  # meters
    ):
        """
        Initialize temporal filter.

        Args:
            max_velocity: Maximum expected object velocity.
            max_acceleration: Maximum expected acceleration.
            history_length: Number of frames to track.
            position_threshold: Maximum position jump threshold.
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.history_length = history_length
        self.position_threshold = position_threshold

        # Track history: {track_id: [(frame_id, position), ...]}
        self.history: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        self.current_frame = 0

    def filter(
        self,
        boxes: List[Optional[BBox3D]],
        frame_id: int,
        dt: float = 0.1,
    ) -> List[Optional[BBox3D]]:
        """
        Filter boxes based on temporal consistency.

        Args:
            boxes: List of 3D boxes.
            frame_id: Current frame ID.
            dt: Time delta between frames.

        Returns:
            Filtered boxes.
        """
        self.current_frame = frame_id
        results = []

        for box in boxes:
            if box is None:
                results.append(None)
                continue

            # Check temporal consistency
            if self._is_temporally_consistent(box, dt):
                results.append(box)
                self._update_history(box, frame_id)
            else:
                results.append(None)

        # Clean old history
        self._clean_history()

        return results

    def _is_temporally_consistent(self, box: BBox3D, dt: float) -> bool:
        """Check if box position is consistent with history."""
        # New tracks are always accepted
        track_id = box.detection_2d.track_id if hasattr(box.detection_2d, 'track_id') else None

        if track_id is None or track_id not in self.history:
            return True

        history = self.history[track_id]
        if len(history) == 0:
            return True

        # Check velocity constraint
        last_frame, last_pos = history[-1]
        frames_elapsed = self.current_frame - last_frame
        if frames_elapsed <= 0:
            return True

        time_elapsed = frames_elapsed * dt
        distance = np.linalg.norm(box.center - last_pos)
        velocity = distance / time_elapsed

        if velocity > self.max_velocity:
            return False

        # Check acceleration if we have enough history
        if len(history) >= 2:
            prev_frame, prev_pos = history[-2]
            prev_velocity = np.linalg.norm(last_pos - prev_pos) / ((last_frame - prev_frame) * dt)
            acceleration = abs(velocity - prev_velocity) / time_elapsed

            if acceleration > self.max_acceleration:
                return False

        return True

    def _update_history(self, box: BBox3D, frame_id: int):
        """Update tracking history."""
        track_id = box.detection_2d.track_id if hasattr(box.detection_2d, 'track_id') else id(box)

        if track_id not in self.history:
            self.history[track_id] = []

        self.history[track_id].append((frame_id, box.center.copy()))

        # Trim history
        if len(self.history[track_id]) > self.history_length:
            self.history[track_id] = self.history[track_id][-self.history_length:]

    def _clean_history(self):
        """Remove old tracks from history."""
        to_remove = []
        for track_id, history in self.history.items():
            if len(history) == 0:
                to_remove.append(track_id)
            elif self.current_frame - history[-1][0] > self.history_length * 2:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.history[track_id]

    def reset(self):
        """Reset tracking history."""
        self.history.clear()
        self.current_frame = 0


class GroundPlaneFilter:
    """
    Filter based on ground plane constraints.

    Objects should be positioned on or near the ground plane.
    """

    def __init__(
        self,
        camera_height: float = 1.65,
        ground_tolerance: float = 0.5,
        max_floating_height: float = 0.3,
    ):
        """
        Initialize ground plane filter.

        Args:
            camera_height: Camera height above ground in meters.
            ground_tolerance: Tolerance for ground plane position.
            max_floating_height: Maximum height above ground for bottom of box.
        """
        self.camera_height = camera_height
        self.ground_tolerance = ground_tolerance
        self.max_floating_height = max_floating_height

    def filter(self, boxes: List[Optional[BBox3D]]) -> List[Optional[BBox3D]]:
        """
        Filter boxes based on ground plane constraints.

        Args:
            boxes: List of 3D boxes.

        Returns:
            Filtered boxes.
        """
        results = []

        for box in boxes:
            if box is None:
                results.append(None)
                continue

            if self._is_grounded(box):
                results.append(box)
            else:
                results.append(None)

        return results

    def _is_grounded(self, box: BBox3D) -> bool:
        """Check if box is properly grounded."""
        # In camera frame, Y points down
        # Bottom of box (center_y + height/2) should be near ground plane
        center_y = box.center[1]
        height = box.dimensions[2]
        bottom_y = center_y + height / 2

        # Ground plane is at y = camera_height (approximately)
        expected_ground = self.camera_height

        # Check if bottom is near ground
        ground_error = abs(bottom_y - expected_ground)

        if ground_error > self.ground_tolerance + self.max_floating_height:
            return False

        # Check if bottom is not below ground (buried)
        if bottom_y > expected_ground + self.ground_tolerance:
            return False

        return True


# =============================================================================
# Visualization Utilities
# =============================================================================

class FilterVisualizer:
    """Visualization utilities for filter debugging."""

    @staticmethod
    def create_filter_summary_image(
        stats: FilterStats,
        image_size: Tuple[int, int] = (400, 300),
    ) -> np.ndarray:
        """
        Create summary visualization of filter statistics.

        Args:
            stats: Filter statistics.
            image_size: Output image size (width, height).

        Returns:
            Visualization image.
        """
        import cv2

        w, h = image_size
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[:] = (40, 40, 40)  # Dark gray background

        # Title
        cv2.putText(
            image, "Filter Summary",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        # Statistics
        y = 60
        cv2.putText(
            image, f"Input: {stats.total_input}",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        y += 25

        cv2.putText(
            image, f"Valid: {stats.total_valid}",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
        y += 25

        filter_rate = 1 - (stats.total_valid / max(1, stats.total_input))
        cv2.putText(
            image, f"Filter Rate: {100*filter_rate:.1f}%",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        y += 35

        # Bar chart of filter reasons
        cv2.putText(
            image, "Filtered by:",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        y += 25

        max_count = max(stats.filtered_by_reason.values()) if stats.filtered_by_reason else 1
        bar_width = w - 150

        colors = {
            "low_score": (100, 100, 255),
            "depth_out_of_range": (100, 255, 100),
            "height_out_of_range": (255, 100, 100),
            "invalid_dimensions": (255, 255, 100),
            "low_point_density": (100, 255, 255),
            "depth_inconsistent": (255, 100, 255),
            "physically_implausible": (200, 200, 200),
            "suppressed_by_nms": (150, 150, 255),
        }

        for reason, count in sorted(stats.filtered_by_reason.items()):
            if y > h - 20:
                break

            # Shorten reason name
            short_name = reason[:15] + "..." if len(reason) > 15 else reason
            cv2.putText(
                image, f"{short_name}:",
                (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1
            )

            # Draw bar
            bar_len = int((count / max_count) * bar_width * 0.6)
            color = colors.get(reason, (128, 128, 128))
            cv2.rectangle(image, (130, y - 12), (130 + bar_len, y + 2), color, -1)

            # Count label
            cv2.putText(
                image, str(count),
                (135 + bar_len, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )

            y += 22

        return image

    @staticmethod
    def annotate_filtered_boxes(
        image: np.ndarray,
        boxes: List[Optional[BBox3D]],
        filter_results: List[FilterResult],
    ) -> np.ndarray:
        """
        Annotate image showing which boxes were filtered and why.

        Args:
            image: Input image.
            boxes: Original boxes (before filtering).
            filter_results: Filter results for each box.

        Returns:
            Annotated image.
        """
        import cv2

        result = image.copy()

        for box, fr in zip(boxes, filter_results):
            if box is None:
                continue

            # Get 2D box from detection
            bbox_2d = box.detection_2d.bbox
            x1, y1, x2, y2 = bbox_2d.astype(int)

            if fr.is_valid:
                # Valid - green box
                color = (0, 255, 0)
                label = f"{box.class_name} {box.score:.2f}"
            else:
                # Filtered - red box with reason
                color = (0, 0, 255)
                label = f"FILTERED: {fr.reason.value}"

            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                result, label,
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

        return result
