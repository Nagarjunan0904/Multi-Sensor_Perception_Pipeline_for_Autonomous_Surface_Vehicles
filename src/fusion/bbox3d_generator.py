"""
3D Bounding Box Generation from 2D Detections and LiDAR Points.

This module provides comprehensive 3D object estimation including:
- Center estimation from LiDAR point clouds
- Dimension estimation using PCA and point statistics
- Orientation estimation using principal axis analysis
- Confidence scoring based on multiple factors
- Visualization utilities for debugging

Author: Multi-Sensor Perception Pipeline
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any

import numpy as np
from scipy import stats
from scipy.spatial import ConvexHull

# Handle imports flexibly for both package and direct usage
try:
    from calibration.intrinsics import CameraIntrinsics
    from perception2d.detector import Detection
except ImportError:
    try:
        from ..calibration.intrinsics import CameraIntrinsics
        from ..perception2d.detector import Detection
    except ImportError:
        if TYPE_CHECKING:
            from calibration.intrinsics import CameraIntrinsics
            from perception2d.detector import Detection


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ConfidenceScore:
    """
    Detailed confidence scoring for 3D estimation.

    Attributes:
        overall: Combined confidence score [0, 1].
        point_density: Score based on number of LiDAR points.
        detection_2d: Score from 2D detector confidence.
        depth_consistency: Score based on depth variance.
        dimension_fit: Score based on how well dimensions match class prior.
        coverage: Score based on point cloud coverage of estimated box.
    """
    overall: float = 0.0
    point_density: float = 0.0
    detection_2d: float = 0.0
    depth_consistency: float = 0.0
    dimension_fit: float = 0.0
    coverage: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "overall": self.overall,
            "point_density": self.point_density,
            "detection_2d": self.detection_2d,
            "depth_consistency": self.depth_consistency,
            "dimension_fit": self.dimension_fit,
            "coverage": self.coverage,
        }


@dataclass
class EstimationStats:
    """
    Statistics from 3D estimation process (for debugging).

    Attributes:
        num_points: Number of LiDAR points used.
        depth_mean: Mean depth of points.
        depth_std: Standard deviation of depth.
        point_spread: Range of points in each dimension [x, y, z].
        eigenvalues: PCA eigenvalues for orientation estimation.
        method_used: Which estimation method was used.
    """
    num_points: int = 0
    depth_mean: float = 0.0
    depth_std: float = 0.0
    point_spread: np.ndarray = field(default_factory=lambda: np.zeros(3))
    eigenvalues: np.ndarray = field(default_factory=lambda: np.zeros(3))
    method_used: str = "prior"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_points": self.num_points,
            "depth_mean": self.depth_mean,
            "depth_std": self.depth_std,
            "point_spread": self.point_spread.tolist(),
            "eigenvalues": self.eigenvalues.tolist(),
            "method_used": self.method_used,
        }


@dataclass
class BBox3D:
    """
    3D Bounding Box representation.

    Coordinate System (KITTI camera frame):
    - X: right
    - Y: down
    - Z: forward (depth)

    Attributes:
        center: (x, y, z) center position in camera frame.
        dimensions: (length, width, height) in meters.
        rotation_y: Rotation around Y-axis in radians.
        class_name: Object class name.
        score: Overall confidence score.
        detection_2d: Associated 2D detection.
        confidence: Detailed confidence breakdown.
        stats: Estimation statistics for debugging.
    """
    center: np.ndarray
    dimensions: np.ndarray
    rotation_y: float
    class_name: str
    score: float
    detection_2d: Detection
    confidence: ConfidenceScore = field(default_factory=ConfidenceScore)
    stats: EstimationStats = field(default_factory=EstimationStats)

    @property
    def corners(self) -> np.ndarray:
        """
        Get 8 corners of the 3D box in camera frame.

        Corner ordering (looking from above, Y pointing into page):
            4 ---- 5
           /|    /|
          7 ---- 6 |
          | 0 --|- 1
          |/    |/
          3 ---- 2

        Returns:
            Corners array (8, 3) in camera frame.
        """
        return compute_corners_3d(self.center, self.dimensions, self.rotation_y)

    @property
    def corners_2d(self) -> Optional[np.ndarray]:
        """
        Project corners to 2D image plane.

        Requires detection_2d to have intrinsics info.
        Returns None if projection not possible.
        """
        # This would need camera intrinsics - return None by default
        return None

    @property
    def volume(self) -> float:
        """Get box volume in cubic meters."""
        return float(np.prod(self.dimensions))

    @property
    def depth(self) -> float:
        """Get depth (Z coordinate) in meters."""
        return float(self.center[2])

    def to_kitti_format(self) -> str:
        """
        Convert to KITTI label format.

        Format: class truncated occluded alpha bbox dimensions location rotation score

        Returns:
            KITTI format string.
        """
        box_2d = self.detection_2d.bbox
        x, y, z = self.center
        l, w, h = self.dimensions

        # Compute alpha (observation angle)
        alpha = self.rotation_y - np.arctan2(x, z)
        # Normalize alpha to [-pi, pi]
        while alpha > np.pi:
            alpha -= 2 * np.pi
        while alpha < -np.pi:
            alpha += 2 * np.pi

        return (
            f"{self.class_name} 0.0 0 {alpha:.2f} "
            f"{box_2d[0]:.2f} {box_2d[1]:.2f} {box_2d[2]:.2f} {box_2d[3]:.2f} "
            f"{h:.2f} {w:.2f} {l:.2f} "
            f"{x:.2f} {y:.2f} {z:.2f} {self.rotation_y:.2f} {self.score:.2f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "center": self.center.tolist(),
            "dimensions": self.dimensions.tolist(),
            "rotation_y": self.rotation_y,
            "class_name": self.class_name,
            "score": self.score,
            "volume": self.volume,
            "depth": self.depth,
            "confidence": self.confidence.to_dict(),
            "stats": self.stats.to_dict(),
        }


# =============================================================================
# Standalone Utility Functions
# =============================================================================

def compute_corners_3d(
    center: np.ndarray,
    dimensions: np.ndarray,
    rotation_y: float
) -> np.ndarray:
    """
    Compute 8 corners of a 3D box.

    Args:
        center: Box center (x, y, z) in camera frame.
        dimensions: Box dimensions (length, width, height).
        rotation_y: Rotation around Y-axis in radians.

    Returns:
        Corners array (8, 3) in camera frame.
    """
    l, w, h = dimensions

    # Box corners in object frame (centered at origin)
    # Bottom face (y = 0, touching ground in KITTI convention)
    # Top face (y = -h, above ground)
    corners = np.array([
        [l/2,  0,  w/2],   # 0: front-right-bottom
        [l/2,  0, -w/2],   # 1: front-left-bottom
        [-l/2, 0, -w/2],   # 2: back-left-bottom
        [-l/2, 0,  w/2],   # 3: back-right-bottom
        [l/2, -h,  w/2],   # 4: front-right-top
        [l/2, -h, -w/2],   # 5: front-left-top
        [-l/2, -h, -w/2],  # 6: back-left-top
        [-l/2, -h,  w/2],  # 7: back-right-top
    ])

    # Rotation matrix around Y-axis
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c],
    ])

    # Transform: rotate then translate
    return (corners @ R.T) + center


def project_corners_to_image(
    corners_3d: np.ndarray,
    intrinsics: CameraIntrinsics,
    image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D box corners to 2D image plane.

    Args:
        corners_3d: 3D corners (8, 3) in camera frame.
        intrinsics: Camera intrinsic parameters.
        image_size: Optional (width, height) for visibility check.

    Returns:
        Tuple of:
            - corners_2d: Projected 2D corners (8, 2).
            - visible_mask: Boolean mask of visible corners.
    """
    # Filter points behind camera
    valid_depth = corners_3d[:, 2] > 0.1

    # Project to 2D
    corners_2d = np.zeros((8, 2))

    for i, (x, y, z) in enumerate(corners_3d):
        if z > 0.1:
            u = intrinsics.fx * x / z + intrinsics.cx
            v = intrinsics.fy * y / z + intrinsics.cy
            corners_2d[i] = [u, v]
        else:
            corners_2d[i] = [-1, -1]

    # Check visibility
    visible_mask = valid_depth.copy()

    if image_size is not None:
        w, h = image_size
        in_bounds = (
            (corners_2d[:, 0] >= 0) & (corners_2d[:, 0] < w) &
            (corners_2d[:, 1] >= 0) & (corners_2d[:, 1] < h)
        )
        visible_mask = visible_mask & in_bounds

    return corners_2d, visible_mask


def compute_box_edges() -> List[Tuple[int, int]]:
    """
    Get edge indices for 3D box wireframe rendering.

    Returns:
        List of (start_idx, end_idx) pairs for box edges.
    """
    return [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]


# =============================================================================
# Main Generator Class
# =============================================================================

class BBox3DGenerator:
    """
    Generate 3D bounding boxes from 2D detections and LiDAR points.

    This class provides multiple estimation methods:
    1. Prior-based: Use class-specific dimension priors with depth from LiDAR.
    2. Point-based: Estimate dimensions directly from point cloud statistics.
    3. Hybrid: Blend prior and point-based estimates based on point density.

    Example:
        >>> generator = BBox3DGenerator(intrinsics)
        >>> bbox3d = generator.generate_from_points(
        ...     detection=detection,
        ...     points=lidar_points,
        ...     depth=15.0
        ... )
    """

    # Default dimensions per class (length, width, height) in meters
    # Based on KITTI dataset statistics
    DEFAULT_DIMENSIONS = {
        "Car": np.array([3.88, 1.63, 1.53]),
        "Van": np.array([5.08, 1.90, 2.30]),
        "Truck": np.array([10.13, 2.58, 3.25]),
        "Pedestrian": np.array([0.88, 0.65, 1.77]),
        "Person_sitting": np.array([0.80, 0.60, 1.28]),
        "Cyclist": np.array([1.76, 0.60, 1.73]),
        "Tram": np.array([16.0, 2.60, 3.50]),
        "Misc": np.array([2.50, 1.50, 1.50]),
    }

    # Dimension standard deviations per class
    DIMENSION_STD = {
        "Car": np.array([0.42, 0.10, 0.14]),
        "Van": np.array([0.58, 0.18, 0.32]),
        "Truck": np.array([2.22, 0.36, 0.61]),
        "Pedestrian": np.array([0.26, 0.16, 0.14]),
        "Person_sitting": np.array([0.20, 0.15, 0.15]),
        "Cyclist": np.array([0.30, 0.12, 0.15]),
        "Tram": np.array([3.00, 0.30, 0.50]),
        "Misc": np.array([1.00, 0.50, 0.50]),
    }

    # Minimum points needed for different estimation methods
    MIN_POINTS_CENTER = 3
    MIN_POINTS_DIMENSIONS = 20
    MIN_POINTS_ORIENTATION = 30
    MIN_POINTS_FULL = 50

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        default_dimensions: Optional[Dict[str, np.ndarray]] = None,
        use_prior_dimensions: bool = True,
        prior_weight: float = 0.7,
        min_points_for_estimation: int = 20,
        ground_height: float = 1.65,  # Camera height above ground
    ):
        """
        Initialize 3D box generator.

        Args:
            intrinsics: Camera intrinsic parameters.
            default_dimensions: Custom default dimensions per class.
            use_prior_dimensions: Whether to use class-based dimension priors.
            prior_weight: Weight for prior dimensions vs estimated (0-1).
            min_points_for_estimation: Minimum points for point-based estimation.
            ground_height: Camera height above ground plane in meters.
        """
        self.intrinsics = intrinsics
        self.default_dimensions = default_dimensions or self.DEFAULT_DIMENSIONS.copy()
        self.use_prior_dimensions = use_prior_dimensions
        self.prior_weight = prior_weight
        self.min_points_for_estimation = min_points_for_estimation
        self.ground_height = ground_height

    # -------------------------------------------------------------------------
    # Main Generation Methods
    # -------------------------------------------------------------------------

    def generate(
        self,
        detection: Detection,
        depth: float,
        rotation_y: float = 0.0,
        dimensions: Optional[np.ndarray] = None,
    ) -> BBox3D:
        """
        Generate 3D box from 2D detection and depth (basic method).

        Uses class priors for dimensions and simple geometry for center.

        Args:
            detection: 2D detection.
            depth: Object depth in meters.
            rotation_y: Rotation around Y-axis (default 0).
            dimensions: Custom dimensions (uses prior if None).

        Returns:
            BBox3D instance.
        """
        # Get dimensions
        if dimensions is None:
            dimensions = self._get_prior_dimensions(detection.class_name)

        # Compute 3D center from 2D bbox center
        center = self._compute_center_from_2d(detection.bbox, depth, dimensions[2])

        # Basic confidence
        confidence = ConfidenceScore(
            overall=detection.confidence,
            detection_2d=detection.confidence,
            dimension_fit=1.0,  # Using prior
        )

        stats = EstimationStats(method_used="prior")

        return BBox3D(
            center=center,
            dimensions=dimensions,
            rotation_y=rotation_y,
            class_name=detection.class_name,
            score=detection.confidence,
            detection_2d=detection,
            confidence=confidence,
            stats=stats,
        )

    def generate_from_points(
        self,
        detection: Detection,
        points: np.ndarray,
        depth: Optional[float] = None,
    ) -> BBox3D:
        """
        Generate 3D box using LiDAR points for enhanced estimation.

        This method uses the point cloud to:
        1. Estimate center from point centroid
        2. Estimate dimensions from point spread
        3. Estimate orientation from principal axis
        4. Compute confidence based on point statistics

        Args:
            detection: 2D detection.
            points: LiDAR points (N, 3) in camera frame within the detection.
            depth: Optional depth override (uses point median if None).

        Returns:
            BBox3D instance with full estimation.
        """
        stats = EstimationStats()
        stats.num_points = len(points)

        # Handle insufficient points
        if len(points) < self.MIN_POINTS_CENTER:
            if depth is None:
                depth = 10.0  # Fallback depth
            return self.generate(detection, depth)

        # Estimate center
        center, center_depth = self.estimate_center_3d(points)
        if depth is None:
            depth = center_depth

        stats.depth_mean = float(np.mean(points[:, 2]))
        stats.depth_std = float(np.std(points[:, 2]))

        # Estimate dimensions
        if len(points) >= self.MIN_POINTS_DIMENSIONS:
            estimated_dims = self.estimate_dimensions(points)
            stats.point_spread = np.ptp(points, axis=0)
            stats.method_used = "point_based"
        else:
            estimated_dims = None
            stats.method_used = "prior_with_center"

        # Get final dimensions (blend prior and estimated)
        prior_dims = self._get_prior_dimensions(detection.class_name)
        dimensions = self._blend_dimensions(
            prior_dims, estimated_dims, detection.class_name, len(points)
        )

        # Estimate orientation
        if len(points) >= self.MIN_POINTS_ORIENTATION:
            rotation_y, eigenvalues = self.estimate_orientation(points, detection.bbox)
            stats.eigenvalues = eigenvalues
        else:
            rotation_y = self._estimate_rotation_from_position(center)

        # Compute confidence scores
        confidence = self._compute_confidence(
            points=points,
            detection=detection,
            center=center,
            dimensions=dimensions,
        )

        stats.num_points = len(points)

        return BBox3D(
            center=center,
            dimensions=dimensions,
            rotation_y=rotation_y,
            class_name=detection.class_name,
            score=confidence.overall,
            detection_2d=detection,
            confidence=confidence,
            stats=stats,
        )

    def generate_batch(
        self,
        detections: List[Detection],
        depths: List[Optional[float]],
        points_list: Optional[List[Optional[np.ndarray]]] = None,
    ) -> List[Optional[BBox3D]]:
        """
        Generate 3D boxes for multiple detections.

        Args:
            detections: List of 2D detections.
            depths: List of depth estimates.
            points_list: Optional list of point clouds per detection.

        Returns:
            List of BBox3D (None where depth is None).
        """
        results = []

        for i, (detection, depth) in enumerate(zip(detections, depths)):
            if depth is None:
                results.append(None)
                continue

            # Use points if available
            if points_list is not None and points_list[i] is not None:
                points = points_list[i]
                if len(points) >= self.MIN_POINTS_CENTER:
                    bbox3d = self.generate_from_points(detection, points, depth)
                else:
                    bbox3d = self.generate(detection, depth)
            else:
                bbox3d = self.generate(detection, depth)

            results.append(bbox3d)

        return results

    # -------------------------------------------------------------------------
    # Core Estimation Methods
    # -------------------------------------------------------------------------

    def estimate_center_3d(
        self,
        points: np.ndarray,
        method: str = "robust_centroid",
    ) -> Tuple[np.ndarray, float]:
        """
        Estimate 3D center from LiDAR points.

        Methods:
        - "centroid": Simple mean of all points.
        - "robust_centroid": Trimmed mean (excludes outliers).
        - "closest_cluster": Center of points closest to camera.

        Args:
            points: LiDAR points (N, 3) in camera frame.
            method: Estimation method to use.

        Returns:
            Tuple of (center (3,), depth).
        """
        if len(points) == 0:
            return np.zeros(3), 0.0

        if method == "centroid":
            center = np.mean(points, axis=0)

        elif method == "robust_centroid":
            # Use trimmed mean to exclude outliers
            center = np.zeros(3)
            for i in range(3):
                center[i] = stats.trim_mean(points[:, i], 0.1)

        elif method == "closest_cluster":
            # Find center of closest points (front surface)
            depths = points[:, 2]
            depth_threshold = np.percentile(depths, 25)
            close_mask = depths <= depth_threshold
            center = np.mean(points[close_mask], axis=0)

        else:
            center = np.mean(points, axis=0)

        depth = float(center[2])

        return center, depth

    def estimate_dimensions(
        self,
        points: np.ndarray,
        method: str = "pca_spread",
    ) -> np.ndarray:
        """
        Estimate object dimensions from LiDAR points.

        Methods:
        - "point_spread": Use range in each axis.
        - "pca_spread": Use PCA-aligned bounding box.
        - "convex_hull": Use convex hull extents.

        Args:
            points: LiDAR points (N, 3) in camera frame.
            method: Estimation method to use.

        Returns:
            Dimensions (length, width, height) in meters.
        """
        if len(points) < 4:
            return np.array([2.0, 1.5, 1.5])

        if method == "point_spread":
            # Simple axis-aligned spread
            spread = np.ptp(points, axis=0)
            # Map to (length, width, height) = (x_range, z_range, y_range)
            # In KITTI: X=right, Y=down, Z=forward
            length = spread[0]  # X spread
            width = spread[2]   # Z spread (depth direction)
            height = spread[1]  # Y spread

        elif method == "pca_spread":
            # PCA-aligned bounding box
            centered = points - np.mean(points, axis=0)

            try:
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)

                # Sort by eigenvalue (largest first)
                idx = np.argsort(eigenvalues)[::-1]
                eigenvectors = eigenvectors[:, idx]

                # Project points onto principal axes
                projected = centered @ eigenvectors
                spread = np.ptp(projected, axis=0)

                # Length is largest spread, width is second, height is third
                sorted_spread = np.sort(spread)[::-1]
                length = sorted_spread[0]
                width = sorted_spread[1]
                height = sorted_spread[2]

            except np.linalg.LinAlgError:
                spread = np.ptp(points, axis=0)
                length, height, width = spread[0], spread[1], spread[2]

        elif method == "convex_hull":
            try:
                # 2D convex hull in XZ plane
                points_xz = points[:, [0, 2]]
                hull = ConvexHull(points_xz)
                hull_points = points_xz[hull.vertices]

                # Find minimum bounding rectangle
                length, width = self._min_bounding_rect(hull_points)
                height = np.ptp(points[:, 1])

            except Exception:
                spread = np.ptp(points, axis=0)
                length, height, width = spread[0], spread[1], spread[2]

        else:
            spread = np.ptp(points, axis=0)
            length, height, width = spread[0], spread[1], spread[2]

        # Ensure reasonable minimum dimensions
        length = max(length, 0.5)
        width = max(width, 0.3)
        height = max(height, 0.5)

        return np.array([length, width, height])

    def estimate_orientation(
        self,
        points: np.ndarray,
        bbox_2d: np.ndarray,
        method: str = "pca",
    ) -> Tuple[float, np.ndarray]:
        """
        Estimate object orientation from point cloud.

        Methods:
        - "pca": Use principal component direction.
        - "l_shape": Fit L-shape to detect front/side edges.
        - "bbox_edges": Use 2D bbox aspect ratio as hint.

        Args:
            points: LiDAR points (N, 3) in camera frame.
            bbox_2d: 2D bounding box [x1, y1, x2, y2].
            method: Estimation method to use.

        Returns:
            Tuple of (rotation_y in radians, eigenvalues for debugging).
        """
        eigenvalues = np.zeros(3)

        if len(points) < self.MIN_POINTS_ORIENTATION:
            # Fall back to position-based estimate
            center = np.mean(points, axis=0)
            rotation_y = self._estimate_rotation_from_position(center)
            return rotation_y, eigenvalues

        if method == "pca":
            # Use PCA on XZ plane (bird's eye view)
            points_xz = points[:, [0, 2]]
            centered = points_xz - np.mean(points_xz, axis=0)

            try:
                cov = np.cov(centered.T)
                eigenvalues_2d, eigenvectors = np.linalg.eigh(cov)

                # Principal direction (largest eigenvalue)
                principal_idx = np.argmax(eigenvalues_2d)
                principal_dir = eigenvectors[:, principal_idx]

                # Convert to rotation around Y
                # atan2(x_component, z_component)
                rotation_y = np.arctan2(principal_dir[0], principal_dir[1])

                eigenvalues[:2] = eigenvalues_2d

            except np.linalg.LinAlgError:
                center = np.mean(points, axis=0)
                rotation_y = self._estimate_rotation_from_position(center)

        elif method == "l_shape":
            rotation_y = self._estimate_orientation_l_shape(points)

        elif method == "bbox_edges":
            # Use 2D bbox as orientation hint
            box_center_x = (bbox_2d[0] + bbox_2d[2]) / 2
            image_center_x = self.intrinsics.width / 2

            # Objects at image center face away, objects on sides face inward
            offset = (box_center_x - image_center_x) / image_center_x
            rotation_y = -offset * np.pi / 4

        else:
            center = np.mean(points, axis=0)
            rotation_y = self._estimate_rotation_from_position(center)

        # Normalize to [-pi, pi]
        while rotation_y > np.pi:
            rotation_y -= 2 * np.pi
        while rotation_y < -np.pi:
            rotation_y += 2 * np.pi

        return rotation_y, eigenvalues

    # -------------------------------------------------------------------------
    # Confidence Scoring
    # -------------------------------------------------------------------------

    def _compute_confidence(
        self,
        points: np.ndarray,
        detection: Detection,
        center: np.ndarray,
        dimensions: np.ndarray,
    ) -> ConfidenceScore:
        """
        Compute detailed confidence scores.

        Args:
            points: LiDAR points used for estimation.
            detection: 2D detection.
            center: Estimated 3D center.
            dimensions: Estimated dimensions.

        Returns:
            ConfidenceScore with all components.
        """
        confidence = ConfidenceScore()

        # 1. Point density score
        # More points = higher confidence, saturates at ~100 points
        num_points = len(points)
        confidence.point_density = min(1.0, num_points / 100.0)

        # 2. 2D detection confidence
        confidence.detection_2d = detection.confidence

        # 3. Depth consistency (low variance = high confidence)
        if num_points > 3:
            depth_std = np.std(points[:, 2])
            depth_mean = np.mean(points[:, 2])
            cv = depth_std / depth_mean if depth_mean > 0 else 1.0
            # Low coefficient of variation = high confidence
            confidence.depth_consistency = max(0.0, 1.0 - cv)
        else:
            confidence.depth_consistency = 0.5

        # 4. Dimension fit score
        prior_dims = self._get_prior_dimensions(detection.class_name)
        prior_std = self.DIMENSION_STD.get(
            detection.class_name, np.array([1.0, 0.5, 0.5])
        )

        # Z-score based fitness
        z_scores = np.abs(dimensions - prior_dims) / (prior_std + 1e-6)
        mean_z = np.mean(z_scores)
        confidence.dimension_fit = max(0.0, 1.0 - mean_z / 3.0)

        # 5. Coverage score (how well points fill the estimated box)
        if num_points >= 10:
            box_volume = np.prod(dimensions)
            point_spread = np.ptp(points, axis=0)
            point_volume = np.prod(point_spread + 0.1)  # Add small epsilon
            coverage = min(1.0, point_volume / (box_volume + 1e-6))
            confidence.coverage = coverage
        else:
            confidence.coverage = 0.5

        # Overall score (weighted combination)
        weights = {
            "point_density": 0.20,
            "detection_2d": 0.30,
            "depth_consistency": 0.20,
            "dimension_fit": 0.15,
            "coverage": 0.15,
        }

        confidence.overall = (
            weights["point_density"] * confidence.point_density +
            weights["detection_2d"] * confidence.detection_2d +
            weights["depth_consistency"] * confidence.depth_consistency +
            weights["dimension_fit"] * confidence.dimension_fit +
            weights["coverage"] * confidence.coverage
        )

        return confidence

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_prior_dimensions(self, class_name: str) -> np.ndarray:
        """Get prior dimensions for a class."""
        if class_name in self.default_dimensions:
            return self.default_dimensions[class_name].copy()
        else:
            return np.array([2.0, 1.5, 1.5])

    def _blend_dimensions(
        self,
        prior_dims: np.ndarray,
        estimated_dims: Optional[np.ndarray],
        class_name: str,
        num_points: int,
    ) -> np.ndarray:
        """Blend prior and estimated dimensions based on confidence."""
        if estimated_dims is None:
            return prior_dims

        # Adjust weight based on point count
        point_factor = min(1.0, num_points / self.MIN_POINTS_FULL)
        adjusted_weight = self.prior_weight * (1 - 0.5 * point_factor)

        # Blend
        blended = adjusted_weight * prior_dims + (1 - adjusted_weight) * estimated_dims

        # Clamp to reasonable bounds
        min_dims = np.array([0.5, 0.3, 0.5])
        max_dims = np.array([20.0, 5.0, 5.0])

        return np.clip(blended, min_dims, max_dims)

    def _compute_center_from_2d(
        self,
        box_2d: np.ndarray,
        depth: float,
        height: float,
    ) -> np.ndarray:
        """
        Compute 3D center from 2D box center and depth.

        Args:
            box_2d: 2D box [x1, y1, x2, y2].
            depth: Object depth.
            height: Object height.

        Returns:
            3D center (x, y, z) in camera frame.
        """
        # Get 2D box bottom center (objects stand on ground)
        center_x = (box_2d[0] + box_2d[2]) / 2
        bottom_y = box_2d[3]  # Bottom of 2D box

        # Unproject bottom center to 3D
        center_2d = np.array([center_x, bottom_y])
        bottom_3d = self.intrinsics.unproject_point(center_2d, depth)

        # Move center up by half the height
        center_3d = bottom_3d.copy()
        center_3d[1] -= height / 2  # Y points down, so subtract to go up

        return center_3d

    def _estimate_rotation_from_position(self, center: np.ndarray) -> float:
        """Estimate rotation based on object position relative to camera."""
        x, y, z = center

        # Assume objects roughly face the camera or perpendicular to view ray
        # Objects on the left tend to face right and vice versa
        viewing_angle = np.arctan2(x, z)

        # Add small offset for more natural orientation
        rotation_y = viewing_angle + np.pi / 2  # Perpendicular to view

        # Normalize
        while rotation_y > np.pi:
            rotation_y -= 2 * np.pi
        while rotation_y < -np.pi:
            rotation_y += 2 * np.pi

        return rotation_y

    def _estimate_orientation_l_shape(self, points: np.ndarray) -> float:
        """
        Estimate orientation using L-shape fitting.

        Vehicles often show two visible sides (L-shape) in point cloud.
        """
        points_xz = points[:, [0, 2]]

        # Find corner point (closest to convex hull corner)
        try:
            hull = ConvexHull(points_xz)
            hull_points = points_xz[hull.vertices]

            # Find the two longest edges
            edges = []
            for i in range(len(hull_points)):
                p1 = hull_points[i]
                p2 = hull_points[(i + 1) % len(hull_points)]
                length = np.linalg.norm(p2 - p1)
                direction = (p2 - p1) / length
                edges.append((length, direction, p1, p2))

            edges.sort(reverse=True, key=lambda x: x[0])

            if len(edges) >= 2:
                # Use direction of longest edge as primary orientation
                primary_dir = edges[0][1]
                rotation_y = np.arctan2(primary_dir[0], primary_dir[1])
            else:
                rotation_y = 0.0

        except Exception:
            rotation_y = 0.0

        return rotation_y

    def _min_bounding_rect(self, points_2d: np.ndarray) -> Tuple[float, float]:
        """
        Find minimum bounding rectangle for 2D points.

        Uses rotating calipers algorithm.
        """
        try:
            hull = ConvexHull(points_2d)
            hull_points = points_2d[hull.vertices]

            min_area = float('inf')
            best_dims = (1.0, 1.0)

            # Try each edge as base
            n = len(hull_points)
            for i in range(n):
                edge = hull_points[(i + 1) % n] - hull_points[i]
                edge_len = np.linalg.norm(edge)

                if edge_len < 1e-6:
                    continue

                # Rotation to align edge with X-axis
                angle = np.arctan2(edge[1], edge[0])
                c, s = np.cos(-angle), np.sin(-angle)
                R = np.array([[c, -s], [s, c]])

                rotated = hull_points @ R.T

                min_xy = rotated.min(axis=0)
                max_xy = rotated.max(axis=0)

                width = max_xy[0] - min_xy[0]
                height = max_xy[1] - min_xy[1]
                area = width * height

                if area < min_area:
                    min_area = area
                    best_dims = (max(width, height), min(width, height))

            return best_dims

        except Exception:
            spread = np.ptp(points_2d, axis=0)
            return (spread[0], spread[1])

    # -------------------------------------------------------------------------
    # Refinement Methods
    # -------------------------------------------------------------------------

    def refine_with_lidar(
        self,
        bbox3d: BBox3D,
        points: np.ndarray,
    ) -> BBox3D:
        """
        Refine 3D box using LiDAR points.

        Args:
            bbox3d: Initial 3D box estimate.
            points: LiDAR points in camera frame.

        Returns:
            Refined BBox3D.
        """
        # Find points within expanded search region
        center = bbox3d.center
        dims = bbox3d.dimensions

        margin = 1.5
        mask = (
            (np.abs(points[:, 0] - center[0]) < dims[0] * margin) &
            (np.abs(points[:, 1] - center[1]) < dims[2] * margin) &
            (np.abs(points[:, 2] - center[2]) < dims[1] * margin)
        )

        box_points = points[mask]

        if len(box_points) < 10:
            return bbox3d

        # Re-estimate with the found points
        return self.generate_from_points(
            bbox3d.detection_2d, box_points, depth=center[2]
        )


# =============================================================================
# Visualization Utilities
# =============================================================================

class BBox3DVisualizer:
    """
    Visualization utilities for 3D bounding boxes.

    Provides methods for:
    - Drawing 3D boxes on images
    - Creating bird's eye view plots
    - Generating debug visualizations
    """

    # Color scheme for different classes
    CLASS_COLORS = {
        "Car": (0, 255, 0),        # Green
        "Van": (0, 200, 100),      # Teal
        "Truck": (0, 150, 255),    # Orange
        "Pedestrian": (255, 0, 0),  # Red
        "Person_sitting": (255, 100, 100),
        "Cyclist": (255, 255, 0),  # Yellow
        "Tram": (128, 0, 128),     # Purple
        "Misc": (128, 128, 128),   # Gray
    }

    # Edge connections for wireframe
    EDGES = compute_box_edges()

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        line_thickness: int = 2,
        font_scale: float = 0.5,
    ):
        """
        Initialize visualizer.

        Args:
            intrinsics: Camera intrinsic parameters.
            line_thickness: Thickness for box edges.
            font_scale: Scale for text labels.
        """
        self.intrinsics = intrinsics
        self.line_thickness = line_thickness
        self.font_scale = font_scale

    def draw_box_3d_on_image(
        self,
        image: np.ndarray,
        bbox3d: BBox3D,
        color: Optional[Tuple[int, int, int]] = None,
        show_label: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Draw 3D bounding box projected onto 2D image.

        Args:
            image: Input image (BGR).
            bbox3d: 3D bounding box to draw.
            color: Override color (BGR). Uses class color if None.
            show_label: Whether to show class label.
            show_confidence: Whether to show confidence breakdown.

        Returns:
            Image with drawn box.
        """
        import cv2

        result = image.copy()

        # Get color
        if color is None:
            color = self.CLASS_COLORS.get(bbox3d.class_name, (0, 255, 0))

        # Get 3D corners
        corners_3d = bbox3d.corners

        # Project to 2D
        corners_2d, visible = project_corners_to_image(
            corners_3d, self.intrinsics, (image.shape[1], image.shape[0])
        )

        # Draw edges
        for start_idx, end_idx in self.EDGES:
            if visible[start_idx] and visible[end_idx]:
                pt1 = tuple(corners_2d[start_idx].astype(int))
                pt2 = tuple(corners_2d[end_idx].astype(int))
                cv2.line(result, pt1, pt2, color, self.line_thickness)

        # Draw front face with different color (indices 0, 1, 4, 5)
        front_color = tuple(min(255, c + 50) for c in color)
        front_edges = [(0, 1), (1, 5), (5, 4), (4, 0)]
        for start_idx, end_idx in front_edges:
            if visible[start_idx] and visible[end_idx]:
                pt1 = tuple(corners_2d[start_idx].astype(int))
                pt2 = tuple(corners_2d[end_idx].astype(int))
                cv2.line(result, pt1, pt2, front_color, self.line_thickness + 1)

        # Draw label
        if show_label and np.any(visible):
            # Find topmost visible corner for label placement
            visible_corners = corners_2d[visible]
            top_idx = np.argmin(visible_corners[:, 1])
            label_pos = visible_corners[top_idx].astype(int)
            label_pos[1] -= 10  # Above the box

            label = f"{bbox3d.class_name}"
            if show_confidence:
                label += f" {bbox3d.score:.2f}"

            cv2.putText(
                result, label, tuple(label_pos),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                color, self.line_thickness
            )

        return result

    def draw_boxes_3d_on_image(
        self,
        image: np.ndarray,
        boxes: List[Optional[BBox3D]],
        show_labels: bool = True,
    ) -> np.ndarray:
        """Draw multiple 3D boxes on image."""
        result = image.copy()

        for bbox3d in boxes:
            if bbox3d is not None:
                result = self.draw_box_3d_on_image(
                    result, bbox3d, show_label=show_labels
                )

        return result

    def create_debug_visualization(
        self,
        image: np.ndarray,
        bbox3d: BBox3D,
        points: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Create detailed debug visualization for a single detection.

        Shows:
        - 3D box projection
        - Point cloud overlay
        - Confidence breakdown
        - Estimation statistics

        Args:
            image: Input image.
            bbox3d: 3D bounding box.
            points: Optional LiDAR points used for estimation.

        Returns:
            Debug visualization image.
        """
        import cv2

        # Create larger canvas
        h, w = image.shape[:2]
        canvas = np.zeros((h + 200, w, 3), dtype=np.uint8)
        canvas[:h, :] = image

        # Draw 3D box
        canvas[:h, :] = self.draw_box_3d_on_image(canvas[:h, :], bbox3d)

        # Draw points if available
        if points is not None and len(points) > 0:
            for point in points:
                x, y, z = point
                if z > 0.1:
                    u = int(self.intrinsics.fx * x / z + self.intrinsics.cx)
                    v = int(self.intrinsics.fy * y / z + self.intrinsics.cy)
                    if 0 <= u < w and 0 <= v < h:
                        cv2.circle(canvas, (u, v), 2, (255, 0, 255), -1)

        # Draw info panel
        info_y = h + 20
        line_height = 20

        # Class and score
        cv2.putText(
            canvas, f"Class: {bbox3d.class_name}  Score: {bbox3d.score:.3f}",
            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        info_y += line_height

        # Center and dimensions
        cv2.putText(
            canvas, f"Center: ({bbox3d.center[0]:.2f}, {bbox3d.center[1]:.2f}, {bbox3d.center[2]:.2f})",
            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        info_y += line_height

        cv2.putText(
            canvas, f"Dims (LWH): ({bbox3d.dimensions[0]:.2f}, {bbox3d.dimensions[1]:.2f}, {bbox3d.dimensions[2]:.2f})",
            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        info_y += line_height

        # Rotation
        cv2.putText(
            canvas, f"Rotation: {np.degrees(bbox3d.rotation_y):.1f} deg",
            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        info_y += line_height

        # Confidence breakdown
        conf = bbox3d.confidence
        cv2.putText(
            canvas, f"Confidence: pts={conf.point_density:.2f} det={conf.detection_2d:.2f} depth={conf.depth_consistency:.2f}",
            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        info_y += line_height

        # Stats
        stats = bbox3d.stats
        cv2.putText(
            canvas, f"Stats: {stats.num_points} pts, method={stats.method_used}",
            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

        return canvas
