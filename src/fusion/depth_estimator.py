"""
Depth Estimation from LiDAR for 2D Detections.

This module provides tools for estimating object depth and 3D dimensions
by fusing 2D image detections with LiDAR point cloud data.

Key Functionality:
==================
1. Extract LiDAR points within 2D bounding boxes
2. Estimate representative depth with outlier handling
3. Estimate 3D object dimensions from point clusters
4. Compute depth confidence based on point density
5. Ground plane filtering and point clustering

The typical pipeline:
    1. Run 2D object detection on image
    2. Project LiDAR points to image plane
    3. Extract points within each detection box
    4. Filter ground plane and cluster object points
    5. Estimate depth and 3D dimensions

Edge Cases Handled:
    - Empty regions (no LiDAR points in box)
    - Occlusions (multiple depth peaks)
    - Multiple objects in same box
    - Sparse point regions (far objects)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

# Handle imports flexibly for both package and direct usage
try:
    from calibration.projection import Projector
    from calibration.intrinsics import CameraIntrinsics
    from calibration.extrinsics import CameraLiDARExtrinsics
    from perception2d.detector import Detection
except ImportError:
    try:
        from ..calibration.projection import Projector
        from ..calibration.intrinsics import CameraIntrinsics
        from ..calibration.extrinsics import CameraLiDARExtrinsics
        from ..perception2d.detector import Detection
    except ImportError:
        # Type hints only - will be provided at runtime
        if TYPE_CHECKING:
            from calibration.projection import Projector
            from calibration.intrinsics import CameraIntrinsics
            from calibration.extrinsics import CameraLiDARExtrinsics
            from perception2d.detector import Detection


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DepthResult:
    """Result of depth estimation for a detection."""
    depth: Optional[float]  # Estimated depth in meters
    confidence: float  # Confidence score [0, 1]
    num_points: int  # Number of LiDAR points in box
    std: Optional[float]  # Standard deviation of depths
    points_3d: Optional[np.ndarray]  # 3D points in camera frame
    points_2d: Optional[np.ndarray]  # Projected 2D points


@dataclass
class DimensionResult:
    """Result of 3D dimension estimation."""
    length: float  # Object length (X direction)
    width: float  # Object width (Z direction, depth)
    height: float  # Object height (Y direction)
    center: np.ndarray  # Object center in 3D
    num_points: int  # Points used for estimation
    confidence: float  # Estimation confidence


# =============================================================================
# Helper Functions
# =============================================================================

def filter_ground_plane(
    points: np.ndarray,
    ground_height: float = -1.5,
    tolerance: float = 0.2,
    method: str = "threshold",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter out ground plane points from a point cloud.

    In camera coordinates (KITTI convention):
        - Y-axis points down
        - Ground is at approximately y = 1.65m (camera height above ground)

    In LiDAR coordinates (Velodyne convention):
        - Z-axis points up
        - Ground is at approximately z = -1.7m

    Args:
        points: Point cloud (N, 3) or (N, 4).
        ground_height: Ground plane height threshold.
                       In camera coords: ~1.5-1.8m (positive, Y down).
                       In LiDAR coords: ~-1.7m (negative, Z up).
        tolerance: Tolerance for ground detection.
        method: Filtering method:
                - 'threshold': Simple height threshold
                - 'ransac': RANSAC plane fitting (if available)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - non_ground: Points above ground (M, 3/4)
            - ground_mask: Boolean mask (N,) of ground points

    Example:
        >>> # For LiDAR points (Z up)
        >>> non_ground, mask = filter_ground_plane(points, ground_height=-1.5)
        >>> # For camera points (Y down)
        >>> non_ground, mask = filter_ground_plane(points_cam, ground_height=1.5)
    """
    points = np.atleast_2d(points)

    if method == "threshold":
        # Simple threshold - works for both coordinate systems
        # For camera coords: keep points where Y < ground_height (Y is down)
        # For LiDAR coords: keep points where Z > ground_height (Z is up)

        # Detect coordinate system based on typical ranges
        z_range = points[:, 2].max() - points[:, 2].min()
        y_range = points[:, 1].max() - points[:, 1].min() if points.shape[1] > 2 else 0

        # Heuristic: LiDAR has larger Z range (forward distance)
        if z_range > 50:  # Likely camera coordinates (Z is depth/forward)
            # Camera coords: Y is down, ground has high Y values
            ground_mask = points[:, 1] > (ground_height - tolerance)
        else:  # Likely LiDAR coordinates (Z is up)
            # LiDAR coords: Z is up, ground has low Z values
            ground_mask = points[:, 2] < (ground_height + tolerance)

    elif method == "ransac":
        ground_mask = _ransac_ground_plane(points, tolerance)

    else:
        raise ValueError(f"Unknown method: {method}")

    non_ground = points[~ground_mask]
    return non_ground, ground_mask


def _ransac_ground_plane(
    points: np.ndarray,
    tolerance: float = 0.2,
    max_iterations: int = 100,
) -> np.ndarray:
    """
    RANSAC-based ground plane detection.

    Fits a plane to the lowest points and marks inliers as ground.

    Args:
        points: Point cloud (N, 3).
        tolerance: Distance threshold for inliers.
        max_iterations: Maximum RANSAC iterations.

    Returns:
        np.ndarray: Boolean mask of ground points.
    """
    best_mask = np.zeros(len(points), dtype=bool)
    best_count = 0

    # Focus on low points (potential ground)
    z_values = points[:, 2] if points.shape[1] >= 3 else points[:, 1]
    low_threshold = np.percentile(z_values, 20)
    low_indices = np.where(z_values < low_threshold)[0]

    if len(low_indices) < 3:
        return best_mask

    for _ in range(max_iterations):
        # Sample 3 points
        sample_idx = np.random.choice(low_indices, 3, replace=False)
        p1, p2, p3 = points[sample_idx, :3]

        # Compute plane normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)

        if norm < 1e-10:
            continue

        normal /= norm
        d = -np.dot(normal, p1)

        # Check if plane is roughly horizontal
        if abs(normal[2]) < 0.8:  # Normal should point mostly up/down
            continue

        # Compute distances to plane
        distances = np.abs(points[:, :3] @ normal + d)
        inlier_mask = distances < tolerance
        inlier_count = inlier_mask.sum()

        if inlier_count > best_count:
            best_count = inlier_count
            best_mask = inlier_mask

    return best_mask


def cluster_points(
    points: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 3,
) -> Tuple[np.ndarray, int]:
    """
    Cluster points using DBSCAN algorithm.

    DBSCAN groups nearby points into clusters, useful for separating
    multiple objects in the same bounding box region.

    Args:
        points: Point cloud (N, 3) or (N, 4).
        eps: Maximum distance between points in same cluster (meters).
        min_samples: Minimum points to form a cluster.

    Returns:
        Tuple[np.ndarray, int]:
            - labels: Cluster labels for each point (-1 = noise)
            - n_clusters: Number of clusters found

    Example:
        >>> labels, n_clusters = cluster_points(points, eps=0.5)
        >>> largest_cluster = points[labels == 0]  # Label 0 is typically largest
    """
    try:
        from sklearn.cluster import DBSCAN

        xyz = points[:, :3]
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        return labels, n_clusters

    except ImportError:
        # Fallback: simple distance-based clustering
        return _simple_cluster(points[:, :3], eps, min_samples)


def _simple_cluster(
    points: np.ndarray,
    eps: float,
    min_samples: int,
) -> Tuple[np.ndarray, int]:
    """
    Simple fallback clustering when sklearn is not available.

    Uses a greedy nearest-neighbor approach.
    """
    n_points = len(points)
    labels = np.full(n_points, -1, dtype=int)
    current_label = 0

    for i in range(n_points):
        if labels[i] != -1:
            continue

        # Find neighbors
        distances = np.linalg.norm(points - points[i], axis=1)
        neighbors = np.where(distances < eps)[0]

        if len(neighbors) >= min_samples:
            labels[neighbors] = current_label
            current_label += 1

    n_clusters = current_label
    return labels, n_clusters


def get_largest_cluster(
    points: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Extract points from the largest cluster.

    Args:
        points: Point cloud (N, 3/4).
        labels: Cluster labels from cluster_points().

    Returns:
        np.ndarray: Points belonging to largest cluster.
    """
    if len(points) == 0:
        return points

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]  # Exclude noise (-1)

    if len(unique_labels) == 0:
        return points  # No valid clusters, return all points

    # Find largest cluster
    largest_label = max(unique_labels, key=lambda l: (labels == l).sum())
    return points[labels == largest_label]


# =============================================================================
# Main Depth Estimator Class
# =============================================================================

class DepthEstimator:
    """
    Estimate depth for 2D detections using LiDAR point cloud data.

    This class provides methods to:
    1. Extract LiDAR points within 2D bounding boxes
    2. Estimate representative depth with outlier handling
    3. Estimate 3D object dimensions
    4. Compute depth confidence metrics

    Attributes:
        projector: Calibrated projector for LiDAR-camera transformation.
        method: Depth aggregation method.
        search_expansion: Box expansion factor for point search.
        min_points: Minimum points required for valid estimate.
        depth_range: Valid depth range (min, max) meters.
        filter_ground: Whether to filter ground plane points.
        use_clustering: Whether to use clustering for multiple objects.

    Example:
        >>> projector = Projector(intrinsics, extrinsics)
        >>> estimator = DepthEstimator(projector, method='median')
        >>> depth = estimator.estimate_depth(detection, lidar_points)
    """

    def __init__(
        self,
        projector: Optional["Projector"] = None,
        intrinsics: Optional["CameraIntrinsics"] = None,
        extrinsics: Optional["CameraLiDARExtrinsics"] = None,
        method: str = "median",
        search_expansion: float = 0.0,
        min_points: int = 3,
        depth_range: Tuple[float, float] = (0.5, 80.0),
        filter_ground: bool = True,
        ground_height: float = -1.5,
        use_clustering: bool = False,
        cluster_eps: float = 0.5,
    ):
        """
        Initialize depth estimator.

        Args:
            projector: Calibrated projector instance (preferred).
            intrinsics: Camera intrinsics (alternative to projector).
            extrinsics: Camera-LiDAR extrinsics (alternative to projector).
            method: Depth aggregation method:
                    - 'median': Median depth (robust to outliers)
                    - 'mean': Mean depth
                    - 'closest': Minimum depth (closest point)
                    - 'weighted': Distance-weighted average
                    - 'percentile_25': 25th percentile (front of object)
            search_expansion: Expand box by this fraction for point search.
            min_points: Minimum LiDAR points required for valid estimate.
            depth_range: Valid depth range (min, max) in meters.
            filter_ground: Whether to filter ground plane points.
            ground_height: Ground plane height threshold (LiDAR Z coord).
            use_clustering: Whether to use DBSCAN clustering.
            cluster_eps: DBSCAN epsilon parameter (meters).
        """
        if projector is not None:
            self.projector = projector
        elif intrinsics is not None and extrinsics is not None:
            self.projector = Projector(intrinsics, extrinsics)
        else:
            self.projector = None  # Will need to be set later

        self.method = method
        self.search_expansion = search_expansion
        self.min_points = min_points
        self.depth_range = depth_range
        self.filter_ground = filter_ground
        self.ground_height = ground_height
        self.use_clustering = use_clustering
        self.cluster_eps = cluster_eps

    def extract_depth_in_bbox(
        self,
        points_3d: np.ndarray,
        bbox_2d: np.ndarray,
        calib: Optional["CameraLiDARExtrinsics"] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract all LiDAR points that project inside a 2D bounding box.

        This is the core method for associating LiDAR points with 2D detections.
        It projects all LiDAR points to the image plane and filters those
        falling within the specified bounding box.

        Args:
            points_3d: LiDAR points (N, 3) or (N, 4) in Velodyne frame.
            bbox_2d: 2D bounding box [x1, y1, x2, y2] in pixels.
            calib: Optional calibration (uses self.projector if None).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - points_in_box_3d: 3D points in camera frame (M, 3)
                - points_in_box_2d: Corresponding 2D projections (M, 2)
                - depths: Depth values of points (M,)

        Example:
            >>> pts_3d, pts_2d, depths = estimator.extract_depth_in_bbox(
            ...     lidar_points, [100, 150, 300, 350])
            >>> print(f"Found {len(depths)} points in box")
        """
        if calib is not None:
            extrinsics = calib
        elif self.projector is not None:
            extrinsics = self.projector.extrinsics
        else:
            raise ValueError("No calibration provided")

        # Expand box if configured
        bbox_2d = np.asarray(bbox_2d)
        if self.search_expansion > 0:
            bbox_2d = self._expand_box(bbox_2d)

        # Project LiDAR to image
        points_2d, depths = extrinsics.project_lidar_to_image(points_3d)

        # Transform to camera coordinates for 3D points
        points_cam = extrinsics.transform_to_camera(points_3d)

        # Filter by depth range
        depth_mask = (depths >= self.depth_range[0]) & (depths <= self.depth_range[1])

        # Filter by bounding box
        x1, y1, x2, y2 = bbox_2d
        in_box = (
            depth_mask &
            (points_2d[:, 0] >= x1) &
            (points_2d[:, 0] <= x2) &
            (points_2d[:, 1] >= y1) &
            (points_2d[:, 1] <= y2)
        )

        points_in_box_3d = points_cam[in_box]
        points_in_box_2d = points_2d[in_box]
        depths_in_box = depths[in_box]

        # Optional: filter ground plane
        if self.filter_ground and len(points_in_box_3d) > 0:
            # In camera coords, Y is down, ground has high Y
            non_ground, ground_mask = filter_ground_plane(
                points_in_box_3d,
                ground_height=1.5,  # Camera coords
                tolerance=0.2,
            )
            non_ground_mask = ~ground_mask
            points_in_box_3d = points_in_box_3d[non_ground_mask]
            points_in_box_2d = points_in_box_2d[non_ground_mask]
            depths_in_box = depths_in_box[non_ground_mask]

        # Optional: cluster points
        if self.use_clustering and len(points_in_box_3d) >= self.min_points:
            labels, n_clusters = cluster_points(
                points_in_box_3d,
                eps=self.cluster_eps
            )
            if n_clusters > 0:
                # Get largest cluster (most likely the main object)
                largest_mask = labels == np.argmax(np.bincount(labels[labels >= 0]))
                points_in_box_3d = points_in_box_3d[largest_mask]
                points_in_box_2d = points_in_box_2d[largest_mask]
                depths_in_box = depths_in_box[largest_mask]

        return points_in_box_3d, points_in_box_2d, depths_in_box

    def get_representative_depth(
        self,
        depths: np.ndarray,
        method: Optional[str] = None,
    ) -> Optional[float]:
        """
        Get representative depth from a set of depth values.

        Handles outliers using robust statistical methods.

        Args:
            depths: Array of depth values (N,).
            method: Override default method for this call.

        Returns:
            Representative depth value, or None if insufficient data.

        Methods:
            - 'median': Median (robust to outliers)
            - 'mean': Simple mean
            - 'closest': Minimum (closest point to camera)
            - 'percentile_25': 25th percentile (front of object)
            - 'trimmed_mean': Mean after removing 10% extremes
            - 'mode_region': Most common depth region (handles occlusions)

        Example:
            >>> depths = estimator.extract_depth_in_bbox(points, box)[2]
            >>> depth = estimator.get_representative_depth(depths, method='median')
        """
        if len(depths) < self.min_points:
            return None

        method = method or self.method

        if method == "median":
            return float(np.median(depths))

        elif method == "mean":
            return float(np.mean(depths))

        elif method == "closest":
            return float(np.min(depths))

        elif method == "percentile_25":
            return float(np.percentile(depths, 25))

        elif method == "trimmed_mean":
            # Remove 10% extreme values
            lower = np.percentile(depths, 10)
            upper = np.percentile(depths, 90)
            trimmed = depths[(depths >= lower) & (depths <= upper)]
            if len(trimmed) == 0:
                return float(np.median(depths))
            return float(np.mean(trimmed))

        elif method == "mode_region":
            # Find most common depth region (handles multiple objects/occlusion)
            # Use histogram to find mode
            hist, bin_edges = np.histogram(depths, bins=20)
            mode_idx = np.argmax(hist)
            mode_range = (bin_edges[mode_idx], bin_edges[mode_idx + 1])
            mode_depths = depths[(depths >= mode_range[0]) & (depths < mode_range[1])]
            if len(mode_depths) == 0:
                return float(np.median(depths))
            return float(np.median(mode_depths))

        else:
            raise ValueError(f"Unknown method: {method}")

    def estimate_3d_dimensions(
        self,
        points: np.ndarray,
        class_name: Optional[str] = None,
    ) -> DimensionResult:
        """
        Estimate 3D object dimensions from a point cluster.

        Computes length, width, and height from the 3D extent of points.
        Uses PCA for orientation-invariant estimation when possible.

        Args:
            points: 3D points (N, 3) in camera frame.
            class_name: Optional class name for dimension priors.

        Returns:
            DimensionResult: Estimated dimensions and metadata.

        Note:
            In camera coordinates (KITTI):
            - Length: X direction (left-right)
            - Width: Z direction (depth/forward)
            - Height: Y direction (up-down, but Y points down)

        Dimension Priors (when few points):
            - Car: ~4.5m x 1.8m x 1.5m (L x W x H)
            - Pedestrian: ~0.6m x 0.6m x 1.7m
            - Cyclist: ~1.8m x 0.6m x 1.7m
        """
        # Default priors
        DIMENSION_PRIORS = {
            "Car": (4.5, 1.8, 1.5),
            "Van": (5.0, 2.0, 1.8),
            "Truck": (8.0, 2.5, 3.0),
            "Pedestrian": (0.6, 0.6, 1.7),
            "Cyclist": (1.8, 0.6, 1.7),
            "Person_sitting": (0.6, 0.6, 1.2),
        }

        if len(points) < 4:
            # Use priors for insufficient points
            if class_name and class_name in DIMENSION_PRIORS:
                l, w, h = DIMENSION_PRIORS[class_name]
            else:
                l, w, h = 2.0, 2.0, 1.5  # Default

            center = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)

            return DimensionResult(
                length=l,
                width=w,
                height=h,
                center=center,
                num_points=len(points),
                confidence=0.1,
            )

        # Compute extent in each dimension
        min_coords = np.min(points[:, :3], axis=0)
        max_coords = np.max(points[:, :3], axis=0)
        extent = max_coords - min_coords

        # In camera coords: X=length, Y=height, Z=width(depth)
        length = float(extent[0])
        height = float(extent[1])
        width = float(extent[2])

        # Center
        center = (min_coords + max_coords) / 2

        # Confidence based on point count and coverage
        # More points = higher confidence
        point_conf = min(1.0, len(points) / 50)

        # Apply constraints based on class priors
        if class_name and class_name in DIMENSION_PRIORS:
            prior_l, prior_w, prior_h = DIMENSION_PRIORS[class_name]

            # Blend with priors if estimate seems off
            if length < 0.3 or length > prior_l * 3:
                length = prior_l
                point_conf *= 0.5
            if width < 0.3 or width > prior_w * 3:
                width = prior_w
                point_conf *= 0.5
            if height < 0.3 or height > prior_h * 2:
                height = prior_h
                point_conf *= 0.5

        return DimensionResult(
            length=length,
            width=width,
            height=height,
            center=center,
            num_points=len(points),
            confidence=point_conf,
        )

    def get_depth_confidence(
        self,
        points_3d: Optional[np.ndarray] = None,
        depths: Optional[np.ndarray] = None,
        detection: Optional["Detection"] = None,
        points: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute depth confidence based on point density and consistency.

        Confidence factors:
        1. Point count: More points = higher confidence
        2. Depth consistency: Lower std = higher confidence
        3. Spatial coverage: Points spread across box = higher confidence
        4. Occlusion detection: Multiple depth peaks = lower confidence

        Args:
            points_3d: 3D points in box (preferred input).
            depths: Depth values (alternative).
            detection: 2D detection (for legacy interface).
            points: Full LiDAR cloud (for legacy interface).

        Returns:
            Dict with confidence metrics:
                - depth: Estimated depth value
                - confidence: Overall confidence [0, 1]
                - num_points: Number of points used
                - std: Depth standard deviation
                - occlusion_score: Likelihood of occlusion
                - coverage_score: Spatial coverage in box
        """
        # Handle legacy interface
        if detection is not None and points is not None:
            box = self._expand_box(detection.bbox)
            points_3d, _, depths = self.extract_depth_in_bbox(points, box)

        if depths is None and points_3d is not None:
            depths = points_3d[:, 2]  # Z coordinate in camera frame

        if depths is None or len(depths) == 0:
            return {
                "depth": None,
                "confidence": 0.0,
                "num_points": 0,
                "std": None,
                "occlusion_score": 0.0,
                "coverage_score": 0.0,
            }

        num_points = len(depths)

        if num_points < self.min_points:
            return {
                "depth": None,
                "confidence": 0.0,
                "num_points": num_points,
                "std": None,
                "occlusion_score": 0.0,
                "coverage_score": 0.0,
            }

        # Compute depth
        depth = self.get_representative_depth(depths)
        std = float(np.std(depths))

        # Point count confidence (saturates at 30 points)
        point_conf = min(1.0, num_points / 30)

        # Consistency confidence (lower std = higher confidence)
        if depth > 0:
            consistency_conf = max(0, 1 - std / (depth * 0.1))
        else:
            consistency_conf = 0.0

        # Occlusion detection (multiple depth peaks)
        hist, bin_edges = np.histogram(depths, bins=10)
        peaks = (hist > num_points * 0.15).sum()
        occlusion_score = min(1.0, (peaks - 1) / 3) if peaks > 1 else 0.0
        occlusion_penalty = 1 - occlusion_score * 0.3

        # Spatial coverage (if we have 3D points)
        if points_3d is not None and len(points_3d) > 0:
            xy_range = np.ptp(points_3d[:, :2], axis=0)
            coverage_score = min(1.0, np.prod(xy_range) / 4.0)  # Normalize by ~2x2m
        else:
            coverage_score = 0.5

        # Combined confidence
        confidence = (
            0.4 * point_conf +
            0.3 * consistency_conf +
            0.2 * coverage_score
        ) * occlusion_penalty

        return {
            "depth": depth,
            "confidence": float(confidence),
            "num_points": num_points,
            "std": std,
            "occlusion_score": float(occlusion_score),
            "coverage_score": float(coverage_score),
        }

    def estimate_depth(
        self,
        detection: "Detection",
        points: np.ndarray,
    ) -> Optional[float]:
        """
        Estimate depth for a single detection.

        Args:
            detection: 2D detection with bbox attribute.
            points: LiDAR points (N, 4).

        Returns:
            Estimated depth or None if insufficient points.
        """
        bbox = detection.bbox if hasattr(detection, 'bbox') else detection.box
        _, _, depths = self.extract_depth_in_bbox(points, bbox)
        return self.get_representative_depth(depths)

    def estimate_depths_batch(
        self,
        detections: List["Detection"],
        points: np.ndarray,
    ) -> List[Optional[float]]:
        """
        Estimate depths for multiple detections efficiently.

        Projects LiDAR once and processes all detections.

        Args:
            detections: List of 2D detections.
            points: LiDAR points (N, 4).

        Returns:
            List of depth estimates (None for failed estimates).
        """
        results = []

        for detection in detections:
            depth = self.estimate_depth(detection, points)
            results.append(depth)

        return results

    def estimate_full(
        self,
        detection: "Detection",
        points: np.ndarray,
    ) -> DepthResult:
        """
        Full depth estimation with all metadata.

        Args:
            detection: 2D detection.
            points: LiDAR points.

        Returns:
            DepthResult with depth, confidence, and point data.
        """
        bbox = detection.bbox if hasattr(detection, 'bbox') else detection.box
        points_3d, points_2d, depths = self.extract_depth_in_bbox(points, bbox)

        if len(depths) < self.min_points:
            return DepthResult(
                depth=None,
                confidence=0.0,
                num_points=len(depths),
                std=None,
                points_3d=points_3d if len(points_3d) > 0 else None,
                points_2d=points_2d if len(points_2d) > 0 else None,
            )

        depth = self.get_representative_depth(depths)
        conf = self.get_depth_confidence(points_3d=points_3d, depths=depths)

        return DepthResult(
            depth=depth,
            confidence=conf["confidence"],
            num_points=len(depths),
            std=conf["std"],
            points_3d=points_3d,
            points_2d=points_2d,
        )

    def _expand_box(self, box: np.ndarray) -> np.ndarray:
        """Expand bounding box by search_expansion fraction."""
        if self.search_expansion == 0:
            return np.asarray(box)

        box = np.asarray(box)
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        expand_w = w * self.search_expansion
        expand_h = h * self.search_expansion

        return np.array([
            x1 - expand_w,
            y1 - expand_h,
            x2 + expand_w,
            y2 + expand_h,
        ])

    def _aggregate_depths(
        self,
        depths: np.ndarray,
        points_2d: np.ndarray,
        box: np.ndarray,
    ) -> float:
        """
        Aggregate depth values using specified method.

        Args:
            depths: Depth values.
            points_2d: Corresponding 2D points.
            box: Detection box for weighted methods.

        Returns:
            Aggregated depth value.
        """
        if self.method == "weighted":
            # Weight by distance to box center
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2

            distances = np.sqrt(
                (points_2d[:, 0] - center_x) ** 2 +
                (points_2d[:, 1] - center_y) ** 2
            )

            # Inverse distance weights
            weights = 1 / (distances + 1e-6)
            weights /= weights.sum()

            return float(np.sum(depths * weights))

        return self.get_representative_depth(depths)
