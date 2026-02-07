"""
3D-2D Projection Utilities Module.

This module provides functions and classes for projecting 3D points to 2D
image coordinates and back-projecting 2D pixels to 3D space.

Mathematical Background:
========================

Pinhole Camera Model:
---------------------
The pinhole camera model projects 3D world points to 2D image coordinates:

    [u]       [X]       [fx  0  cx] [X]
    [v] = K * [Y] / Z = [ 0 fy  cy] [Y] / Z
    [1]       [Z]       [ 0  0   1] [Z]

Where:
    - (X, Y, Z): 3D point in camera coordinates
    - (u, v): 2D pixel coordinates
    - K: Camera intrinsic matrix
    - fx, fy: Focal lengths
    - cx, cy: Principal point

Full Projection Pipeline:
-------------------------
For points in world/LiDAR coordinates:

    p_2d = K * [R|t] * P_3d

Or using homogeneous coordinates:
    [u']       [X]
    [v'] = P * [Y]    where P is 3x4 projection matrix
    [w ]       [Z]
               [1]

Then: u = u'/w, v = v'/w

Back-projection (with known depth):
-----------------------------------
Given pixel (u, v) and depth Z:

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

This gives a 3D point in camera coordinates.

Coordinate Transformations:
---------------------------
A 3D point P in frame A can be transformed to frame B:

    P_B = R @ P_A + t

Where:
    - R: 3x3 rotation matrix
    - t: 3x1 translation vector

In homogeneous form:
    [P_B]   [R  t] [P_A]
    [ 1 ] = [0  1] [ 1 ]
"""

from typing import Optional, Tuple, Union

import numpy as np

from .intrinsics import CameraIntrinsics
from .extrinsics import CameraExtrinsics, CameraLiDARExtrinsics


# =============================================================================
# Standalone Projection Functions
# =============================================================================

def project_3d_to_2d(
    points_3d: np.ndarray,
    K: np.ndarray,
    R: Optional[np.ndarray] = None,
    t: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to 2D pixel coordinates.

    This function implements the full camera projection pipeline:
    1. Transform from world to camera coordinates (if R, t provided)
    2. Project using intrinsic matrix K
    3. Perspective division

    Mathematical Form:
        If R, t are provided:
            P_cam = R @ P_world + t
        Then:
            [u]       P_cam       [fx  0  cx] [X]
            [v] = K * ----- = K * [ 0 fy  cy] [Y] / Z
            [1]         Z         [ 0  0   1] [Z]

    Args:
        points_3d: 3D points (N, 3) in world coordinates (or camera if R,t=None).
        K: 3x3 camera intrinsic matrix.
        R: 3x3 rotation matrix (optional, world to camera).
        t: 3x1 or (3,) translation vector (optional).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - points_2d: Pixel coordinates (N, 2)
            - depths: Depth values Z in camera frame (N,)

    Warning:
        Points behind the camera (Z <= 0) will produce invalid results.
        Filter these points before or after projection.

    Example:
        >>> K = np.array([[721.5, 0, 609.5], [0, 721.5, 172.8], [0, 0, 1]])
        >>> points_3d = np.array([[1, 0, 10], [2, 1, 20], [-1, 0, 15]])
        >>> points_2d, depths = project_3d_to_2d(points_3d, K)
        >>> print(f"Projected to pixels: {points_2d[0]}")
    """
    points_3d = np.atleast_2d(points_3d)

    # Transform to camera coordinates if extrinsics provided
    if R is not None and t is not None:
        t = np.asarray(t).flatten()
        points_cam = points_3d @ R.T + t
    else:
        points_cam = points_3d

    # Extract depths
    depths = points_cam[:, 2].copy()

    # Avoid division by zero
    depths_safe = np.where(depths > 0, depths, 1e-10)

    # Perspective projection: normalize by Z
    points_norm = points_cam[:, :2] / depths_safe[:, np.newaxis]

    # Apply intrinsics
    # u = fx * x + cx, v = fy * y + cy
    points_2d = np.zeros((len(points_3d), 2))
    points_2d[:, 0] = K[0, 0] * points_norm[:, 0] + K[0, 2]
    points_2d[:, 1] = K[1, 1] * points_norm[:, 1] + K[1, 2]

    return points_2d, depths


def backproject_2d_to_3d(
    pixels: np.ndarray,
    depths: Union[float, np.ndarray],
    K: np.ndarray,
) -> np.ndarray:
    """
    Back-project 2D pixels to 3D points using depth values.

    This is the inverse of perspective projection. Given pixel coordinates
    and depth, recover the 3D point in camera coordinates.

    Mathematical Form:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth

    Args:
        pixels: 2D pixel coordinates (N, 2) or (2,).
        depths: Depth value(s) - scalar or array (N,).
        K: 3x3 camera intrinsic matrix.

    Returns:
        np.ndarray: 3D points (N, 3) or (3,) in camera coordinates.

    Note:
        The returned points are in camera coordinate frame:
        - X: Right
        - Y: Down
        - Z: Forward (depth direction)

    Example:
        >>> K = np.array([[721.5, 0, 609.5], [0, 721.5, 172.8], [0, 0, 1]])
        >>> pixels = np.array([[700, 200], [500, 150]])
        >>> depths = np.array([10.0, 15.0])
        >>> points_3d = backproject_2d_to_3d(pixels, depths, K)
        >>> print(f"3D point: {points_3d[0]}")

    Mathematical Derivation:
        From projection: u = fx * X/Z + cx
        Solving for X: X = (u - cx) * Z / fx
        Similarly: Y = (v - cy) * Z / fy
    """
    pixels = np.atleast_2d(pixels)
    depths = np.atleast_1d(depths)

    # Ensure depths broadcast correctly
    if len(depths) == 1:
        depths = np.full(len(pixels), depths[0])

    # Extract intrinsic parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Back-project
    X = (pixels[:, 0] - cx) * depths / fx
    Y = (pixels[:, 1] - cy) * depths / fy
    Z = depths

    points_3d = np.stack([X, Y, Z], axis=1)

    return points_3d.squeeze()


def transform_points(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Apply rigid body transformation to 3D points.

    Transforms points from one coordinate frame to another using
    rotation and translation.

    Mathematical Form:
        P_out = R @ P_in + t

    Args:
        points: 3D points (N, 3) or (3,) in source frame.
        R: 3x3 rotation matrix.
        t: 3x1 or (3,) translation vector.

    Returns:
        np.ndarray: Transformed points (N, 3) or (3,).

    Properties of Rigid Transformations:
        - Preserves distances between points
        - Preserves angles between vectors
        - R is orthonormal: R^T @ R = I, det(R) = 1

    Example:
        >>> R = np.eye(3)  # Identity rotation
        >>> t = np.array([1, 0, 0])  # Translate 1 unit in X
        >>> points = np.array([[0, 0, 0], [1, 1, 1]])
        >>> transformed = transform_points(points, R, t)
        >>> print(transformed)  # [[1, 0, 0], [2, 1, 1]]

    Note:
        For the inverse transformation (from target back to source frame):
            R_inv = R.T
            t_inv = -R.T @ t
            P_source = R_inv @ P_target + t_inv
    """
    points = np.atleast_2d(points)
    t = np.asarray(t).flatten()

    transformed = points @ R.T + t

    return transformed.squeeze()


def filter_fov(
    points: np.ndarray,
    image_shape: Tuple[int, int],
    K: np.ndarray,
    R: Optional[np.ndarray] = None,
    t: Optional[np.ndarray] = None,
    depth_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter 3D points to keep only those visible in the camera's field of view.

    This function:
    1. Projects all points to image coordinates
    2. Filters out points behind camera (Z <= 0)
    3. Filters out points outside image bounds
    4. Optionally filters by depth range

    Args:
        points: 3D points (N, 3) in world/LiDAR coordinates.
        image_shape: (height, width) of the image.
        K: 3x3 camera intrinsic matrix.
        R: 3x3 rotation matrix (world to camera), optional.
        t: 3x1 translation vector, optional.
        depth_range: Optional (min_depth, max_depth) filter.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - points_2d: Filtered pixel coordinates (M, 2)
            - depths: Filtered depth values (M,)
            - mask: Boolean mask (N,) indicating which original points are valid

    Example:
        >>> K = np.array([[721.5, 0, 609.5], [0, 721.5, 172.8], [0, 0, 1]])
        >>> points = np.random.randn(1000, 3) * 20  # Random 3D points
        >>> points[:, 2] = np.abs(points[:, 2]) + 1  # Ensure positive depth
        >>> pts_2d, depths, mask = filter_fov(points, (375, 1242), K)
        >>> print(f"Kept {mask.sum()} of {len(points)} points")

    Filtering Criteria:
        1. depth > 0 (in front of camera)
        2. 0 <= u < width
        3. 0 <= v < height
        4. min_depth <= depth <= max_depth (if depth_range provided)
    """
    height, width = image_shape

    # Project to 2D
    points_2d, depths = project_3d_to_2d(points, K, R, t)

    # Initialize mask
    mask = np.ones(len(points), dtype=bool)

    # Filter by positive depth (in front of camera)
    mask &= depths > 0

    # Filter by image bounds
    mask &= (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width)
    mask &= (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)

    # Filter by depth range if provided
    if depth_range is not None:
        min_depth, max_depth = depth_range
        mask &= (depths >= min_depth) & (depths <= max_depth)

    return points_2d[mask], depths[mask], mask


def compute_frustum_points(
    image_shape: Tuple[int, int],
    K: np.ndarray,
    depth: float,
) -> np.ndarray:
    """
    Compute the 3D points of the camera frustum at a given depth.

    Returns the 4 corner points of the image plane at the specified depth,
    useful for visualizing camera field of view.

    Args:
        image_shape: (height, width) of the image.
        K: 3x3 camera intrinsic matrix.
        depth: Depth at which to compute frustum corners.

    Returns:
        np.ndarray: 4 corner points (4, 3) at the given depth.
                    Order: top-left, top-right, bottom-right, bottom-left.

    Example:
        >>> K = np.array([[721.5, 0, 609.5], [0, 721.5, 172.8], [0, 0, 1]])
        >>> corners = compute_frustum_points((375, 1242), K, depth=50.0)
        >>> print(f"Frustum width at 50m: {corners[1, 0] - corners[0, 0]:.1f}m")
    """
    height, width = image_shape

    # Image corners (pixel coordinates)
    corners_2d = np.array([
        [0, 0],              # Top-left
        [width - 1, 0],      # Top-right
        [width - 1, height - 1],  # Bottom-right
        [0, height - 1],     # Bottom-left
    ])

    # Back-project to 3D
    corners_3d = backproject_2d_to_3d(corners_2d, depth, K)

    return corners_3d


# =============================================================================
# Projector Class (combines intrinsics and extrinsics)
# =============================================================================

class Projector:
    """
    Handle 3D-2D projections between sensor frames.

    This class combines camera intrinsics and camera-LiDAR extrinsics to
    provide convenient methods for projecting LiDAR points to images and
    estimating 3D positions from 2D detections.

    The projection pipeline:
        1. LiDAR points → Transform to camera frame
        2. Camera points → Project to image pixels

    The back-projection pipeline:
        1. Image pixels + depth → 3D camera points
        2. Camera points → Transform to LiDAR frame (optional)

    Attributes:
        intrinsics: Camera intrinsic parameters.
        extrinsics: Camera-LiDAR extrinsic calibration.

    Example:
        >>> intrinsics = CameraIntrinsics(fx=721.5, fy=721.5, cx=609.5, cy=172.8,
        ...                                width=1242, height=375)
        >>> extrinsics = CameraLiDARExtrinsics("path/to/calib.txt")
        >>> projector = Projector(intrinsics, extrinsics)
        >>> points_2d, depths, mask = projector.project_lidar_to_image(lidar_points)
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraLiDARExtrinsics,
    ):
        """
        Initialize the projector.

        Args:
            intrinsics: Camera intrinsic parameters.
            extrinsics: Camera-LiDAR extrinsic calibration.
        """
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    def project_lidar_to_image(
        self,
        points: np.ndarray,
        filter_fov: bool = True,
        filter_depth: bool = True,
        depth_range: Tuple[float, float] = (0.1, 100.0),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project LiDAR points to image coordinates.

        Pipeline:
        1. Transform points from LiDAR to camera frame
        2. Project using camera intrinsics
        3. Optionally filter by FOV and depth

        Args:
            points: LiDAR points (N, 3) or (N, 4) with optional intensity.
            filter_fov: Whether to filter points outside camera FOV.
            filter_depth: Whether to filter by depth range.
            depth_range: (min_depth, max_depth) in meters.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - points_2d: Image coordinates (M, 2) of valid points
                - depths: Depth values (M,) of valid points
                - mask: Boolean mask (N,) indicating which points are valid

        Example:
            >>> projector = Projector(intrinsics, extrinsics)
            >>> pts_2d, depths, mask = projector.project_lidar_to_image(lidar_points)
            >>> print(f"Projected {mask.sum()} of {len(lidar_points)} points")
        """
        # Project to image
        points_2d, depths = self.extrinsics.project_lidar_to_image(points)

        # Initialize mask
        mask = np.ones(len(points), dtype=bool)

        # Filter by depth
        if filter_depth:
            depth_mask = (depths >= depth_range[0]) & (depths <= depth_range[1])
            mask &= depth_mask

        # Filter by FOV
        if filter_fov:
            fov_mask = self.intrinsics.is_in_image(points_2d)
            mask &= fov_mask

        return points_2d[mask], depths[mask], mask

    def project_3d_box_to_image(
        self,
        center: np.ndarray,
        dimensions: np.ndarray,
        rotation_y: float,
    ) -> np.ndarray:
        """
        Project 3D bounding box corners to image coordinates.

        Creates the 8 corners of a 3D bounding box in camera coordinates
        and projects them to the image.

        Box corner ordering (when viewed from above):
            3 ---- 2        4 ---- 5
            |      |   =>   |      |
            0 ---- 1        7 ---- 6
            (bottom)        (top)

        Args:
            center: Box center (x, y, z) in camera frame.
            dimensions: Box dimensions [length, width, height].
            rotation_y: Rotation around Y-axis (radians), yaw angle.

        Returns:
            np.ndarray: 8 corners projected to image (8, 2).

        Note:
            In KITTI, objects are defined in camera coordinates with:
            - X: right, Y: down, Z: forward
            - rotation_y rotates around the Y (down) axis
        """
        l, w, h = dimensions

        # 3D box corners in object-centered coordinates
        # Order: bottom corners (0-3), top corners (4-7)
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

        corners_3d = np.array([x_corners, y_corners, z_corners])  # (3, 8)

        # Rotation matrix around Y-axis
        c, s = np.cos(rotation_y), np.sin(rotation_y)
        R_y = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ])

        # Rotate and translate to camera frame
        corners_cam = R_y @ corners_3d + center.reshape(3, 1)  # (3, 8)
        corners_cam = corners_cam.T  # (8, 3)

        # Project to image
        corners_2d = self.intrinsics.project_point(corners_cam)

        return corners_2d

    def get_depth_at_pixel(
        self,
        points: np.ndarray,
        pixel: Tuple[int, int],
        search_radius: int = 5,
    ) -> Optional[float]:
        """
        Get LiDAR depth at a specific pixel location.

        Finds the median depth of LiDAR points that project near the
        specified pixel.

        Args:
            points: LiDAR points (N, 4).
            pixel: (u, v) pixel coordinates.
            search_radius: Search radius in pixels.

        Returns:
            float or None: Median depth of nearby points, or None if no points found.

        Example:
            >>> depth = projector.get_depth_at_pixel(lidar_points, (600, 180))
            >>> if depth is not None:
            ...     print(f"Depth at pixel: {depth:.2f}m")
        """
        points_2d, depths, _ = self.project_lidar_to_image(points)

        if len(points_2d) == 0:
            return None

        # Find points near pixel
        u, v = pixel
        distances = np.sqrt(
            (points_2d[:, 0] - u) ** 2 +
            (points_2d[:, 1] - v) ** 2
        )

        nearby_mask = distances <= search_radius

        if not np.any(nearby_mask):
            return None

        # Return median depth of nearby points
        return float(np.median(depths[nearby_mask]))

    def get_depth_in_box(
        self,
        points: np.ndarray,
        box_2d: np.ndarray,
        method: str = "median",
    ) -> Optional[float]:
        """
        Get depth estimate within a 2D bounding box.

        Projects LiDAR points to image and computes aggregate depth
        of points falling within the specified bounding box.

        Args:
            points: LiDAR points (N, 4).
            box_2d: 2D bounding box [x1, y1, x2, y2].
            method: Aggregation method:
                    - 'median': Median depth (robust to outliers)
                    - 'mean': Mean depth
                    - 'closest': Minimum depth (closest point)
                    - 'percentile_25': 25th percentile (front of object)

        Returns:
            float or None: Estimated depth, or None if no points in box.

        Example:
            >>> box = np.array([500, 150, 650, 280])  # x1, y1, x2, y2
            >>> depth = projector.get_depth_in_box(lidar_points, box)
            >>> if depth:
            ...     print(f"Object depth: {depth:.2f}m")
        """
        points_2d, depths, _ = self.project_lidar_to_image(points)

        if len(points_2d) == 0:
            return None

        x1, y1, x2, y2 = box_2d

        # Find points in box
        in_box = (
            (points_2d[:, 0] >= x1) &
            (points_2d[:, 0] <= x2) &
            (points_2d[:, 1] >= y1) &
            (points_2d[:, 1] <= y2)
        )

        if not np.any(in_box):
            return None

        box_depths = depths[in_box]

        if method == "median":
            return float(np.median(box_depths))
        elif method == "mean":
            return float(np.mean(box_depths))
        elif method == "closest":
            return float(np.min(box_depths))
        elif method == "percentile_25":
            return float(np.percentile(box_depths, 25))
        else:
            raise ValueError(f"Unknown method: {method}")

    def unproject_box_to_3d(
        self,
        box_2d: np.ndarray,
        depth: float,
        dimensions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unproject 2D bounding box to 3D using depth estimate.

        Estimates 3D object center from 2D box center and depth,
        assuming the object bottom is at ground level.

        Args:
            box_2d: 2D bounding box [x1, y1, x2, y2].
            depth: Estimated object depth (Z in camera frame).
            dimensions: Default object dimensions [length, width, height].

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - center_3d: Estimated 3D center in camera coordinates
                - dimensions: Object dimensions (passed through)

        Note:
            This is an approximation. The 2D box center projects to a ray,
            and depth gives the Z coordinate. X is estimated from this,
            while Y is adjusted assuming the object sits on the ground.
        """
        x1, y1, x2, y2 = box_2d

        # Get box center in image
        center_2d = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

        # Unproject bottom center of 2D box to 3D
        bottom_2d = np.array([(x1 + x2) / 2, y2])
        center_3d = self.intrinsics.unproject_point(bottom_2d, depth)

        # Adjust Y to be at object center (move up by half height)
        center_3d[1] -= dimensions[2] / 2

        return center_3d, dimensions

    def camera_to_lidar(self, points_cam: np.ndarray) -> np.ndarray:
        """
        Transform points from camera to LiDAR coordinate frame.

        Args:
            points_cam: Points in camera frame (N, 3).

        Returns:
            np.ndarray: Points in LiDAR frame (N, 3).
        """
        return self.extrinsics.transform_to_lidar(points_cam)

    def lidar_to_camera(self, points_lidar: np.ndarray) -> np.ndarray:
        """
        Transform points from LiDAR to camera coordinate frame.

        Args:
            points_lidar: Points in LiDAR frame (N, 3) or (N, 4).

        Returns:
            np.ndarray: Points in camera frame (N, 3).
        """
        return self.extrinsics.transform_to_camera(points_lidar)
