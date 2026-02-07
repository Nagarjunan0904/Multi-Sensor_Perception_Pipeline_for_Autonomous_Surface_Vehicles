"""
Camera-LiDAR Extrinsic Calibration Module.

This module handles the extrinsic calibration between sensors, specifically
the transformation between LiDAR (Velodyne) and camera coordinate frames.

Mathematical Background:
========================

Rigid Body Transformation:
--------------------------
A rigid body transformation consists of a rotation R (3x3 orthonormal matrix)
and translation t (3x1 vector). For a point P in frame A, its coordinates
in frame B are:

    P_B = R * P_A + t

This can be written as a 4x4 homogeneous transformation matrix:

    T = | R   t |    where T transforms points: P_B = T * P_A (homogeneous)
        | 0   1 |

Inverse Transformation:
-----------------------
The inverse transformation (from B to A) is:

    T^(-1) = | R^T  -R^T * t |
             |  0       1    |

Since R is orthonormal: R^(-1) = R^T

KITTI Coordinate Systems:
=========================

Camera Coordinate Frame (Reference):
    - X: Right
    - Y: Down
    - Z: Forward (into the scene)
    - Origin: At camera optical center

LiDAR (Velodyne) Coordinate Frame:
    - X: Forward
    - Y: Left
    - Z: Up
    - Origin: At LiDAR sensor center

The transformation Tr_velo_to_cam converts points from Velodyne to camera frame.

KITTI Calibration Files:
========================

KITTI calibration files contain:

    P0, P1, P2, P3: 3x4 projection matrices for cameras 0-3
        - P2 is typically used (left color camera)
        - Combines intrinsics K with baseline offset for stereo

    R0_rect: 3x3 rectification matrix
        - Aligns camera axes to be parallel
        - Applied after Tr_velo_to_cam

    Tr_velo_to_cam: 3x4 rigid body transformation
        - First 3x3 is rotation R
        - Last column is translation t

Complete LiDAR to Image Projection:
    1. P_velo (in Velodyne frame)
    2. P_cam_unrect = Tr_velo_to_cam * P_velo
    3. P_cam = R0_rect * P_cam_unrect
    4. p_img = P2 * P_cam (with perspective division)

Or combined:
    p_img = P2 * R0_rect * Tr_velo_to_cam * [P_velo; 1]
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np


@dataclass
class CameraExtrinsics:
    """
    Camera extrinsic parameters (rotation and translation).

    This class represents the pose of one coordinate frame relative to another,
    commonly used for the transformation from LiDAR to camera frame.

    Attributes:
        R: Rotation matrix (3x3) - transforms vectors from source to target frame.
        t: Translation vector (3,) - position of source origin in target frame.

    Mathematical Details:
        For a point P in the source frame:
            P_target = R @ P_source + t

    Example:
        >>> R = np.eye(3)  # Identity rotation
        >>> t = np.array([0.27, 0.06, -0.12])  # Typical KITTI values
        >>> extrinsics = CameraExtrinsics(R=R, t=t)
        >>> T = extrinsics.get_transform_matrix()
    """

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        """Validate inputs after initialization."""
        self.R = np.asarray(self.R, dtype=np.float64)
        self.t = np.asarray(self.t, dtype=np.float64).flatten()

        if self.R.shape != (3, 3):
            raise ValueError(f"R must be 3x3, got {self.R.shape}")
        if self.t.shape != (3,):
            raise ValueError(f"t must be (3,), got {self.t.shape}")

    def get_transform_matrix(self) -> np.ndarray:
        """
        Get the 4x4 homogeneous transformation matrix.

        The transformation matrix combines rotation and translation:

            T = | R  t |
                | 0  1 |

        For a point P = [x, y, z, 1]^T in homogeneous coordinates:
            P_transformed = T @ P

        Returns:
            np.ndarray: 4x4 transformation matrix.

        Example:
            >>> extrinsics = CameraExtrinsics(R=np.eye(3), t=np.array([1, 2, 3]))
            >>> T = extrinsics.get_transform_matrix()
            >>> print(T)
            [[1. 0. 0. 1.]
             [0. 1. 0. 2.]
             [0. 0. 1. 3.]
             [0. 0. 0. 1.]]
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def get_3x4_matrix(self) -> np.ndarray:
        """
        Get the 3x4 transformation matrix (without homogeneous row).

        Returns:
            np.ndarray: 3x4 matrix [R | t].
        """
        return np.hstack([self.R, self.t.reshape(3, 1)])

    def inverse(self) -> "CameraExtrinsics":
        """
        Get the inverse transformation.

        For transformation T = [R, t], the inverse is:
            T^(-1) = [R^T, -R^T @ t]

        This is useful for converting from camera frame back to LiDAR frame.

        Returns:
            CameraExtrinsics: New instance representing the inverse transform.

        Mathematical Derivation:
            Given: P_cam = R @ P_velo + t
            Solving for P_velo:
                P_velo = R^T @ (P_cam - t)
                       = R^T @ P_cam - R^T @ t
            So: R_inv = R^T, t_inv = -R^T @ t

        Example:
            >>> extrinsics = CameraExtrinsics(R=np.eye(3), t=np.array([1, 2, 3]))
            >>> inv = extrinsics.inverse()
            >>> print(inv.t)  # [-1, -2, -3]
        """
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return CameraExtrinsics(R=R_inv, t=t_inv)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform 3D points using this extrinsic transformation.

        Args:
            points: 3D points (N, 3) or (3,) in source frame.

        Returns:
            np.ndarray: Transformed points in target frame.

        Example:
            >>> extrinsics = CameraExtrinsics(R=np.eye(3), t=np.array([1, 0, 0]))
            >>> pts = np.array([[0, 0, 0], [1, 1, 1]])
            >>> pts_transformed = extrinsics.transform_points(pts)
        """
        points = np.atleast_2d(points)
        transformed = points @ self.R.T + self.t
        return transformed.squeeze()

    def compose(self, other: "CameraExtrinsics") -> "CameraExtrinsics":
        """
        Compose this transformation with another (chain transformations).

        If this is T1 and other is T2, result is T2 @ T1
        (applies T1 first, then T2).

        Args:
            other: The transformation to apply after this one.

        Returns:
            CameraExtrinsics: Combined transformation.
        """
        R_combined = other.R @ self.R
        t_combined = other.R @ self.t + other.t
        return CameraExtrinsics(R=R_combined, t=t_combined)

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> "CameraExtrinsics":
        """
        Create from 4x4 or 3x4 transformation matrix.

        Args:
            T: 4x4 homogeneous or 3x4 transformation matrix.

        Returns:
            CameraExtrinsics: Instance with extracted R and t.
        """
        T = np.asarray(T)
        if T.shape == (4, 4):
            return cls(R=T[:3, :3], t=T[:3, 3])
        elif T.shape == (3, 4):
            return cls(R=T[:, :3], t=T[:, 3])
        else:
            raise ValueError(f"Expected 4x4 or 3x4 matrix, got {T.shape}")

    @classmethod
    def from_kitti_calib(
        cls,
        calib_dict: Dict[str, np.ndarray],
        key: str = "Tr_velo_to_cam",
    ) -> "CameraExtrinsics":
        """
        Create extrinsics from KITTI calibration dictionary.

        KITTI provides Tr_velo_to_cam as a 12-element vector that reshapes
        to a 3x4 matrix [R | t].

        Args:
            calib_dict: Dictionary with calibration data.
            key: Key for the transformation matrix (default: 'Tr_velo_to_cam').

        Returns:
            CameraExtrinsics: Instance with KITTI calibration.

        Raises:
            KeyError: If the specified key not found.
        """
        # Handle alternate key names
        if key not in calib_dict:
            alternate_keys = ["Tr_velo_cam", "Tr_velo2cam"]
            for alt_key in alternate_keys:
                if alt_key in calib_dict:
                    key = alt_key
                    break
            else:
                raise KeyError(f"Transformation '{key}' not found in calibration")

        T = calib_dict[key]
        if T.ndim == 1:
            T = T.reshape(3, 4)

        return cls(R=T[:, :3], t=T[:, 3])


class CameraLiDARExtrinsics:
    """
    Handle extrinsic calibration between camera and LiDAR for KITTI dataset.

    This class loads and manages the complete calibration for projecting
    LiDAR points to camera images, including:
    - Projection matrices P0-P3
    - Rectification matrix R0_rect
    - Velodyne to camera transformation Tr_velo_to_cam
    - IMU to Velodyne transformation Tr_imu_to_velo (optional)

    The full projection pipeline for LiDAR point to image pixel:
        1. Transform from Velodyne to unrectified camera: Tr_velo_to_cam
        2. Rectify camera coordinates: R0_rect
        3. Project to image: P2 (includes intrinsics and stereo offset)

    Attributes:
        P0, P1, P2, P3: 3x4 projection matrices for each camera.
        R0_rect: 3x3 rectification rotation matrix.
        Tr_velo_to_cam: 3x4 Velodyne to camera transformation.
        Tr_imu_to_velo: 3x4 IMU to Velodyne transformation (optional).

    Example:
        >>> extrinsics = CameraLiDARExtrinsics("path/to/calib.txt")
        >>> points_2d, depths = extrinsics.project_lidar_to_image(lidar_points)
    """

    def __init__(
        self,
        calib_file: Optional[str] = None,
        R: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
    ):
        """
        Initialize extrinsic calibration.

        Args:
            calib_file: Path to KITTI calibration file.
            R: Rotation matrix (3x3) - alternative to loading from file.
            T: Translation vector (3,) - alternative to loading from file.
        """
        # Initialize all calibration matrices to None
        self.P0: Optional[np.ndarray] = None
        self.P1: Optional[np.ndarray] = None
        self.P2: Optional[np.ndarray] = None
        self.P3: Optional[np.ndarray] = None
        self.R0_rect: Optional[np.ndarray] = None
        self.Tr_velo_to_cam: Optional[np.ndarray] = None
        self.Tr_imu_to_velo: Optional[np.ndarray] = None

        if calib_file is not None:
            self.load_kitti_calib(calib_file)
        elif R is not None and T is not None:
            self.set_transform(R, T)

    def load_kitti_calib(self, calib_file: str) -> None:
        """
        Load calibration from KITTI format file.

        KITTI calibration file format:
            P0: [12 values] (flattened 3x4 matrix)
            P1: [12 values]
            P2: [12 values]
            P3: [12 values]
            R0_rect: [9 values] (flattened 3x3 matrix)
            Tr_velo_to_cam: [12 values] (flattened 3x4 matrix)
            Tr_imu_to_velo: [12 values] (optional)

        Args:
            calib_file: Path to calibration file.

        Raises:
            FileNotFoundError: If calibration file doesn't exist.
        """
        calib_file = Path(calib_file)
        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_file}")

        calibs = {}
        with open(calib_file, "r") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                calibs[key.strip()] = np.array([float(x) for x in value.split()])

        # Parse projection matrices (3x4)
        for i in range(4):
            key = f"P{i}"
            if key in calibs:
                setattr(self, key, calibs[key].reshape(3, 4))

        # Rectification matrix (3x3)
        if "R0_rect" in calibs:
            self.R0_rect = calibs["R0_rect"].reshape(3, 3)

        # Velodyne to camera transformation (3x4)
        for key in ["Tr_velo_to_cam", "Tr_velo_cam"]:
            if key in calibs:
                self.Tr_velo_to_cam = calibs[key].reshape(3, 4)
                break

        # IMU to Velodyne (optional, 3x4)
        if "Tr_imu_to_velo" in calibs:
            self.Tr_imu_to_velo = calibs["Tr_imu_to_velo"].reshape(3, 4)

    def set_transform(self, R: np.ndarray, T: np.ndarray) -> None:
        """
        Set transformation directly from rotation and translation.

        Args:
            R: Rotation matrix (3x3).
            T: Translation vector (3,).
        """
        self.Tr_velo_to_cam = np.hstack([R, T.reshape(3, 1)])

    @property
    def rotation(self) -> np.ndarray:
        """
        Get rotation matrix (3x3) from Velodyne to camera frame.

        Returns:
            np.ndarray: 3x3 rotation matrix, or identity if not set.
        """
        if self.Tr_velo_to_cam is None:
            return np.eye(3)
        return self.Tr_velo_to_cam[:, :3]

    @property
    def translation(self) -> np.ndarray:
        """
        Get translation vector (3,) from Velodyne to camera frame.

        Returns:
            np.ndarray: Translation vector, or zeros if not set.
        """
        if self.Tr_velo_to_cam is None:
            return np.zeros(3)
        return self.Tr_velo_to_cam[:, 3]

    def get_transform_matrix(self) -> np.ndarray:
        """
        Get the 4x4 homogeneous transformation matrix (Velodyne to camera).

        Returns:
            np.ndarray: 4x4 transformation matrix.

        Mathematical Form:
            T = | R  t |
                | 0  1 |

        Where R is 3x3 rotation and t is 3x1 translation from Tr_velo_to_cam.
        """
        T = np.eye(4, dtype=np.float64)
        if self.Tr_velo_to_cam is not None:
            T[:3, :4] = self.Tr_velo_to_cam
        return T

    def inverse(self) -> "CameraExtrinsics":
        """
        Get the inverse transformation (camera to Velodyne).

        Useful for converting points from camera coordinates back to LiDAR.

        Returns:
            CameraExtrinsics: Inverse transformation.
        """
        R_inv = self.rotation.T
        t_inv = -R_inv @ self.translation
        return CameraExtrinsics(R=R_inv, t=t_inv)

    def transform_to_camera(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from LiDAR (Velodyne) to camera coordinate frame.

        The transformation includes:
        1. Rigid body transform: P_cam = R @ P_velo + t
        2. Rectification: P_rect = R0_rect @ P_cam (if available)

        KITTI Coordinate Systems:
            Velodyne: X-forward, Y-left, Z-up
            Camera:   X-right, Y-down, Z-forward

        Args:
            points: Points in LiDAR frame (N, 3) or (N, 4).
                    If (N, 4), the 4th column (intensity) is ignored.

        Returns:
            np.ndarray: Points in camera frame (N, 3).

        Raises:
            ValueError: If transformation not initialized.
        """
        if self.Tr_velo_to_cam is None:
            raise ValueError("Transformation not initialized")

        # Extract XYZ (ignore intensity if present)
        xyz = points[:, :3]

        # Apply rigid body transformation: P_cam = R @ P_velo + t
        points_cam = xyz @ self.rotation.T + self.translation

        # Apply rectification if available
        if self.R0_rect is not None:
            points_cam = points_cam @ self.R0_rect.T

        return points_cam

    def transform_to_lidar(self, points: np.ndarray) -> np.ndarray:
        """
        Transform points from camera to LiDAR (Velodyne) coordinate frame.

        This is the inverse of transform_to_camera().

        Args:
            points: Points in camera frame (N, 3).

        Returns:
            np.ndarray: Points in LiDAR frame (N, 3).

        Raises:
            ValueError: If transformation not initialized.
        """
        if self.Tr_velo_to_cam is None:
            raise ValueError("Transformation not initialized")

        # Inverse rectification
        if self.R0_rect is not None:
            points = points @ self.R0_rect  # R0_rect^T^T = R0_rect

        # Inverse rigid body transformation
        R_inv = self.rotation.T
        t_inv = -R_inv @ self.translation

        points_lidar = points @ R_inv.T + t_inv

        return points_lidar

    def get_projection_matrix(self, camera_id: int = 2) -> np.ndarray:
        """
        Get full projection matrix for specified camera.

        KITTI cameras:
            0: Left grayscale
            1: Right grayscale
            2: Left color (most commonly used)
            3: Right color

        Args:
            camera_id: Camera index (0-3, default 2 for left color camera).

        Returns:
            np.ndarray: 3x4 projection matrix.

        Raises:
            ValueError: If requested camera not calibrated.
        """
        P = getattr(self, f"P{camera_id}", None)
        if P is None:
            raise ValueError(f"Camera P{camera_id} not calibrated")
        return P

    def project_lidar_to_image(
        self,
        points: np.ndarray,
        camera_id: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project LiDAR points to image coordinates.

        Pipeline:
        1. Transform to camera frame (with rectification)
        2. Project using camera projection matrix
        3. Perspective division

        Args:
            points: LiDAR points (N, 3) or (N, 4).
            camera_id: Camera index.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - points_2d: Image coordinates (N, 2)
                - depths: Depth values (N,) - Z coordinate in camera frame
        """
        # Transform to camera frame
        points_cam = self.transform_to_camera(points)

        # Get depth (Z in camera frame)
        depths = points_cam[:, 2]

        # Get projection matrix
        P = self.get_projection_matrix(camera_id)

        # Project to image: [u, v, w] = P @ [X, Y, Z, 1]
        points_hom = np.hstack([points_cam, np.ones((len(points_cam), 1))])
        points_img = points_hom @ P.T

        # Perspective division: [u, v] = [u/w, v/w]
        points_2d = points_img[:, :2] / points_img[:, 2:3]

        return points_2d, depths

    def get_full_projection_matrix(self, camera_id: int = 2) -> np.ndarray:
        """
        Get the complete projection matrix from Velodyne to image.

        This combines all transformations:
            P_full = P_camera @ R0_rect_4x4 @ Tr_velo_to_cam_4x4

        Args:
            camera_id: Camera index.

        Returns:
            np.ndarray: 3x4 complete projection matrix.

        Note:
            This allows direct projection: p_img = P_full @ [x_velo; y_velo; z_velo; 1]
        """
        P = self.get_projection_matrix(camera_id)

        # Build 4x4 matrices
        R0_4x4 = np.eye(4)
        if self.R0_rect is not None:
            R0_4x4[:3, :3] = self.R0_rect

        Tr_4x4 = np.eye(4)
        if self.Tr_velo_to_cam is not None:
            Tr_4x4[:3, :4] = self.Tr_velo_to_cam

        # Combine: P @ R0 @ Tr
        P_4x4 = np.vstack([P, [0, 0, 0, 1]])
        full = P_4x4 @ R0_4x4 @ Tr_4x4

        return full[:3, :]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CameraLiDARExtrinsics("
            f"P2={'set' if self.P2 is not None else 'None'}, "
            f"R0_rect={'set' if self.R0_rect is not None else 'None'}, "
            f"Tr_velo_to_cam={'set' if self.Tr_velo_to_cam is not None else 'None'})"
        )
