"""
Camera Intrinsic Parameters Module.

This module handles camera intrinsic parameters which describe the internal
characteristics of a camera, including focal length and principal point.

Mathematical Background:
========================

The camera intrinsic matrix K (also called the calibration matrix) transforms
3D points in the camera coordinate frame to 2D pixel coordinates:

    K = | fx   0  cx |
        |  0  fy  cy |
        |  0   0   1 |

Where:
    - fx, fy: Focal lengths in pixel units (fx = f * mx, fy = f * my)
              f is the physical focal length, mx/my are pixels per unit length
    - cx, cy: Principal point coordinates (usually near image center)

The projection equation (pinhole camera model):

    | u |       | X |
    | v | = K * | Y | / Z
    | 1 |       | Z |

Where (X, Y, Z) is a 3D point in camera coordinates, and (u, v) is the
resulting pixel coordinate.

Inverse projection (given depth d):
    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d

KITTI Dataset:
==============
KITTI provides a 3x4 projection matrix P2 for the left color camera:

    P2 = | fx   0  cx  tx |
         |  0  fy  cy  ty |
         |  0   0   1  tz |

The first 3x3 block contains the intrinsic parameters. The 4th column
represents baseline offsets for stereo cameras (tx for horizontal offset).

Field of View:
==============
Horizontal FOV: 2 * atan(width / (2 * fx))
Vertical FOV:   2 * atan(height / (2 * fy))
Diagonal FOV:   2 * atan(sqrt(width² + height²) / (2 * f))

where f ≈ (fx + fy) / 2 for approximately square pixels.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np


@dataclass
class CameraIntrinsics:
    """
    Camera intrinsic parameters.

    This class stores and manipulates camera intrinsic parameters including
    focal lengths, principal point, and image dimensions.

    Attributes:
        fx: Focal length in x direction (pixels).
        fy: Focal length in y direction (pixels).
        cx: Principal point x coordinate (pixels).
        cy: Principal point y coordinate (pixels).
        width: Image width in pixels.
        height: Image height in pixels.

    Example:
        >>> intrinsics = CameraIntrinsics(fx=721.5, fy=721.5, cx=609.5, cy=172.8,
        ...                                width=1242, height=375)
        >>> K = intrinsics.get_K_matrix()
        >>> fov_h, fov_v = intrinsics.get_fov()
        >>> print(f"Horizontal FOV: {np.degrees(fov_h):.1f}°")
    """

    fx: float  # Focal length x (pixels)
    fy: float  # Focal length y (pixels)
    cx: float  # Principal point x (pixels)
    cy: float  # Principal point y (pixels)
    width: int  # Image width (pixels)
    height: int  # Image height (pixels)

    @property
    def K(self) -> np.ndarray:
        """
        Get camera intrinsic matrix (3x3).

        Returns:
            3x3 intrinsic matrix K.

        Note:
            Alias for get_K_matrix() for convenience.
        """
        return self.get_K_matrix()

    def get_K_matrix(self) -> np.ndarray:
        """
        Get the 3x3 camera intrinsic (calibration) matrix.

        The intrinsic matrix K maps 3D camera coordinates to 2D pixel coordinates:

            [u]       [X]       [fx  0  cx] [X]
            [v] = K * [Y] / Z = [ 0 fy  cy] [Y] / Z
            [1]       [Z]       [ 0  0   1] [Z]

        Returns:
            np.ndarray: 3x3 intrinsic matrix K with dtype float64.

        Mathematical Details:
            - fx, fy: Focal lengths determine the scaling from normalized
                      image coordinates to pixels.
            - cx, cy: Principal point offsets the image center.

        Example:
            >>> intrinsics = CameraIntrinsics(fx=721.5, fy=721.5, cx=609.5, cy=172.8,
            ...                                width=1242, height=375)
            >>> K = intrinsics.get_K_matrix()
            >>> print(K)
            [[721.5   0.  609.5]
             [  0.  721.5 172.8]
             [  0.    0.    1. ]]
        """
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def get_K_inverse(self) -> np.ndarray:
        """
        Get the inverse of the intrinsic matrix.

        The inverse is used for back-projection from pixels to camera rays:

            K^(-1) = | 1/fx    0   -cx/fx |
                     |   0   1/fy  -cy/fy |
                     |   0     0      1   |

        Returns:
            np.ndarray: 3x3 inverse intrinsic matrix.

        Note:
            For a point (u, v, 1), K^(-1) gives the normalized ray direction
            in camera coordinates (before scaling by depth).
        """
        return np.array([
            [1/self.fx, 0, -self.cx/self.fx],
            [0, 1/self.fy, -self.cy/self.fy],
            [0, 0, 1]
        ], dtype=np.float64)

    def get_fov(self) -> Tuple[float, float]:
        """
        Calculate the camera field of view.

        The field of view (FOV) is the angular extent of the scene visible
        through the camera lens.

        Horizontal FOV:
            θ_h = 2 * arctan(width / (2 * fx))

        Vertical FOV:
            θ_v = 2 * arctan(height / (2 * fy))

        Returns:
            Tuple[float, float]: (horizontal_fov, vertical_fov) in radians.

        Example:
            >>> intrinsics = CameraIntrinsics(fx=721.5, fy=721.5, cx=609.5, cy=172.8,
            ...                                width=1242, height=375)
            >>> fov_h, fov_v = intrinsics.get_fov()
            >>> print(f"H-FOV: {np.degrees(fov_h):.1f}°, V-FOV: {np.degrees(fov_v):.1f}°")
            H-FOV: 81.5°, V-FOV: 29.1°

        Note:
            KITTI camera typically has ~80° horizontal FOV.
        """
        horizontal_fov = 2 * np.arctan(self.width / (2 * self.fx))
        vertical_fov = 2 * np.arctan(self.height / (2 * self.fy))
        return horizontal_fov, vertical_fov

    def get_diagonal_fov(self) -> float:
        """
        Calculate the diagonal field of view.

        The diagonal FOV spans from corner to corner of the image:

            diagonal = sqrt(width² + height²)
            f_avg = (fx + fy) / 2
            θ_d = 2 * arctan(diagonal / (2 * f_avg))

        Returns:
            float: Diagonal FOV in radians.
        """
        diagonal = np.sqrt(self.width**2 + self.height**2)
        f_avg = (self.fx + self.fy) / 2
        return 2 * np.arctan(diagonal / (2 * f_avg))

    @classmethod
    def from_kitti_calib(
        cls,
        calib_dict: Dict[str, np.ndarray],
        width: int = 1242,
        height: int = 375,
        camera_id: int = 2,
    ) -> "CameraIntrinsics":
        """
        Create intrinsics from KITTI calibration dictionary.

        KITTI calibration files contain projection matrices P0-P3 for
        different cameras:
            - P0: Left grayscale camera
            - P1: Right grayscale camera
            - P2: Left color camera (most commonly used)
            - P3: Right color camera

        Each P matrix is 3x4 and combines intrinsics with extrinsic offset:

            P2 = | fx   0  cx  tx |    where tx accounts for stereo baseline
                 |  0  fy  cy  ty |
                 |  0   0   1  tz |

        Args:
            calib_dict: Dictionary containing calibration matrices.
                        Keys should include 'P0', 'P1', 'P2', 'P3'.
            width: Image width in pixels (default: KITTI standard 1242).
            height: Image height in pixels (default: KITTI standard 375).
            camera_id: Which camera to use (0-3, default 2 for left color).

        Returns:
            CameraIntrinsics: Instance with extracted parameters.

        Raises:
            KeyError: If required projection matrix not found.

        Example:
            >>> # Parse KITTI calib file into dict first
            >>> calib_dict = {'P2': np.array([721.5, 0, 609.5, 44.8, ...])}
            >>> intrinsics = CameraIntrinsics.from_kitti_calib(calib_dict)
        """
        key = f"P{camera_id}"
        if key not in calib_dict:
            raise KeyError(f"Projection matrix {key} not found in calibration")

        P = calib_dict[key]
        if P.ndim == 1:
            P = P.reshape(3, 4)

        return cls(
            fx=float(P[0, 0]),
            fy=float(P[1, 1]),
            cx=float(P[0, 2]),
            cy=float(P[1, 2]),
            width=width,
            height=height,
        )

    @classmethod
    def from_matrix(
        cls,
        K: np.ndarray,
        width: int,
        height: int,
    ) -> "CameraIntrinsics":
        """
        Create intrinsics from a 3x3 intrinsic matrix.

        Args:
            K: 3x3 intrinsic matrix.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            CameraIntrinsics: Instance with extracted parameters.

        Example:
            >>> K = np.array([[721.5, 0, 609.5],
            ...               [0, 721.5, 172.8],
            ...               [0, 0, 1]])
            >>> intrinsics = CameraIntrinsics.from_matrix(K, 1242, 375)
        """
        return cls(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            width=width,
            height=height,
        )

    @classmethod
    def from_projection_matrix(
        cls,
        P: np.ndarray,
        width: int,
        height: int,
    ) -> "CameraIntrinsics":
        """
        Create intrinsics from a 3x4 projection matrix.

        The projection matrix P combines intrinsics and extrinsics:

            P = K [R|t]

        For KITTI's P2 matrix, R is identity for the reference camera,
        so the first 3x3 block directly contains the intrinsics.

        Args:
            P: 3x4 projection matrix.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            CameraIntrinsics: Instance with extracted parameters.

        Note:
            This assumes the rotation component is identity, which is
            true for KITTI's P2 (reference camera). For other cameras
            with rotation, this extracts approximate intrinsics.
        """
        return cls(
            fx=float(P[0, 0]),
            fy=float(P[1, 1]),
            cx=float(P[0, 2]),
            cy=float(P[1, 2]),
            width=width,
            height=height,
        )

    def project_point(self, point_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D point(s) in camera frame to 2D pixel coordinates.

        Implements the pinhole camera model:

            u = fx * (X/Z) + cx
            v = fy * (Y/Z) + cy

        Args:
            point_3d: 3D point (3,) or points (N, 3) in camera coordinates.
                      Points should have positive Z (in front of camera).

        Returns:
            np.ndarray: 2D pixel coordinates (2,) or (N, 2).

        Warning:
            Points with Z <= 0 (behind camera) will produce invalid results.
            Filter these before projection.

        Example:
            >>> intrinsics = CameraIntrinsics(fx=721.5, fy=721.5, cx=609.5, cy=172.8,
            ...                                width=1242, height=375)
            >>> point_3d = np.array([1.0, 0.5, 10.0])  # X, Y, Z in camera frame
            >>> pixel = intrinsics.project_point(point_3d)
            >>> print(f"Pixel: ({pixel[0]:.1f}, {pixel[1]:.1f})")
        """
        point_3d = np.atleast_2d(point_3d)

        # Perspective division
        x = point_3d[:, 0] / point_3d[:, 2]
        y = point_3d[:, 1] / point_3d[:, 2]

        # Apply intrinsics
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy

        points_2d = np.stack([u, v], axis=1)

        return points_2d.squeeze()

    def unproject_point(
        self,
        point_2d: np.ndarray,
        depth: Union[float, np.ndarray],
    ) -> np.ndarray:
        """
        Back-project 2D pixel(s) to 3D using depth.

        Implements inverse projection:

            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth

        Args:
            point_2d: 2D pixel coordinate (2,) or coordinates (N, 2).
            depth: Depth value(s) in the same units as desired output.

        Returns:
            np.ndarray: 3D point(s) (3,) or (N, 3) in camera coordinates.

        Mathematical Derivation:
            From the projection equations:
                u = fx * X/Z + cx  →  X = (u - cx) * Z / fx
                v = fy * Y/Z + cy  →  Y = (v - cy) * Z / fy

        Example:
            >>> intrinsics = CameraIntrinsics(fx=721.5, fy=721.5, cx=609.5, cy=172.8,
            ...                                width=1242, height=375)
            >>> pixel = np.array([700, 200])
            >>> depth = 15.0  # meters
            >>> point_3d = intrinsics.unproject_point(pixel, depth)
            >>> print(f"3D point: ({point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f})")
        """
        point_2d = np.atleast_2d(point_2d)
        depth = np.atleast_1d(depth)

        # Inverse projection
        x = (point_2d[:, 0] - self.cx) * depth / self.fx
        y = (point_2d[:, 1] - self.cy) * depth / self.fy
        z = depth

        points_3d = np.stack([x, y, z], axis=1)

        return points_3d.squeeze()

    def pixel_to_ray(self, point_2d: np.ndarray) -> np.ndarray:
        """
        Convert pixel coordinates to normalized ray directions.

        Returns unit vectors pointing from camera origin through each pixel.

        Args:
            point_2d: 2D pixel coordinate(s) (2,) or (N, 2).

        Returns:
            np.ndarray: Unit ray direction(s) (3,) or (N, 3).

        Note:
            Ray = normalize([(u-cx)/fx, (v-cy)/fy, 1])
        """
        point_2d = np.atleast_2d(point_2d)

        # Compute ray direction (at unit depth)
        x = (point_2d[:, 0] - self.cx) / self.fx
        y = (point_2d[:, 1] - self.cy) / self.fy
        z = np.ones(len(point_2d))

        rays = np.stack([x, y, z], axis=1)

        # Normalize to unit vectors
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        rays = rays / norms

        return rays.squeeze()

    def is_in_image(
        self,
        points_2d: np.ndarray,
        margin: int = 0,
    ) -> np.ndarray:
        """
        Check if 2D points are within image bounds.

        Args:
            points_2d: 2D points (N, 2) in pixel coordinates.
            margin: Additional margin from image border (pixels).

        Returns:
            np.ndarray: Boolean mask (N,) indicating valid points.

        Example:
            >>> intrinsics = CameraIntrinsics(fx=721.5, fy=721.5, cx=609.5, cy=172.8,
            ...                                width=1242, height=375)
            >>> points = np.array([[100, 200], [1500, 100], [600, 180]])
            >>> mask = intrinsics.is_in_image(points)
            >>> print(mask)  # [True, False, True]
        """
        points_2d = np.atleast_2d(points_2d)

        valid = (
            (points_2d[:, 0] >= margin) &
            (points_2d[:, 0] < self.width - margin) &
            (points_2d[:, 1] >= margin) &
            (points_2d[:, 1] < self.height - margin)
        )

        return valid

    def get_pixel_size_at_depth(self, depth: float) -> Tuple[float, float]:
        """
        Calculate the physical size of a pixel at a given depth.

        At depth Z, the physical width and height covered by a single pixel:
            pixel_width = Z / fx
            pixel_height = Z / fy

        Args:
            depth: Distance from camera in the same units as desired output.

        Returns:
            Tuple[float, float]: (pixel_width, pixel_height) at the given depth.

        Note:
            Useful for understanding detection resolution at different distances.
            For example, at 50m with fx=720, a pixel covers ~7cm horizontally.
        """
        pixel_width = depth / self.fx
        pixel_height = depth / self.fy
        return pixel_width, pixel_height

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CameraIntrinsics(fx={self.fx:.2f}, fy={self.fy:.2f}, "
            f"cx={self.cx:.2f}, cy={self.cy:.2f}, "
            f"width={self.width}, height={self.height})"
        )
