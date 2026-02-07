"""
Calibration modules for camera-LiDAR geometry.

This package provides tools for handling camera calibration, coordinate
transformations, and 3D-2D projections for multi-sensor perception.

Classes:
    CameraIntrinsics: Camera intrinsic parameters (focal length, principal point).
    CameraExtrinsics: Camera extrinsic parameters (rotation, translation).
    CameraLiDARExtrinsics: Full KITTI calibration handling.
    Projector: High-level projection utilities.

Standalone Functions:
    project_3d_to_2d: Project 3D points to 2D pixels.
    backproject_2d_to_3d: Back-project 2D pixels to 3D using depth.
    transform_points: Apply rigid body transformation.
    filter_fov: Filter points to camera field of view.
    compute_frustum_points: Compute camera frustum corners.

Example Usage:
    >>> from calibration import CameraIntrinsics, CameraLiDARExtrinsics, Projector
    >>> from calibration import project_3d_to_2d, backproject_2d_to_3d
    >>>
    >>> # Load calibration
    >>> intrinsics = CameraIntrinsics.from_kitti_calib(calib_dict)
    >>> extrinsics = CameraLiDARExtrinsics("path/to/calib.txt")
    >>>
    >>> # Create projector for convenient operations
    >>> projector = Projector(intrinsics, extrinsics)
    >>> points_2d, depths, mask = projector.project_lidar_to_image(lidar_points)
"""

from .intrinsics import CameraIntrinsics
from .extrinsics import CameraExtrinsics, CameraLiDARExtrinsics
from .projection import (
    Projector,
    project_3d_to_2d,
    backproject_2d_to_3d,
    transform_points,
    filter_fov,
    compute_frustum_points,
)

__all__ = [
    # Classes
    "CameraIntrinsics",
    "CameraExtrinsics",
    "CameraLiDARExtrinsics",
    "Projector",
    # Standalone functions
    "project_3d_to_2d",
    "backproject_2d_to_3d",
    "transform_points",
    "filter_fov",
    "compute_frustum_points",
]
