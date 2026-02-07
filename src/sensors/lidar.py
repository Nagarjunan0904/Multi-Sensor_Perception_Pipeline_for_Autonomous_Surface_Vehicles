"""LiDAR point cloud loader for Velodyne data."""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


class LiDARLoader:
    """Load and preprocess LiDAR point clouds from KITTI dataset."""

    def __init__(
        self,
        data_root: str,
        split: str = "training",
        lidar_dir: str = "velodyne",
        range_min: float = 0.5,
        range_max: float = 120.0,
    ):
        """
        Initialize the LiDAR loader.

        Args:
            data_root: Root directory of the KITTI dataset.
            split: Dataset split ('training' or 'testing').
            lidar_dir: Subdirectory containing point clouds.
            range_min: Minimum range filter (meters).
            range_max: Maximum range filter (meters).
        """
        self.data_root = Path(data_root)
        self.split = split
        self.lidar_dir = lidar_dir
        self.range_min = range_min
        self.range_max = range_max

        self.lidar_path = self.data_root / split / lidar_dir
        self._validate_path()
        self._index_pointclouds()

    def _validate_path(self) -> None:
        """Validate that the LiDAR directory exists."""
        if not self.lidar_path.exists():
            raise FileNotFoundError(f"LiDAR directory not found: {self.lidar_path}")

    def _index_pointclouds(self) -> None:
        """Index all available point cloud files."""
        self.lidar_files = sorted(self.lidar_path.glob("*.bin"))

    def __len__(self) -> int:
        """Return the number of available point clouds."""
        return len(self.lidar_files)

    def __getitem__(self, index: int) -> np.ndarray:
        """Load point cloud by index."""
        return self.load_pointcloud(index)

    def load_pointcloud(
        self,
        index: Union[int, str],
        remove_invalid: bool = True,
    ) -> np.ndarray:
        """
        Load a single point cloud.

        Args:
            index: Point cloud index (int) or filename (str).
            remove_invalid: Whether to remove points with invalid reflectance.

        Returns:
            Point cloud as numpy array (N, 4) with [x, y, z, intensity].
        """
        if isinstance(index, int):
            if index < 0 or index >= len(self.lidar_files):
                raise IndexError(f"Point cloud index {index} out of range [0, {len(self) - 1}]")
            lidar_path = self.lidar_files[index]
        else:
            lidar_path = self.lidar_path / index
            if not lidar_path.exists():
                raise FileNotFoundError(f"Point cloud not found: {lidar_path}")

        # Load binary point cloud (x, y, z, intensity)
        points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)

        if remove_invalid:
            # Remove points with zero intensity (usually invalid)
            valid_mask = points[:, 3] >= 0
            points = points[valid_mask]

        return points

    def filter_by_range(
        self,
        points: np.ndarray,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
    ) -> np.ndarray:
        """
        Filter points by distance from sensor.

        Args:
            points: Point cloud (N, 4).
            range_min: Minimum range (uses instance default if None).
            range_max: Maximum range (uses instance default if None).

        Returns:
            Filtered point cloud.
        """
        range_min = range_min or self.range_min
        range_max = range_max or self.range_max

        distances = np.linalg.norm(points[:, :3], axis=1)
        mask = (distances >= range_min) & (distances <= range_max)

        return points[mask]

    def filter_by_fov(
        self,
        points: np.ndarray,
        fov_left: float = 45.0,
        fov_right: float = 45.0,
    ) -> np.ndarray:
        """
        Filter points by horizontal field of view.

        Args:
            points: Point cloud (N, 4).
            fov_left: Left FOV angle in degrees.
            fov_right: Right FOV angle in degrees.

        Returns:
            Filtered point cloud within FOV.
        """
        angles = np.arctan2(points[:, 1], points[:, 0]) * 180 / np.pi
        mask = (angles >= -fov_right) & (angles <= fov_left)

        return points[mask]

    def filter_by_height(
        self,
        points: np.ndarray,
        z_min: float = -3.0,
        z_max: float = 2.0,
    ) -> np.ndarray:
        """
        Filter points by height (z-coordinate).

        Args:
            points: Point cloud (N, 4).
            z_min: Minimum height.
            z_max: Maximum height.

        Returns:
            Filtered point cloud.
        """
        mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        return points[mask]

    def remove_ground(
        self,
        points: np.ndarray,
        ground_threshold: float = -1.5,
    ) -> np.ndarray:
        """
        Remove ground points using simple height threshold.

        Args:
            points: Point cloud (N, 4).
            ground_threshold: Height below which points are considered ground.

        Returns:
            Point cloud with ground removed.
        """
        mask = points[:, 2] > ground_threshold
        return points[mask]

    def voxel_downsample(
        self,
        points: np.ndarray,
        voxel_size: float = 0.1,
    ) -> np.ndarray:
        """
        Downsample point cloud using voxel grid.

        Args:
            points: Point cloud (N, 4).
            voxel_size: Voxel size in meters.

        Returns:
            Downsampled point cloud.
        """
        # Compute voxel indices
        voxel_indices = np.floor(points[:, :3] / voxel_size).astype(np.int32)

        # Get unique voxels and their first occurrence indices
        _, unique_indices = np.unique(
            voxel_indices, axis=0, return_index=True
        )

        return points[unique_indices]

    def get_frame_id(self, index: int) -> str:
        """
        Get frame ID for given index.

        Args:
            index: Point cloud index.

        Returns:
            Frame ID string (e.g., '000000').
        """
        return self.lidar_files[index].stem

    def get_statistics(self, points: np.ndarray) -> dict:
        """
        Compute point cloud statistics.

        Args:
            points: Point cloud (N, 4).

        Returns:
            Dictionary with statistics.
        """
        distances = np.linalg.norm(points[:, :3], axis=1)

        return {
            "num_points": len(points),
            "x_range": (points[:, 0].min(), points[:, 0].max()),
            "y_range": (points[:, 1].min(), points[:, 1].max()),
            "z_range": (points[:, 2].min(), points[:, 2].max()),
            "distance_range": (distances.min(), distances.max()),
            "mean_intensity": points[:, 3].mean(),
        }
