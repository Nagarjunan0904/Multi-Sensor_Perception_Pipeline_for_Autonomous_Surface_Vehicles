"""Sensor synchronization for camera and LiDAR data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .camera import CameraLoader
from .lidar import LiDARLoader


@dataclass
class SynchronizedFrame:
    """Container for synchronized sensor data."""

    frame_id: str
    image: np.ndarray
    pointcloud: np.ndarray
    timestamp: Optional[float] = None


class SensorSynchronizer:
    """Synchronize camera and LiDAR data streams."""

    def __init__(
        self,
        camera_loader: CameraLoader,
        lidar_loader: LiDARLoader,
        max_time_diff: float = 0.05,
    ):
        """
        Initialize the sensor synchronizer.

        Args:
            camera_loader: Camera data loader instance.
            lidar_loader: LiDAR data loader instance.
            max_time_diff: Maximum time difference for synchronization (seconds).
        """
        self.camera = camera_loader
        self.lidar = lidar_loader
        self.max_time_diff = max_time_diff

        self._build_frame_index()

    def _build_frame_index(self) -> None:
        """Build index of synchronized frames based on frame IDs."""
        camera_frames = {self.camera.get_frame_id(i): i for i in range(len(self.camera))}
        lidar_frames = {self.lidar.get_frame_id(i): i for i in range(len(self.lidar))}

        # Find common frames
        common_ids = set(camera_frames.keys()) & set(lidar_frames.keys())
        self.frame_ids = sorted(common_ids)

        self.camera_indices = {fid: camera_frames[fid] for fid in self.frame_ids}
        self.lidar_indices = {fid: lidar_frames[fid] for fid in self.frame_ids}

    def __len__(self) -> int:
        """Return number of synchronized frames."""
        return len(self.frame_ids)

    def __getitem__(self, index: int) -> SynchronizedFrame:
        """Get synchronized frame by index."""
        return self.get_frame(index)

    def get_frame(self, index: int) -> SynchronizedFrame:
        """
        Get synchronized camera and LiDAR data for a frame.

        Args:
            index: Frame index.

        Returns:
            SynchronizedFrame containing image and point cloud.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Frame index {index} out of range [0, {len(self) - 1}]")

        frame_id = self.frame_ids[index]

        image = self.camera.load_image(self.camera_indices[frame_id])
        pointcloud = self.lidar.load_pointcloud(self.lidar_indices[frame_id])

        return SynchronizedFrame(
            frame_id=frame_id,
            image=image,
            pointcloud=pointcloud,
        )

    def get_frame_by_id(self, frame_id: str) -> SynchronizedFrame:
        """
        Get synchronized data by frame ID.

        Args:
            frame_id: Frame identifier (e.g., '000000').

        Returns:
            SynchronizedFrame containing image and point cloud.
        """
        if frame_id not in self.frame_ids:
            raise KeyError(f"Frame ID '{frame_id}' not found")

        index = self.frame_ids.index(frame_id)
        return self.get_frame(index)

    def iterate_frames(
        self,
        start: int = 0,
        end: Optional[int] = None,
        step: int = 1,
    ):
        """
        Iterate over synchronized frames.

        Args:
            start: Starting frame index.
            end: Ending frame index (exclusive).
            step: Step size.

        Yields:
            SynchronizedFrame for each frame.
        """
        end = end or len(self)

        for i in range(start, end, step):
            yield self.get_frame(i)

    def get_frame_ids(self) -> List[str]:
        """Return list of all synchronized frame IDs."""
        return self.frame_ids.copy()

    def get_statistics(self) -> Dict:
        """
        Get synchronization statistics.

        Returns:
            Dictionary with sync statistics.
        """
        return {
            "total_camera_frames": len(self.camera),
            "total_lidar_frames": len(self.lidar),
            "synchronized_frames": len(self),
            "sync_ratio": len(self) / max(len(self.camera), len(self.lidar)),
        }
