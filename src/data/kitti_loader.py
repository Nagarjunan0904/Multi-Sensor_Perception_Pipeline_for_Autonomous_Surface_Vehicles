"""
KITTI 3D Object Detection Dataset Loader.

KITTI File Formats:
====================

1. Images (training/image_2/XXXXXX.png):
   - Left color camera images
   - PNG format, typically 1242x375 pixels
   - RGB color, 8-bit per channel

2. Point Clouds (training/velodyne/XXXXXX.bin):
   - Velodyne HDL-64E LiDAR scans
   - Binary format: sequence of float32 values
   - Each point: [x, y, z, intensity] (4 x float32 = 16 bytes)
   - x: forward, y: left, z: up (LiDAR coordinate system)
   - Intensity: reflectance value [0, 1]
   - Typical: 100,000+ points per frame

3. Calibration (training/calib/XXXXXX.txt):
   - P0, P1, P2, P3: 3x4 projection matrices for cameras 0-3
   - R0_rect: 3x3 rectification matrix
   - Tr_velo_to_cam: 3x4 Velodyne to camera transformation
   - Tr_imu_to_velo: 3x4 IMU to Velodyne transformation

   To project LiDAR point X_velo to image:
   X_cam = R0_rect @ Tr_velo_to_cam @ X_velo
   x_img = P2 @ X_cam

4. Labels (training/label_2/XXXXXX.txt):
   Each line: type truncated occluded alpha bbox3d dimensions location rotation
   - type: 'Car', 'Pedestrian', 'Cyclist', etc.
   - truncated: float [0, 1], 0 = fully visible
   - occluded: int {0, 1, 2, 3}, 0 = fully visible
   - alpha: observation angle [-pi, pi]
   - bbox: 2D box [left, top, right, bottom] in pixels
   - dimensions: 3D box [height, width, length] in meters
   - location: 3D center [x, y, z] in camera coordinates
   - rotation_y: rotation around Y-axis in camera coords [-pi, pi]

Coordinate Systems:
===================
- Camera: x=right, y=down, z=forward
- Velodyne: x=forward, y=left, z=up
- Image: u=right, v=down (origin at top-left)
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np


class Object3D:
    """Represents a 3D object annotation from KITTI labels."""

    def __init__(self, label_line: str):
        """
        Parse a single line from KITTI label file.

        Args:
            label_line: Single line from label_2/*.txt file.
        """
        parts = label_line.strip().split()

        self.type = parts[0]
        self.truncation = float(parts[1])
        self.occlusion = int(parts[2])
        self.alpha = float(parts[3])

        # 2D bounding box in image coordinates
        self.bbox2d = np.array([
            float(parts[4]),  # left
            float(parts[5]),  # top
            float(parts[6]),  # right
            float(parts[7]),  # bottom
        ])

        # 3D dimensions: height, width, length (in meters)
        self.h = float(parts[8])
        self.w = float(parts[9])
        self.l = float(parts[10])

        # 3D location in camera coordinates (center of bottom face)
        self.location = np.array([
            float(parts[11]),  # x
            float(parts[12]),  # y
            float(parts[13]),  # z
        ])

        # Rotation around Y-axis in camera coordinates
        self.rotation_y = float(parts[14])

        # Score (for detection results, not in ground truth)
        self.score = float(parts[15]) if len(parts) > 15 else 1.0

    @property
    def dimensions(self) -> np.ndarray:
        """Return dimensions as [h, w, l] array."""
        return np.array([self.h, self.w, self.l])

    def get_3d_box_corners(self) -> np.ndarray:
        """
        Compute 8 corners of 3D bounding box in camera coordinates.

        Returns:
            corners: (8, 3) array of corner coordinates.

        Corner order:
            4 -------- 5
           /|         /|
          7 -------- 6 .
          | |        | |
          . 0 -------- 1
          |/         |/
          3 -------- 2

        Where 0-3 are bottom face, 4-7 are top face.
        """
        # 3D box in object coordinate system (centered at origin)
        l, w, h = self.l, self.w, self.h
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]  # y points down in camera coords
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        corners = np.array([x_corners, y_corners, z_corners])  # (3, 8)

        # Rotation matrix around Y-axis
        ry = self.rotation_y
        R = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)],
        ])

        # Rotate and translate
        corners = R @ corners  # (3, 8)
        corners = corners.T + self.location  # (8, 3)

        return corners

    def __repr__(self) -> str:
        return (
            f"Object3D(type={self.type}, "
            f"loc=[{self.location[0]:.1f}, {self.location[1]:.1f}, {self.location[2]:.1f}], "
            f"dim=[{self.h:.1f}, {self.w:.1f}, {self.l:.1f}])"
        )


class Calibration:
    """KITTI calibration data handler."""

    def __init__(self, calib_path: Union[str, Path]):
        """
        Load calibration from file.

        Args:
            calib_path: Path to calibration file (calib/XXXXXX.txt).
        """
        self.calib_path = Path(calib_path)
        self._load_calib()

    def _load_calib(self):
        """Parse calibration file."""
        calib_data = {}

        with open(self.calib_path, "r") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                calib_data[key.strip()] = np.array(
                    [float(x) for x in value.strip().split()]
                )

        # Camera projection matrices (3x4)
        self.P0 = calib_data.get("P0", np.zeros(12)).reshape(3, 4)
        self.P1 = calib_data.get("P1", np.zeros(12)).reshape(3, 4)
        self.P2 = calib_data.get("P2", np.zeros(12)).reshape(3, 4)  # Left color camera
        self.P3 = calib_data.get("P3", np.zeros(12)).reshape(3, 4)

        # Rectification matrix (3x3)
        R0_rect = calib_data.get("R0_rect", np.eye(3).flatten())
        self.R0_rect = R0_rect.reshape(3, 3)

        # Velodyne to camera transformation (3x4)
        Tr_velo_to_cam = calib_data.get("Tr_velo_to_cam", np.zeros(12))
        self.Tr_velo_to_cam = Tr_velo_to_cam.reshape(3, 4)

        # IMU to Velodyne transformation (3x4)
        Tr_imu_to_velo = calib_data.get("Tr_imu_to_velo", np.zeros(12))
        self.Tr_imu_to_velo = Tr_imu_to_velo.reshape(3, 4)

        # Precompute useful matrices
        self._compute_transforms()

    def _compute_transforms(self):
        """Precompute transformation matrices."""
        # 4x4 versions for homogeneous coordinates
        self.Tr_velo_to_cam_4x4 = np.eye(4)
        self.Tr_velo_to_cam_4x4[:3, :] = self.Tr_velo_to_cam

        self.R0_rect_4x4 = np.eye(4)
        self.R0_rect_4x4[:3, :3] = self.R0_rect

        # Combined: velo -> rect_cam
        self.Tr_velo_to_rect = self.R0_rect_4x4 @ self.Tr_velo_to_cam_4x4

    def project_velo_to_image(self, points: np.ndarray) -> np.ndarray:
        """
        Project Velodyne points to image plane.

        Args:
            points: (N, 3) or (N, 4) Velodyne points [x, y, z, (intensity)].

        Returns:
            pts_2d: (N, 2) image coordinates [u, v].
        """
        pts_3d = points[:, :3]
        n_points = pts_3d.shape[0]

        # Convert to homogeneous coordinates
        pts_3d_hom = np.hstack([pts_3d, np.ones((n_points, 1))])  # (N, 4)

        # Velodyne -> rectified camera
        pts_rect = (self.Tr_velo_to_rect @ pts_3d_hom.T).T  # (N, 4)

        # Project to image using P2
        pts_2d_hom = (self.P2 @ pts_rect.T).T  # (N, 3)

        # Normalize by depth
        pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]

        return pts_2d

    def project_velo_to_rect(self, points: np.ndarray) -> np.ndarray:
        """
        Transform Velodyne points to rectified camera coordinates.

        Args:
            points: (N, 3) or (N, 4) Velodyne points.

        Returns:
            pts_rect: (N, 3) points in rectified camera coordinates.
        """
        pts_3d = points[:, :3]
        n_points = pts_3d.shape[0]

        pts_3d_hom = np.hstack([pts_3d, np.ones((n_points, 1))])
        pts_rect = (self.Tr_velo_to_rect @ pts_3d_hom.T).T

        return pts_rect[:, :3]

    def get_fov_mask(
        self,
        points: np.ndarray,
        image_shape: tuple,
    ) -> np.ndarray:
        """
        Get mask for points within camera field of view.

        Args:
            points: (N, 3) or (N, 4) Velodyne points.
            image_shape: (height, width) of image.

        Returns:
            mask: (N,) boolean mask.
        """
        pts_rect = self.project_velo_to_rect(points)
        pts_2d = self.project_velo_to_image(points)

        h, w = image_shape[:2]

        # Points must be in front of camera and within image bounds
        mask = (
            (pts_rect[:, 2] > 0) &  # positive depth
            (pts_2d[:, 0] >= 0) &
            (pts_2d[:, 0] < w) &
            (pts_2d[:, 1] >= 0) &
            (pts_2d[:, 1] < h)
        )

        return mask

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Return calibration as dictionary."""
        return {
            "P0": self.P0,
            "P1": self.P1,
            "P2": self.P2,
            "P3": self.P3,
            "R0_rect": self.R0_rect,
            "Tr_velo_to_cam": self.Tr_velo_to_cam,
            "Tr_imu_to_velo": self.Tr_imu_to_velo,
        }


class KITTILoader:
    """
    KITTI 3D Object Detection Dataset Loader.

    Usage:
        loader = KITTILoader("data/KITTI")
        sample = loader[0]
        # sample['image']: (H, W, 3) uint8 numpy array
        # sample['points']: (N, 4) float32 numpy array
        # sample['calib']: Calibration object
        # sample['objects']: list of Object3D (if labels exist)

        for sample in loader:
            process(sample)
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "training",
        load_labels: bool = True,
    ):
        """
        Initialize KITTI loader.

        Args:
            root_dir: Path to KITTI dataset root (contains training/, testing/).
            split: Dataset split ('training' or 'testing').
            load_labels: Whether to load label files (only for training).
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.load_labels = load_labels and (split == "training")

        # Setup paths
        self.split_dir = self.root_dir / split
        self.image_dir = self.split_dir / "image_2"
        self.velodyne_dir = self.split_dir / "velodyne"
        self.calib_dir = self.split_dir / "calib"
        self.label_dir = self.split_dir / "label_2"

        # Get list of frames
        self._load_frame_list()

    def _load_frame_list(self):
        """Load list of available frame indices."""
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}\n"
                f"Run 'python scripts/download_kitti.py' to download the dataset."
            )

        # Get frame indices from image files
        image_files = sorted(self.image_dir.glob("*.png"))
        self.frame_ids = [f.stem for f in image_files]

        if len(self.frame_ids) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        """Return number of frames in dataset."""
        return len(self.frame_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load a single frame.

        Args:
            idx: Frame index (0 to len-1) or frame ID string.

        Returns:
            Dictionary containing:
                - 'frame_id': str, frame identifier (e.g., '000000')
                - 'image': (H, W, 3) uint8 numpy array (RGB)
                - 'points': (N, 4) float32 numpy array [x, y, z, intensity]
                - 'calib': Calibration object
                - 'objects': list of Object3D (if labels loaded)
        """
        if isinstance(idx, str):
            frame_id = idx
        else:
            frame_id = self.frame_ids[idx]

        sample = {
            "frame_id": frame_id,
            "image": self.load_image(frame_id),
            "points": self.load_velodyne(frame_id),
            "calib": self.load_calib(frame_id),
        }

        if self.load_labels:
            sample["objects"] = self.load_labels_for_frame(frame_id)

        return sample

    def __iter__(self):
        """Iterate over all frames."""
        for idx in range(len(self)):
            yield self[idx]

    def get_image_path(self, frame_id: str) -> Path:
        """Get path to image file for given frame."""
        return self.image_dir / f"{frame_id}.png"

    def get_velodyne_path(self, frame_id: str) -> Path:
        """Get path to velodyne point cloud for given frame."""
        return self.velodyne_dir / f"{frame_id}.bin"

    def get_calib_path(self, frame_id: str) -> Path:
        """Get path to calibration file for given frame."""
        return self.calib_dir / f"{frame_id}.txt"

    def get_label_path(self, frame_id: str) -> Path:
        """Get path to label file for given frame."""
        return self.label_dir / f"{frame_id}.txt"

    def load_image(self, frame_id: str) -> np.ndarray:
        """
        Load image for given frame.

        Args:
            frame_id: Frame identifier (e.g., '000000').

        Returns:
            image: (H, W, 3) uint8 numpy array in RGB format.
        """
        image_path = self.get_image_path(frame_id)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # OpenCV loads as BGR, convert to RGB
        image = cv2.imread(str(image_path))
        if image is None:
            raise IOError(f"Failed to load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_velodyne(self, frame_id: str) -> np.ndarray:
        """
        Load Velodyne point cloud for given frame.

        Args:
            frame_id: Frame identifier (e.g., '000000').

        Returns:
            points: (N, 4) float32 numpy array [x, y, z, intensity].

        File format:
            Binary file containing N points as float32 values.
            Each point: x, y, z, intensity (4 x 4 bytes = 16 bytes per point).
        """
        velodyne_path = self.get_velodyne_path(frame_id)
        if not velodyne_path.exists():
            raise FileNotFoundError(f"Point cloud not found: {velodyne_path}")

        # Load binary file (float32, 4 values per point)
        points = np.fromfile(str(velodyne_path), dtype=np.float32)
        points = points.reshape(-1, 4)

        return points

    def load_calib(self, frame_id: str) -> Calibration:
        """
        Load calibration for given frame.

        Args:
            frame_id: Frame identifier (e.g., '000000').

        Returns:
            calib: Calibration object with projection matrices.
        """
        calib_path = self.get_calib_path(frame_id)
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration not found: {calib_path}")

        return Calibration(calib_path)

    def load_labels_for_frame(self, frame_id: str) -> List[Object3D]:
        """
        Load 3D object labels for given frame.

        Args:
            frame_id: Frame identifier (e.g., '000000').

        Returns:
            objects: List of Object3D instances.
        """
        label_path = self.get_label_path(frame_id)
        if not label_path.exists():
            return []

        objects = []
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                obj = Object3D(line)
                objects.append(obj)

        return objects

    def get_frame_by_id(self, frame_id: str) -> Dict[str, Any]:
        """
        Load frame by string ID.

        Args:
            frame_id: Frame identifier (e.g., '000000').

        Returns:
            Sample dictionary.
        """
        return self[frame_id]

    def get_points_in_fov(
        self,
        frame_id: str,
        return_indices: bool = False,
    ) -> Union[np.ndarray, tuple]:
        """
        Get point cloud filtered to camera field of view.

        Args:
            frame_id: Frame identifier.
            return_indices: If True, also return indices of FOV points.

        Returns:
            points_fov: (M, 4) points within camera FOV.
            indices: (optional) indices of FOV points in original array.
        """
        points = self.load_velodyne(frame_id)
        image = self.load_image(frame_id)
        calib = self.load_calib(frame_id)

        mask = calib.get_fov_mask(points, image.shape)
        points_fov = points[mask]

        if return_indices:
            indices = np.where(mask)[0]
            return points_fov, indices

        return points_fov

    def statistics(self) -> Dict[str, Any]:
        """
        Compute dataset statistics.

        Returns:
            Dictionary with dataset statistics.
        """
        stats = {
            "split": self.split,
            "num_frames": len(self),
            "has_labels": self.load_labels,
        }

        if len(self) > 0:
            # Sample first frame for typical values
            sample = self[0]
            stats["image_shape"] = sample["image"].shape
            stats["num_points_sample"] = sample["points"].shape[0]

            if self.load_labels and "objects" in sample:
                # Count object types
                object_types = {}
                for obj in sample["objects"]:
                    object_types[obj.type] = object_types.get(obj.type, 0) + 1
                stats["objects_sample"] = object_types

        return stats
