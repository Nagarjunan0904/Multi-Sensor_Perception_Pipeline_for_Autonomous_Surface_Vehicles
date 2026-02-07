"""Tests for KITTI data loader."""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.kitti_loader import KITTILoader, Calibration, Object3D


# Default data directory - override with pytest --data-dir=path
DATA_DIR = Path("data/KITTI")


@pytest.fixture
def data_dir(request):
    """Get data directory from command line or use default."""
    return Path(request.config.getoption("--data-dir", default=str(DATA_DIR)))


@pytest.fixture
def loader(data_dir):
    """Create KITTILoader instance."""
    if not data_dir.exists():
        pytest.skip(f"KITTI data not found at {data_dir}")
    try:
        return KITTILoader(data_dir, split="training")
    except FileNotFoundError:
        pytest.skip(f"KITTI training data not found at {data_dir}")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--data-dir",
        action="store",
        default=str(DATA_DIR),
        help="Path to KITTI dataset directory",
    )


class TestKITTILoader:
    """Tests for KITTILoader class."""

    def test_loader_initialization(self, loader):
        """Test loader initializes correctly."""
        assert loader is not None
        assert len(loader) > 0
        assert loader.split == "training"

    def test_len(self, loader):
        """Test __len__ returns correct count."""
        num_frames = len(loader)
        assert isinstance(num_frames, int)
        assert num_frames > 0
        # KITTI training set has 7481 frames (or less in quick mode)
        assert num_frames <= 7481

    def test_getitem(self, loader):
        """Test __getitem__ returns correct structure."""
        sample = loader[0]

        assert "frame_id" in sample
        assert "image" in sample
        assert "points" in sample
        assert "calib" in sample
        assert "objects" in sample

        assert isinstance(sample["frame_id"], str)
        assert len(sample["frame_id"]) == 6  # e.g., '000000'

    def test_load_image(self, loader):
        """Test image loading."""
        sample = loader[0]
        image = sample["image"]

        assert isinstance(image, np.ndarray)
        assert image.ndim == 3
        assert image.shape[2] == 3  # RGB
        assert image.dtype == np.uint8
        # KITTI images are typically ~1242x375
        assert image.shape[0] > 300  # height
        assert image.shape[1] > 1000  # width

    def test_load_point_cloud(self, loader):
        """Test point cloud loading."""
        sample = loader[0]
        points = sample["points"]

        assert isinstance(points, np.ndarray)
        assert points.ndim == 2
        assert points.shape[1] == 4  # x, y, z, intensity
        assert points.dtype == np.float32
        # Typical KITTI frame has 100k+ points
        assert points.shape[0] > 1000

    def test_point_cloud_values(self, loader):
        """Test point cloud values are reasonable."""
        sample = loader[0]
        points = sample["points"]

        # x, y, z should be within reasonable range (meters)
        # KITTI Velodyne range is ~120m
        assert np.abs(points[:, :3]).max() < 200

        # Intensity should be in [0, 1] range (normalized)
        # Some KITTI files have intensity in [0, 255]
        assert points[:, 3].min() >= 0
        assert points[:, 3].max() <= 255

    def test_load_calibration(self, loader):
        """Test calibration loading."""
        sample = loader[0]
        calib = sample["calib"]

        assert isinstance(calib, Calibration)
        assert hasattr(calib, "P2")
        assert hasattr(calib, "R0_rect")
        assert hasattr(calib, "Tr_velo_to_cam")


class TestCalibration:
    """Tests for Calibration class."""

    def test_calibration_shapes(self, loader):
        """Test calibration matrix shapes."""
        sample = loader[0]
        calib = sample["calib"]

        assert calib.P0.shape == (3, 4)
        assert calib.P1.shape == (3, 4)
        assert calib.P2.shape == (3, 4)
        assert calib.P3.shape == (3, 4)
        assert calib.R0_rect.shape == (3, 3)
        assert calib.Tr_velo_to_cam.shape == (3, 4)

    def test_calibration_to_dict(self, loader):
        """Test calibration to_dict method."""
        sample = loader[0]
        calib_dict = sample["calib"].to_dict()

        assert "P2" in calib_dict
        assert "R0_rect" in calib_dict
        assert "Tr_velo_to_cam" in calib_dict
        assert isinstance(calib_dict["P2"], np.ndarray)

    def test_projection_output_shape(self, loader):
        """Test point projection output shape."""
        sample = loader[0]
        points = sample["points"][:100]  # Use subset for speed
        calib = sample["calib"]

        pts_2d = calib.project_velo_to_image(points)

        assert pts_2d.shape == (100, 2)
        assert pts_2d.dtype in [np.float32, np.float64]

    def test_fov_mask(self, loader):
        """Test FOV mask generation."""
        sample = loader[0]
        points = sample["points"]
        image = sample["image"]
        calib = sample["calib"]

        mask = calib.get_fov_mask(points, image.shape)

        assert mask.shape == (len(points),)
        assert mask.dtype == bool
        # Some points should be in FOV, some should not
        assert mask.sum() > 0
        assert mask.sum() < len(points)


class TestObject3D:
    """Tests for Object3D class."""

    def test_object_parsing(self):
        """Test Object3D parses label line correctly."""
        # Example KITTI label line
        line = "Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57"
        obj = Object3D(line)

        assert obj.type == "Car"
        assert obj.truncation == 0.0
        assert obj.occlusion == 0
        assert abs(obj.alpha - 1.85) < 0.01
        assert obj.bbox2d.shape == (4,)
        assert abs(obj.h - 1.67) < 0.01
        assert abs(obj.w - 1.87) < 0.01
        assert abs(obj.l - 3.69) < 0.01
        assert obj.location.shape == (3,)
        assert abs(obj.rotation_y - 1.57) < 0.01

    def test_object_3d_corners(self):
        """Test 3D box corner computation."""
        line = "Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57"
        obj = Object3D(line)

        corners = obj.get_3d_box_corners()

        assert corners.shape == (8, 3)
        # Corners should be near the object location
        center = corners.mean(axis=0)
        assert np.linalg.norm(center - obj.location) < obj.l

    def test_load_objects(self, loader):
        """Test loading objects from label file."""
        sample = loader[0]

        if "objects" in sample:
            objects = sample["objects"]
            assert isinstance(objects, list)

            if len(objects) > 0:
                obj = objects[0]
                assert isinstance(obj, Object3D)
                assert hasattr(obj, "type")
                assert hasattr(obj, "location")


class TestIterator:
    """Tests for dataset iteration."""

    def test_iteration(self, loader):
        """Test iterating over dataset."""
        count = 0
        for sample in loader:
            assert "image" in sample
            count += 1
            if count >= 3:  # Only test first 3
                break

        assert count == 3

    def test_index_access(self, loader):
        """Test accessing by index."""
        sample0 = loader[0]
        sample1 = loader[1]

        # Different frames should have different IDs
        assert sample0["frame_id"] != sample1["frame_id"]


class TestPathHelpers:
    """Tests for path helper methods."""

    def test_get_image_path(self, loader):
        """Test get_image_path returns valid path."""
        frame_id = loader.frame_ids[0]
        path = loader.get_image_path(frame_id)

        assert isinstance(path, Path)
        assert path.suffix == ".png"
        assert path.exists()

    def test_get_velodyne_path(self, loader):
        """Test get_velodyne_path returns valid path."""
        frame_id = loader.frame_ids[0]
        path = loader.get_velodyne_path(frame_id)

        assert isinstance(path, Path)
        assert path.suffix == ".bin"
        # May not exist if only images were downloaded

    def test_get_calib_path(self, loader):
        """Test get_calib_path returns valid path."""
        frame_id = loader.frame_ids[0]
        path = loader.get_calib_path(frame_id)

        assert isinstance(path, Path)
        assert path.suffix == ".txt"


class TestStatistics:
    """Tests for statistics method."""

    def test_statistics(self, loader):
        """Test statistics computation."""
        stats = loader.statistics()

        assert "split" in stats
        assert "num_frames" in stats
        assert "has_labels" in stats

        assert stats["split"] == "training"
        assert stats["num_frames"] > 0
        assert stats["has_labels"] is True


# Run with: pytest tests/test_kitti_loader.py -v
# With custom data dir: pytest tests/test_kitti_loader.py -v --data-dir=/path/to/KITTI
