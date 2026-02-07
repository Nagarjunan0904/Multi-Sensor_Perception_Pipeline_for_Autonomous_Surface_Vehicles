"""Tests for camera module."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestCameraLoader:
    """Tests for CameraLoader class."""

    def test_init_with_missing_path(self):
        """Test initialization with missing path raises error."""
        from src.sensors.camera import CameraLoader

        with pytest.raises(FileNotFoundError):
            CameraLoader(data_root="/nonexistent/path")

    @patch("src.sensors.camera.Path.exists")
    @patch("src.sensors.camera.Path.glob")
    def test_index_images(self, mock_glob, mock_exists):
        """Test image indexing."""
        from src.sensors.camera import CameraLoader

        mock_exists.return_value = True
        mock_glob.return_value = [
            Path("000000.png"),
            Path("000001.png"),
            Path("000002.png"),
        ]

        with patch.object(CameraLoader, "_validate_path"):
            loader = CameraLoader.__new__(CameraLoader)
            loader.image_path = Path("/fake/path")
            loader._index_images()

        assert len(loader.image_files) == 3

    def test_preprocess_normalizes_image(self):
        """Test image preprocessing with normalization."""
        from src.sensors.camera import CameraLoader

        # Create mock loader
        loader = CameraLoader.__new__(CameraLoader)
        loader.normalize = True
        loader.mean = np.array([0.5, 0.5, 0.5])
        loader.std = np.array([0.25, 0.25, 0.25])

        # Test image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128

        result = loader.preprocess(image)

        assert result.dtype == np.float32
        # (0.5 - 0.5) / 0.25 = 0.0 approximately
        assert np.allclose(result, 0.0, atol=0.1)

    def test_preprocess_resizes_image(self):
        """Test image preprocessing with resize."""
        from src.sensors.camera import CameraLoader

        loader = CameraLoader.__new__(CameraLoader)
        loader.normalize = False
        loader.mean = np.array([0.0, 0.0, 0.0])
        loader.std = np.array([1.0, 1.0, 1.0])

        image = np.ones((100, 200, 3), dtype=np.uint8)

        result = loader.preprocess(image, target_size=(50, 25))

        assert result.shape == (25, 50, 3)


class TestCameraIntegration:
    """Integration tests for camera module (require actual data)."""

    @pytest.mark.skipif(
        not Path("data/KITTI/training/image_2").exists(),
        reason="KITTI data not available"
    )
    def test_load_real_image(self):
        """Test loading real KITTI image."""
        from src.sensors.camera import CameraLoader

        loader = CameraLoader(data_root="data/KITTI", split="training")

        assert len(loader) > 0

        image = loader.load_image(0)
        assert image is not None
        assert len(image.shape) == 3
        assert image.shape[2] == 3
