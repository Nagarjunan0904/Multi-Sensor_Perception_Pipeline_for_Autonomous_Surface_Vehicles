"""Tests for fusion modules."""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock


class TestDepthEstimator:
    """Tests for DepthEstimator class."""

    def test_median_aggregation(self):
        """Test median depth aggregation."""
        from src.fusion.depth_estimator import DepthEstimator

        estimator = DepthEstimator.__new__(DepthEstimator)
        estimator.method = "median"

        depths = np.array([10.0, 11.0, 12.0, 100.0])  # Outlier at 100
        points_2d = np.zeros((4, 2))  # Dummy points
        box = np.array([0, 0, 100, 100])

        result = estimator._aggregate_depths(depths, points_2d, box)

        assert result == 11.5  # Median of [10, 11, 12, 100]

    def test_mean_aggregation(self):
        """Test mean depth aggregation."""
        from src.fusion.depth_estimator import DepthEstimator

        estimator = DepthEstimator.__new__(DepthEstimator)
        estimator.method = "mean"

        depths = np.array([10.0, 20.0, 30.0])
        points_2d = np.zeros((3, 2))
        box = np.array([0, 0, 100, 100])

        result = estimator._aggregate_depths(depths, points_2d, box)

        assert result == 20.0

    def test_closest_aggregation(self):
        """Test closest depth selection."""
        from src.fusion.depth_estimator import DepthEstimator

        estimator = DepthEstimator.__new__(DepthEstimator)
        estimator.method = "closest"

        depths = np.array([10.0, 5.0, 15.0])
        points_2d = np.zeros((3, 2))
        box = np.array([0, 0, 100, 100])

        result = estimator._aggregate_depths(depths, points_2d, box)

        assert result == 5.0

    def test_box_expansion(self):
        """Test bounding box expansion."""
        from src.fusion.depth_estimator import DepthEstimator

        estimator = DepthEstimator.__new__(DepthEstimator)
        estimator.search_expansion = 0.1  # 10% expansion

        box = np.array([100.0, 100.0, 200.0, 200.0])

        expanded = estimator._expand_box(box)

        # Width and height = 100, expansion = 10
        assert expanded[0] == 90.0   # x1 - 10
        assert expanded[1] == 90.0   # y1 - 10
        assert expanded[2] == 210.0  # x2 + 10
        assert expanded[3] == 210.0  # y2 + 10


class TestBBox3DGenerator:
    """Tests for BBox3DGenerator class."""

    def test_default_dimensions(self):
        """Test default dimension lookup."""
        from src.fusion.bbox3d_generator import BBox3DGenerator

        generator = BBox3DGenerator.__new__(BBox3DGenerator)
        generator.use_prior_dimensions = True
        generator.default_dimensions = {
            "Car": np.array([3.88, 1.63, 1.53]),
            "Pedestrian": np.array([0.88, 0.65, 1.77]),
        }

        car_dims = generator._get_dimensions("Car")
        ped_dims = generator._get_dimensions("Pedestrian")

        assert np.allclose(car_dims, [3.88, 1.63, 1.53])
        assert np.allclose(ped_dims, [0.88, 0.65, 1.77])

    def test_3d_box_corners(self):
        """Test 3D box corner calculation."""
        from src.fusion.bbox3d_generator import BBox3D
        from src.perception2d.detector import Detection

        det = Detection(
            box=np.array([0, 0, 100, 100]),
            score=0.9,
            class_id=2,
            class_name="Car",
        )

        box = BBox3D(
            center=np.array([0.0, 0.0, 10.0]),
            dimensions=np.array([4.0, 2.0, 1.5]),  # L, W, H
            rotation_y=0.0,
            class_name="Car",
            score=0.9,
            detection_2d=det,
        )

        corners = box.corners

        assert corners.shape == (8, 3)

        # Check that corners span the expected dimensions
        x_range = corners[:, 0].max() - corners[:, 0].min()
        z_range = corners[:, 2].max() - corners[:, 2].min()

        assert np.isclose(x_range, 4.0)  # Length
        assert np.isclose(z_range, 2.0)  # Width

    def test_kitti_format_output(self):
        """Test KITTI format string generation."""
        from src.fusion.bbox3d_generator import BBox3D
        from src.perception2d.detector import Detection

        det = Detection(
            box=np.array([100.0, 200.0, 300.0, 400.0]),
            score=0.85,
            class_id=2,
            class_name="Car",
        )

        box = BBox3D(
            center=np.array([1.0, 2.0, 10.0]),
            dimensions=np.array([4.0, 1.5, 1.5]),
            rotation_y=0.5,
            class_name="Car",
            score=0.85,
            detection_2d=det,
        )

        kitti_str = box.to_kitti_format()

        assert "Car" in kitti_str
        assert "100.00" in kitti_str  # bbox x1
        assert "0.85" in kitti_str    # score


class TestOutlierFilter:
    """Tests for OutlierFilter class."""

    def test_depth_filtering(self):
        """Test depth range filtering."""
        from src.fusion.outlier_filter import OutlierFilter
        from src.fusion.bbox3d_generator import BBox3D
        from src.perception2d.detector import Detection

        filter = OutlierFilter(depth_range=(1.0, 50.0))

        det = Detection(
            box=np.array([0, 0, 100, 100]),
            score=0.9,
            class_id=2,
            class_name="Car",
        )

        # Valid box
        box_valid = BBox3D(
            center=np.array([0, 0, 25.0]),
            dimensions=np.array([4.0, 1.5, 1.5]),
            rotation_y=0.0,
            class_name="Car",
            score=0.9,
            detection_2d=det,
        )

        # Invalid box (too far)
        box_far = BBox3D(
            center=np.array([0, 0, 100.0]),
            dimensions=np.array([4.0, 1.5, 1.5]),
            rotation_y=0.0,
            class_name="Car",
            score=0.9,
            detection_2d=det,
        )

        result = filter.filter([box_valid, box_far, None])

        assert result[0] is not None  # Valid kept
        assert result[1] is None      # Far filtered
        assert result[2] is None      # None preserved

    def test_score_filtering(self):
        """Test score threshold filtering."""
        from src.fusion.outlier_filter import OutlierFilter
        from src.fusion.bbox3d_generator import BBox3D
        from src.perception2d.detector import Detection

        filter = OutlierFilter(score_threshold=0.5)

        det_high = Detection(
            box=np.array([0, 0, 100, 100]),
            score=0.8,
            class_id=2,
            class_name="Car",
        )

        det_low = Detection(
            box=np.array([0, 0, 100, 100]),
            score=0.3,
            class_id=2,
            class_name="Car",
        )

        box_high = BBox3D(
            center=np.array([0, 0, 25.0]),
            dimensions=np.array([4.0, 1.5, 1.5]),
            rotation_y=0.0,
            class_name="Car",
            score=0.8,
            detection_2d=det_high,
        )

        box_low = BBox3D(
            center=np.array([0, 0, 25.0]),
            dimensions=np.array([4.0, 1.5, 1.5]),
            rotation_y=0.0,
            class_name="Car",
            score=0.3,
            detection_2d=det_low,
        )

        result = filter.filter([box_high, box_low])

        assert result[0] is not None  # High score kept
        assert result[1] is None      # Low score filtered


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_3d_iou_perfect_overlap(self):
        """Test 3D IoU with identical boxes."""
        from src.eval.metrics import MetricsCalculator
        from src.fusion.bbox3d_generator import BBox3D
        from src.perception2d.detector import Detection

        calc = MetricsCalculator()

        det = Detection(
            box=np.array([0, 0, 100, 100]),
            score=0.9,
            class_id=2,
            class_name="Car",
        )

        box = BBox3D(
            center=np.array([0, 0, 10.0]),
            dimensions=np.array([4.0, 2.0, 1.5]),
            rotation_y=0.0,
            class_name="Car",
            score=0.9,
            detection_2d=det,
        )

        iou = calc.compute_3d_iou(box, box)

        assert np.isclose(iou, 1.0)

    def test_distance_error(self):
        """Test distance error calculation."""
        from src.eval.metrics import MetricsCalculator
        from src.fusion.bbox3d_generator import BBox3D
        from src.perception2d.detector import Detection

        calc = MetricsCalculator()

        det = Detection(
            box=np.array([0, 0, 100, 100]),
            score=0.9,
            class_id=2,
            class_name="Car",
        )

        pred = BBox3D(
            center=np.array([1.0, 0.0, 10.0]),
            dimensions=np.array([4.0, 2.0, 1.5]),
            rotation_y=0.0,
            class_name="Car",
            score=0.9,
            detection_2d=det,
        )

        gt = BBox3D(
            center=np.array([0.0, 0.0, 11.0]),
            dimensions=np.array([4.0, 2.0, 1.5]),
            rotation_y=0.0,
            class_name="Car",
            score=1.0,
            detection_2d=det,
        )

        errors = calc.compute_distance_error(pred, gt)

        assert np.isclose(errors["lateral_error"], 1.0)
        assert np.isclose(errors["depth_error"], 1.0)
