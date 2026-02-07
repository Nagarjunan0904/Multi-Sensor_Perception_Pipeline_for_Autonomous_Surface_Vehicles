"""Tests for projection utilities."""

import numpy as np
import pytest


class TestProjectionMath:
    """Tests for projection mathematics."""

    def test_perspective_projection_formula(self):
        """Test basic perspective projection formula."""
        # P = K * [R|t] * X
        # For identity R and zero t:
        # x = fx * X/Z + cx
        # y = fy * Y/Z + cy

        fx, fy = 500.0, 500.0
        cx, cy = 320.0, 240.0

        X, Y, Z = 1.0, 2.0, 10.0

        x = fx * X / Z + cx
        y = fy * Y / Z + cy

        assert x == 320.0 + 50.0  # 370
        assert y == 240.0 + 100.0  # 340

    def test_inverse_projection_formula(self):
        """Test inverse projection (2D to 3D)."""
        fx, fy = 500.0, 500.0
        cx, cy = 320.0, 240.0
        depth = 10.0

        # Given pixel (370, 340) and depth 10
        u, v = 370.0, 340.0

        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth

        assert np.isclose(X, 1.0)
        assert np.isclose(Y, 2.0)
        assert Z == 10.0

    def test_projection_roundtrip(self):
        """Test that project -> unproject gives original point."""
        from src.calibration.intrinsics import CameraIntrinsics

        intrinsics = CameraIntrinsics(
            fx=721.0, fy=721.0,
            cx=609.0, cy=172.0,
            width=1242, height=375,
        )

        # Original 3D point
        point_3d = np.array([5.0, 2.0, 20.0])

        # Project to 2D
        point_2d = intrinsics.project_point(point_3d)

        # Unproject back to 3D
        recovered = intrinsics.unproject_point(point_2d, point_3d[2])

        assert np.allclose(recovered, point_3d)

    def test_batch_projection(self):
        """Test batch point projection."""
        from src.calibration.intrinsics import CameraIntrinsics

        intrinsics = CameraIntrinsics(
            fx=100.0, fy=100.0,
            cx=50.0, cy=50.0,
            width=100, height=100,
        )

        points_3d = np.array([
            [0, 0, 10],
            [1, 0, 10],
            [0, 1, 10],
            [1, 1, 10],
        ], dtype=float)

        points_2d = intrinsics.project_point(points_3d)

        assert points_2d.shape == (4, 2)

        # Point at origin should project to (cx, cy)
        assert np.allclose(points_2d[0], [50, 50])

        # Point at (1, 0, 10) should project to (cx + fx*0.1, cy)
        assert np.allclose(points_2d[1], [60, 50])


class TestDepthInBox:
    """Tests for depth estimation within bounding boxes."""

    def test_median_depth(self):
        """Test median depth calculation."""
        depths = np.array([10.0, 11.0, 12.0, 100.0])  # One outlier

        median = np.median(depths)

        # Median should be robust to outlier
        assert median == 11.5

    def test_closest_depth(self):
        """Test closest depth selection."""
        depths = np.array([10.0, 11.0, 12.0, 5.0])

        closest = np.min(depths)

        assert closest == 5.0

    def test_weighted_depth(self):
        """Test distance-weighted depth."""
        # Points and their distances from box center
        depths = np.array([10.0, 20.0])
        distances = np.array([1.0, 10.0])

        # Inverse distance weighting
        weights = 1 / (distances + 1e-6)
        weights = weights / weights.sum()

        weighted_depth = np.sum(depths * weights)

        # Closer point (10m) should dominate
        assert weighted_depth < 15.0


class TestIoU:
    """Tests for IoU calculations."""

    def test_2d_iou_perfect_overlap(self):
        """Test 2D IoU with perfect overlap."""
        from src.perception2d.postprocess import compute_iou

        box = np.array([0, 0, 10, 10])

        iou = compute_iou(box, box)

        assert np.isclose(iou, 1.0)

    def test_2d_iou_no_overlap(self):
        """Test 2D IoU with no overlap."""
        from src.perception2d.postprocess import compute_iou

        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([20, 20, 30, 30])

        iou = compute_iou(box1, box2)

        assert iou == 0.0

    def test_2d_iou_partial_overlap(self):
        """Test 2D IoU with partial overlap."""
        from src.perception2d.postprocess import compute_iou

        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([5, 5, 15, 15])

        iou = compute_iou(box1, box2)

        # Intersection: 5x5 = 25
        # Union: 100 + 100 - 25 = 175
        # IoU: 25/175 ≈ 0.143
        assert np.isclose(iou, 25.0 / 175.0, atol=0.01)
