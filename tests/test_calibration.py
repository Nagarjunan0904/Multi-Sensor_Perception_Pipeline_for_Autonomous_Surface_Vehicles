"""
Comprehensive tests for calibration modules.

Test Coverage:
- Projection: known 3D point → expected 2D pixel
- Back-projection: pixel + depth → 3D point → project back
- Coordinate transforms: camera ↔ LiDAR consistency
- Edge cases: points behind camera, outside FOV
"""

import numpy as np
import pytest
from pathlib import Path


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_intrinsics():
    """Simple camera intrinsics for testing."""
    from src.calibration.intrinsics import CameraIntrinsics

    return CameraIntrinsics(
        fx=100.0, fy=100.0,
        cx=50.0, cy=50.0,
        width=100, height=100,
    )


@pytest.fixture
def kitti_intrinsics():
    """Typical KITTI camera intrinsics."""
    from src.calibration.intrinsics import CameraIntrinsics

    return CameraIntrinsics(
        fx=721.5377, fy=721.5377,
        cx=609.5593, cy=172.854,
        width=1242, height=375,
    )


@pytest.fixture
def identity_extrinsics():
    """Identity extrinsic transformation with P2 matrix set."""
    from src.calibration.extrinsics import CameraLiDARExtrinsics

    R = np.eye(3)
    T = np.zeros(3)
    extrinsics = CameraLiDARExtrinsics(R=R, T=T)
    # Set P2 matrix for projection (simple 100x100 camera)
    extrinsics.P2 = np.array([
        [100.0, 0, 50.0, 0],
        [0, 100.0, 50.0, 0],
        [0, 0, 1.0, 0],
    ])
    return extrinsics


@pytest.fixture
def rotation_90_z_extrinsics():
    """90-degree rotation around Z axis."""
    from src.calibration.extrinsics import CameraLiDARExtrinsics

    # 90 degree rotation around Z axis
    R = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ], dtype=float)
    T = np.zeros(3)
    return CameraLiDARExtrinsics(R=R, T=T)


@pytest.fixture
def typical_extrinsics():
    """Typical KITTI-like extrinsic calibration."""
    from src.calibration.extrinsics import CameraLiDARExtrinsics

    # Typical rotation (LiDAR to camera coordinate conversion)
    R = np.array([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ], dtype=float)
    # Typical translation (camera is offset from LiDAR)
    T = np.array([0.27, 0.06, -0.12])
    return CameraLiDARExtrinsics(R=R, T=T)


# =============================================================================
# Test CameraIntrinsics
# =============================================================================

class TestCameraIntrinsics:
    """Tests for CameraIntrinsics class."""

    def test_intrinsic_matrix_construction(self, simple_intrinsics):
        """Test intrinsic matrix K has correct structure."""
        K = simple_intrinsics.K

        assert K.shape == (3, 3)
        assert K[0, 0] == 100.0  # fx
        assert K[1, 1] == 100.0  # fy
        assert K[0, 2] == 50.0   # cx
        assert K[1, 2] == 50.0   # cy
        assert K[2, 2] == 1.0    # homogeneous
        assert K[0, 1] == 0.0    # no skew
        assert K[1, 0] == 0.0

    def test_intrinsic_matrix_inverse(self, simple_intrinsics):
        """Test K @ K^-1 = I."""
        K = simple_intrinsics.K
        K_inv = simple_intrinsics.get_K_inverse()

        identity = K @ K_inv

        assert np.allclose(identity, np.eye(3), atol=1e-10)

    def test_fov_calculation(self, simple_intrinsics):
        """Test field of view calculation."""
        fov_h, fov_v = simple_intrinsics.get_fov()

        # For 100x100 image with f=100:
        # FOV = 2 * atan(50/100) = 2 * atan(0.5) ≈ 53.13°
        expected_fov = 2 * np.arctan(50 / 100)

        assert np.isclose(fov_h, expected_fov)
        assert np.isclose(fov_v, expected_fov)

    def test_fov_kitti(self, kitti_intrinsics):
        """Test KITTI FOV is approximately correct."""
        fov_h, fov_v = kitti_intrinsics.get_fov()

        # KITTI has ~81° horizontal FOV
        assert 75 < np.degrees(fov_h) < 85
        # Vertical is smaller due to image aspect ratio
        assert 25 < np.degrees(fov_v) < 35


# =============================================================================
# Test 3D to 2D Projection
# =============================================================================

class TestProjection3Dto2D:
    """Test 3D point to 2D pixel projection."""

    def test_project_point_at_principal_point(self, simple_intrinsics):
        """Point on optical axis projects to principal point."""
        # Point at (0, 0, 10) should project to (cx, cy) = (50, 50)
        point_3d = np.array([0.0, 0.0, 10.0])
        point_2d = simple_intrinsics.project_point(point_3d)

        assert np.allclose(point_2d, [50.0, 50.0])

    def test_project_point_offset_x(self, simple_intrinsics):
        """Point offset in X projects correctly."""
        # Point at (1, 0, 10): u = fx * (1/10) + cx = 100*0.1 + 50 = 60
        point_3d = np.array([1.0, 0.0, 10.0])
        point_2d = simple_intrinsics.project_point(point_3d)

        assert np.allclose(point_2d, [60.0, 50.0])

    def test_project_point_offset_y(self, simple_intrinsics):
        """Point offset in Y projects correctly."""
        # Point at (0, 2, 10): v = fy * (2/10) + cy = 100*0.2 + 50 = 70
        point_3d = np.array([0.0, 2.0, 10.0])
        point_2d = simple_intrinsics.project_point(point_3d)

        assert np.allclose(point_2d, [50.0, 70.0])

    def test_project_known_kitti_point(self, kitti_intrinsics):
        """Project a known 3D point with KITTI parameters."""
        # Point at (5, 1, 30) - typical object position
        # u = 721.5377 * (5/30) + 609.5593 ≈ 729.8
        # v = 721.5377 * (1/30) + 172.854 ≈ 196.9
        point_3d = np.array([5.0, 1.0, 30.0])
        point_2d = kitti_intrinsics.project_point(point_3d)

        expected_u = 721.5377 * (5 / 30) + 609.5593
        expected_v = 721.5377 * (1 / 30) + 172.854

        assert np.isclose(point_2d[0], expected_u, atol=0.01)
        assert np.isclose(point_2d[1], expected_v, atol=0.01)

    def test_project_batch_points(self, simple_intrinsics):
        """Test batch projection of multiple points."""
        points_3d = np.array([
            [0, 0, 10],
            [1, 0, 10],
            [0, 1, 10],
            [1, 1, 10],
        ], dtype=float)

        points_2d = simple_intrinsics.project_point(points_3d)

        assert points_2d.shape == (4, 2)
        assert np.allclose(points_2d[0], [50, 50])
        assert np.allclose(points_2d[1], [60, 50])
        assert np.allclose(points_2d[2], [50, 60])
        assert np.allclose(points_2d[3], [60, 60])

    def test_project_depth_scaling(self, simple_intrinsics):
        """Verify depth scaling: doubling distance halves apparent size."""
        # Two points with same X,Y but different Z
        point_near = np.array([2.0, 0.0, 10.0])
        point_far = np.array([4.0, 0.0, 20.0])  # Same angle, double distance

        pixel_near = simple_intrinsics.project_point(point_near)
        pixel_far = simple_intrinsics.project_point(point_far)

        # Both should project to same pixel (same ray direction)
        assert np.allclose(pixel_near, pixel_far)


# =============================================================================
# Test 2D to 3D Back-Projection
# =============================================================================

class TestBackProjection2Dto3D:
    """Test 2D pixel to 3D point back-projection."""

    def test_unproject_principal_point(self, simple_intrinsics):
        """Principal point unprojets to optical axis."""
        pixel = np.array([50.0, 50.0])  # (cx, cy)
        depth = 10.0

        point_3d = simple_intrinsics.unproject_point(pixel, depth)

        # Should give (0, 0, depth)
        assert np.allclose(point_3d, [0.0, 0.0, 10.0])

    def test_unproject_offset_pixel(self, simple_intrinsics):
        """Offset pixel unprojets correctly."""
        # Pixel at (60, 70) with depth 10
        # X = (60 - 50) * 10 / 100 = 1.0
        # Y = (70 - 50) * 10 / 100 = 2.0
        pixel = np.array([60.0, 70.0])
        depth = 10.0

        point_3d = simple_intrinsics.unproject_point(pixel, depth)

        assert np.allclose(point_3d, [1.0, 2.0, 10.0])

    def test_unproject_batch_pixels(self, simple_intrinsics):
        """Test batch unprojection."""
        pixels = np.array([
            [50, 50],
            [60, 50],
            [50, 60],
        ], dtype=float)
        depths = np.array([10.0, 10.0, 10.0])

        points_3d = simple_intrinsics.unproject_point(pixels, depths)

        assert points_3d.shape == (3, 3)
        assert np.allclose(points_3d[0], [0, 0, 10])
        assert np.allclose(points_3d[1], [1, 0, 10])
        assert np.allclose(points_3d[2], [0, 1, 10])


# =============================================================================
# Test Projection Round-Trip
# =============================================================================

class TestProjectionRoundTrip:
    """Test project → unproject gives original point."""

    def test_roundtrip_simple(self, simple_intrinsics):
        """Simple round-trip test."""
        original_3d = np.array([5.0, 3.0, 20.0])

        # Project to 2D
        pixel = simple_intrinsics.project_point(original_3d)

        # Unproject back to 3D
        recovered_3d = simple_intrinsics.unproject_point(pixel, original_3d[2])

        assert np.allclose(recovered_3d, original_3d)

    def test_roundtrip_kitti(self, kitti_intrinsics):
        """Round-trip test with KITTI parameters."""
        original_3d = np.array([8.5, 2.0, 35.0])

        pixel = kitti_intrinsics.project_point(original_3d)
        recovered_3d = kitti_intrinsics.unproject_point(pixel, original_3d[2])

        assert np.allclose(recovered_3d, original_3d, atol=1e-10)

    def test_roundtrip_batch(self, kitti_intrinsics):
        """Batch round-trip test."""
        np.random.seed(42)
        # Generate random points in front of camera
        original_3d = np.random.randn(100, 3)
        original_3d[:, 2] = np.abs(original_3d[:, 2]) + 1  # Ensure positive Z

        pixels = kitti_intrinsics.project_point(original_3d)
        recovered_3d = kitti_intrinsics.unproject_point(pixels, original_3d[:, 2])

        assert np.allclose(recovered_3d, original_3d, atol=1e-10)

    def test_roundtrip_edge_of_image(self, kitti_intrinsics):
        """Round-trip at image corners."""
        # Point that projects near image corner
        original_3d = np.array([-30.0, -10.0, 50.0])

        pixel = kitti_intrinsics.project_point(original_3d)
        recovered_3d = kitti_intrinsics.unproject_point(pixel, original_3d[2])

        assert np.allclose(recovered_3d, original_3d, atol=1e-10)


# =============================================================================
# Test Coordinate Transformations
# =============================================================================

class TestCoordinateTransforms:
    """Test camera ↔ LiDAR coordinate transformations."""

    def test_transform_identity(self, identity_extrinsics):
        """Identity transformation preserves points."""
        points_lidar = np.array([[1.0, 2.0, 3.0]])
        points_cam = identity_extrinsics.transform_to_camera(points_lidar)

        assert np.allclose(points_cam, points_lidar)

    def test_transform_rotation_90_z(self, rotation_90_z_extrinsics):
        """90-degree Z rotation transforms correctly."""
        # Point at (1, 0, 0) should go to (0, 1, 0)
        points_lidar = np.array([[1.0, 0.0, 0.0]])
        points_cam = rotation_90_z_extrinsics.transform_to_camera(points_lidar)

        assert np.allclose(points_cam, [[0.0, 1.0, 0.0]], atol=1e-10)

    def test_transform_translation(self):
        """Translation-only transformation."""
        from src.calibration.extrinsics import CameraLiDARExtrinsics

        R = np.eye(3)
        T = np.array([1.0, 2.0, 3.0])
        extrinsics = CameraLiDARExtrinsics(R=R, T=T)

        points_lidar = np.array([[0.0, 0.0, 0.0]])
        points_cam = extrinsics.transform_to_camera(points_lidar)

        assert np.allclose(points_cam, [[1.0, 2.0, 3.0]])

    def test_transform_inverse_roundtrip(self):
        """Camera → LiDAR → Camera gives original point."""
        from src.calibration.extrinsics import CameraExtrinsics

        # Arbitrary rotation and translation
        angle = np.pi / 6
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        t = np.array([1.5, -0.5, 2.0])

        extrinsics = CameraExtrinsics(R=R, t=t)
        extrinsics_inv = extrinsics.inverse()

        # Original point in source frame
        point_original = np.array([5.0, 3.0, 10.0])

        # Transform to target, then back
        point_target = extrinsics.transform_points(point_original)
        point_recovered = extrinsics_inv.transform_points(point_target)

        assert np.allclose(point_recovered, point_original, atol=1e-10)

    def test_camera_lidar_bidirectional(self):
        """LiDAR → Camera → LiDAR gives original point."""
        from src.calibration.extrinsics import CameraLiDARExtrinsics

        # Typical KITTI-like transformation
        R = np.array([
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ], dtype=float)
        T = np.array([0.27, 0.06, -0.12])

        extrinsics = CameraLiDARExtrinsics(R=R, T=T)

        # Points in LiDAR frame
        points_lidar = np.array([
            [10.0, 1.0, -0.5],
            [20.0, -2.0, 1.5],
            [5.0, 0.5, 0.0],
        ])

        # Transform to camera
        points_cam = extrinsics.transform_to_camera(points_lidar)

        # Transform back to LiDAR
        points_lidar_recovered = extrinsics.transform_to_lidar(points_cam)

        assert np.allclose(points_lidar_recovered, points_lidar, atol=1e-10)

    def test_rotation_matrix_orthonormality(self, typical_extrinsics):
        """Rotation matrix should be orthonormal."""
        R = typical_extrinsics.rotation

        # R^T @ R should be identity
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)

        # Determinant should be +1 (proper rotation)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_compose_transformations(self):
        """Test composition of two transformations."""
        from src.calibration.extrinsics import CameraExtrinsics

        # First transformation: translate then rotate
        R1 = np.array([
            [np.cos(np.pi/4), -np.sin(np.pi/4), 0],
            [np.sin(np.pi/4), np.cos(np.pi/4), 0],
            [0, 0, 1],
        ])
        t1 = np.array([1.0, 0.0, 0.0])
        ext1 = CameraExtrinsics(R=R1, t=t1)

        # Second transformation
        R2 = np.eye(3)
        t2 = np.array([0.0, 1.0, 0.0])
        ext2 = CameraExtrinsics(R=R2, t=t2)

        # Composed transformation
        composed = ext1.compose(ext2)

        # Apply both sequentially
        point = np.array([0.0, 0.0, 0.0])
        point_seq = ext2.transform_points(ext1.transform_points(point))

        # Apply composed
        point_composed = composed.transform_points(point)

        assert np.allclose(point_seq, point_composed, atol=1e-10)


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases in projection and transformation."""

    def test_points_behind_camera(self, simple_intrinsics):
        """Points with Z <= 0 produce invalid projections."""
        # Point behind camera (negative Z)
        point_behind = np.array([0.0, 0.0, -10.0])
        pixel = simple_intrinsics.project_point(point_behind)

        # Result should be at or near principal point (due to -Z)
        # but the projection is mathematically invalid
        # This verifies we don't crash, but results are not meaningful

    def test_point_at_camera(self, simple_intrinsics):
        """Point at Z=0 produces degenerate projection."""
        point_at_camera = np.array([1.0, 1.0, 0.0])

        # This would cause division by zero
        # The function should handle it gracefully (or we document the limitation)
        pixel = simple_intrinsics.project_point(point_at_camera)

        # Result will be inf or very large - test it doesn't crash
        assert pixel is not None

    def test_points_outside_fov(self, simple_intrinsics):
        """Test detection of points outside field of view."""
        points = np.array([
            [50, 50],   # In image
            [-10, 50],  # Left of image
            [110, 50],  # Right of image
            [50, -10],  # Above image
            [50, 110],  # Below image
        ])

        valid = simple_intrinsics.is_in_image(points)

        assert valid[0] == True
        assert valid[1] == False
        assert valid[2] == False
        assert valid[3] == False
        assert valid[4] == False

    def test_points_on_boundary(self, simple_intrinsics):
        """Test points exactly on image boundary."""
        points = np.array([
            [0, 0],       # Top-left corner (valid)
            [99, 99],     # Near bottom-right (valid)
            [100, 50],    # At right edge (invalid, >= width)
            [50, 100],    # At bottom edge (invalid, >= height)
        ])

        valid = simple_intrinsics.is_in_image(points)

        assert valid[0] == True
        assert valid[1] == True
        assert valid[2] == False
        assert valid[3] == False

    def test_margin_filtering(self, simple_intrinsics):
        """Test FOV filtering with margin."""
        points = np.array([
            [5, 5],    # Near edge
            [50, 50],  # Center
            [95, 95],  # Near other edge
        ])

        # With margin=10, only center should be valid
        valid_no_margin = simple_intrinsics.is_in_image(points, margin=0)
        valid_with_margin = simple_intrinsics.is_in_image(points, margin=10)

        assert all(valid_no_margin)  # All valid without margin
        assert valid_with_margin[0] == False  # Too close to edge
        assert valid_with_margin[1] == True   # Center is fine
        assert valid_with_margin[2] == False  # Too close to other edge

    def test_very_far_points(self, kitti_intrinsics):
        """Test projection of very distant points."""
        # Point at 500 meters
        point_far = np.array([10.0, 2.0, 500.0])
        pixel = kitti_intrinsics.project_point(point_far)

        # Should still be valid and near principal point
        assert 0 < pixel[0] < kitti_intrinsics.width
        assert 0 < pixel[1] < kitti_intrinsics.height

    def test_empty_point_array(self, simple_intrinsics):
        """Test handling of empty point arrays."""
        from src.calibration.projection import project_3d_to_2d

        empty_points = np.zeros((0, 3))
        K = simple_intrinsics.K

        points_2d, depths = project_3d_to_2d(empty_points, K)

        assert points_2d.shape == (0, 2)
        assert depths.shape == (0,)


# =============================================================================
# Test Standalone Projection Functions
# =============================================================================

class TestStandaloneFunctions:
    """Test standalone projection utility functions."""

    def test_project_3d_to_2d(self, simple_intrinsics):
        """Test standalone projection function."""
        from src.calibration.projection import project_3d_to_2d

        K = simple_intrinsics.K
        points_3d = np.array([[0, 0, 10], [1, 0, 10]])

        points_2d, depths = project_3d_to_2d(points_3d, K)

        assert points_2d.shape == (2, 2)
        assert np.allclose(points_2d[0], [50, 50])
        assert np.allclose(depths, [10, 10])

    def test_backproject_2d_to_3d(self, simple_intrinsics):
        """Test standalone backprojection function."""
        from src.calibration.projection import backproject_2d_to_3d

        K = simple_intrinsics.K
        pixels = np.array([[50, 50], [60, 70]])
        depths = np.array([10.0, 10.0])

        points_3d = backproject_2d_to_3d(pixels, depths, K)

        assert points_3d.shape == (2, 3)
        assert np.allclose(points_3d[0], [0, 0, 10])
        assert np.allclose(points_3d[1], [1, 2, 10])

    def test_transform_points(self):
        """Test standalone transform function."""
        from src.calibration.projection import transform_points

        R = np.eye(3)
        t = np.array([1, 2, 3])
        points = np.array([[0, 0, 0], [1, 1, 1]])

        transformed = transform_points(points, R, t)

        assert np.allclose(transformed[0], [1, 2, 3])
        assert np.allclose(transformed[1], [2, 3, 4])

    def test_filter_fov(self, simple_intrinsics):
        """Test FOV filtering function."""
        from src.calibration.projection import filter_fov

        K = simple_intrinsics.K
        image_shape = (100, 100)  # height, width

        # Mix of valid and invalid points
        points = np.array([
            [0, 0, 10],      # Projects to (50, 50) - valid
            [100, 0, 10],    # Projects far right - invalid
            [-100, 0, 10],   # Projects far left - invalid
            [0, 0, -10],     # Behind camera - invalid
        ])

        points_2d, depths, mask = filter_fov(points, image_shape, K)

        assert mask.sum() == 1  # Only first point is valid
        assert np.allclose(points_2d[0], [50, 50])

    def test_filter_fov_with_depth_range(self, simple_intrinsics):
        """Test FOV filtering with depth constraints."""
        from src.calibration.projection import filter_fov

        K = simple_intrinsics.K
        image_shape = (100, 100)

        points = np.array([
            [0, 0, 5],    # Too close
            [0, 0, 15],   # In range
            [0, 0, 25],   # Too far
        ])

        _, depths, mask = filter_fov(
            points, image_shape, K,
            depth_range=(10, 20)
        )

        assert mask.sum() == 1
        assert depths[0] == 15.0

    def test_compute_frustum_points(self, kitti_intrinsics):
        """Test camera frustum computation."""
        from src.calibration.projection import compute_frustum_points

        K = kitti_intrinsics.K
        image_shape = (kitti_intrinsics.height, kitti_intrinsics.width)

        corners = compute_frustum_points(image_shape, K, depth=50.0)

        assert corners.shape == (4, 3)

        # All corners should be at depth 50
        assert np.allclose(corners[:, 2], 50.0)

        # Width at 50m should be calculable
        frustum_width = corners[1, 0] - corners[0, 0]
        assert frustum_width > 0


# =============================================================================
# Test Projector Class
# =============================================================================

class TestProjector:
    """Tests for the Projector class."""

    def test_project_lidar_to_image(self, simple_intrinsics, identity_extrinsics):
        """Test full LiDAR to image projection."""
        from src.calibration.projection import Projector

        projector = Projector(simple_intrinsics, identity_extrinsics)

        # Point at (0, 0, 10) should project to image center
        points = np.array([[0.0, 0.0, 10.0]])

        points_2d, depths, mask = projector.project_lidar_to_image(
            points, filter_fov=False, filter_depth=False
        )

        assert np.allclose(points_2d, [[50.0, 50.0]], atol=1.0)
        assert np.allclose(depths, [10.0])

    def test_projector_with_filtering(self, simple_intrinsics, identity_extrinsics):
        """Test projector with FOV and depth filtering."""
        from src.calibration.projection import Projector

        projector = Projector(simple_intrinsics, identity_extrinsics)

        points = np.array([
            [0.0, 0.0, 10.0],    # Valid
            [100.0, 0.0, 10.0],  # Outside FOV
            [0.0, 0.0, 200.0],   # Beyond depth range
        ])

        points_2d, depths, mask = projector.project_lidar_to_image(
            points, filter_fov=True, filter_depth=True,
            depth_range=(0.1, 100.0)
        )

        assert mask.sum() == 1

    def test_get_depth_at_pixel(self, simple_intrinsics, identity_extrinsics):
        """Test depth extraction at pixel location."""
        from src.calibration.projection import Projector

        projector = Projector(simple_intrinsics, identity_extrinsics)

        # Create points that project near center
        points = np.array([
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 11.0],
            [0.0, 0.0, 12.0],
            [0.0, 0.0, 10.0],  # Extra intensity column
        ])

        depth = projector.get_depth_at_pixel(points, (50, 50), search_radius=10)

        assert depth is not None
        assert 10.0 <= depth <= 12.0

    def test_get_depth_in_box(self, simple_intrinsics, identity_extrinsics):
        """Test depth estimation within bounding box."""
        from src.calibration.projection import Projector

        projector = Projector(simple_intrinsics, identity_extrinsics)

        # Points that project within box [40, 40, 60, 60]
        points = np.array([
            [0.0, 0.0, 10.0],   # Projects to (50, 50)
            [0.0, 0.0, 15.0],   # Projects to (50, 50)
            [0.0, 0.0, 100.0],  # Projects to (50, 50) but outlier
        ])

        depth = projector.get_depth_in_box(
            points,
            box_2d=np.array([40, 40, 60, 60]),
            method="median"
        )

        assert depth is not None
        assert depth == 15.0  # Median of [10, 15, 100]


# =============================================================================
# Test from KITTI Calibration
# =============================================================================

class TestKITTICalibration:
    """Tests for KITTI-specific calibration loading."""

    def test_from_kitti_calib_dict(self):
        """Test creating intrinsics from KITTI calibration dictionary."""
        from src.calibration.intrinsics import CameraIntrinsics

        # Simulated KITTI P2 matrix
        P2 = np.array([
            721.5377, 0, 609.5593, 44.85728,
            0, 721.5377, 172.854, 0.2163791,
            0, 0, 1, 0.002745884,
        ]).reshape(3, 4)

        calib_dict = {"P2": P2}

        intrinsics = CameraIntrinsics.from_kitti_calib(calib_dict)

        assert np.isclose(intrinsics.fx, 721.5377)
        assert np.isclose(intrinsics.fy, 721.5377)
        assert np.isclose(intrinsics.cx, 609.5593)
        assert np.isclose(intrinsics.cy, 172.854)

    def test_from_projection_matrix(self):
        """Test creating intrinsics from projection matrix."""
        from src.calibration.intrinsics import CameraIntrinsics

        P = np.array([
            [721.0, 0, 609.0, 0],
            [0, 721.0, 172.0, 0],
            [0, 0, 1, 0],
        ])

        intrinsics = CameraIntrinsics.from_projection_matrix(P, 1242, 375)

        assert intrinsics.fx == 721.0
        assert intrinsics.fy == 721.0
        assert intrinsics.cx == 609.0
        assert intrinsics.cy == 172.0

    def test_extrinsics_from_kitti_calib(self):
        """Test creating extrinsics from KITTI calibration dictionary."""
        from src.calibration.extrinsics import CameraExtrinsics

        # Simulated Tr_velo_to_cam
        Tr = np.array([
            7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03,
            1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02,
            9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01,
        ]).reshape(3, 4)

        calib_dict = {"Tr_velo_to_cam": Tr}

        extrinsics = CameraExtrinsics.from_kitti_calib(calib_dict)

        assert extrinsics.R.shape == (3, 3)
        assert extrinsics.t.shape == (3,)


# =============================================================================
# Test Pixel Size and Physical Units
# =============================================================================

class TestPhysicalUnits:
    """Test physical unit conversions."""

    def test_pixel_size_at_depth(self, kitti_intrinsics):
        """Test pixel physical size calculation."""
        # At 50 meters with fx≈721, pixel width = 50/721 ≈ 0.069m ≈ 7cm
        pixel_w, pixel_h = kitti_intrinsics.get_pixel_size_at_depth(50.0)

        assert np.isclose(pixel_w, 50.0 / 721.5377, atol=0.001)
        assert np.isclose(pixel_h, 50.0 / 721.5377, atol=0.001)

    def test_pixel_size_scales_with_depth(self, kitti_intrinsics):
        """Pixel size should scale linearly with depth."""
        pw_10, _ = kitti_intrinsics.get_pixel_size_at_depth(10.0)
        pw_20, _ = kitti_intrinsics.get_pixel_size_at_depth(20.0)

        assert np.isclose(pw_20, 2 * pw_10)

    def test_pixel_to_ray_unit_vector(self, simple_intrinsics):
        """Pixel to ray should produce unit vectors."""
        pixels = np.array([
            [50, 50],  # Center
            [0, 0],    # Corner
            [99, 99],  # Other corner
        ])

        rays = simple_intrinsics.pixel_to_ray(pixels)

        # All rays should have unit length
        norms = np.linalg.norm(rays, axis=1)
        assert np.allclose(norms, 1.0)

    def test_ray_direction_center(self, simple_intrinsics):
        """Center pixel should give forward-pointing ray."""
        pixel = np.array([50, 50])  # Principal point
        ray = simple_intrinsics.pixel_to_ray(pixel)

        # Should point along Z axis (0, 0, 1) when normalized
        assert np.allclose(ray, [0, 0, 1])


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
