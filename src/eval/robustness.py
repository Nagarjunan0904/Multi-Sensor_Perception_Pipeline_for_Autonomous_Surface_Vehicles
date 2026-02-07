"""Robustness evaluation with noise injection."""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np

# Handle imports flexibly for both package and direct usage
try:
    from fusion.bbox3d_generator import BBox3D
    from eval.metrics import MetricsCalculator
except ImportError:
    try:
        from ..fusion.bbox3d_generator import BBox3D
        from .metrics import MetricsCalculator
    except ImportError:
        if TYPE_CHECKING:
            from fusion.bbox3d_generator import BBox3D
            from eval.metrics import MetricsCalculator


class RobustnessEvaluator:
    """Evaluate pipeline robustness under various failure modes."""

    def __init__(
        self,
        metrics_calculator: Optional[MetricsCalculator] = None,
    ):
        """
        Initialize robustness evaluator.

        Args:
            metrics_calculator: Metrics calculator instance.
        """
        self.metrics = metrics_calculator or MetricsCalculator()

    def add_lidar_noise(
        self,
        points: np.ndarray,
        noise_std: float = 0.1,
    ) -> np.ndarray:
        """
        Add Gaussian noise to LiDAR points.

        Args:
            points: Original point cloud (N, 4).
            noise_std: Standard deviation of noise (meters).

        Returns:
            Noisy point cloud.
        """
        noisy = points.copy()
        noisy[:, :3] += np.random.normal(0, noise_std, noisy[:, :3].shape)
        return noisy

    def add_lidar_dropout(
        self,
        points: np.ndarray,
        dropout_rate: float = 0.1,
    ) -> np.ndarray:
        """
        Randomly drop LiDAR points.

        Args:
            points: Original point cloud (N, 4).
            dropout_rate: Fraction of points to drop.

        Returns:
            Point cloud with dropped points.
        """
        mask = np.random.random(len(points)) > dropout_rate
        return points[mask]

    def add_lidar_systematic_error(
        self,
        points: np.ndarray,
        offset: np.ndarray = np.array([0.1, 0.0, 0.05]),
    ) -> np.ndarray:
        """
        Add systematic calibration error to LiDAR.

        Args:
            points: Original point cloud (N, 4).
            offset: Systematic offset (dx, dy, dz).

        Returns:
            Point cloud with offset.
        """
        shifted = points.copy()
        shifted[:, :3] += offset
        return shifted

    def add_image_noise(
        self,
        image: np.ndarray,
        noise_std: float = 10.0,
    ) -> np.ndarray:
        """
        Add Gaussian noise to image.

        Args:
            image: Original image (H, W, C).
            noise_std: Standard deviation of noise.

        Returns:
            Noisy image.
        """
        noise = np.random.normal(0, noise_std, image.shape)
        noisy = np.clip(image.astype(float) + noise, 0, 255)
        return noisy.astype(np.uint8)

    def add_image_blur(
        self,
        image: np.ndarray,
        kernel_size: int = 5,
    ) -> np.ndarray:
        """
        Add motion blur to image.

        Args:
            image: Original image.
            kernel_size: Blur kernel size.

        Returns:
            Blurred image.
        """
        try:
            import cv2
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        except ImportError:
            return image

    def add_image_occlusion(
        self,
        image: np.ndarray,
        occlusion_ratio: float = 0.1,
    ) -> np.ndarray:
        """
        Add random occlusion patches to image.

        Args:
            image: Original image.
            occlusion_ratio: Fraction of image to occlude.

        Returns:
            Image with occlusions.
        """
        h, w = image.shape[:2]
        occluded = image.copy()

        # Calculate patch size
        patch_area = int(h * w * occlusion_ratio)
        patch_size = int(np.sqrt(patch_area))

        # Random patch location
        x = np.random.randint(0, max(1, w - patch_size))
        y = np.random.randint(0, max(1, h - patch_size))

        # Apply occlusion (gray patch)
        occluded[y:y+patch_size, x:x+patch_size] = 128

        return occluded

    def simulate_sensor_failure(
        self,
        points: np.ndarray,
        failure_type: str = "partial",
        failure_region: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        """
        Simulate sensor failure scenarios.

        Args:
            points: Original point cloud.
            failure_type: Type of failure ('partial', 'sector', 'near', 'far').
            failure_region: Region of failure (x_min, x_max, y_min, y_max).

        Returns:
            Point cloud with simulated failure.
        """
        if failure_type == "partial":
            # Random 50% dropout
            return self.add_lidar_dropout(points, 0.5)

        elif failure_type == "sector":
            # Drop a sector (angular region)
            angles = np.arctan2(points[:, 1], points[:, 0])
            sector_start = np.random.uniform(-np.pi, np.pi)
            sector_end = sector_start + np.pi / 4  # 45 degree sector

            mask = ~((angles >= sector_start) & (angles < sector_end))
            return points[mask]

        elif failure_type == "near":
            # Drop near points (sensor contamination)
            distances = np.linalg.norm(points[:, :3], axis=1)
            mask = distances > 10.0
            return points[mask]

        elif failure_type == "far":
            # Drop far points (reduced range)
            distances = np.linalg.norm(points[:, :3], axis=1)
            mask = distances < 40.0
            return points[mask]

        else:
            return points

    def evaluate_noise_robustness(
        self,
        pipeline_func: Callable,
        images: List[np.ndarray],
        pointclouds: List[np.ndarray],
        ground_truths: List[List[BBox3D]],
        noise_levels: List[float] = [0.0, 0.05, 0.1, 0.2, 0.5],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate pipeline robustness to varying noise levels.

        Args:
            pipeline_func: Function that takes (image, points) and returns predictions.
            images: List of input images.
            pointclouds: List of point clouds.
            ground_truths: Ground truth boxes for each frame.
            noise_levels: Noise standard deviations to test.

        Returns:
            Dictionary with metrics at each noise level.
        """
        results = {}

        for noise_std in noise_levels:
            all_preds = []
            all_gts = []

            for image, points, gts in zip(images, pointclouds, ground_truths):
                # Add noise
                noisy_points = self.add_lidar_noise(points, noise_std)

                # Run pipeline
                predictions = pipeline_func(image, noisy_points)

                all_preds.extend(predictions)
                all_gts.extend(gts)

            # Compute metrics
            metrics = self.metrics.compute_map(all_preds, all_gts)
            results[f"noise_{noise_std}"] = metrics

        return results

    def evaluate_failure_modes(
        self,
        pipeline_func: Callable,
        images: List[np.ndarray],
        pointclouds: List[np.ndarray],
        ground_truths: List[List[BBox3D]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate pipeline under different failure modes.

        Args:
            pipeline_func: Pipeline function.
            images: Input images.
            pointclouds: Point clouds.
            ground_truths: Ground truth boxes.

        Returns:
            Metrics under each failure mode.
        """
        failure_modes = ["partial", "sector", "near", "far"]
        results = {}

        for mode in failure_modes:
            all_preds = []
            all_gts = []

            for image, points, gts in zip(images, pointclouds, ground_truths):
                failed_points = self.simulate_sensor_failure(points, mode)
                predictions = pipeline_func(image, failed_points)

                all_preds.extend(predictions)
                all_gts.extend(gts)

            metrics = self.metrics.compute_map(all_preds, all_gts)
            results[mode] = metrics

        return results
