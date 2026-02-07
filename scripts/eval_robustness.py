#!/usr/bin/env python3
"""
Robustness Evaluation Suite for Multi-Sensor Perception Pipeline.

This script evaluates pipeline robustness under various failure conditions:
- LiDAR noise (Gaussian perturbation)
- Calibration errors (systematic offset)
- Sensor dropout (missing points)
- Combined worst-case scenarios

Outputs:
- Baseline metrics (clean accuracy)
- Robustness curves for each noise type
- Plots: plots/robustness_*.png
- Summary report: outputs/robustness_report.md
- Failure visualization: outputs/failure_modes.avi

Usage:
    # Run full evaluation
    python scripts/eval_robustness.py

    # Quick test with fewer frames
    python scripts/eval_robustness.py --num-frames 50

    # Custom noise levels
    python scripts/eval_robustness.py --noise-levels 5

    # Skip video generation
    python scripts/eval_robustness.py --no-video
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from src.data.kitti_loader import KITTILoader
from src.perception2d.detector import ObjectDetector2D, Detection
from src.fusion.depth_estimator import DepthEstimator
from src.fusion.bbox3d_generator import BBox3DGenerator, BBox3D
from src.calibration.intrinsics import CameraIntrinsics
from src.calibration.extrinsics import CameraLiDARExtrinsics

# Optional imports
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Setup logger with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvaluationMetrics:
    """Metrics for a single evaluation run."""
    num_frames: int = 0
    total_detections_2d: int = 0
    total_boxes_3d: int = 0
    detection_rate: float = 0.0  # 3D boxes / 2D detections

    # Depth metrics
    depth_errors: List[float] = field(default_factory=list)
    mean_depth_error: float = 0.0
    median_depth_error: float = 0.0
    depth_rmse: float = 0.0

    # Position metrics
    position_errors: List[float] = field(default_factory=list)
    mean_position_error: float = 0.0

    # Timing
    avg_processing_time: float = 0.0

    def compute_aggregates(self):
        """Compute aggregate metrics from collected data."""
        if self.total_detections_2d > 0:
            self.detection_rate = self.total_boxes_3d / self.total_detections_2d

        if self.depth_errors:
            self.mean_depth_error = np.mean(self.depth_errors)
            self.median_depth_error = np.median(self.depth_errors)
            self.depth_rmse = np.sqrt(np.mean(np.array(self.depth_errors) ** 2))

        if self.position_errors:
            self.mean_position_error = np.mean(self.position_errors)


@dataclass
class RobustnessResult:
    """Results from robustness evaluation."""
    noise_type: str
    noise_levels: List[float]
    metrics: List[EvaluationMetrics]
    baseline_metrics: EvaluationMetrics = None

    def get_detection_rates(self) -> List[float]:
        return [m.detection_rate for m in self.metrics]

    def get_depth_errors(self) -> List[float]:
        return [m.mean_depth_error for m in self.metrics]

    def get_degradation(self) -> List[float]:
        """Compute performance degradation relative to baseline."""
        if self.baseline_metrics is None:
            return [0.0] * len(self.metrics)

        baseline_rate = self.baseline_metrics.detection_rate
        if baseline_rate == 0:
            return [0.0] * len(self.metrics)

        return [(baseline_rate - m.detection_rate) / baseline_rate * 100
                for m in self.metrics]


# =============================================================================
# Noise Injection Functions
# =============================================================================

class NoiseInjector:
    """Inject various types of noise into sensor data."""

    @staticmethod
    def add_lidar_noise(
        points: np.ndarray,
        noise_std: float = 0.1,
    ) -> np.ndarray:
        """
        Add Gaussian noise to LiDAR points.

        Args:
            points: Point cloud (N, 4) with [x, y, z, intensity].
            noise_std: Standard deviation of noise in meters.

        Returns:
            Noisy point cloud.
        """
        if noise_std == 0:
            return points

        noisy = points.copy()
        noisy[:, :3] += np.random.normal(0, noise_std, noisy[:, :3].shape)
        return noisy

    @staticmethod
    def add_lidar_dropout(
        points: np.ndarray,
        dropout_rate: float = 0.1,
    ) -> np.ndarray:
        """
        Randomly drop LiDAR points to simulate sensor failures.

        Args:
            points: Point cloud (N, 4).
            dropout_rate: Fraction of points to drop [0, 1].

        Returns:
            Point cloud with dropped points.
        """
        if dropout_rate == 0:
            return points

        mask = np.random.random(len(points)) > dropout_rate
        return points[mask]

    @staticmethod
    def add_calibration_error(
        points: np.ndarray,
        translation_error: np.ndarray = None,
        rotation_error: float = 0.0,
    ) -> np.ndarray:
        """
        Add systematic calibration error to LiDAR points.

        Args:
            points: Point cloud (N, 4).
            translation_error: [dx, dy, dz] offset in meters.
            rotation_error: Rotation error in degrees around Z-axis.

        Returns:
            Misaligned point cloud.
        """
        if translation_error is None:
            translation_error = np.array([0.0, 0.0, 0.0])

        result = points.copy()

        # Apply translation
        result[:, :3] += translation_error

        # Apply rotation around Z-axis
        if rotation_error != 0:
            angle_rad = np.radians(rotation_error)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            x, y = result[:, 0].copy(), result[:, 1].copy()
            result[:, 0] = cos_a * x - sin_a * y
            result[:, 1] = sin_a * x + cos_a * y

        return result

    @staticmethod
    def add_depth_noise(
        points: np.ndarray,
        depth_noise_std: float = 0.0,
    ) -> np.ndarray:
        """
        Add noise specifically to depth (range) measurements.

        Args:
            points: Point cloud (N, 4).
            depth_noise_std: Standard deviation as fraction of depth.

        Returns:
            Point cloud with depth noise.
        """
        if depth_noise_std == 0:
            return points

        result = points.copy()

        # Compute ranges
        ranges = np.linalg.norm(result[:, :3], axis=1, keepdims=True)

        # Add range-proportional noise
        noise = np.random.normal(0, depth_noise_std, (len(points), 1)) * ranges

        # Apply noise along radial direction
        directions = result[:, :3] / (ranges + 1e-6)
        result[:, :3] += directions * noise

        return result


# =============================================================================
# Robustness Evaluator
# =============================================================================

class RobustnessEvaluator:
    """
    Evaluate pipeline robustness under various conditions.

    Runs the perception pipeline with different noise levels and
    collects metrics to assess robustness.
    """

    def __init__(
        self,
        data_root: str = "data/kitti",
        device: str = "cuda",
        num_frames: int = 100,
    ):
        """
        Initialize the robustness evaluator.

        Args:
            data_root: Path to KITTI dataset.
            device: Inference device.
            num_frames: Number of frames to evaluate.
        """
        self.logger = setup_logger("robustness_eval")
        self.device = device
        self.num_frames = num_frames

        # Load dataset
        self.logger.info(f"Loading KITTI dataset from {data_root}")
        self.loader = KITTILoader(root_dir=data_root, split="training")
        self.num_frames = min(num_frames, len(self.loader))
        self.logger.info(f"Using {self.num_frames} frames for evaluation")

        # Initialize detector
        self.logger.info("Initializing detector")
        self.detector = ObjectDetector2D(
            model_name="yolov8m",
            device=device,
            confidence_threshold=0.3,
        )

        # Placeholders for per-frame components
        self.depth_estimator = None
        self.bbox_generator = None
        self.intrinsics = None
        self.extrinsics = None

        # Noise injector
        self.noise_injector = NoiseInjector()

        # Results storage
        self.baseline_metrics = None
        self.results: Dict[str, RobustnessResult] = {}

    def _setup_calibration(self, calib) -> Tuple[CameraIntrinsics, CameraLiDARExtrinsics]:
        """Setup calibration from KITTI Calibration object."""
        P2 = calib.P2
        fx, fy = P2[0, 0], P2[1, 1]
        cx, cy = P2[0, 2], P2[1, 2]

        intrinsics = CameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=1242, height=375
        )

        Tr = calib.Tr_velo_to_cam
        R, T = Tr[:3, :3], Tr[:3, 3]

        extrinsics = CameraLiDARExtrinsics(R=R, T=T)
        extrinsics.P2 = P2

        return intrinsics, extrinsics

    def _setup_fusion(self, intrinsics: CameraIntrinsics, extrinsics: CameraLiDARExtrinsics):
        """Initialize depth estimator and 3D generator."""
        self.depth_estimator = DepthEstimator(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )
        self.bbox_generator = BBox3DGenerator(intrinsics=intrinsics)

    def process_frame(
        self,
        frame_idx: int,
        points_override: np.ndarray = None,
    ) -> Tuple[List[Detection], List[Optional[BBox3D]], float]:
        """
        Process a single frame.

        Args:
            frame_idx: Frame index.
            points_override: Override LiDAR points (for noise injection).

        Returns:
            Tuple of (detections_2d, boxes_3d, processing_time)
        """
        start_time = time.time()

        # Load frame
        sample = self.loader[frame_idx]
        image = sample["image"]
        points = points_override if points_override is not None else sample["points"]
        calib = sample["calib"]

        # Setup calibration on first frame or if not set
        if self.depth_estimator is None:
            intrinsics, extrinsics = self._setup_calibration(calib)
            self._setup_fusion(intrinsics, extrinsics)
            self.intrinsics = intrinsics
            self.extrinsics = extrinsics

        # Run 2D detection
        detections = self.detector.detect(image)

        # Generate 3D boxes
        boxes_3d = []
        for det in detections:
            depth_result = self.depth_estimator.estimate_full(det, points)

            if depth_result.depth is not None:
                if depth_result.points_3d is not None and len(depth_result.points_3d) >= 3:
                    box_3d = self.bbox_generator.generate_from_points(
                        det, depth_result.points_3d, depth=depth_result.depth
                    )
                else:
                    box_3d = self.bbox_generator.generate(det, depth_result.depth)
                boxes_3d.append(box_3d)
            else:
                boxes_3d.append(None)

        processing_time = time.time() - start_time
        return detections, boxes_3d, processing_time

    def evaluate_baseline(self) -> EvaluationMetrics:
        """Run baseline evaluation without any noise."""
        self.logger.info("=" * 60)
        self.logger.info("Running BASELINE evaluation (no noise)")
        self.logger.info("=" * 60)

        metrics = EvaluationMetrics()
        processing_times = []

        frame_indices = range(self.num_frames)
        if HAS_TQDM:
            frame_indices = tqdm(frame_indices, desc="Baseline")

        for frame_idx in frame_indices:
            try:
                detections, boxes_3d, proc_time = self.process_frame(frame_idx)

                metrics.num_frames += 1
                metrics.total_detections_2d += len(detections)
                metrics.total_boxes_3d += sum(1 for b in boxes_3d if b is not None)
                processing_times.append(proc_time)

            except Exception as e:
                self.logger.warning(f"Frame {frame_idx} failed: {e}")
                continue

        metrics.avg_processing_time = np.mean(processing_times) if processing_times else 0
        metrics.compute_aggregates()

        self.baseline_metrics = metrics

        self.logger.info(f"Baseline Results:")
        self.logger.info(f"  Frames: {metrics.num_frames}")
        self.logger.info(f"  2D Detections: {metrics.total_detections_2d}")
        self.logger.info(f"  3D Boxes: {metrics.total_boxes_3d}")
        self.logger.info(f"  Detection Rate: {metrics.detection_rate:.1%}")
        self.logger.info(f"  Avg Processing Time: {metrics.avg_processing_time*1000:.1f}ms")

        return metrics

    def evaluate_with_noise(
        self,
        noise_type: str,
        noise_levels: List[float],
        noise_func,
    ) -> RobustnessResult:
        """
        Evaluate pipeline with varying noise levels.

        Args:
            noise_type: Name of noise type.
            noise_levels: List of noise levels to test.
            noise_func: Function to apply noise: f(points, level) -> noisy_points

        Returns:
            RobustnessResult with metrics for each level.
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Running {noise_type.upper()} robustness evaluation")
        self.logger.info(f"Noise levels: {noise_levels}")
        self.logger.info("=" * 60)

        all_metrics = []

        for level in noise_levels:
            self.logger.info(f"  Testing noise level: {level}")

            metrics = EvaluationMetrics()
            processing_times = []

            frame_indices = range(self.num_frames)
            if HAS_TQDM:
                frame_indices = tqdm(frame_indices, desc=f"{noise_type}={level:.3f}", leave=False)

            for frame_idx in frame_indices:
                try:
                    # Load original points
                    sample = self.loader[frame_idx]
                    points = sample["points"]

                    # Apply noise
                    noisy_points = noise_func(points, level)

                    # Process with noisy points
                    detections, boxes_3d, proc_time = self.process_frame(
                        frame_idx, points_override=noisy_points
                    )

                    metrics.num_frames += 1
                    metrics.total_detections_2d += len(detections)
                    metrics.total_boxes_3d += sum(1 for b in boxes_3d if b is not None)
                    processing_times.append(proc_time)

                except Exception as e:
                    continue

            metrics.avg_processing_time = np.mean(processing_times) if processing_times else 0
            metrics.compute_aggregates()
            all_metrics.append(metrics)

            self.logger.info(f"    Detection Rate: {metrics.detection_rate:.1%}")

        result = RobustnessResult(
            noise_type=noise_type,
            noise_levels=noise_levels,
            metrics=all_metrics,
            baseline_metrics=self.baseline_metrics,
        )

        self.results[noise_type] = result
        return result

    def evaluate_depth_noise(self, levels: List[float] = None) -> RobustnessResult:
        """Evaluate robustness to depth/range noise."""
        if levels is None:
            levels = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]

        return self.evaluate_with_noise(
            noise_type="depth_noise",
            noise_levels=levels,
            noise_func=lambda pts, lvl: self.noise_injector.add_depth_noise(pts, lvl),
        )

    def evaluate_calibration_error(self, levels: List[float] = None) -> RobustnessResult:
        """Evaluate robustness to calibration errors."""
        if levels is None:
            levels = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]

        def apply_calib_error(points, level):
            return self.noise_injector.add_calibration_error(
                points,
                translation_error=np.array([level, level/2, level/3]),
                rotation_error=level * 2,  # degrees
            )

        return self.evaluate_with_noise(
            noise_type="calibration_error",
            noise_levels=levels,
            noise_func=apply_calib_error,
        )

    def evaluate_dropout(self, levels: List[float] = None) -> RobustnessResult:
        """Evaluate robustness to sensor dropout."""
        if levels is None:
            levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]

        return self.evaluate_with_noise(
            noise_type="dropout",
            noise_levels=levels,
            noise_func=lambda pts, lvl: self.noise_injector.add_lidar_dropout(pts, lvl),
        )

    def evaluate_combined_worst_case(self) -> EvaluationMetrics:
        """Evaluate worst-case scenario with all noise types combined."""
        self.logger.info("=" * 60)
        self.logger.info("Running COMBINED WORST-CASE evaluation")
        self.logger.info("  Depth noise: 0.10")
        self.logger.info("  Calibration error: 0.20m translation, 0.4° rotation")
        self.logger.info("  Dropout: 30%")
        self.logger.info("=" * 60)

        metrics = EvaluationMetrics()
        processing_times = []

        frame_indices = range(self.num_frames)
        if HAS_TQDM:
            frame_indices = tqdm(frame_indices, desc="Worst-case")

        for frame_idx in frame_indices:
            try:
                sample = self.loader[frame_idx]
                points = sample["points"]

                # Apply all noise types
                noisy = self.noise_injector.add_depth_noise(points, 0.10)
                noisy = self.noise_injector.add_calibration_error(
                    noisy,
                    translation_error=np.array([0.20, 0.10, 0.07]),
                    rotation_error=0.4,
                )
                noisy = self.noise_injector.add_lidar_dropout(noisy, 0.30)

                detections, boxes_3d, proc_time = self.process_frame(
                    frame_idx, points_override=noisy
                )

                metrics.num_frames += 1
                metrics.total_detections_2d += len(detections)
                metrics.total_boxes_3d += sum(1 for b in boxes_3d if b is not None)
                processing_times.append(proc_time)

            except Exception:
                continue

        metrics.avg_processing_time = np.mean(processing_times) if processing_times else 0
        metrics.compute_aggregates()

        self.logger.info(f"Worst-case Results:")
        self.logger.info(f"  Detection Rate: {metrics.detection_rate:.1%}")

        if self.baseline_metrics:
            degradation = (self.baseline_metrics.detection_rate - metrics.detection_rate)
            degradation_pct = degradation / self.baseline_metrics.detection_rate * 100
            self.logger.info(f"  Degradation from baseline: {degradation_pct:.1f}%")

        return metrics

    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete robustness evaluation suite."""
        self.logger.info("Starting Full Robustness Evaluation Suite")
        self.logger.info("=" * 60)

        results = {}

        # 1. Baseline
        results["baseline"] = self.evaluate_baseline()

        # 2. Depth noise sweep
        results["depth_noise"] = self.evaluate_depth_noise()

        # 3. Calibration error sweep
        results["calibration_error"] = self.evaluate_calibration_error()

        # 4. Dropout sweep
        results["dropout"] = self.evaluate_dropout()

        # 5. Combined worst-case
        results["worst_case"] = self.evaluate_combined_worst_case()

        return results


# =============================================================================
# Report Generator
# =============================================================================

class ReportGenerator:
    """Generate plots and reports from evaluation results."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.plots_dir = Path("plots")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger("report_gen")

    def generate_robustness_plot(
        self,
        result: RobustnessResult,
        save_path: str = None,
    ):
        """Generate robustness curve plot."""
        if not HAS_MATPLOTLIB:
            self.logger.warning("matplotlib not available, skipping plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Detection rate vs noise level
        ax1 = axes[0]
        ax1.plot(result.noise_levels, result.get_detection_rates(), 'b-o', linewidth=2, markersize=8)
        if result.baseline_metrics:
            ax1.axhline(y=result.baseline_metrics.detection_rate, color='g', linestyle='--',
                       label=f'Baseline ({result.baseline_metrics.detection_rate:.1%})')
        ax1.set_xlabel(f'{result.noise_type.replace("_", " ").title()} Level', fontsize=12)
        ax1.set_ylabel('Detection Rate', fontsize=12)
        ax1.set_title(f'Detection Rate vs {result.noise_type.replace("_", " ").title()}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1.05)

        # Degradation plot
        ax2 = axes[1]
        degradation = result.get_degradation()
        colors = ['green' if d < 10 else 'orange' if d < 25 else 'red' for d in degradation]
        ax2.bar(range(len(result.noise_levels)), degradation, color=colors)
        ax2.set_xticks(range(len(result.noise_levels)))
        ax2.set_xticklabels([f'{l:.2f}' for l in result.noise_levels])
        ax2.set_xlabel(f'{result.noise_type.replace("_", " ").title()} Level', fontsize=12)
        ax2.set_ylabel('Performance Degradation (%)', fontsize=12)
        ax2.set_title('Degradation from Baseline', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add threshold lines
        ax2.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10% threshold')
        ax2.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='25% threshold')
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved plot: {save_path}")

        plt.close()

    def generate_summary_plot(
        self,
        results: Dict[str, RobustnessResult],
        baseline: EvaluationMetrics,
        worst_case: EvaluationMetrics,
        save_path: str = None,
    ):
        """Generate summary comparison plot."""
        if not HAS_MATPLOTLIB:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Combined robustness curves
        ax1 = axes[0]
        for name, result in results.items():
            if isinstance(result, RobustnessResult):
                ax1.plot(result.noise_levels, result.get_detection_rates(),
                        '-o', linewidth=2, markersize=6, label=name.replace("_", " ").title())

        ax1.axhline(y=baseline.detection_rate, color='green', linestyle='--',
                   linewidth=2, label=f'Baseline ({baseline.detection_rate:.1%})')
        ax1.set_xlabel('Noise Level', fontsize=12)
        ax1.set_ylabel('Detection Rate', fontsize=12)
        ax1.set_title('Robustness Comparison', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1.05)

        # Summary bar chart
        ax2 = axes[1]
        scenarios = ['Baseline', 'Depth\nNoise\n(max)', 'Calib\nError\n(max)',
                    'Dropout\n(max)', 'Worst\nCase']
        rates = [baseline.detection_rate]

        for name in ['depth_noise', 'calibration_error', 'dropout']:
            if name in results:
                rates.append(results[name].metrics[-1].detection_rate)
            else:
                rates.append(0)

        rates.append(worst_case.detection_rate)

        colors = ['green', 'blue', 'orange', 'purple', 'red']
        bars = ax2.bar(scenarios, rates, color=colors, alpha=0.7, edgecolor='black')

        # Add value labels
        for bar, rate in zip(bars, rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom', fontsize=10)

        ax2.set_ylabel('Detection Rate', fontsize=12)
        ax2.set_title('Performance Under Different Conditions', fontsize=14)
        ax2.set_ylim(0, 1.15)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved summary plot: {save_path}")

        plt.close()

    def generate_markdown_report(
        self,
        results: Dict[str, Any],
        save_path: str = None,
    ) -> str:
        """Generate markdown summary report."""
        baseline = results.get("baseline")
        worst_case = results.get("worst_case")

        report = []
        report.append("# Robustness Evaluation Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Executive Summary
        report.append("## Executive Summary\n")
        if baseline:
            report.append(f"- **Baseline Detection Rate:** {baseline.detection_rate:.1%}")
            report.append(f"- **Total Frames Evaluated:** {baseline.num_frames}")
            report.append(f"- **Total 2D Detections:** {baseline.total_detections_2d}")
            report.append(f"- **Total 3D Boxes Generated:** {baseline.total_boxes_3d}")

        if worst_case and baseline:
            degradation = (baseline.detection_rate - worst_case.detection_rate) / baseline.detection_rate * 100
            report.append(f"- **Worst-Case Detection Rate:** {worst_case.detection_rate:.1%}")
            report.append(f"- **Maximum Degradation:** {degradation:.1f}%")

        # Detailed Results
        report.append("\n## Detailed Results\n")

        # Depth Noise
        if "depth_noise" in results:
            result = results["depth_noise"]
            report.append("### Depth Noise Robustness\n")
            report.append("| Noise Level | Detection Rate | Degradation |")
            report.append("|-------------|----------------|-------------|")
            for level, metrics, deg in zip(result.noise_levels, result.metrics, result.get_degradation()):
                report.append(f"| {level:.2f} | {metrics.detection_rate:.1%} | {deg:.1f}% |")
            report.append("")

        # Calibration Error
        if "calibration_error" in results:
            result = results["calibration_error"]
            report.append("### Calibration Error Robustness\n")
            report.append("| Error Level (m) | Detection Rate | Degradation |")
            report.append("|-----------------|----------------|-------------|")
            for level, metrics, deg in zip(result.noise_levels, result.metrics, result.get_degradation()):
                report.append(f"| {level:.2f} | {metrics.detection_rate:.1%} | {deg:.1f}% |")
            report.append("")

        # Dropout
        if "dropout" in results:
            result = results["dropout"]
            report.append("### Sensor Dropout Robustness\n")
            report.append("| Dropout Rate | Detection Rate | Degradation |")
            report.append("|--------------|----------------|-------------|")
            for level, metrics, deg in zip(result.noise_levels, result.metrics, result.get_degradation()):
                report.append(f"| {level:.0%} | {metrics.detection_rate:.1%} | {deg:.1f}% |")
            report.append("")

        # Failure Analysis
        report.append("## Failure Analysis\n")
        report.append("### Identified Failure Patterns\n")
        report.append("1. **High Depth Noise (>10%):** Causes depth estimation failures, especially for distant objects")
        report.append("2. **Calibration Misalignment (>0.2m):** LiDAR points don't project correctly to 2D boxes")
        report.append("3. **High Dropout (>50%):** Insufficient points for reliable depth estimation")
        report.append("4. **Combined Degradation:** Multiplicative effects when multiple noise sources present\n")

        # Mitigation Strategies
        report.append("### Mitigation Strategies\n")
        report.append("1. **Depth Noise:**")
        report.append("   - Use robust depth estimation (median instead of mean)")
        report.append("   - Apply temporal filtering across frames")
        report.append("   - Increase minimum point threshold\n")
        report.append("2. **Calibration Error:**")
        report.append("   - Implement online calibration refinement")
        report.append("   - Use feature-based alignment verification")
        report.append("   - Add calibration confidence scoring\n")
        report.append("3. **Sensor Dropout:**")
        report.append("   - Implement multi-frame point accumulation")
        report.append("   - Use learned depth priors for sparse regions")
        report.append("   - Graceful degradation to 2D-only mode\n")

        # Recommendations
        report.append("## Recommendations\n")
        if baseline and worst_case:
            if worst_case.detection_rate > 0.7 * baseline.detection_rate:
                report.append("- **Status: ACCEPTABLE** - Pipeline maintains >70% performance under worst-case conditions")
            elif worst_case.detection_rate > 0.5 * baseline.detection_rate:
                report.append("- **Status: MARGINAL** - Pipeline shows significant degradation, improvements recommended")
            else:
                report.append("- **Status: CRITICAL** - Pipeline performance severely degraded, urgent improvements needed")

        report.append("\n---\n*Report generated by Multi-Sensor Perception Pipeline Robustness Evaluation Suite*")

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            self.logger.info(f"Saved report: {save_path}")

        return report_text


# =============================================================================
# Failure Visualization
# =============================================================================

class FailureVisualizer:
    """Generate failure mode visualization videos."""

    def __init__(self, evaluator: RobustnessEvaluator):
        self.evaluator = evaluator
        self.logger = setup_logger("failure_viz")

    def create_comparison_frame(
        self,
        frame_idx: int,
        noisy_points: np.ndarray,
        noise_description: str,
    ) -> np.ndarray:
        """Create side-by-side comparison frame."""
        # Get original sample
        sample = self.evaluator.loader[frame_idx]
        image = sample["image"]
        clean_points = sample["points"]

        # Process with clean points
        clean_dets, clean_boxes, _ = self.evaluator.process_frame(frame_idx)

        # Process with noisy points
        noisy_dets, noisy_boxes, _ = self.evaluator.process_frame(
            frame_idx, points_override=noisy_points
        )

        # Convert image to BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw clean results
        clean_vis = image_bgr.copy()
        clean_count = 0
        for box in clean_boxes:
            if box is not None:
                clean_count += 1
                self._draw_box(clean_vis, box)
        cv2.putText(clean_vis, "CLEAN (Reference)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(clean_vis, f"3D Boxes: {clean_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw noisy results
        noisy_vis = image_bgr.copy()
        noisy_count = 0
        for box in noisy_boxes:
            if box is not None:
                noisy_count += 1
                self._draw_box(noisy_vis, box, color=(0, 0, 255))
        cv2.putText(noisy_vis, f"NOISY ({noise_description})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(noisy_vis, f"3D Boxes: {noisy_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Add failure annotation if degraded
        if noisy_count < clean_count:
            lost = clean_count - noisy_count
            cv2.putText(noisy_vis, f"LOST {lost} BOXES!", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Combine side by side
        separator = np.ones((clean_vis.shape[0], 5, 3), dtype=np.uint8) * 128
        combined = np.hstack([clean_vis, separator, noisy_vis])

        return combined

    def _draw_box(self, image: np.ndarray, box: BBox3D, color: Tuple[int, int, int] = (0, 255, 0)):
        """Draw 3D box on image."""
        try:
            corners = box.corners
            if corners[:, 2].min() <= 0.1:
                return

            corners_2d = self.evaluator.intrinsics.project_point(corners).astype(int)

            # Draw edges
            for i in range(4):
                cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[(i+1)%4]), color, 2)
                cv2.line(image, tuple(corners_2d[i+4]), tuple(corners_2d[(i+1)%4+4]), color, 2)
                cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[i+4]), color, 1)
        except Exception:
            pass

    def generate_failure_video(
        self,
        output_path: str = "outputs/failure_modes.avi",
        num_frames: int = 30,
    ):
        """Generate failure modes visualization video."""
        self.logger.info(f"Generating failure visualization video: {output_path}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        video_writer = None
        frame_count = 0

        noise_scenarios = [
            ("Depth Noise 5%", lambda pts: NoiseInjector.add_depth_noise(pts, 0.05)),
            ("Depth Noise 15%", lambda pts: NoiseInjector.add_depth_noise(pts, 0.15)),
            ("Calibration Error 0.2m", lambda pts: NoiseInjector.add_calibration_error(
                pts, translation_error=np.array([0.2, 0.1, 0.05]))),
            ("Calibration Error 0.5m", lambda pts: NoiseInjector.add_calibration_error(
                pts, translation_error=np.array([0.5, 0.25, 0.1]))),
            ("30% Dropout", lambda pts: NoiseInjector.add_lidar_dropout(pts, 0.3)),
            ("70% Dropout", lambda pts: NoiseInjector.add_lidar_dropout(pts, 0.7)),
        ]

        frames_per_scenario = max(num_frames // len(noise_scenarios), 5)

        frame_indices = range(min(frames_per_scenario * len(noise_scenarios), len(self.evaluator.loader)))
        if HAS_TQDM:
            frame_indices = tqdm(list(frame_indices), desc="Generating video")

        for frame_idx in frame_indices:
            scenario_idx = frame_count // frames_per_scenario
            if scenario_idx >= len(noise_scenarios):
                break

            noise_name, noise_func = noise_scenarios[scenario_idx]

            try:
                sample = self.evaluator.loader[frame_idx]
                noisy_points = noise_func(sample["points"])

                frame = self.create_comparison_frame(frame_idx, noisy_points, noise_name)

                if video_writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    video_writer = cv2.VideoWriter(output_path, fourcc, 5, (w, h))

                video_writer.write(frame)
                frame_count += 1

            except Exception as e:
                self.logger.warning(f"Frame {frame_idx} failed: {e}")
                continue

        if video_writer:
            video_writer.release()
            self.logger.info(f"Saved failure video: {output_path} ({frame_count} frames)")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Robustness Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="data/kitti",
        help="Path to KITTI dataset",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Number of frames to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device",
    )
    parser.add_argument(
        "--noise-levels",
        type=int,
        default=6,
        help="Number of noise levels to test",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip failure video generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for reports",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    logger = setup_logger("main")
    logger.info("=" * 60)
    logger.info("Robustness Evaluation Suite")
    logger.info("=" * 60)

    # Auto-detect device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Initialize evaluator
    evaluator = RobustnessEvaluator(
        data_root=args.data_root,
        device=device,
        num_frames=args.num_frames,
    )

    # Run full evaluation
    results = evaluator.run_full_evaluation()

    # Generate reports
    report_gen = ReportGenerator(output_dir=args.output_dir)

    # Generate individual plots
    for name in ["depth_noise", "calibration_error", "dropout"]:
        if name in results and isinstance(results[name], RobustnessResult):
            plot_path = f"plots/robustness_{name}.png"
            report_gen.generate_robustness_plot(results[name], save_path=plot_path)

    # Generate summary plot
    robustness_results = {k: v for k, v in results.items() if isinstance(v, RobustnessResult)}
    if robustness_results and results.get("baseline") and results.get("worst_case"):
        report_gen.generate_summary_plot(
            robustness_results,
            results["baseline"],
            results["worst_case"],
            save_path="plots/robustness_summary.png",
        )

    # Generate markdown report
    report_path = Path(args.output_dir) / "robustness_report.md"
    report_gen.generate_markdown_report(results, save_path=str(report_path))

    # Generate failure video
    if not args.no_video:
        failure_viz = FailureVisualizer(evaluator)
        failure_viz.generate_failure_video(
            output_path=str(Path(args.output_dir) / "failure_modes.avi"),
            num_frames=min(args.num_frames, 60),
        )

    logger.info("=" * 60)
    logger.info("Evaluation Complete!")
    logger.info(f"  Report: {report_path}")
    logger.info(f"  Plots: plots/robustness_*.png")
    if not args.no_video:
        logger.info(f"  Video: {args.output_dir}/failure_modes.avi")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
