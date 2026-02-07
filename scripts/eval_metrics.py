#!/usr/bin/env python3
"""
Comprehensive Metrics Evaluation for Multi-Sensor Perception Pipeline.

This script evaluates the perception pipeline against KITTI ground truth labels
and generates detailed metrics reports with visualizations.

Outputs:
- outputs/metrics_report.md: Comprehensive metrics report
- plots/metrics_*.png: Visualization plots
  - Error distribution histograms
  - Per-distance-bin accuracy
  - Confusion matrices
  - PR curves

Usage:
    # Run evaluation on 100 frames
    python scripts/eval_metrics.py --num-frames 100

    # Full evaluation
    python scripts/eval_metrics.py --num-frames 200

    # Quick test
    python scripts/eval_metrics.py --num-frames 20 --device cpu
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
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
from src.perception2d.detector import ObjectDetector2D
from src.fusion.depth_estimator import DepthEstimator
from src.fusion.bbox3d_generator import BBox3DGenerator
from src.calibration.intrinsics import CameraIntrinsics
from src.calibration.extrinsics import CameraLiDARExtrinsics
from src.eval.metrics import (
    MetricsCalculator, Metrics2DCalculator, Metrics3DCalculator,
    Detection2D, Detection3D, GroundTruth2D, GroundTruth3D,
    parse_kitti_label, compute_iou_2d, compute_iou_3d,
)

# Optional imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_pdf import PdfPages
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
# Metrics Evaluator
# =============================================================================

class MetricsEvaluator:
    """
    Comprehensive metrics evaluator for perception pipeline.

    Evaluates 2D and 3D detection against KITTI ground truth.
    """

    def __init__(
        self,
        data_root: str = "data/kitti",
        device: str = "cuda",
        num_frames: int = 100,
    ):
        """
        Initialize metrics evaluator.

        Args:
            data_root: Path to KITTI dataset.
            device: Inference device.
            num_frames: Number of frames to evaluate.
        """
        self.logger = setup_logger("metrics_eval")
        self.device = device
        self.num_frames = num_frames
        self.data_root = Path(data_root)

        # Load dataset
        self.logger.info(f"Loading KITTI dataset from {data_root}")
        self.loader = KITTILoader(root_dir=data_root, split="training")
        self.num_frames = min(num_frames, len(self.loader))
        self.logger.info(f"Using {self.num_frames} frames for evaluation")

        # Label directory
        self.label_dir = self.data_root / "training" / "label_2"

        # Initialize detector
        self.logger.info("Initializing detector")
        self.detector = ObjectDetector2D(
            model_name="yolov8m",
            device=device,
            confidence_threshold=0.3,
        )

        # Placeholders for fusion components
        self.depth_estimator = None
        self.bbox_generator = None
        self.intrinsics = None
        self.extrinsics = None

        # Metrics calculators
        self.calc_2d = Metrics2DCalculator(iou_threshold=0.5)
        self.calc_3d = Metrics3DCalculator(iou_threshold=0.5, distance_threshold=2.0)

        # Raw data storage for visualization
        self.all_center_errors: List[float] = []
        self.all_orientation_errors: List[float] = []
        self.all_dimension_errors: List[np.ndarray] = []
        self.all_iou_3d: List[float] = []
        self.all_distances: List[float] = []
        self.all_classes: List[str] = []

        # Confusion matrix data
        self.class_predictions: List[str] = []
        self.class_ground_truths: List[str] = []

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

    def process_frame(self, frame_idx: int) -> Tuple[List, List, List, List]:
        """
        Process a single frame and get detections + ground truths.

        Returns:
            Tuple of (detections_2d, detections_3d, gt_2d, gt_3d)
        """
        # Load frame
        sample = self.loader[frame_idx]
        image = sample["image"]
        points = sample["points"]
        calib = sample["calib"]
        frame_id = sample.get("frame_id", f"{frame_idx:06d}")

        # Setup calibration
        if self.depth_estimator is None:
            intrinsics, extrinsics = self._setup_calibration(calib)
            self._setup_fusion(intrinsics, extrinsics)
            self.intrinsics = intrinsics
            self.extrinsics = extrinsics

        # Run 2D detection
        raw_detections = self.detector.detect(image)

        # Convert to metrics format
        detections_2d = []
        for det in raw_detections:
            detections_2d.append(Detection2D(
                bbox=det.bbox,
                score=det.confidence,
                class_name=det.class_name,
                class_id=getattr(det, 'class_id', 0),
            ))

        # Generate 3D boxes
        detections_3d = []
        for det in raw_detections:
            depth_result = self.depth_estimator.estimate_full(det, points)

            if depth_result.depth is not None:
                if depth_result.points_3d is not None and len(depth_result.points_3d) >= 3:
                    box_3d = self.bbox_generator.generate_from_points(
                        det, depth_result.points_3d, depth=depth_result.depth
                    )
                else:
                    box_3d = self.bbox_generator.generate(det, depth_result.depth)

                if box_3d is not None:
                    detections_3d.append(Detection3D(
                        center=box_3d.center,
                        dimensions=box_3d.dimensions,
                        rotation_y=box_3d.rotation_y,
                        score=det.confidence,
                        class_name=det.class_name,
                    ))

        # Load ground truth
        label_path = self.label_dir / f"{frame_id}.txt"
        gt_2d, gt_3d = parse_kitti_label(str(label_path))

        return detections_2d, detections_3d, gt_2d, gt_3d

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run full evaluation on dataset.

        Returns:
            Dictionary with all metrics.
        """
        self.logger.info("=" * 60)
        self.logger.info("Running Metrics Evaluation")
        self.logger.info("=" * 60)

        # Reset calculators
        self.calc_2d.reset()
        self.calc_3d.reset()
        self.all_center_errors = []
        self.all_orientation_errors = []
        self.all_dimension_errors = []
        self.all_iou_3d = []
        self.all_distances = []
        self.all_classes = []

        frame_indices = range(self.num_frames)
        if HAS_TQDM:
            frame_indices = tqdm(frame_indices, desc="Evaluating")

        for frame_idx in frame_indices:
            try:
                det_2d, det_3d, gt_2d, gt_3d = self.process_frame(frame_idx)

                # Add to 2D calculator
                self.calc_2d.add_frame(det_2d, gt_2d)

                # Add to 3D calculator
                self.calc_3d.add_frame(det_3d, gt_3d)

                # Collect raw data for visualization
                self._collect_raw_data(det_3d, gt_3d)

            except Exception as e:
                self.logger.warning(f"Frame {frame_idx} failed: {e}")
                continue

        # Compute final metrics
        metrics_2d = self.calc_2d.compute_metrics()
        metrics_3d = self.calc_3d.compute_metrics()

        # Log summary
        self._log_summary(metrics_2d, metrics_3d)

        return {
            "2d": metrics_2d,
            "3d": metrics_3d,
            "raw_data": {
                "center_errors": self.all_center_errors,
                "orientation_errors": self.all_orientation_errors,
                "dimension_errors": self.all_dimension_errors,
                "iou_3d": self.all_iou_3d,
                "distances": self.all_distances,
                "classes": self.all_classes,
            }
        }

    def _collect_raw_data(self, det_3d: List[Detection3D], gt_3d: List[GroundTruth3D]):
        """Collect raw data for visualization."""
        # Match detections to GTs
        matches, _, _ = self.calc_3d.match_detections(det_3d, gt_3d)

        for match in matches:
            det = det_3d[match.det_idx]
            gt = gt_3d[match.gt_idx]

            # Center error
            center_error = np.linalg.norm(det.center - gt.center)
            self.all_center_errors.append(center_error)

            # Orientation error
            orient_diff = det.rotation_y - gt.rotation_y
            orient_diff = (orient_diff + np.pi) % (2 * np.pi) - np.pi
            self.all_orientation_errors.append(abs(orient_diff))

            # Dimension error
            dim_error = np.abs(det.dimensions - gt.dimensions)
            self.all_dimension_errors.append(dim_error)

            # 3D IoU
            iou = compute_iou_3d(
                det.center, det.dimensions, det.rotation_y,
                gt.center, gt.dimensions, gt.rotation_y
            )
            self.all_iou_3d.append(iou)

            # Distance
            distance = np.linalg.norm(gt.center)
            self.all_distances.append(distance)

            # Class
            self.all_classes.append(det.class_name)

    def _log_summary(self, metrics_2d: Dict, metrics_3d):
        """Log metrics summary."""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("=" * 60)

        # 2D Metrics
        self.logger.info("\n2D Detection Metrics:")
        self.logger.info(f"  mAP@0.5: {metrics_2d['overall']['mAP']:.3f}")
        self.logger.info(f"  Precision: {metrics_2d['overall']['precision']:.3f}")
        self.logger.info(f"  Recall: {metrics_2d['overall']['recall']:.3f}")
        self.logger.info(f"  F1: {metrics_2d['overall']['f1']:.3f}")

        self.logger.info("\n  Per-Class AP:")
        for cls in ["Car", "Pedestrian", "Cyclist"]:
            if cls in metrics_2d:
                self.logger.info(f"    {cls}: {metrics_2d[cls]['ap']:.3f}")

        # 3D Metrics
        self.logger.info("\n3D Detection Metrics:")
        self.logger.info(f"  Precision: {metrics_3d.precision_2d:.3f}")
        self.logger.info(f"  Recall: {metrics_3d.recall_2d:.3f}")
        self.logger.info(f"  Mean Center Error: {metrics_3d.mean_center_error:.2f}m")
        self.logger.info(f"  Median Center Error: {metrics_3d.median_center_error:.2f}m")
        self.logger.info(f"  Mean Orientation Error: {np.degrees(metrics_3d.mean_orientation_error):.1f}°")
        self.logger.info(f"  Mean 3D IoU: {metrics_3d.mean_iou_3d:.3f}")

        if metrics_3d.mean_dimension_error is not None and len(metrics_3d.mean_dimension_error) == 3:
            self.logger.info(f"  Mean Dimension Error (L/W/H): {metrics_3d.mean_dimension_error[0]:.2f}m / "
                           f"{metrics_3d.mean_dimension_error[1]:.2f}m / {metrics_3d.mean_dimension_error[2]:.2f}m")

        # Distance-binned metrics
        self.logger.info("\n  Accuracy by Distance:")
        for bin_name, bin_metrics in metrics_3d.metrics_by_distance.items():
            self.logger.info(f"    {bin_name}: IoU={bin_metrics['mean_iou_3d']:.3f}, "
                           f"CenterErr={bin_metrics['mean_center_error']:.2f}m "
                           f"(n={bin_metrics['count']})")

        self.logger.info("=" * 60)


# =============================================================================
# Report Generator
# =============================================================================

class ReportGenerator:
    """Generate comprehensive metrics report with visualizations."""

    def __init__(self, output_dir: str = "outputs", plots_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.plots_dir = Path(plots_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger("report_gen")

    def generate_all(self, results: Dict[str, Any]):
        """Generate all reports and plots."""
        # Generate plots
        if HAS_MATPLOTLIB:
            self.plot_error_distributions(results)
            self.plot_per_distance_accuracy(results)
            self.plot_confusion_matrix(results)
            self.plot_pr_curves(results)
            self.plot_iou_distribution(results)

        # Generate markdown report
        self.generate_markdown_report(results)

    def plot_error_distributions(self, results: Dict[str, Any]):
        """Plot error distribution histograms."""
        if not HAS_MATPLOTLIB:
            return

        raw = results.get("raw_data", {})

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Center Error Distribution
        ax = axes[0, 0]
        center_errors = raw.get("center_errors", [])
        if center_errors:
            ax.hist(center_errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(center_errors), color='red', linestyle='--',
                      label=f'Mean: {np.mean(center_errors):.2f}m')
            ax.axvline(np.median(center_errors), color='orange', linestyle='--',
                      label=f'Median: {np.median(center_errors):.2f}m')
        ax.set_xlabel('Center Distance Error (m)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Center Distance Error Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Orientation Error Distribution
        ax = axes[0, 1]
        orient_errors = raw.get("orientation_errors", [])
        if orient_errors:
            orient_errors_deg = np.degrees(orient_errors)
            ax.hist(orient_errors_deg, bins=50, color='coral', edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(orient_errors_deg), color='red', linestyle='--',
                      label=f'Mean: {np.mean(orient_errors_deg):.1f}°')
        ax.set_xlabel('Orientation Error (degrees)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Orientation Error Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Dimension Error Distribution
        ax = axes[1, 0]
        dim_errors = raw.get("dimension_errors", [])
        if dim_errors:
            dim_errors = np.array(dim_errors)
            if len(dim_errors) > 0:
                labels = ['Length', 'Width', 'Height']
                colors = ['green', 'blue', 'purple']
                for i, (label, color) in enumerate(zip(labels, colors)):
                    ax.hist(dim_errors[:, i], bins=30, alpha=0.5, label=label, color=color)
        ax.set_xlabel('Dimension Error (m)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Dimension Error Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3D IoU Distribution
        ax = axes[1, 1]
        iou_values = raw.get("iou_3d", [])
        if iou_values:
            ax.hist(iou_values, bins=50, color='teal', edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(iou_values), color='red', linestyle='--',
                      label=f'Mean: {np.mean(iou_values):.3f}')
            ax.axvline(0.5, color='green', linestyle='-', linewidth=2, label='Threshold (0.5)')
        ax.set_xlabel('3D IoU', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('3D IoU Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.plots_dir / "metrics_error_distributions.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved: {save_path}")
        plt.close()

    def plot_per_distance_accuracy(self, results: Dict[str, Any]):
        """Plot accuracy metrics by distance bins."""
        if not HAS_MATPLOTLIB:
            return

        raw = results.get("raw_data", {})
        distances = raw.get("distances", [])
        center_errors = raw.get("center_errors", [])
        iou_values = raw.get("iou_3d", [])

        if not distances or not center_errors:
            return

        # Define distance bins
        bins = [(0, 20), (20, 40), (40, 60), (60, 80)]
        bin_names = ['0-20m', '20-40m', '40-60m', '60-80m']

        # Compute metrics per bin
        bin_center_errors = []
        bin_ious = []
        bin_counts = []

        for bin_min, bin_max in bins:
            mask = [(bin_min <= d < bin_max) for d in distances]
            bin_ce = [e for e, m in zip(center_errors, mask) if m]
            bin_iou = [i for i, m in zip(iou_values, mask) if m]

            bin_center_errors.append(np.mean(bin_ce) if bin_ce else 0)
            bin_ious.append(np.mean(bin_iou) if bin_iou else 0)
            bin_counts.append(sum(mask))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Center Error by Distance
        ax = axes[0]
        bars = ax.bar(bin_names, bin_center_errors, color='steelblue', edgecolor='black')
        ax.set_xlabel('Distance Range', fontsize=12)
        ax.set_ylabel('Mean Center Error (m)', fontsize=12)
        ax.set_title('Center Error by Distance', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, bin_center_errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{val:.2f}', ha='center', fontsize=10)

        # IoU by Distance
        ax = axes[1]
        colors = ['green' if iou >= 0.5 else 'orange' if iou >= 0.3 else 'red' for iou in bin_ious]
        bars = ax.bar(bin_names, bin_ious, color=colors, edgecolor='black')
        ax.axhline(y=0.5, color='green', linestyle='--', label='Good (0.5)')
        ax.axhline(y=0.3, color='orange', linestyle='--', label='Acceptable (0.3)')
        ax.set_xlabel('Distance Range', fontsize=12)
        ax.set_ylabel('Mean 3D IoU', fontsize=12)
        ax.set_title('3D IoU by Distance', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, bin_ious):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', fontsize=10)

        # Sample Count by Distance
        ax = axes[2]
        bars = ax.bar(bin_names, bin_counts, color='gray', edgecolor='black')
        ax.set_xlabel('Distance Range', fontsize=12)
        ax.set_ylabel('Sample Count', fontsize=12)
        ax.set_title('Detection Count by Distance', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, bin_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val}', ha='center', fontsize=10)

        plt.tight_layout()
        save_path = self.plots_dir / "metrics_per_distance.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved: {save_path}")
        plt.close()

    def plot_confusion_matrix(self, results: Dict[str, Any]):
        """Plot confusion matrix for class predictions."""
        if not HAS_MATPLOTLIB:
            return

        metrics_2d = results.get("2d", {})
        classes = ["Car", "Pedestrian", "Cyclist"]

        # Build confusion-like data from per-class metrics
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create metrics matrix
        metrics_matrix = np.zeros((len(classes), 4))
        labels = ['TP', 'FP', 'FN', 'Precision']

        for i, cls in enumerate(classes):
            if cls in metrics_2d:
                m = metrics_2d[cls]
                metrics_matrix[i, 0] = m.get('num_tp', 0)
                metrics_matrix[i, 1] = m.get('num_fp', 0)
                metrics_matrix[i, 2] = m.get('num_gt', 0) - m.get('num_tp', 0)  # FN
                metrics_matrix[i, 3] = m.get('precision', 0) * 100

        # Plot as heatmap
        im = ax.imshow(metrics_matrix[:, :3], cmap='Blues', aspect='auto')

        # Labels
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(['True Positives', 'False Positives', 'False Negatives'])
        ax.set_yticklabels(classes)

        # Add text annotations
        for i in range(len(classes)):
            for j in range(3):
                text = ax.text(j, i, f'{int(metrics_matrix[i, j])}',
                              ha='center', va='center', color='black', fontsize=12)

        ax.set_title('Detection Results by Class', fontsize=14)
        plt.colorbar(im, ax=ax, label='Count')

        plt.tight_layout()
        save_path = self.plots_dir / "metrics_confusion_matrix.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved: {save_path}")
        plt.close()

    def plot_pr_curves(self, results: Dict[str, Any]):
        """Plot Precision-Recall curves."""
        if not HAS_MATPLOTLIB:
            return

        metrics_2d = results.get("2d", {})
        classes = ["Car", "Pedestrian", "Cyclist"]

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = {'Car': 'green', 'Pedestrian': 'red', 'Cyclist': 'blue'}

        for cls in classes:
            if cls in metrics_2d:
                m = metrics_2d[cls]
                precision = m.get('precision', 0)
                recall = m.get('recall', 0)
                ap = m.get('ap', 0)

                # Plot single point (precision, recall) since we don't have full PR curve
                ax.scatter([recall], [precision], s=200, c=colors[cls],
                          label=f'{cls} (AP={ap:.3f})', marker='o', edgecolors='black')

                # Add text label
                ax.annotate(f'{cls}\nP={precision:.2f}\nR={recall:.2f}',
                           (recall, precision), textcoords="offset points",
                           xytext=(10, 10), fontsize=10)

        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')

        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title('Precision vs Recall by Class', fontsize=16)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower left', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add overall mAP
        mAP = metrics_2d.get('overall', {}).get('mAP', 0)
        ax.text(0.95, 0.95, f'mAP@0.5 = {mAP:.3f}', transform=ax.transAxes,
               fontsize=14, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        save_path = self.plots_dir / "metrics_pr_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved: {save_path}")
        plt.close()

    def plot_iou_distribution(self, results: Dict[str, Any]):
        """Plot IoU distribution with thresholds."""
        if not HAS_MATPLOTLIB:
            return

        raw = results.get("raw_data", {})
        iou_values = raw.get("iou_3d", [])
        classes = raw.get("classes", [])

        if not iou_values:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Overall IoU distribution with thresholds
        ax = axes[0]
        ax.hist(iou_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)

        # Threshold lines
        thresholds = [(0.3, 'Easy', 'green'), (0.5, 'Moderate', 'orange'), (0.7, 'Hard', 'red')]
        for thresh, label, color in thresholds:
            above = sum(1 for v in iou_values if v >= thresh)
            pct = above / len(iou_values) * 100 if iou_values else 0
            ax.axvline(thresh, color=color, linestyle='--', linewidth=2,
                      label=f'{label} (≥{thresh}): {pct:.1f}%')

        ax.set_xlabel('3D IoU', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('3D IoU Distribution with Thresholds', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # IoU by class
        ax = axes[1]
        class_ious = {}
        for cls, iou in zip(classes, iou_values):
            if cls not in class_ious:
                class_ious[cls] = []
            class_ious[cls].append(iou)

        if class_ious:
            class_names = list(class_ious.keys())
            class_means = [np.mean(class_ious[c]) for c in class_names]
            colors = ['green', 'red', 'blue', 'orange', 'purple'][:len(class_names)]

            bars = ax.bar(class_names, class_means, color=colors, edgecolor='black', alpha=0.7)
            ax.axhline(0.5, color='black', linestyle='--', label='Threshold (0.5)')

            for bar, val in zip(bars, class_means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.3f}', ha='center', fontsize=11)

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Mean 3D IoU', fontsize=12)
        ax.set_title('Mean 3D IoU by Class', fontsize=14)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = self.plots_dir / "metrics_iou_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved: {save_path}")
        plt.close()

    def generate_markdown_report(self, results: Dict[str, Any]):
        """Generate comprehensive markdown report."""
        metrics_2d = results.get("2d", {})
        metrics_3d = results.get("3d")
        raw = results.get("raw_data", {})

        report = []
        report.append("# Perception Pipeline Metrics Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Executive Summary
        report.append("## Executive Summary\n")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| mAP@0.5 (2D) | {metrics_2d.get('overall', {}).get('mAP', 0):.3f} |")
        report.append(f"| 2D Precision | {metrics_2d.get('overall', {}).get('precision', 0):.3f} |")
        report.append(f"| 2D Recall | {metrics_2d.get('overall', {}).get('recall', 0):.3f} |")
        if metrics_3d:
            report.append(f"| 3D Precision | {metrics_3d.precision_2d:.3f} |")
            report.append(f"| 3D Recall | {metrics_3d.recall_2d:.3f} |")
            report.append(f"| Mean Center Error | {metrics_3d.mean_center_error:.2f}m |")
            report.append(f"| Mean 3D IoU | {metrics_3d.mean_iou_3d:.3f} |")
        report.append("")

        # 2D Detection Metrics
        report.append("## 2D Detection Metrics\n")
        report.append("### Overall Performance\n")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        overall = metrics_2d.get('overall', {})
        report.append(f"| mAP@0.5 | {overall.get('mAP', 0):.3f} |")
        report.append(f"| Precision | {overall.get('precision', 0):.3f} |")
        report.append(f"| Recall | {overall.get('recall', 0):.3f} |")
        report.append(f"| F1 Score | {overall.get('f1', 0):.3f} |")
        report.append("")

        report.append("### Per-Class Performance\n")
        report.append("| Class | AP | Precision | Recall | F1 | TP | FP | GT |")
        report.append("|-------|-----|-----------|--------|-----|-----|-----|-----|")
        for cls in ["Car", "Pedestrian", "Cyclist"]:
            if cls in metrics_2d:
                m = metrics_2d[cls]
                report.append(f"| {cls} | {m.get('ap', 0):.3f} | {m.get('precision', 0):.3f} | "
                            f"{m.get('recall', 0):.3f} | {m.get('f1', 0):.3f} | "
                            f"{m.get('num_tp', 0)} | {m.get('num_fp', 0)} | {m.get('num_gt', 0)} |")
        report.append("")

        # 3D Detection Metrics
        report.append("## 3D Detection Metrics\n")
        if metrics_3d:
            report.append("### Error Metrics\n")
            report.append("| Metric | Value |")
            report.append("|--------|-------|")
            report.append(f"| Mean Center Error | {metrics_3d.mean_center_error:.3f}m |")
            report.append(f"| Median Center Error | {metrics_3d.median_center_error:.3f}m |")
            report.append(f"| Mean Orientation Error | {np.degrees(metrics_3d.mean_orientation_error):.1f}° |")
            report.append(f"| Mean 3D IoU | {metrics_3d.mean_iou_3d:.3f} |")
            if metrics_3d.mean_dimension_error is not None and len(metrics_3d.mean_dimension_error) == 3:
                report.append(f"| Dimension Error (L) | {metrics_3d.mean_dimension_error[0]:.3f}m |")
                report.append(f"| Dimension Error (W) | {metrics_3d.mean_dimension_error[1]:.3f}m |")
                report.append(f"| Dimension Error (H) | {metrics_3d.mean_dimension_error[2]:.3f}m |")
            report.append("")

            report.append("### Accuracy by Distance\n")
            report.append("| Distance | Count | Mean Center Error | Mean 3D IoU |")
            report.append("|----------|-------|-------------------|-------------|")
            for bin_name, bin_metrics in metrics_3d.metrics_by_distance.items():
                report.append(f"| {bin_name} | {bin_metrics['count']} | "
                            f"{bin_metrics['mean_center_error']:.3f}m | "
                            f"{bin_metrics['mean_iou_3d']:.3f} |")
            report.append("")

        # Metric Definitions
        report.append("## Metric Definitions\n")
        report.append("### 2D Metrics\n")
        report.append("- **Precision**: TP / (TP + FP) - Fraction of detections that are correct")
        report.append("- **Recall**: TP / (TP + FN) - Fraction of ground truths that are detected")
        report.append("- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5")
        report.append("- **AP**: Area under the Precision-Recall curve\n")

        report.append("### 3D Metrics\n")
        report.append("- **Center Distance Error**: Euclidean distance |predicted_center - GT_center| in meters")
        report.append("- **Orientation Error**: Angular difference in yaw, wrapped to [0, π]")
        report.append("- **3D IoU**: Intersection over Union of 3D bounding boxes")
        report.append("- **Dimension Error**: |predicted_dimensions - GT_dimensions| for L/W/H\n")

        report.append("### Distance Bins\n")
        report.append("- **Near (0-20m)**: Close-range objects, typically highest accuracy")
        report.append("- **Medium (20-40m)**: Mid-range objects")
        report.append("- **Far (40m+)**: Distant objects, challenging due to sparse LiDAR points\n")

        # Interpretation Guide
        report.append("## Interpretation Guide\n")
        report.append("### Quality Thresholds\n")
        report.append("| Metric | Good | Acceptable | Poor |")
        report.append("|--------|------|------------|------|")
        report.append("| mAP@0.5 | > 0.7 | 0.5-0.7 | < 0.5 |")
        report.append("| 3D IoU | > 0.5 | 0.3-0.5 | < 0.3 |")
        report.append("| Center Error | < 1m | 1-2m | > 2m |")
        report.append("| Orientation Error | < 10° | 10-30° | > 30° |")
        report.append("")

        # Save report
        report_text = "\n".join(report)
        report_path = self.output_dir / "metrics_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        self.logger.info(f"Saved report: {report_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Metrics Evaluation",
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
    logger.info("Comprehensive Metrics Evaluation")
    logger.info("=" * 60)

    # Auto-detect device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Initialize evaluator
    evaluator = MetricsEvaluator(
        data_root=args.data_root,
        device=device,
        num_frames=args.num_frames,
    )

    # Run evaluation
    results = evaluator.run_evaluation()

    # Generate reports and plots
    report_gen = ReportGenerator(output_dir=args.output_dir)
    report_gen.generate_all(results)

    logger.info("=" * 60)
    logger.info("Evaluation Complete!")
    logger.info(f"  Report: {args.output_dir}/metrics_report.md")
    logger.info(f"  Plots: plots/metrics_*.png")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
