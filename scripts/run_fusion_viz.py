#!/usr/bin/env python3
"""
Multi-Sensor Fusion Visualization Pipeline.

This script runs the complete perception pipeline with professional
visualization inspired by Waymo/Tesla perception displays.

Features:
- Side-by-side comparison: 2D detection vs 3D fusion
- Bird's eye view with LiDAR points and 3D boxes
- Multi-panel dashboard layout
- Metrics overlay showing detection statistics
- Video output generation

Usage:
    # Run on sequence with default settings
    python scripts/run_fusion_viz.py

    # Process specific frame range
    python scripts/run_fusion_viz.py --start 0 --end 200

    # Custom BEV range
    python scripts/run_fusion_viz.py --bev-range 60

    # Generate comparison video
    python scripts/run_fusion_viz.py --mode comparison --output fusion_comparison.avi

    # Multi-panel Waymo/Tesla style
    python scripts/run_fusion_viz.py --mode multi_panel --output fusion_output.avi

Author: Multi-Sensor Perception Pipeline
"""

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

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
from src.viz.bev_viz import (
    BEVVisualizer, BEVConfig, ColorScheme,
    MultiPanelDisplay, create_comparison_view, draw_depth_improvement_indicator
)

# Optional tqdm
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
# Visualization Helpers
# =============================================================================

def draw_2d_detections(
    image: np.ndarray,
    detections: List[Any],
    thickness: int = 2,
) -> np.ndarray:
    """Draw 2D bounding boxes on image."""
    result = image.copy()

    colors = {
        "Car": (0, 255, 0),
        "Pedestrian": (0, 0, 255),
        "Cyclist": (255, 165, 0),
        "Van": (255, 255, 0),
        "Truck": (128, 0, 128),
    }

    for det in detections:
        color = colors.get(det.class_name, (255, 255, 255))
        x1, y1, x2, y2 = det.bbox.astype(int)

        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        label = f"{det.class_name} {det.confidence:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return result


def draw_3d_boxes_on_image(
    image: np.ndarray,
    boxes_3d: List[Any],
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Project and draw 3D boxes on image."""
    result = image.copy()

    colors = {
        "Car": (0, 255, 0),
        "Pedestrian": (0, 0, 255),
        "Cyclist": (255, 165, 0),
        "Van": (255, 255, 0),
        "Truck": (128, 0, 128),
    }

    for box in boxes_3d:
        if box is None:
            continue

        color = colors.get(box.class_name, (255, 255, 255))

        try:
            corners = box.corners  # (8, 3)

            # Check if box is in front of camera
            if corners[:, 2].min() <= 0.1:
                continue

            # Project to image
            corners_2d = intrinsics.project_point(corners)
            corners_2d = corners_2d.astype(int)

            # Check if within image bounds
            h, w = image.shape[:2]
            if (corners_2d[:, 0].max() < 0 or corners_2d[:, 0].min() > w or
                corners_2d[:, 1].max() < 0 or corners_2d[:, 1].min() > h):
                continue

            # Draw 3D box edges
            # Bottom face (0-1-2-3)
            for i in range(4):
                pt1 = tuple(corners_2d[i])
                pt2 = tuple(corners_2d[(i + 1) % 4])
                cv2.line(result, pt1, pt2, color, 2)

            # Top face (4-5-6-7)
            for i in range(4):
                pt1 = tuple(corners_2d[i + 4])
                pt2 = tuple(corners_2d[(i + 1) % 4 + 4])
                cv2.line(result, pt1, pt2, color, 2)

            # Vertical edges
            for i in range(4):
                pt1 = tuple(corners_2d[i])
                pt2 = tuple(corners_2d[i + 4])
                cv2.line(result, pt1, pt2, color, 1)

            # Add depth label
            center_2d = corners_2d.mean(axis=0).astype(int)
            depth = box.center[2]
            label = f"{depth:.1f}m"
            cv2.putText(result, label, tuple(center_2d),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        except Exception:
            continue

    return result


# =============================================================================
# Main Fusion Visualization Class
# =============================================================================

class FusionVisualizer:
    """
    Multi-sensor fusion visualization pipeline.

    Provides professional-grade visualization of perception results
    with multiple display modes.
    """

    def __init__(
        self,
        data_root: str = "data/kitti",
        split: str = "training",
        device: str = "cuda",
        bev_range: float = 50.0,
        color_scheme: str = "dark",
    ):
        """
        Initialize the fusion visualizer.

        Args:
            data_root: Path to KITTI dataset.
            split: Dataset split ("training" or "testing").
            device: Inference device ("cuda" or "cpu").
            bev_range: BEV visualization range in meters.
            color_scheme: BEV color scheme ("dark", "light", "midnight").
        """
        self.logger = setup_logger("fusion_viz")

        # Initialize data loader
        self.logger.info(f"Loading KITTI dataset from {data_root}")
        self.loader = KITTILoader(root_dir=data_root, split=split)
        self.logger.info(f"Found {len(self.loader)} frames")

        # Initialize detector
        self.logger.info("Initializing YOLOv8 detector")
        self.detector = ObjectDetector2D(
            model_name="yolov8m",
            device=device,
            confidence_threshold=0.3,
        )

        # Initialize depth estimator and 3D generator (placeholders)
        self.depth_estimator = None
        self.bbox_generator = None

        # BEV configuration
        try:
            scheme = ColorScheme[color_scheme.upper()]
        except KeyError:
            scheme = ColorScheme.DARK

        self.bev_config = BEVConfig(
            x_range=(-bev_range, bev_range),
            y_range=(0, bev_range * 1.2),
            resolution=0.1,
            color_scheme=scheme,
        )
        self.bev_viz = BEVVisualizer(config=self.bev_config)

        # Multi-panel display
        self.multi_panel = MultiPanelDisplay(
            camera_size=(640, 384),
            bev_size=(600, 600),
            metrics_height=120,
        )

        self.logger.info("Initialization complete")

    def _setup_calibration(self, calib) -> Tuple[CameraIntrinsics, CameraLiDARExtrinsics]:
        """Setup calibration from KITTI Calibration object."""
        # Extract intrinsics from P2 (access as attribute)
        P2 = calib.P2
        fx = P2[0, 0]
        fy = P2[1, 1]
        cx = P2[0, 2]
        cy = P2[1, 2]

        intrinsics = CameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=1242, height=375
        )

        # Extract extrinsics from Tr_velo_to_cam (access as attribute)
        Tr = calib.Tr_velo_to_cam
        R = Tr[:3, :3]
        T = Tr[:3, 3]

        extrinsics = CameraLiDARExtrinsics(R=R, T=T)
        extrinsics.P2 = P2

        return intrinsics, extrinsics

    def _setup_fusion(self, intrinsics: CameraIntrinsics, extrinsics: CameraLiDARExtrinsics):
        """Initialize depth estimator and 3D generator."""
        self.depth_estimator = DepthEstimator(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )

        self.bbox_generator = BBox3DGenerator(
            intrinsics=intrinsics,
        )

    def process_frame(
        self,
        frame_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Process a single frame through the complete pipeline.

        Args:
            frame_idx: Frame index.

        Returns:
            Tuple of (camera_visualization, bev_visualization, metrics_dict)
        """
        start_time = time.time()

        # Load frame data
        sample = self.loader[frame_idx]
        image = sample["image"]
        points = sample["points"]
        calib = sample["calib"]

        # Setup calibration on first frame
        intrinsics, extrinsics = self._setup_calibration(calib)
        if self.depth_estimator is None:
            self._setup_fusion(intrinsics, extrinsics)

        # Transform points to camera frame
        points_cam = extrinsics.transform_to_camera(points[:, :3])

        # Run 2D detection
        det_start = time.time()
        detections = self.detector.detect(image)
        det_time = time.time() - det_start

        # Convert image to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Estimate depth and generate 3D boxes
        boxes_3d = []
        fusion_start = time.time()

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

        fusion_time = time.time() - fusion_start
        total_time = time.time() - start_time

        # Create camera visualization
        camera_vis = image_bgr.copy()
        camera_vis = draw_2d_detections(camera_vis, detections)
        valid_boxes = [b for b in boxes_3d if b is not None]
        camera_vis = draw_3d_boxes_on_image(camera_vis, valid_boxes, intrinsics)

        # Create BEV visualization
        bev_vis = self.bev_viz.render(points=points_cam, boxes=valid_boxes)

        # Build metrics
        metrics = {
            "frame_id": sample.get("frame_id", str(frame_idx)),
            "detections_2d": len(detections),
            "boxes_3d": len(valid_boxes),
            "lidar_points": len(points),
            "fps": 1.0 / total_time if total_time > 0 else 0,
            "detection_time": det_time * 1000,
            "fusion_time": fusion_time * 1000,
            "processing_time": total_time * 1000,
        }

        return camera_vis, bev_vis, metrics

    def create_comparison_frame(self, frame_idx: int) -> np.ndarray:
        """
        Create before/after fusion comparison frame.

        Args:
            frame_idx: Frame index.

        Returns:
            Comparison visualization image.
        """
        # Load frame data
        sample = self.loader[frame_idx]
        image = sample["image"]
        points = sample["points"]
        calib = sample["calib"]

        # Setup calibration
        intrinsics, extrinsics = self._setup_calibration(calib)
        if self.depth_estimator is None:
            self._setup_fusion(intrinsics, extrinsics)

        # Transform points to camera frame
        points_cam = extrinsics.transform_to_camera(points[:, :3])

        # Run 2D detection
        detections = self.detector.detect(image)

        # Convert to BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Before: just 2D detections
        before_image = draw_2d_detections(image_bgr.copy(), detections)
        cv2.putText(before_image, "2D Detection Only", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

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

        # After: 3D fusion
        after_image = draw_3d_boxes_on_image(image_bgr.copy(), boxes_3d, intrinsics)
        cv2.putText(after_image, "LiDAR-Camera Fusion", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Add depth annotations
        valid_count = sum(1 for b in boxes_3d if b is not None)
        coverage = f"Depth Coverage: {valid_count}/{len(detections)}"
        cv2.putText(after_image, coverage, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

        return create_comparison_view(before_image, after_image)

    def run(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        output_path: Optional[str] = None,
        show_display: bool = True,
        mode: str = "multi_panel",
    ):
        """
        Run the fusion visualization pipeline.

        Args:
            start_frame: Starting frame index.
            end_frame: Ending frame index (None for all).
            output_path: Path for video output.
            show_display: Whether to show live display.
            mode: Visualization mode ("multi_panel", "comparison", "bev_only", "camera_only").
        """
        if end_frame is None:
            end_frame = len(self.loader)

        end_frame = min(end_frame, len(self.loader))
        frame_indices = range(start_frame, end_frame)

        # Progress bar
        if HAS_TQDM:
            frame_indices = tqdm(frame_indices, desc="Processing")

        # Video writer
        video_writer = None
        first_frame = True
        self._video_size = None

        self.logger.info(f"Processing frames {start_frame} to {end_frame-1}")
        self.logger.info(f"Mode: {mode}")

        total_fps = []

        try:
            for frame_idx in frame_indices:
                # Process frame based on mode
                if mode == "multi_panel":
                    camera_vis, bev_vis, metrics = self.process_frame(frame_idx)
                    vis_frame = self.multi_panel.render(camera_vis, bev_vis, metrics)
                    total_fps.append(metrics.get("fps", 0))

                elif mode == "comparison":
                    vis_frame = self.create_comparison_frame(frame_idx)

                elif mode == "bev_only":
                    _, bev_vis, metrics = self.process_frame(frame_idx)
                    vis_frame = bev_vis
                    total_fps.append(metrics.get("fps", 0))

                else:  # camera_only
                    camera_vis, _, metrics = self.process_frame(frame_idx)
                    vis_frame = camera_vis
                    total_fps.append(metrics.get("fps", 0))

                # Initialize video writer
                if output_path and first_frame:
                    h, w = vis_frame.shape[:2]
                    # Ensure dimensions are even for codec compatibility
                    self._video_size = (w - w % 2, h - h % 2)
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    video_writer = cv2.VideoWriter(output_path, fourcc, 10, self._video_size)
                    self.logger.info(f"Writing video to {output_path}")
                    first_frame = False

                # Write to video (resize to ensure consistent dimensions)
                if video_writer is not None:
                    h, w = vis_frame.shape[:2]
                    if (w, h) != self._video_size:
                        vis_frame = cv2.resize(vis_frame, self._video_size)
                    video_writer.write(vis_frame)

                # Display
                if show_display:
                    cv2.imshow("Fusion Visualization", vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        cv2.waitKey(0)

        finally:
            if video_writer is not None:
                video_writer.release()
                self.logger.info(f"Video saved to {output_path}")

            if show_display:
                cv2.destroyAllWindows()

        # Print summary
        if total_fps:
            avg_fps = np.mean(total_fps)
            self.logger.info(f"Average FPS: {avg_fps:.2f}")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Sensor Fusion Visualization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="data/kitti",
        help="Path to KITTI dataset",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting frame index",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending frame index",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable live display",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="multi_panel",
        choices=["multi_panel", "comparison", "bev_only", "camera_only"],
        help="Visualization mode",
    )
    parser.add_argument(
        "--bev-range",
        type=float,
        default=50.0,
        help="BEV visualization range in meters",
    )
    parser.add_argument(
        "--color-scheme",
        type=str,
        default="dark",
        choices=["dark", "light", "midnight"],
        help="BEV color scheme",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_dir = Path("outputs/visualization")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"fusion_{args.mode}.avi")

    # Initialize visualizer
    viz = FusionVisualizer(
        data_root=args.data_root,
        device=args.device,
        bev_range=args.bev_range,
        color_scheme=args.color_scheme,
    )

    # Run visualization
    viz.run(
        start_frame=args.start,
        end_frame=args.end,
        output_path=output_path,
        show_display=not args.no_display,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
