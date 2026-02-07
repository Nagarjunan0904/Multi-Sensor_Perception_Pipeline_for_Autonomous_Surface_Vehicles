#!/usr/bin/env python3
"""
Multi-Sensor Perception Pipeline for Autonomous Surface Vehicles.

This script implements the complete perception pipeline:
1. Load KITTI frame (image + LiDAR + calibration)
2. Run 2D object detection (YOLOv8)
3. For each detection:
   - Extract LiDAR points in bounding box
   - Estimate depth and 3D dimensions
   - Generate 3D bounding box
4. Visualize results (multi-panel: image + BEV)
5. Save outputs (images, video, logs)

Usage:
    # Process all frames with default config
    python scripts/run_pipeline.py

    # Process specific frame range
    python scripts/run_pipeline.py --start 0 --end 100

    # Process single frame with visualization
    python scripts/run_pipeline.py --frame 42

    # Use custom config
    python scripts/run_pipeline.py --config configs/custom.yaml

    # Generate video output
    python scripts/run_pipeline.py --start 0 --end 500 --video
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from src.data.kitti_loader import KITTILoader, Calibration
from src.perception2d.detector import ObjectDetector2D, Detection
from src.fusion.depth_estimator import DepthEstimator, DepthResult
from src.fusion.bbox3d_generator import BBox3DGenerator, BBox3D
from src.calibration.intrinsics import CameraIntrinsics
from src.calibration.extrinsics import CameraLiDARExtrinsics
from src.viz.bev_viz import (
    BEVVisualizer, BEVConfig, ColorScheme,
    MultiPanelDisplay, create_comparison_view
)
from src.utils.config_loader import load_config

# Optional: tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FrameResult:
    """Results from processing a single frame."""
    frame_id: str
    frame_idx: int
    image: np.ndarray
    points: np.ndarray
    detections_2d: List[Detection]
    depth_results: List[DepthResult]
    boxes_3d: List[Optional[BBox3D]]
    processing_time: float
    detection_time: float
    fusion_time: float


@dataclass
class PipelineMetrics:
    """Aggregated pipeline metrics."""
    total_frames: int = 0
    total_detections: int = 0
    total_3d_boxes: int = 0
    total_time: float = 0.0
    detection_time: float = 0.0
    fusion_time: float = 0.0

    @property
    def fps(self) -> float:
        return self.total_frames / self.total_time if self.total_time > 0 else 0.0

    @property
    def avg_detections(self) -> float:
        return self.total_detections / self.total_frames if self.total_frames > 0 else 0.0

    @property
    def detection_rate(self) -> float:
        return self.total_3d_boxes / self.total_detections if self.total_detections > 0 else 0.0


# =============================================================================
# Logger Setup
# =============================================================================

def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """Configure logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# Visualization
# =============================================================================

class PipelineVisualizer:
    """Multi-panel visualization for pipeline results."""

    # Class colors (BGR for OpenCV)
    CLASS_COLORS = {
        "Car": (0, 255, 127),        # Spring green
        "Pedestrian": (82, 82, 255),  # Red
        "Cyclist": (255, 156, 64),    # Light blue
        "Van": (209, 206, 0),         # Cyan
        "Truck": (37, 193, 255),      # Orange
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize visualizer with config."""
        self.config = config
        viz_config = config.get("viz", {})

        self.show_2d_boxes = viz_config.get("show_2d_boxes", True)
        self.show_3d_boxes = viz_config.get("show_3d_boxes", True)
        self.show_lidar = viz_config.get("show_lidar_points", True)
        self.show_depth = viz_config.get("show_depth_text", True)
        self.show_metrics = viz_config.get("show_metrics", True)
        self.box_thickness = viz_config.get("box_thickness", 2)
        self.font_scale = viz_config.get("font_scale", 0.5)
        self.lidar_point_size = viz_config.get("lidar_point_size", 2)
        self.lidar_depth_range = tuple(viz_config.get("lidar_depth_range", [0.5, 60.0]))
        self.layout = viz_config.get("layout", "side_by_side")
        self.panel_gap = viz_config.get("panel_gap", 10)

        # Visualization mode
        self.mode = viz_config.get("mode", "multi_panel")
        bev_range = viz_config.get("bev_range", 50.0)
        color_scheme = viz_config.get("color_scheme", "dark")

        # BEV visualizer with configurable range and color scheme
        bev_config = viz_config.get("bev", {})
        if bev_config.get("enabled", True):
            # Create BEV config
            try:
                scheme = ColorScheme[color_scheme.upper()]
            except KeyError:
                scheme = ColorScheme.DARK

            self.bev_config = BEVConfig(
                x_range=(-bev_range, bev_range),
                y_range=(0, bev_range * 1.2),
                resolution=bev_config.get("resolution", 0.1),
                color_scheme=scheme,
            )
            self.bev_viz = BEVVisualizer(config=self.bev_config)
        else:
            self.bev_viz = None
            self.bev_config = None

        # Multi-panel display for professional visualization
        if self.mode == "multi_panel":
            self.multi_panel = MultiPanelDisplay(
                camera_size=(640, 384),
                bev_size=(600, 600),
                metrics_height=120,
            )
        else:
            self.multi_panel = None

    def get_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get BGR color for class."""
        return self.CLASS_COLORS.get(class_name, (255, 255, 255))

    def draw_2d_boxes(
        self,
        image: np.ndarray,
        detections: List[Detection],
        depth_results: List[DepthResult],
    ) -> np.ndarray:
        """Draw 2D detection boxes with depth labels."""
        result = image.copy()

        for det, depth_res in zip(detections, depth_results):
            color = self.get_color(det.class_name)
            x1, y1, x2, y2 = det.bbox.astype(int)

            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, self.box_thickness)

            # Label with class and depth
            if depth_res.depth is not None and self.show_depth:
                label = f"{det.class_name} {depth_res.depth:.1f}m"
            else:
                label = f"{det.class_name} {det.confidence:.2f}"

            # Background for label
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
            cv2.rectangle(result, (x1, y1 - h - 6), (x1 + w + 4, y1), color, -1)
            cv2.putText(result, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 0, 0), 1)

        return result

    def draw_3d_boxes_on_image(
        self,
        image: np.ndarray,
        boxes_3d: List[Optional[BBox3D]],
        intrinsics: CameraIntrinsics,
    ) -> np.ndarray:
        """Project and draw 3D boxes on image."""
        result = image.copy()

        for box in boxes_3d:
            if box is None:
                continue

            color = self.get_color(box.class_name)
            corners = box.corners  # (8, 3) in camera frame

            # Project corners to image
            corners_2d = intrinsics.project_point(corners)

            # Filter invalid projections
            if corners[:, 2].min() <= 0:
                continue

            corners_2d = corners_2d.astype(int)

            # Draw 3D box edges
            # Bottom face (0-1-2-3)
            for i in range(4):
                pt1 = tuple(corners_2d[i])
                pt2 = tuple(corners_2d[(i + 1) % 4])
                cv2.line(result, pt1, pt2, color, 1)

            # Top face (4-5-6-7)
            for i in range(4):
                pt1 = tuple(corners_2d[i + 4])
                pt2 = tuple(corners_2d[(i + 1) % 4 + 4])
                cv2.line(result, pt1, pt2, color, 1)

            # Vertical edges
            for i in range(4):
                pt1 = tuple(corners_2d[i])
                pt2 = tuple(corners_2d[i + 4])
                cv2.line(result, pt1, pt2, color, 1)

        return result

    def draw_lidar_overlay(
        self,
        image: np.ndarray,
        points: np.ndarray,
        calib: Calibration,
    ) -> np.ndarray:
        """Overlay LiDAR points colored by depth."""
        result = image.copy()

        # Get FOV mask and project
        fov_mask = calib.get_fov_mask(points, image.shape)
        points_fov = points[fov_mask]

        if len(points_fov) == 0:
            return result

        points_2d = calib.project_velo_to_image(points_fov)
        pts_rect = calib.project_velo_to_rect(points_fov)
        depths = pts_rect[:, 2]

        # Filter by depth range
        depth_mask = (depths >= self.lidar_depth_range[0]) & (depths <= self.lidar_depth_range[1])
        points_2d = points_2d[depth_mask]
        depths = depths[depth_mask]

        if len(depths) == 0:
            return result

        # Normalize depths to colors
        depths_norm = (depths - self.lidar_depth_range[0]) / (self.lidar_depth_range[1] - self.lidar_depth_range[0])
        depths_norm = np.clip(depths_norm, 0, 1)
        colors = cv2.applyColorMap((depths_norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)

        # Draw points (far to near for proper occlusion)
        sort_idx = np.argsort(-depths)
        for idx in sort_idx:
            u, v = int(points_2d[idx, 0]), int(points_2d[idx, 1])
            if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                cv2.circle(result, (u, v), self.lidar_point_size, colors[idx].tolist(), -1)

        return result

    def add_metrics_overlay(
        self,
        image: np.ndarray,
        frame_result: FrameResult,
        metrics: PipelineMetrics,
    ) -> np.ndarray:
        """Add metrics overlay to image."""
        result = image.copy()

        # Build info lines
        lines = [
            f"Frame: {frame_result.frame_id}",
            f"Detections: {len(frame_result.detections_2d)}",
            f"3D Boxes: {sum(1 for b in frame_result.boxes_3d if b is not None)}",
            f"FPS: {1.0 / frame_result.processing_time:.1f}",
            f"Det: {frame_result.detection_time * 1000:.0f}ms",
            f"Fusion: {frame_result.fusion_time * 1000:.0f}ms",
        ]

        # Draw background
        line_height = 18
        box_height = len(lines) * line_height + 15
        box_width = 160
        cv2.rectangle(result, (5, 5), (5 + box_width, 5 + box_height), (0, 0, 0), -1)
        cv2.rectangle(result, (5, 5), (5 + box_width, 5 + box_height), (255, 255, 255), 1)

        # Draw text
        for i, line in enumerate(lines):
            y = 22 + i * line_height
            cv2.putText(result, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return result

    def create_visualization(
        self,
        frame_result: FrameResult,
        calib: Calibration,
        intrinsics: CameraIntrinsics,
        metrics: PipelineMetrics,
    ) -> np.ndarray:
        """Create complete visualization based on mode."""
        # Start with original image (convert RGB to BGR for OpenCV)
        image = cv2.cvtColor(frame_result.image, cv2.COLOR_RGB2BGR)

        # Transform points to camera frame for BEV (X-right, Z-forward)
        pts_cam = calib.project_velo_to_rect(frame_result.points)
        valid_boxes = [b for b in frame_result.boxes_3d if b is not None]

        # Handle different visualization modes
        if self.mode == "multi_panel" and self.multi_panel is not None and self.bev_viz is not None:
            # Professional Waymo/Tesla style multi-panel display
            # Create camera view with detections
            camera_vis = image.copy()
            if self.show_lidar:
                camera_vis = self.draw_lidar_overlay(camera_vis, frame_result.points, calib)
            if self.show_2d_boxes:
                camera_vis = self.draw_2d_boxes(camera_vis, frame_result.detections_2d, frame_result.depth_results)
            if self.show_3d_boxes:
                camera_vis = self.draw_3d_boxes_on_image(camera_vis, valid_boxes, intrinsics)

            # Create BEV view
            bev_vis = self.bev_viz.render(points=pts_cam, boxes=valid_boxes)

            # Build metrics dictionary
            metrics_dict = {
                "frame_id": frame_result.frame_id,
                "detections_2d": len(frame_result.detections_2d),
                "boxes_3d": len(valid_boxes),
                "lidar_points": len(frame_result.points),
                "fps": metrics.fps,
                "processing_time": frame_result.processing_time * 1000,
            }

            # Render multi-panel display
            return self.multi_panel.render(camera_vis, bev_vis, metrics_dict)

        elif self.mode == "comparison":
            # Before/After fusion comparison
            # Before: just 2D detections
            before_image = image.copy()
            before_image = self.draw_2d_boxes(before_image, frame_result.detections_2d, frame_result.depth_results)
            cv2.putText(before_image, "2D Detection Only", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # After: 3D boxes with LiDAR fusion
            after_image = image.copy()
            if self.show_lidar:
                after_image = self.draw_lidar_overlay(after_image, frame_result.points, calib)
            after_image = self.draw_3d_boxes_on_image(after_image, valid_boxes, intrinsics)
            cv2.putText(after_image, "LiDAR-Camera Fusion", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Add depth coverage annotation
            coverage = f"3D Coverage: {len(valid_boxes)}/{len(frame_result.detections_2d)}"
            cv2.putText(after_image, coverage, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

            return create_comparison_view(before_image, after_image)

        elif self.mode == "bev_only" and self.bev_viz is not None:
            # BEV only mode
            return self.bev_viz.render(points=pts_cam, boxes=valid_boxes)

        else:
            # Basic mode (original behavior)
            # Draw LiDAR overlay
            if self.show_lidar:
                image = self.draw_lidar_overlay(image, frame_result.points, calib)

            # Draw 2D boxes
            if self.show_2d_boxes:
                image = self.draw_2d_boxes(image, frame_result.detections_2d, frame_result.depth_results)

            # Draw 3D boxes
            if self.show_3d_boxes:
                image = self.draw_3d_boxes_on_image(image, frame_result.boxes_3d, intrinsics)

            # Add metrics
            if self.show_metrics:
                image = self.add_metrics_overlay(image, frame_result, metrics)

            # Create BEV view
            if self.bev_viz is not None:
                bev_image = self.bev_viz.render(points=pts_cam, boxes=valid_boxes)

                # Resize BEV to match image height
                bev_h = image.shape[0]
                bev_w = int(bev_image.shape[1] * bev_h / bev_image.shape[0])
                bev_image = cv2.resize(bev_image, (bev_w, bev_h))

                # Combine panels
                if self.layout == "side_by_side":
                    gap = np.ones((image.shape[0], self.panel_gap, 3), dtype=np.uint8) * 40
                    combined = np.hstack([image, gap, bev_image])
                elif self.layout == "stacked":
                    gap = np.ones((self.panel_gap, max(image.shape[1], bev_image.shape[1]), 3), dtype=np.uint8) * 40
                    if image.shape[1] != bev_image.shape[1]:
                        bev_image = cv2.resize(bev_image, (image.shape[1], bev_image.shape[0]))
                    combined = np.vstack([image, gap, bev_image])
                else:
                    combined = image
            else:
                combined = image

            return combined


# =============================================================================
# Main Pipeline
# =============================================================================

class PerceptionPipeline:
    """Complete multi-sensor perception pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: str = "configs/default.yaml"):
        """
        Initialize the perception pipeline.

        Args:
            config: Configuration dictionary (takes precedence over config_path).
            config_path: Path to YAML configuration file (used if config is None).
        """
        # Load configuration
        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_path)

        # Setup logging
        log_config = self.config.get("logging", {})
        self.logger = setup_logger(
            name="pipeline",
            level=log_config.get("level", "INFO"),
            log_file=log_config.get("file"),
            console=log_config.get("console", True),
        )

        self.logger.info("=" * 60)
        self.logger.info("Multi-Sensor Perception Pipeline")
        self.logger.info("=" * 60)

        # Initialize components
        self._init_data_loader()
        self._init_detector()
        self._init_fusion()
        self._init_visualizer()
        self._init_output()

        # Metrics
        self.metrics = PipelineMetrics()

        self.logger.info("Pipeline initialization complete")

    def _init_data_loader(self):
        """Initialize KITTI data loader."""
        data_config = self.config.get("data", {})
        data_root = data_config.get("root", "data/kitti")
        split = data_config.get("split", "training")

        self.logger.info(f"Loading KITTI dataset from {data_root} ({split})")

        try:
            self.loader = KITTILoader(data_root, split=split)
            self.logger.info(f"Found {len(self.loader)} frames")
        except FileNotFoundError as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise

    def _init_detector(self):
        """Initialize 2D object detector."""
        det_config = self.config.get("detector", {})
        perf_config = self.config.get("performance", {})

        self.logger.info(f"Loading {det_config.get('model', 'yolov8m')} detector")

        self.detector = ObjectDetector2D(
            model_name=det_config.get("model", "yolov8m"),
            weights_path=det_config.get("weights"),
            confidence_threshold=det_config.get("confidence_threshold", 0.3),
            iou_threshold=det_config.get("iou_threshold", 0.45),
            device=perf_config.get("device", "cuda"),
            classes=det_config.get("classes"),
            half_precision=det_config.get("half_precision", False),
        )

        # Warmup
        if perf_config.get("warmup", True):
            self.logger.info("Warming up detector...")
            self.detector.warmup()

    def _init_fusion(self):
        """Initialize depth estimation and 3D box generation."""
        depth_config = self.config.get("depth", {})
        bbox_config = self.config.get("bbox3d", {})

        # Depth estimator (will be configured per-frame with calibration)
        self.depth_config = depth_config
        self.depth_estimator = None

        # 3D box generator (will be configured per-frame with intrinsics)
        self.bbox_config = bbox_config
        self.bbox_generator = None

    def _init_visualizer(self):
        """Initialize visualization components."""
        self.visualizer = PipelineVisualizer(self.config)

    def _init_output(self):
        """Initialize output directories and video writer."""
        viz_config = self.config.get("viz", {})
        self.output_dir = Path(viz_config.get("output_dir", "outputs/pipeline"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_images = viz_config.get("save_images", False)
        self.save_video = viz_config.get("save_video", True)
        self.show_display = viz_config.get("show_display", True)

        self.video_writer = None
        self.video_fps = viz_config.get("video_fps", 10)
        self.video_name = viz_config.get("video_name", "pipeline_output.mp4")

    def _setup_frame_calibration(self, calib: Calibration, image_shape: Tuple[int, int]):
        """Configure fusion components with frame calibration."""
        # Create intrinsics from P2
        self.intrinsics = CameraIntrinsics.from_projection_matrix(
            calib.P2,
            width=image_shape[1],
            height=image_shape[0],
        )

        # Create extrinsics
        self.extrinsics = CameraLiDARExtrinsics()
        self.extrinsics.P2 = calib.P2
        self.extrinsics.R0_rect = calib.R0_rect
        self.extrinsics.Tr_velo_to_cam = calib.Tr_velo_to_cam

        # Configure depth estimator
        self.depth_estimator = DepthEstimator(
            intrinsics=self.intrinsics,
            extrinsics=self.extrinsics,
            method=self.depth_config.get("method", "median"),
            min_points=self.depth_config.get("min_points", 5),
            depth_range=(
                self.depth_config.get("min_depth", 0.5),
                self.depth_config.get("max_depth", 80.0),
            ),
            search_expansion=self.depth_config.get("search_expansion", 0.05),
            filter_ground=self.depth_config.get("filter_ground", True),
            ground_height=self.depth_config.get("ground_height", -1.5),
            use_clustering=self.depth_config.get("use_clustering", False),
            cluster_eps=self.depth_config.get("cluster_eps", 0.5),
        )

        # Configure 3D box generator
        default_dims = self.bbox_config.get("default_dimensions", {})
        dims_array = {k: np.array(v) for k, v in default_dims.items()}

        self.bbox_generator = BBox3DGenerator(
            intrinsics=self.intrinsics,
            default_dimensions=dims_array if dims_array else None,
            use_prior_dimensions=self.bbox_config.get("use_prior_dimensions", True),
        )

    def process_frame(self, frame_idx: int) -> FrameResult:
        """
        Process a single frame through the pipeline.

        Args:
            frame_idx: Frame index to process.

        Returns:
            FrameResult with all processing outputs.
        """
        frame_start = time.time()

        # Load frame data
        sample = self.loader[frame_idx]
        frame_id = sample["frame_id"]
        image = sample["image"]
        points = sample["points"]
        calib = sample["calib"]

        # Configure fusion with this frame's calibration
        self._setup_frame_calibration(calib, image.shape[:2])

        # 2D Detection
        det_start = time.time()
        detections = self.detector.detect(image)
        detection_time = time.time() - det_start

        # Depth estimation and 3D box generation
        fusion_start = time.time()
        depth_results = []
        boxes_3d = []

        for det in detections:
            # Estimate depth
            depth_res = self.depth_estimator.estimate_full(det, points)
            depth_results.append(depth_res)

            # Generate 3D box if valid depth
            if depth_res.depth is not None:
                # Use point-based generation if we have enough points
                if depth_res.points_3d is not None and len(depth_res.points_3d) >= 3:
                    box_3d = self.bbox_generator.generate_from_points(
                        det,
                        depth_res.points_3d,
                        depth=depth_res.depth,
                    )
                else:
                    # Fall back to basic generation
                    box_3d = self.bbox_generator.generate(det, depth_res.depth)

                # Optional: refine with LiDAR
                if self.bbox_config.get("refine_with_lidar", False) and depth_res.points_3d is not None:
                    box_3d = self.bbox_generator.refine_with_lidar(box_3d, depth_res.points_3d)

                boxes_3d.append(box_3d)
            else:
                boxes_3d.append(None)

        fusion_time = time.time() - fusion_start
        total_time = time.time() - frame_start

        # Create result
        result = FrameResult(
            frame_id=frame_id,
            frame_idx=frame_idx,
            image=image,
            points=points,
            detections_2d=detections,
            depth_results=depth_results,
            boxes_3d=boxes_3d,
            processing_time=total_time,
            detection_time=detection_time,
            fusion_time=fusion_time,
        )

        # Update metrics
        self.metrics.total_frames += 1
        self.metrics.total_detections += len(detections)
        self.metrics.total_3d_boxes += sum(1 for b in boxes_3d if b is not None)
        self.metrics.total_time += total_time
        self.metrics.detection_time += detection_time
        self.metrics.fusion_time += fusion_time

        return result

    def visualize_frame(
        self,
        result: FrameResult,
        calib: Calibration,
    ) -> np.ndarray:
        """Create visualization for frame result."""
        return self.visualizer.create_visualization(
            result, calib, self.intrinsics, self.metrics
        )

    def run(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        frame_skip: int = 1,
        visualize: bool = True,
    ) -> None:
        """
        Run the pipeline on a range of frames.

        Args:
            start_frame: Starting frame index.
            end_frame: Ending frame index (None for all).
            frame_skip: Process every Nth frame.
            visualize: Whether to generate visualizations.
        """
        end_frame = end_frame or len(self.loader)
        end_frame = min(end_frame, len(self.loader))

        frame_indices = list(range(start_frame, end_frame, frame_skip))
        total_frames = len(frame_indices)

        self.logger.info(f"Processing {total_frames} frames ({start_frame} to {end_frame - 1}, skip={frame_skip})")

        # Progress iterator
        if HAS_TQDM:
            iterator = tqdm(frame_indices, desc="Processing", unit="frame")
        else:
            iterator = frame_indices

        try:
            for frame_idx in iterator:
                # Process frame
                try:
                    result = self.process_frame(frame_idx)
                except Exception as e:
                    self.logger.error(f"Error processing frame {frame_idx}: {e}")
                    continue

                # Visualize
                if visualize:
                    sample = self.loader[frame_idx]
                    vis_image = self.visualize_frame(result, sample["calib"])

                    # Initialize video writer on first frame
                    if self.save_video and self.video_writer is None:
                        h, w = vis_image.shape[:2]
                        self._video_frame_size = (w, h)
                        video_path = self.output_dir / self.video_name
                        # Use XVID codec for better compatibility
                        fourcc = cv2.VideoWriter_fourcc(*"XVID")
                        video_path_avi = video_path.with_suffix(".avi")
                        self.video_writer = cv2.VideoWriter(str(video_path_avi), fourcc, self.video_fps, (w, h))
                        self.logger.info(f"Writing video to {video_path_avi}")

                    # Write video frame (resize if needed to match first frame)
                    if self.video_writer is not None:
                        h, w = vis_image.shape[:2]
                        if (w, h) != self._video_frame_size:
                            vis_image = cv2.resize(vis_image, self._video_frame_size)
                        self.video_writer.write(vis_image)

                    # Save image
                    if self.save_images:
                        img_path = self.output_dir / "frames" / f"{result.frame_id}.png"
                        img_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(img_path), vis_image)

                    # Display
                    if self.show_display:
                        cv2.imshow("Perception Pipeline", vis_image)
                        key = cv2.waitKey(1)
                        if key == ord("q") or key == 27:
                            self.logger.info("User requested stop")
                            break
                        elif key == ord("s"):
                            save_path = self.output_dir / f"frame_{result.frame_id}_saved.png"
                            cv2.imwrite(str(save_path), vis_image)
                            self.logger.info(f"Saved: {save_path}")

                # Log progress (non-tqdm)
                if not HAS_TQDM and (frame_idx - start_frame + 1) % 100 == 0:
                    self.logger.info(
                        f"Processed {frame_idx - start_frame + 1}/{total_frames} frames "
                        f"({self.metrics.fps:.1f} FPS)"
                    )

        finally:
            # Cleanup
            if self.video_writer is not None:
                self.video_writer.release()
                video_path = (self.output_dir / self.video_name).with_suffix(".avi")
                self.logger.info(f"Video saved: {video_path}")

            if self.show_display:
                cv2.destroyAllWindows()

        # Final statistics
        self._print_summary()

    def _print_summary(self):
        """Print final processing summary."""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Processing Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Total frames processed: {self.metrics.total_frames}")
        self.logger.info(f"Total processing time: {self.metrics.total_time:.2f}s")
        self.logger.info(f"Average FPS: {self.metrics.fps:.2f}")
        self.logger.info(f"Total 2D detections: {self.metrics.total_detections}")
        self.logger.info(f"Total 3D boxes: {self.metrics.total_3d_boxes}")
        self.logger.info(f"Detection rate: {self.metrics.detection_rate * 100:.1f}%")
        self.logger.info(f"Avg detections/frame: {self.metrics.avg_detections:.1f}")
        if self.metrics.total_frames > 0:
            self.logger.info(f"Detection time: {self.metrics.detection_time / self.metrics.total_frames * 1000:.1f}ms/frame")
            self.logger.info(f"Fusion time: {self.metrics.fusion_time / self.metrics.total_frames * 1000:.1f}ms/frame")
        self.logger.info("=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Sensor Perception Pipeline for Autonomous Surface Vehicles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting frame index (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending frame index (default: all frames)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Process single frame (overrides --start/--end)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Frame skip interval (default: 1, process all)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't show display window",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Save video output",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save individual frame images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Override inference device",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="multi_panel",
        choices=["basic", "multi_panel", "comparison", "bev_only"],
        help="Visualization mode: basic (image+BEV), multi_panel (Waymo/Tesla style), comparison (before/after), bev_only",
    )
    parser.add_argument(
        "--bev-range",
        type=float,
        default=50.0,
        help="BEV visualization range in meters (default: 50)",
    )
    parser.add_argument(
        "--color-scheme",
        type=str,
        default="dark",
        choices=["dark", "light", "midnight"],
        help="BEV color scheme (default: dark)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load and modify config
    config = load_config(args.config)

    # Apply command-line overrides
    if args.no_display:
        config["viz"]["show_display"] = False

    if args.video:
        config["viz"]["save_video"] = True

    if args.save_images:
        config["viz"]["save_images"] = True

    if args.output_dir:
        config["viz"]["output_dir"] = args.output_dir

    if args.device:
        config["performance"]["device"] = args.device

    # Visualization mode options
    if "viz" not in config:
        config["viz"] = {}
    config["viz"]["mode"] = args.mode
    config["viz"]["bev_range"] = args.bev_range
    config["viz"]["color_scheme"] = args.color_scheme

    # Save modified config for this run
    from src.utils.config_loader import ConfigLoader
    loader = ConfigLoader()
    output_dir = Path(config["viz"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    loader.save(config, output_dir / "config_used.yaml")

    # Initialize pipeline with modified config
    pipeline = PerceptionPipeline(config=config)

    # Apply runtime config overrides
    pipeline.show_display = config["viz"]["show_display"]
    pipeline.save_video = config["viz"]["save_video"]
    pipeline.save_images = config["viz"]["save_images"]
    pipeline.output_dir = output_dir

    # Run pipeline
    if args.frame is not None:
        # Single frame mode
        result = pipeline.process_frame(args.frame)

        if not args.no_viz:
            sample = pipeline.loader[args.frame]
            vis_image = pipeline.visualize_frame(result, sample["calib"])

            # Save
            output_path = output_dir / f"frame_{result.frame_id}.png"
            cv2.imwrite(str(output_path), vis_image)
            print(f"Saved: {output_path}")

            # Display
            if not args.no_display:
                cv2.imshow("Pipeline Result", vis_image)
                print("Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        # Print results
        print(f"\nFrame {result.frame_id}:")
        print(f"  2D Detections: {len(result.detections_2d)}")
        print(f"  3D Boxes: {sum(1 for b in result.boxes_3d if b is not None)}")
        print(f"  Processing time: {result.processing_time * 1000:.1f}ms")

        for i, (det, depth, box) in enumerate(zip(result.detections_2d, result.depth_results, result.boxes_3d)):
            depth_str = f"{depth.depth:.1f}m" if depth.depth else "N/A"
            print(f"    [{i}] {det.class_name}: conf={det.confidence:.2f}, depth={depth_str}")

    else:
        # Batch mode
        pipeline.run(
            start_frame=args.start,
            end_frame=args.end,
            frame_skip=args.skip,
            visualize=not args.no_viz,
        )


if __name__ == "__main__":
    main()
