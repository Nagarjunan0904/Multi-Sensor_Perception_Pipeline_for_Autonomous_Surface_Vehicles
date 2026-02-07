#!/usr/bin/env python3
"""Sensor fusion pipeline for 3D object detection."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sensors import CameraLoader, LiDARLoader, SensorSynchronizer
from src.calibration import CameraIntrinsics, CameraLiDARExtrinsics, Projector
from src.perception2d import YOLOv8Detector, DetectionPostProcessor
from src.fusion import DepthEstimator, BBox3DGenerator, OutlierFilter
from src.viz import ImageOverlay, BEVVisualizer, PointCloudVisualizer


def main():
    parser = argparse.ArgumentParser(description="Sensor Fusion Pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/KITTI",
        help="Path to KITTI data directory",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Frame index to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/fusion",
        help="Output directory",
    )
    parser.add_argument(
        "--show-3d",
        action="store_true",
        help="Show 3D point cloud visualization",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check data exists
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Run 'python scripts/download_kitti.py' to download the dataset.")
        return 1

    # Load data
    print("Loading data loaders...")
    camera = CameraLoader(data_root=str(data_dir), split="training")
    lidar = LiDARLoader(data_root=str(data_dir), split="training")
    sync = SensorSynchronizer(camera, lidar)

    print(f"Dataset has {len(sync)} synchronized frames")

    if args.index >= len(sync):
        print(f"Error: Index {args.index} out of range [0, {len(sync)-1}]")
        return 1

    # Load frame
    print(f"Loading frame {args.index}...")
    frame = sync.get_frame(args.index)
    print(f"Frame ID: {frame.frame_id}")
    print(f"Image shape: {frame.image.shape}")
    print(f"Point cloud shape: {frame.pointcloud.shape}")

    # Load calibration
    calib_file = data_dir / "training" / "calib" / f"{frame.frame_id}.txt"
    if not calib_file.exists():
        print(f"Error: Calibration file not found: {calib_file}")
        return 1

    extrinsics = CameraLiDARExtrinsics(str(calib_file))
    width, height = camera.get_image_size()
    intrinsics = CameraIntrinsics.from_projection_matrix(extrinsics.P2, width, height)
    projector = Projector(intrinsics, extrinsics)

    print("Calibration loaded")

    # Run 2D detection
    print("Running 2D detection...")
    detector = YOLOv8Detector(device=args.device)
    detections = detector.detect(frame.image)
    postprocessor = DetectionPostProcessor()
    detections = postprocessor.process(detections)
    print(f"Detected {len(detections)} objects")

    # Depth estimation and 3D box generation
    print("Estimating depth and generating 3D boxes...")
    depth_estimator = DepthEstimator(projector=projector)
    bbox_generator = BBox3DGenerator(intrinsics=intrinsics)
    outlier_filter = OutlierFilter()

    depths = depth_estimator.estimate_depths_batch(detections, frame.pointcloud)
    boxes_3d = bbox_generator.generate_batch(detections, depths)
    boxes_3d = outlier_filter.filter(boxes_3d)

    valid_boxes = [b for b in boxes_3d if b is not None]
    print(f"Generated {len(valid_boxes)} valid 3D boxes")

    for box in valid_boxes:
        print(f"  - {box.class_name}: depth={box.center[2]:.1f}m")

    # Visualizations
    print("Generating visualizations...")

    # 1. Image with 2D boxes
    overlay = ImageOverlay(intrinsics=intrinsics)
    img_2d = overlay.draw_2d_boxes(frame.image, detections)
    cv2.imwrite(str(output_dir / f"{frame.frame_id}_2d.png"), img_2d)

    # 2. Image with 3D boxes
    img_3d = overlay.draw_3d_boxes(frame.image.copy(), boxes_3d)
    cv2.imwrite(str(output_dir / f"{frame.frame_id}_3d.png"), img_3d)

    # 3. LiDAR projection
    points_2d, depths_proj, _ = projector.project_lidar_to_image(frame.pointcloud)
    img_lidar = overlay.draw_lidar_projection(frame.image.copy(), points_2d, depths_proj)
    cv2.imwrite(str(output_dir / f"{frame.frame_id}_lidar.png"), img_lidar)

    # 4. Composite
    img_composite = overlay.create_composite(
        frame.image,
        detections_2d=detections,
        boxes_3d=boxes_3d,
        points_2d=points_2d,
        depths=depths_proj,
    )
    cv2.imwrite(str(output_dir / f"{frame.frame_id}_composite.png"), img_composite)

    # 5. Bird's-eye view
    bev_viz = BEVVisualizer()
    # Transform points to camera frame for BEV
    points_cam = extrinsics.transform_to_camera(frame.pointcloud)
    bev_image = bev_viz.create_bev_visualization(points_cam, boxes=boxes_3d)
    cv2.imwrite(str(output_dir / f"{frame.frame_id}_bev.png"), bev_image)

    print(f"\nResults saved to {output_dir}/")
    print("  - *_2d.png: 2D detection boxes")
    print("  - *_3d.png: 3D detection boxes")
    print("  - *_lidar.png: LiDAR projection")
    print("  - *_composite.png: Combined visualization")
    print("  - *_bev.png: Bird's-eye view")

    # 3D visualization (optional)
    if args.show_3d:
        print("\nShowing 3D visualization (close window to exit)...")
        pc_viz = PointCloudVisualizer()
        pc_viz.visualize(points_cam, boxes_3d=boxes_3d)

    return 0


if __name__ == "__main__":
    exit(main())
