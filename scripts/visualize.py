#!/usr/bin/env python3
"""
Visualization utilities for KITTI dataset.

This script provides:
1. Loading KITTI frames
2. Projecting LiDAR points onto image (classic colored dots overlay)
3. Creating bird's eye view visualizations
4. Drawing 2D and 3D bounding boxes
5. Interactive 3D visualization with Open3D

Usage:
    python scripts/visualize.py
    python scripts/visualize.py --data-dir /path/to/KITTI
    python scripts/visualize.py --frame-id 000050
    python scripts/visualize.py --no-3d  # Skip Open3D visualization
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.kitti_loader import KITTILoader
from viz.image_overlay import (
    ImageOverlay,
    draw_2d_boxes,
    draw_3d_boxes,
    draw_legend,
    draw_text,
    save_image,
    KITTI_COLORS,
)
from viz.pointcloud_viz import (
    project_points_to_image,
    create_bev_image,
    save_pointcloud_image,
    visualize_pointcloud,
    HAS_OPEN3D,
)


def create_lidar_overlay(
    sample: dict,
    output_dir: Path,
) -> None:
    """
    Demonstrate LiDAR-to-image projection with different color modes.

    Creates the classic "colored dots" visualization showing LiDAR points
    projected onto the camera image.
    """
    print("\n--- LiDAR Overlay ---")

    image = sample["image"]
    points = sample["points"]
    calib = sample["calib"]
    frame_id = sample["frame_id"]

    # 1. Depth-colored overlay (blue=near, red=far)
    print("  Creating depth-colored overlay...")
    overlay_depth = project_points_to_image(
        image, points, calib,
        color_mode="depth",
        point_size=2,
        min_depth=1.0,
        max_depth=50.0,
    )

    # Add info text
    overlay_depth = draw_text(
        overlay_depth,
        f"Frame: {frame_id} | LiDAR Overlay (Depth)",
        (10, 25),
        color=(255, 255, 255),
        font_scale=0.7,
        background=True,
    )
    overlay_depth = draw_text(
        overlay_depth,
        f"Points: {len(points):,} | FOV: {calib.get_fov_mask(points, image.shape).sum():,}",
        (10, 55),
        color=(200, 200, 200),
        font_scale=0.5,
        background=True,
    )

    save_path = output_dir / f"{frame_id}_lidar_depth.png"
    save_image(overlay_depth, save_path)
    print(f"  Saved: {save_path}")

    # 2. Intensity-colored overlay
    print("  Creating intensity-colored overlay...")
    overlay_intensity = project_points_to_image(
        image, points, calib,
        color_mode="intensity",
        point_size=2,
    )
    overlay_intensity = draw_text(
        overlay_intensity,
        f"Frame: {frame_id} | LiDAR Overlay (Intensity)",
        (10, 25),
        color=(255, 255, 255),
        font_scale=0.7,
        background=True,
    )
    save_path = output_dir / f"{frame_id}_lidar_intensity.png"
    save_image(overlay_intensity, save_path)
    print(f"  Saved: {save_path}")

    # 3. Height-colored overlay
    print("  Creating height-colored overlay...")
    overlay_height = project_points_to_image(
        image, points, calib,
        color_mode="height",
        point_size=2,
    )
    overlay_height = draw_text(
        overlay_height,
        f"Frame: {frame_id} | LiDAR Overlay (Height)",
        (10, 25),
        color=(255, 255, 255),
        font_scale=0.7,
        background=True,
    )
    save_path = output_dir / f"{frame_id}_lidar_height.png"
    save_image(overlay_height, save_path)
    print(f"  Saved: {save_path}")


def create_bounding_boxes(
    sample: dict,
    output_dir: Path,
) -> None:
    """
    Demonstrate 2D and 3D bounding box visualization.
    """
    print("\n--- Bounding Boxes ---")

    image = sample["image"]
    calib = sample["calib"]
    frame_id = sample["frame_id"]
    objects = sample.get("objects", [])

    if not objects:
        print("  No objects in this frame, skipping box visualization")
        return

    # Filter out DontCare objects
    objects = [obj for obj in objects if obj.type != "DontCare"]

    if not objects:
        print("  No valid objects after filtering")
        return

    # 1. 2D Bounding Boxes
    print(f"  Drawing {len(objects)} 2D boxes...")
    boxes_2d = np.array([obj.bbox2d for obj in objects])
    labels = [obj.type for obj in objects]

    result_2d = draw_2d_boxes(
        image.copy(),
        boxes_2d,
        labels=labels,
        colors=KITTI_COLORS,
        thickness=2,
        alpha=0.15,
    )
    result_2d = draw_text(
        result_2d,
        f"Frame: {frame_id} | 2D Boxes ({len(objects)} objects)",
        (10, 25),
        color=(255, 255, 255),
        font_scale=0.7,
        background=True,
    )

    # Add legend
    unique_labels = list(set(labels))
    result_2d = draw_legend(result_2d, unique_labels, KITTI_COLORS, position="top-right")

    save_path = output_dir / f"{frame_id}_boxes_2d.png"
    save_image(result_2d, save_path)
    print(f"  Saved: {save_path}")

    # 2. 3D Bounding Boxes
    print(f"  Drawing {len(objects)} 3D boxes...")

    # Project 3D box corners to image
    corners_list = []
    valid_labels = []
    for obj, label in zip(objects, labels):
        corners_3d = obj.get_3d_box_corners()  # (8, 3) in camera coords

        # Project to image
        corners_hom = np.hstack([corners_3d, np.ones((8, 1))])
        corners_2d_hom = (calib.P2 @ corners_hom.T).T

        # Skip if behind camera
        if np.any(corners_2d_hom[:, 2] <= 0):
            continue

        corners_2d = corners_2d_hom[:, :2] / corners_2d_hom[:, 2:3]
        corners_list.append(corners_2d)
        valid_labels.append(label)

    result_3d = draw_3d_boxes(
        image.copy(),
        corners_list,
        labels=valid_labels,
        colors=KITTI_COLORS,
        thickness=2,
    )
    result_3d = draw_text(
        result_3d,
        f"Frame: {frame_id} | 3D Boxes ({len(corners_list)} visible)",
        (10, 25),
        color=(255, 255, 255),
        font_scale=0.7,
        background=True,
    )
    result_3d = draw_legend(result_3d, unique_labels, KITTI_COLORS, position="top-right")

    save_path = output_dir / f"{frame_id}_boxes_3d.png"
    save_image(result_3d, save_path)
    print(f"  Saved: {save_path}")

    # 3. Combined: LiDAR + 3D Boxes
    print("  Creating combined LiDAR + 3D box visualization...")
    combined = project_points_to_image(
        image, sample["points"], calib,
        color_mode="depth",
        point_size=2,
    )
    combined = draw_3d_boxes(combined, corners_list, labels=valid_labels, colors=KITTI_COLORS)
    combined = draw_text(
        combined,
        f"Frame: {frame_id} | LiDAR + 3D Boxes",
        (10, 25),
        color=(255, 255, 255),
        font_scale=0.7,
        background=True,
    )
    combined = draw_legend(combined, unique_labels, KITTI_COLORS, position="top-right")

    save_path = output_dir / f"{frame_id}_combined.png"
    save_image(combined, save_path)
    print(f"  Saved: {save_path}")


def create_bev(
    sample: dict,
    output_dir: Path,
) -> None:
    """
    Demonstrate bird's eye view visualization.
    """
    print("\n--- Bird's Eye View ---")

    points = sample["points"]
    frame_id = sample["frame_id"]

    # 1. BEV with height coloring
    print("  Creating BEV image (height-colored)...")
    bev = create_bev_image(
        points,
        x_range=(0, 70),
        y_range=(-40, 40),
        z_range=(-3, 1),
        resolution=0.1,
        color_mode="height",
    )

    # Add info
    bev = cv2.flip(bev, 0)  # Flip so forward is up
    bev = draw_text(
        bev,
        f"BEV | Frame: {frame_id}",
        (10, 25),
        color=(255, 255, 255),
        font_scale=0.6,
        background=True,
    )
    bev = draw_text(
        bev,
        "Range: 70m x 80m",
        (10, 50),
        color=(200, 200, 200),
        font_scale=0.5,
        background=True,
    )

    save_path = output_dir / f"{frame_id}_bev.png"
    save_image(bev, save_path)
    print(f"  Saved: {save_path}")

    # 2. Side view
    print("  Creating side view image...")
    save_path = output_dir / f"{frame_id}_side.png"
    save_pointcloud_image(
        points, save_path,
        image_size=(800, 400),
        view="side",
        color_mode="height",
    )
    print(f"  Saved: {save_path}")


def create_fluent_interface(
    sample: dict,
    output_dir: Path,
) -> None:
    """
    Demonstrate the fluent ImageOverlay interface.
    """
    print("\n--- Fluent Interface ---")

    image = sample["image"]
    frame_id = sample["frame_id"]
    objects = sample.get("objects", [])
    objects = [obj for obj in objects if obj.type != "DontCare"]

    # Build visualization using fluent interface
    overlay = ImageOverlay(image)

    # Add LiDAR projection first (as background)
    overlay.image = project_points_to_image(
        overlay.image,
        sample["points"],
        sample["calib"],
        color_mode="depth",
        point_size=2,
        alpha=0.8,
    )

    if objects:
        # Add 2D boxes
        boxes_2d = np.array([obj.bbox2d for obj in objects])
        labels = [obj.type for obj in objects]

        overlay.draw_boxes(
            boxes_2d,
            labels=labels,
            colors=KITTI_COLORS,
            thickness=2,
            show_scores=False,
        )

        # Add legend
        unique_labels = list(set(labels))
        overlay.draw_legend(unique_labels, colors=KITTI_COLORS)

    # Add title
    overlay.draw_text(
        f"Multi-Sensor Fusion | Frame: {frame_id}",
        (10, 25),
        color=(255, 255, 255),
        font_scale=0.7,
    )

    # Add stats
    points = sample["points"]
    calib = sample["calib"]
    fov_count = calib.get_fov_mask(points, image.shape).sum()
    overlay.draw_text(
        f"LiDAR: {len(points):,} pts | FOV: {fov_count:,} | Objects: {len(objects)}",
        (10, 55),
        color=(200, 200, 200),
        font_scale=0.5,
    )

    # Save
    save_path = output_dir / f"{frame_id}_fluent.png"
    overlay.save(save_path)
    print(f"  Saved: {save_path}")


def visualize_3d(
    sample: dict,
) -> None:
    """
    Demonstrate interactive 3D visualization with Open3D.
    """
    print("\n--- 3D Visualization (Open3D) ---")

    if not HAS_OPEN3D:
        print("  Open3D not installed, skipping 3D visualization")
        print("  Install with: pip install open3d")
        return

    points = sample["points"]
    objects = sample.get("objects", [])

    # Prepare 3D boxes (in LiDAR coordinates)
    boxes = []
    box_colors = []
    for obj in objects:
        if obj.type == "DontCare":
            continue

        # Get corners in camera coordinates
        corners_cam = obj.get_3d_box_corners()  # (8, 3)

        # Transform to LiDAR coordinates (approximate inverse)
        # For visualization, we use camera coordinates directly
        boxes.append(corners_cam)

        color = KITTI_COLORS.get(obj.type, (255, 255, 255))
        box_colors.append(tuple(c / 255 for c in color))

    print(f"  Displaying {len(points):,} points with {len(boxes)} boxes")
    print("  Controls: Left-click + drag to rotate, scroll to zoom, Ctrl+click to pan")
    print("  Press Q or Esc to close window")

    # Note: boxes are in camera coords, points in LiDAR coords
    # For proper visualization, we'd need to transform one to the other
    # Here we just show points colored by height
    visualize_pointcloud(
        points,
        color_mode="height",
        window_name=f"KITTI Frame {sample['frame_id']}",
        point_size=2.0,
        show_axes=True,
        # boxes=boxes,  # Would need coordinate transform
        # box_colors=box_colors,
    )


def main():
    parser = argparse.ArgumentParser(
        description="KITTI Visualization Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/KITTI",
        help="Path to KITTI dataset",
    )
    parser.add_argument(
        "--frame-id",
        type=str,
        default=None,
        help="Specific frame ID to visualize (e.g., '000050')",
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=0,
        help="Frame index to visualize (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualization",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--no-3d",
        action="store_true",
        help="Skip Open3D 3D visualization",
    )

    args = parser.parse_args()

    # Check data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("\nTo download KITTI dataset, run:")
        print("  python scripts/download_kitti.py --quick")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize loader
    print("Loading KITTI dataset...")
    try:
        loader = KITTILoader(data_dir, split="training")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"Dataset loaded: {len(loader)} frames")

    # Load specific frame
    if args.frame_id:
        print(f"\nLoading frame: {args.frame_id}")
        try:
            sample = loader.get_frame_by_id(args.frame_id)
        except (ValueError, IndexError):
            print(f"Error: Frame {args.frame_id} not found")
            return 1
    else:
        print(f"\nLoading frame index: {args.frame_idx}")
        sample = loader[args.frame_idx]

    print(f"Frame ID: {sample['frame_id']}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Point cloud: {sample['points'].shape[0]:,} points")
    if "objects" in sample:
        print(f"Objects: {len(sample['objects'])}")

    # Run visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    create_lidar_overlay(sample, output_dir)
    create_bounding_boxes(sample, output_dir)
    create_bev(sample, output_dir)
    create_fluent_interface(sample, output_dir)

    if not args.no_3d:
        visualize_3d(sample)

    # Summary
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\nVisualizations saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob(f"{sample['frame_id']}_*.png")):
        print(f"  - {f.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
