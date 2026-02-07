#!/usr/bin/env python3
"""
Test script for KITTI data loading functionality.

This script tests:
1. Loading KITTI frames (image, point cloud, calibration, labels)
2. Projecting point cloud onto image
3. Visualizing 3D bounding boxes
4. Saving overlay visualizations

Usage:
    python scripts/test_data_loader.py
    python scripts/test_data_loader.py --data-dir /path/to/KITTI
    python scripts/test_data_loader.py --num-frames 10 --output-dir output/tests
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.kitti_loader import KITTILoader


def colorize_depth(depth: np.ndarray, max_depth: float = 50.0) -> np.ndarray:
    """
    Colorize depth values using a colormap.

    Args:
        depth: (N,) array of depth values.
        max_depth: Maximum depth for normalization.

    Returns:
        colors: (N, 3) array of RGB colors.
    """
    # Normalize depth to [0, 1]
    depth_norm = np.clip(depth / max_depth, 0, 1)

    # Apply colormap (use turbo-like coloring: blue=near, red=far)
    colors = np.zeros((len(depth), 3), dtype=np.uint8)
    colors[:, 0] = (255 * depth_norm).astype(np.uint8)  # R increases with depth
    colors[:, 2] = (255 * (1 - depth_norm)).astype(np.uint8)  # B decreases with depth
    colors[:, 1] = (255 * (1 - np.abs(depth_norm - 0.5) * 2)).astype(np.uint8)  # G peaks at mid-depth

    return colors


def project_points_to_image(
    image: np.ndarray,
    points: np.ndarray,
    calib,
    point_size: int = 2,
) -> np.ndarray:
    """
    Project point cloud onto image with depth coloring.

    Args:
        image: (H, W, 3) RGB image.
        points: (N, 4) point cloud [x, y, z, intensity].
        calib: Calibration object.
        point_size: Size of projected points.

    Returns:
        overlay: Image with projected points.
    """
    # Get points in FOV
    mask = calib.get_fov_mask(points, image.shape)
    points_fov = points[mask]

    if len(points_fov) == 0:
        return image.copy()

    # Project to image
    pts_2d = calib.project_velo_to_image(points_fov)
    pts_rect = calib.project_velo_to_rect(points_fov)

    # Get depth (z in camera coordinates)
    depth = pts_rect[:, 2]

    # Colorize by depth
    colors = colorize_depth(depth)

    # Draw on image
    overlay = image.copy()
    for i in range(len(pts_2d)):
        u, v = int(pts_2d[i, 0]), int(pts_2d[i, 1])
        if 0 <= u < overlay.shape[1] and 0 <= v < overlay.shape[0]:
            color = tuple(map(int, colors[i]))
            cv2.circle(overlay, (u, v), point_size, color, -1)

    return overlay


def draw_3d_box(
    image: np.ndarray,
    corners_2d: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw 3D bounding box on image.

    Args:
        image: Image to draw on.
        corners_2d: (8, 2) projected corner coordinates.
        color: Box color (RGB).
        thickness: Line thickness.

    Returns:
        Image with box drawn.
    """
    corners_2d = corners_2d.astype(int)

    # Draw bottom face (0-1-2-3)
    for i in range(4):
        pt1 = tuple(corners_2d[i])
        pt2 = tuple(corners_2d[(i + 1) % 4])
        cv2.line(image, pt1, pt2, color, thickness)

    # Draw top face (4-5-6-7)
    for i in range(4):
        pt1 = tuple(corners_2d[i + 4])
        pt2 = tuple(corners_2d[(i + 1) % 4 + 4])
        cv2.line(image, pt1, pt2, color, thickness)

    # Draw vertical edges
    for i in range(4):
        pt1 = tuple(corners_2d[i])
        pt2 = tuple(corners_2d[i + 4])
        cv2.line(image, pt1, pt2, color, thickness)

    return image


def draw_objects(
    image: np.ndarray,
    objects: list,
    calib,
) -> np.ndarray:
    """
    Draw 3D object boxes on image.

    Args:
        image: Image to draw on.
        objects: List of Object3D instances.
        calib: Calibration object.

    Returns:
        Image with objects drawn.
    """
    # Color map for object types
    colors = {
        "Car": (0, 255, 0),  # Green
        "Pedestrian": (255, 0, 0),  # Red
        "Cyclist": (0, 0, 255),  # Blue
        "Van": (0, 255, 255),  # Cyan
        "Truck": (255, 255, 0),  # Yellow
        "Person_sitting": (255, 0, 255),  # Magenta
        "Tram": (128, 128, 0),  # Olive
        "Misc": (128, 128, 128),  # Gray
        "DontCare": (64, 64, 64),  # Dark gray
    }

    result = image.copy()

    for obj in objects:
        if obj.type == "DontCare":
            continue

        # Get 3D corners
        corners_3d = obj.get_3d_box_corners()  # (8, 3) in camera coords

        # Add homogeneous coordinate
        corners_3d_hom = np.hstack([corners_3d, np.ones((8, 1))])  # (8, 4)

        # Project to image using P2 (corners are already in camera coords)
        corners_2d_hom = (calib.P2 @ corners_3d_hom.T).T  # (8, 3)

        # Skip if any point is behind camera
        if np.any(corners_2d_hom[:, 2] <= 0):
            continue

        # Normalize
        corners_2d = corners_2d_hom[:, :2] / corners_2d_hom[:, 2:3]  # (8, 2)

        # Check if box is within image
        h, w = image.shape[:2]
        if (corners_2d[:, 0].max() < 0 or corners_2d[:, 0].min() > w or
                corners_2d[:, 1].max() < 0 or corners_2d[:, 1].min() > h):
            continue

        # Draw box
        color = colors.get(obj.type, (255, 255, 255))
        result = draw_3d_box(result, corners_2d, color, thickness=2)

        # Draw label
        label_pos = (int(corners_2d[4, 0]), int(corners_2d[4, 1]) - 5)
        cv2.putText(
            result,
            f"{obj.type}",
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return result


def print_frame_info(sample: dict, idx: int):
    """Print information about a frame."""
    print(f"\n{'='*60}")
    print(f"Frame {idx}: {sample['frame_id']}")
    print('='*60)

    # Image info
    image = sample["image"]
    print(f"\nImage:")
    print(f"  Shape: {image.shape}")
    print(f"  Dtype: {image.dtype}")
    print(f"  Range: [{image.min()}, {image.max()}]")

    # Point cloud info
    points = sample["points"]
    print(f"\nPoint Cloud:")
    print(f"  Num points: {points.shape[0]:,}")
    print(f"  Shape: {points.shape}")
    print(f"  Dtype: {points.dtype}")
    print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] m")
    print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] m")
    print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}] m")
    print(f"  Intensity range: [{points[:, 3].min():.3f}, {points[:, 3].max():.3f}]")

    # Calibration info
    calib = sample["calib"]
    print(f"\nCalibration:")
    print(f"  P2 shape: {calib.P2.shape}")
    print(f"  R0_rect shape: {calib.R0_rect.shape}")
    print(f"  Tr_velo_to_cam shape: {calib.Tr_velo_to_cam.shape}")

    # Objects info
    if "objects" in sample:
        objects = sample["objects"]
        print(f"\nObjects: {len(objects)}")
        type_counts = {}
        for obj in objects:
            type_counts[obj.type] = type_counts.get(obj.type, 0) + 1
        for obj_type, count in sorted(type_counts.items()):
            print(f"  {obj_type}: {count}")

    # FOV points
    mask = calib.get_fov_mask(points, image.shape)
    print(f"\nPoints in camera FOV: {mask.sum():,} ({100*mask.sum()/len(points):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Test KITTI data loading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/KITTI",
        help="Path to KITTI dataset directory",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=5,
        help="Number of frames to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/data_test",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save visualization images",
    )

    args = parser.parse_args()

    # Check data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("\nTo download KITTI dataset, run:")
        print("  python scripts/download_kitti.py --quick  # For 100 samples")
        print("  python scripts/download_kitti.py          # For full dataset")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize loader
    print("Initializing KITTI loader...")
    try:
        loader = KITTILoader(data_dir, split="training")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"Dataset loaded: {len(loader)} frames")

    # Process frames
    num_frames = min(args.num_frames, len(loader))
    print(f"\nProcessing {num_frames} frames...")

    for idx in range(num_frames):
        sample = loader[idx]
        print_frame_info(sample, idx)

        if not args.no_save:
            # Create point cloud overlay
            overlay = project_points_to_image(
                sample["image"],
                sample["points"],
                sample["calib"],
            )

            # Draw 3D boxes if available
            if "objects" in sample and len(sample["objects"]) > 0:
                overlay = draw_objects(overlay, sample["objects"], sample["calib"])

            # Save visualization
            output_path = output_dir / f"frame_{sample['frame_id']}.png"
            # Convert RGB to BGR for OpenCV
            cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"\nSaved: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    stats = loader.statistics()
    print(f"Split: {stats['split']}")
    print(f"Total frames: {stats['num_frames']}")
    print(f"Image shape: {stats.get('image_shape', 'N/A')}")
    print(f"Points per frame (sample): {stats.get('num_points_sample', 'N/A'):,}")

    if not args.no_save:
        print(f"\nVisualizations saved to: {output_dir.absolute()}")

    print("\nData loader test complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
