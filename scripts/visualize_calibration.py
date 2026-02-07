#!/usr/bin/env python
"""
Camera-LiDAR Calibration Visualization Tool.

Visualizes and validates camera-LiDAR calibration by projecting LiDAR points
onto camera images and displaying the camera frustum in 3D space.

Features:
    - LiDAR to camera projection with depth-colored overlay
    - Camera frustum visualization in 3D
    - Projection accuracy metrics
    - Batch processing for multiple frames
    - Video output generation

Usage:
    # Visualize single frame
    python scripts/visualize_calibration.py --frame_idx 0

    # Visualize frame range
    python scripts/visualize_calibration.py --start_frame 0 --end_frame 100

    # Generate video output
    python scripts/visualize_calibration.py --start_frame 0 --end_frame 500 --video

    # Show 3D visualization (requires Open3D)
    python scripts/visualize_calibration.py --frame_idx 0 --show_3d
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kitti_loader import KITTILoader, Calibration
from src.calibration.intrinsics import CameraIntrinsics
from src.calibration.projection import compute_frustum_points


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize camera-LiDAR calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/kitti",
        help="Path to KITTI dataset directory (default: data/kitti)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        choices=["training", "testing"],
        help="Dataset split (default: training)",
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=None,
        help="Single frame index to visualize",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Start frame index for batch processing (default: 0)",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=None,
        help="End frame index for batch processing (default: all frames)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/calibration",
        help="Output directory for visualizations (default: outputs/calibration)",
    )
    parser.add_argument(
        "--depth_range",
        type=float,
        nargs=2,
        default=[0.5, 60.0],
        metavar=("MIN", "MAX"),
        help="Depth range for coloring in meters (default: 0.5 60.0)",
    )
    parser.add_argument(
        "--point_size",
        type=int,
        default=2,
        help="Point size for LiDAR overlay (default: 2)",
    )
    parser.add_argument(
        "--show_3d",
        action="store_true",
        help="Show 3D visualization using Open3D",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Generate video output from frame range",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Video frame rate (default: 10)",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Don't display images interactively",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save individual frame images",
    )
    return parser.parse_args()


def depth_to_color(depths: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    """
    Convert depth values to BGR colors using jet colormap.

    Args:
        depths: Array of depth values (N,).
        min_depth: Minimum depth for normalization.
        max_depth: Maximum depth for normalization.

    Returns:
        BGR colors as (N, 3) uint8 array.
    """
    depths_norm = np.clip((depths - min_depth) / (max_depth - min_depth), 0, 1)
    colors = cv2.applyColorMap(
        (depths_norm * 255).astype(np.uint8).reshape(-1, 1),
        cv2.COLORMAP_JET
    ).reshape(-1, 3)
    return colors


def overlay_lidar_on_image(
    image: np.ndarray,
    points_2d: np.ndarray,
    depths: np.ndarray,
    depth_range: Tuple[float, float] = (0.5, 60.0),
    point_size: int = 2,
) -> np.ndarray:
    """
    Overlay LiDAR points on image with depth-based coloring.

    Args:
        image: RGB image (H, W, 3).
        points_2d: Projected 2D points (N, 2).
        depths: Depth values (N,).
        depth_range: (min, max) depth for color normalization.
        point_size: Radius of drawn points.

    Returns:
        Image with LiDAR overlay (BGR format for cv2).
    """
    # Convert RGB to BGR for OpenCV
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if len(points_2d) == 0:
        return result

    colors = depth_to_color(depths, depth_range[0], depth_range[1])

    # Sort by depth (draw far points first, close points on top)
    sort_idx = np.argsort(-depths)

    for idx in sort_idx:
        u, v = int(points_2d[idx, 0]), int(points_2d[idx, 1])
        color = colors[idx].tolist()
        cv2.circle(result, (u, v), point_size, color, -1)

    return result


def add_colorbar(
    image: np.ndarray,
    min_val: float,
    max_val: float,
    label: str = "Depth (m)",
) -> np.ndarray:
    """Add depth colorbar to image."""
    h, w = image.shape[:2]
    bar_width, bar_height = 25, min(200, h - 60)
    bar_x = w - bar_width - 15
    bar_y = (h - bar_height) // 2

    # Background
    cv2.rectangle(
        image,
        (bar_x - 5, bar_y - 25),
        (bar_x + bar_width + 40, bar_y + bar_height + 25),
        (255, 255, 255), -1
    )

    # Gradient
    for i in range(bar_height):
        norm_val = 1 - i / bar_height
        color_idx = int(norm_val * 255)
        color = cv2.applyColorMap(
            np.array([[color_idx]], dtype=np.uint8),
            cv2.COLORMAP_JET
        )[0, 0].tolist()
        cv2.line(image, (bar_x, bar_y + i), (bar_x + bar_width, bar_y + i), color, 1)

    # Border and labels
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 1)
    cv2.putText(image, f"{max_val:.0f}m", (bar_x + bar_width + 3, bar_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(image, f"{min_val:.0f}m", (bar_x + bar_width + 3, bar_y + bar_height + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(image, label, (bar_x - 5, bar_y + bar_height + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    return image


def add_info_overlay(image: np.ndarray, info: dict) -> np.ndarray:
    """Add information overlay to image."""
    # Semi-transparent background
    overlay = image.copy()
    box_height = 25 + 18 * len(info)
    cv2.rectangle(overlay, (8, 8), (280, box_height), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    # Text
    y = 25
    for key, value in info.items():
        cv2.putText(image, f"{key}: {value}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        y += 18

    return image


def compute_projection_metrics(
    calib: Calibration,
    points: np.ndarray,
    image_shape: Tuple[int, int],
) -> dict:
    """
    Compute projection metrics for a frame.

    Args:
        calib: Calibration object.
        points: LiDAR points (N, 4).
        image_shape: (height, width).

    Returns:
        Dictionary with projection metrics.
    """
    # Get FOV mask
    fov_mask = calib.get_fov_mask(points, image_shape)
    points_fov = points[fov_mask]

    # Get depths in camera frame
    pts_rect = calib.project_velo_to_rect(points_fov)
    depths = pts_rect[:, 2]

    return {
        "total_points": len(points),
        "points_in_fov": len(points_fov),
        "fov_percentage": f"{100 * len(points_fov) / len(points):.1f}%",
        "depth_min": f"{depths.min():.1f}m" if len(depths) > 0 else "N/A",
        "depth_max": f"{depths.max():.1f}m" if len(depths) > 0 else "N/A",
        "depth_mean": f"{depths.mean():.1f}m" if len(depths) > 0 else "N/A",
    }


def visualize_frame(
    loader: KITTILoader,
    frame_idx: int,
    depth_range: Tuple[float, float],
    point_size: int,
) -> Tuple[np.ndarray, dict]:
    """
    Create visualization for a single frame.

    Args:
        loader: KITTI data loader.
        frame_idx: Frame index.
        depth_range: (min, max) depth for coloring.
        point_size: Point size for overlay.

    Returns:
        Tuple of (visualization image, info dict).
    """
    # Load frame data
    sample = loader[frame_idx]
    image = sample["image"]
    points = sample["points"]
    calib = sample["calib"]
    frame_id = sample["frame_id"]

    # Get FOV mask and project points
    fov_mask = calib.get_fov_mask(points, image.shape)
    points_fov = points[fov_mask]

    # Project to image and get depths
    points_2d = calib.project_velo_to_image(points_fov)
    pts_rect = calib.project_velo_to_rect(points_fov)
    depths = pts_rect[:, 2]

    # Filter by depth range
    depth_mask = (depths >= depth_range[0]) & (depths <= depth_range[1])
    points_2d = points_2d[depth_mask]
    depths = depths[depth_mask]

    # Create visualization
    result = overlay_lidar_on_image(image, points_2d, depths, depth_range, point_size)
    result = add_colorbar(result, depth_range[0], depth_range[1])

    # Compute metrics
    metrics = compute_projection_metrics(calib, points, image.shape)

    # Get camera parameters
    fx = calib.P2[0, 0]
    fy = calib.P2[1, 1]
    cx = calib.P2[0, 2]
    cy = calib.P2[1, 2]
    h_fov = np.degrees(2 * np.arctan(image.shape[1] / (2 * fx)))
    v_fov = np.degrees(2 * np.arctan(image.shape[0] / (2 * fy)))

    # Info overlay
    info = {
        "Frame": frame_id,
        "Points in FOV": f"{metrics['points_in_fov']} ({metrics['fov_percentage']})",
        "Depth range": f"{depth_range[0]:.1f} - {depth_range[1]:.1f}m",
        "H-FOV / V-FOV": f"{h_fov:.1f}° / {v_fov:.1f}°",
    }
    result = add_info_overlay(result, info)

    return result, metrics


def visualize_3d(
    loader: KITTILoader,
    frame_idx: int,
    depth_range: Tuple[float, float],
):
    """
    Show 3D visualization of LiDAR points and camera frustum.

    Args:
        loader: KITTI data loader.
        frame_idx: Frame index.
        depth_range: Depth range for frustum visualization.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not installed. Install with: pip install open3d")
        return

    # Load data
    sample = loader[frame_idx]
    points = sample["points"]
    calib = sample["calib"]
    image_shape = sample["image"].shape

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # Color by height (Z in Velodyne = up)
    z_vals = points[:, 2]
    z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-10)
    colors = np.zeros((len(points), 3))
    colors[:, 0] = z_norm  # Red channel based on height
    colors[:, 2] = 1 - z_norm  # Blue channel inverse
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create camera frustum
    # Get intrinsics from P2
    fx = calib.P2[0, 0]
    fy = calib.P2[1, 1]
    cx = calib.P2[0, 2]
    cy = calib.P2[1, 2]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Compute frustum corners at max depth
    frustum_cam = compute_frustum_points(image_shape[:2], K, depth_range[1])

    # Transform frustum from camera to Velodyne frame
    # Inverse of Tr_velo_to_cam
    R = calib.Tr_velo_to_cam[:, :3]
    t = calib.Tr_velo_to_cam[:, 3]
    R0 = calib.R0_rect

    # Camera to unrectified camera
    frustum_unrect = frustum_cam @ R0.T

    # Unrectified camera to Velodyne
    R_inv = R.T
    t_inv = -R_inv @ t
    frustum_velo = frustum_unrect @ R_inv.T + t_inv

    # Camera origin in Velodyne frame
    cam_origin_velo = t_inv

    # Create frustum lines
    frustum_all = np.vstack([cam_origin_velo.reshape(1, 3), frustum_velo])
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Camera to corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # Frustum edges
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(frustum_all)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))

    # Coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)

    print("\n3D Visualization Controls:")
    print("  Mouse: Rotate (left), Pan (right), Zoom (scroll)")
    print("  Q: Close window")
    print(f"\nViewing frame {sample['frame_id']}")
    print(f"  {len(points)} LiDAR points")
    print(f"  Camera frustum at {depth_range[1]}m depth")

    o3d.visualization.draw_geometries(
        [pcd, line_set, coord_frame],
        window_name=f"Frame {sample['frame_id']} - Camera Frustum and LiDAR",
        width=1280,
        height=720,
    )


def process_frames(
    loader: KITTILoader,
    args,
    output_dir: Path,
):
    """
    Process multiple frames for visualization.

    Args:
        loader: KITTI data loader.
        args: Command line arguments.
        output_dir: Output directory.
    """
    start_frame = args.start_frame
    end_frame = args.end_frame if args.end_frame is not None else len(loader)
    end_frame = min(end_frame, len(loader))

    print(f"\nProcessing frames {start_frame} to {end_frame - 1} ({end_frame - start_frame} frames)")

    video_writer = None
    if args.video:
        sample = loader[start_frame]
        h, w = sample["image"].shape[:2]
        video_path = output_dir / "calibration_visualization.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
        print(f"Writing video to: {video_path}")

    for idx in range(start_frame, end_frame):
        result, metrics = visualize_frame(
            loader, idx, tuple(args.depth_range), args.point_size
        )

        if args.save_images:
            frame_id = loader.frame_ids[idx]
            output_path = output_dir / "frames" / f"{frame_id}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), result)

        if video_writer is not None:
            video_writer.write(result)

        if not args.no_display:
            cv2.imshow("Calibration Visualization", result)
            key = cv2.waitKey(1 if args.video else 0)
            if key == ord("q") or key == 27:  # q or ESC
                break
            elif key == ord("s"):  # Save current frame
                frame_id = loader.frame_ids[idx]
                save_path = output_dir / f"frame_{frame_id}.png"
                cv2.imwrite(str(save_path), result)
                print(f"Saved: {save_path}")

        # Progress
        if (idx - start_frame + 1) % 50 == 0:
            print(f"  Processed {idx - start_frame + 1}/{end_frame - start_frame} frames")

    if video_writer is not None:
        video_writer.release()
        print(f"Video saved: {output_dir / 'calibration_visualization.mp4'}")

    if not args.no_display:
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize data loader
    print(f"Loading KITTI dataset from: {args.data_dir}")
    print(f"Split: {args.split}")

    try:
        loader = KITTILoader(args.data_dir, split=args.split)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo use this tool, ensure KITTI dataset is available at the specified path.")
        print("Download KITTI Object Detection dataset from: http://www.cvlibs.net/datasets/kitti/")
        sys.exit(1)

    print(f"Found {len(loader)} frames")

    # Single frame mode
    if args.frame_idx is not None:
        print(f"\nVisualizing frame {args.frame_idx}")

        result, metrics = visualize_frame(
            loader, args.frame_idx, tuple(args.depth_range), args.point_size
        )

        # Print metrics
        print("\nProjection Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Save
        frame_id = loader.frame_ids[args.frame_idx]
        output_path = output_dir / f"calibration_{frame_id}.png"
        cv2.imwrite(str(output_path), result)
        print(f"\nSaved: {output_path}")

        # Display
        if not args.no_display:
            cv2.imshow("Calibration Visualization", result)
            print("\nPress any key to close (or 's' to save, 'q' to quit)")
            key = cv2.waitKey(0)
            if key == ord("s"):
                extra_path = output_dir / f"calibration_{frame_id}_saved.png"
                cv2.imwrite(str(extra_path), result)
                print(f"Saved: {extra_path}")
            cv2.destroyAllWindows()

        # 3D visualization
        if args.show_3d:
            visualize_3d(loader, args.frame_idx, tuple(args.depth_range))

    # Batch processing mode
    else:
        process_frames(loader, args, output_dir)


if __name__ == "__main__":
    main()
