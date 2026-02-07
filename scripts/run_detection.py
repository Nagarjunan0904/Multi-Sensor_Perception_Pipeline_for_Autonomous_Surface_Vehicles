#!/usr/bin/env python3
"""
2D Object Detection Pipeline for KITTI Dataset.

This script performs:
1. Loading KITTI sequences
2. Running YOLOv8 detection on each frame
3. Drawing professional bounding box visualizations
4. Creating output video with overlays
5. Computing detection statistics

Usage:
    python scripts/run_detection.py
    python scripts/run_detection.py --model yolov8x --num-frames 200
    python scripts/run_detection.py --no-video  # Just statistics
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.kitti_loader import KITTILoader
from perception2d.detector import ObjectDetector2D, Detection
from perception2d.postprocess import DetectionPostProcessor
from viz.image_overlay import (
    draw_2d_boxes,
    draw_text,
    draw_legend,
    save_image,
    KITTI_COLORS,
)


def create_info_panel(
    width: int,
    height: int,
    frame_idx: int,
    total_frames: int,
    fps: float,
    num_detections: int,
    class_counts: Dict[str, int],
) -> np.ndarray:
    """
    Create an information panel for the video overlay.

    Args:
        width: Panel width.
        height: Panel height.
        frame_idx: Current frame index.
        total_frames: Total number of frames.
        fps: Current processing FPS.
        num_detections: Number of detections in current frame.
        class_counts: Detection counts by class.

    Returns:
        RGB image of the info panel.
    """
    # Create dark background
    panel = np.full((height, width, 3), (30, 30, 30), dtype=np.uint8)

    # Draw border
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (80, 80, 80), 1)

    # Title
    cv2.putText(
        panel,
        "DETECTION STATS",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Frame counter
    y = 55
    cv2.putText(
        panel,
        f"Frame: {frame_idx + 1}/{total_frames}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # FPS
    y += 25
    fps_color = (0, 255, 127) if fps >= 20 else (255, 193, 37) if fps >= 10 else (255, 82, 82)
    cv2.putText(
        panel,
        f"FPS: {fps:.1f}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        fps_color,
        1,
        cv2.LINE_AA,
    )

    # Detection count
    y += 25
    cv2.putText(
        panel,
        f"Detections: {num_detections}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Class breakdown
    y += 30
    cv2.putText(
        panel,
        "Classes:",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )

    for class_name in ["Car", "Pedestrian", "Cyclist", "Van", "Truck"]:
        count = class_counts.get(class_name, 0)
        if count > 0 or class_name in ["Car", "Pedestrian", "Cyclist"]:
            y += 20
            color = KITTI_COLORS.get(class_name, (255, 255, 255))
            cv2.rectangle(panel, (10, y - 10), (22, y + 2), color, -1)
            cv2.putText(
                panel,
                f"{class_name}: {count}",
                (28, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

    return panel


def draw_frame_overlay(
    image: np.ndarray,
    detections: List[Detection],
    frame_idx: int,
    total_frames: int,
    fps: float,
    show_panel: bool = True,
) -> np.ndarray:
    """
    Draw complete frame overlay with detections and info panel.

    Args:
        image: Input RGB image.
        detections: List of detections.
        frame_idx: Current frame index.
        total_frames: Total frames.
        fps: Processing FPS.
        show_panel: Whether to show info panel.

    Returns:
        Annotated image.
    """
    result = image.copy()
    h, w = result.shape[:2]

    # Draw bounding boxes
    if detections:
        boxes = np.array([det.bbox for det in detections])
        labels = [det.class_name for det in detections]
        scores = np.array([det.confidence for det in detections])

        result = draw_2d_boxes(
            result,
            boxes,
            labels=labels,
            scores=scores,
            colors=KITTI_COLORS,
            thickness=2,
            font_scale=0.5,
            alpha=0.15,
        )

    # Draw legend in top-right
    unique_classes = list(set(det.class_name for det in detections)) if detections else []
    if unique_classes:
        result = draw_legend(
            result,
            unique_classes,
            colors=KITTI_COLORS,
            position="top-right",
            font_scale=0.45,
        )

    # Draw info panel
    if show_panel:
        class_counts = defaultdict(int)
        for det in detections:
            class_counts[det.class_name] += 1

        panel_width = 160
        panel_height = 220
        panel = create_info_panel(
            panel_width,
            panel_height,
            frame_idx,
            total_frames,
            fps,
            len(detections),
            dict(class_counts),
        )

        # Place panel in bottom-left
        x_offset = 10
        y_offset = h - panel_height - 10
        result[y_offset:y_offset + panel_height, x_offset:x_offset + panel_width] = panel

    # Draw frame number in top-left
    cv2.putText(
        result,
        f"KITTI 2D Object Detection",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Progress bar at bottom
    progress = (frame_idx + 1) / total_frames
    bar_width = w - 20
    bar_height = 4
    bar_y = h - 5

    # Background
    cv2.rectangle(
        result,
        (10, bar_y - bar_height),
        (10 + bar_width, bar_y),
        (60, 60, 60),
        -1,
    )
    # Progress
    cv2.rectangle(
        result,
        (10, bar_y - bar_height),
        (10 + int(bar_width * progress), bar_y),
        (0, 255, 127),
        -1,
    )

    return result


def run_detection(
    data_dir: Path,
    output_dir: Path,
    model_name: str = "yolov8m",
    confidence_threshold: float = 0.3,
    num_frames: int = 100,
    start_frame: int = 0,
    create_video: bool = True,
    video_fps: int = 30,
    device: str = "cuda",
    save_frames: bool = False,
) -> Dict:
    """
    Run detection pipeline on KITTI sequence.

    Args:
        data_dir: Path to KITTI dataset.
        output_dir: Output directory.
        model_name: YOLOv8 model variant.
        confidence_threshold: Detection confidence threshold.
        num_frames: Number of frames to process.
        start_frame: Starting frame index.
        create_video: Whether to create output video.
        video_fps: Output video FPS.
        device: Inference device.
        save_frames: Save individual frames as images.

    Returns:
        Dictionary with statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize loader
    print(f"Loading KITTI dataset from {data_dir}...")
    loader = KITTILoader(data_dir, split="training")
    total_available = len(loader)
    print(f"  Available frames: {total_available}")

    # Adjust frame range (-1 means all frames)
    if num_frames <= 0:
        num_frames = total_available - start_frame
    end_frame = min(start_frame + num_frames, total_available)
    actual_frames = end_frame - start_frame
    print(f"  Processing frames: {start_frame} to {end_frame - 1} ({actual_frames} frames)")

    # Initialize detector
    print(f"\nInitializing {model_name} detector...")
    detector = ObjectDetector2D(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        device=device,
    )

    # Warmup
    print("  Warming up model...")
    detector.warmup()

    # Initialize post-processor
    processor = DetectionPostProcessor(
        min_confidence=confidence_threshold,
        min_box_size=(20, 20),
        nms_threshold=0.5,
    )

    # Initialize video writer
    video_writer = None
    video_path = None
    use_imageio = False
    video_target_size = None
    if create_video:
        # Get frame size from first frame
        sample = loader[start_frame]
        h, w = sample["image"].shape[:2]
        video_path = output_dir / "detection_output.mp4"

        # Prefer imageio for more reliable video writing on Windows
        if HAS_IMAGEIO:
            try:
                # Ensure dimensions are divisible by 16 for codec compatibility
                target_h = ((h + 15) // 16) * 16
                target_w = ((w + 15) // 16) * 16
                video_writer = imageio.get_writer(
                    str(video_path),
                    fps=video_fps,
                    codec='libx264',
                    quality=8,  # Higher = better quality
                    pixelformat='yuv420p',  # Compatibility
                    macro_block_size=16,
                    output_params=['-s', f'{target_w}x{target_h}'],  # Force output size
                )
                use_imageio = True
                # Store target size for frame resizing
                video_target_size = (target_w, target_h)
                print(f"\nVideo will be saved to: {video_path} (using imageio)")
                if target_w != w or target_h != h:
                    print(f"  Note: Frames will be resized from {w}x{h} to {target_w}x{target_h} for codec compatibility")
            except Exception as e:
                print(f"  Warning: imageio failed ({e}), falling back to OpenCV")
                use_imageio = False

        # Fallback to OpenCV VideoWriter
        if not use_imageio:
            # Try different codecs for better compatibility
            codecs = [
                ('avc1', '.mp4'),  # H.264 - best compatibility
                ('XVID', '.avi'),  # XVID - reliable fallback
                ('mp4v', '.mp4'),  # MPEG-4
            ]
            for codec, ext in codecs:
                try:
                    test_path = output_dir / f"detection_output{ext}"
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    video_writer = cv2.VideoWriter(str(test_path), fourcc, video_fps, (w, h))
                    if video_writer.isOpened():
                        video_path = test_path
                        print(f"\nVideo will be saved to: {video_path} (codec: {codec})")
                        break
                    else:
                        video_writer.release()
                        video_writer = None
                except Exception:
                    continue

            if video_writer is None:
                print("\nWarning: Could not initialize video writer, skipping video creation")
                create_video = False

    # Statistics tracking
    stats = {
        "model": model_name,
        "confidence_threshold": confidence_threshold,
        "total_frames": actual_frames,
        "total_detections": 0,
        "detections_per_frame": [],
        "class_counts": defaultdict(int),
        "processing_times": [],
        "fps_history": [],
    }

    # Process frames
    print(f"\nProcessing {actual_frames} frames...")
    print("-" * 60)

    fps_window = []
    window_size = 10

    for i, frame_idx in enumerate(range(start_frame, end_frame)):
        # Load frame
        sample = loader[frame_idx]
        image = sample["image"]

        # Run detection
        t_start = time.perf_counter()
        detections = detector.detect(image)
        detections = processor.process(detections)
        t_end = time.perf_counter()

        # Calculate FPS
        frame_time = t_end - t_start
        stats["processing_times"].append(frame_time)
        fps_window.append(1.0 / frame_time if frame_time > 0 else 0)
        if len(fps_window) > window_size:
            fps_window.pop(0)
        current_fps = np.mean(fps_window)
        stats["fps_history"].append(current_fps)

        # Update statistics
        stats["total_detections"] += len(detections)
        stats["detections_per_frame"].append(len(detections))
        for det in detections:
            stats["class_counts"][det.class_name] += 1

        # Create visualization
        vis_image = draw_frame_overlay(
            image,
            detections,
            i,
            actual_frames,
            current_fps,
            show_panel=True,
        )

        # Write to video
        if video_writer is not None:
            if use_imageio:
                # Resize frame if needed for codec compatibility
                frame_to_write = vis_image
                if video_target_size is not None:
                    target_w, target_h = video_target_size
                    if frame_to_write.shape[1] != target_w or frame_to_write.shape[0] != target_h:
                        frame_to_write = cv2.resize(frame_to_write, (target_w, target_h))
                # imageio expects RGB
                video_writer.append_data(frame_to_write)
            else:
                # OpenCV expects BGR
                video_writer.write(cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

        # Save frame
        if save_frames:
            frame_path = output_dir / "frames" / f"frame_{frame_idx:06d}.jpg"
            frame_path.parent.mkdir(exist_ok=True)
            save_image(vis_image, frame_path, quality=90)

        # Progress output
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  Frame {i + 1:4d}/{actual_frames} | "
                f"Detections: {len(detections):2d} | "
                f"FPS: {current_fps:5.1f}"
            )

    # Cleanup
    if video_writer is not None:
        if use_imageio:
            video_writer.close()
        else:
            video_writer.release()

    # Compute final statistics
    stats["class_counts"] = dict(stats["class_counts"])
    stats["avg_detections_per_frame"] = np.mean(stats["detections_per_frame"])
    stats["avg_fps"] = np.mean(stats["fps_history"])
    stats["avg_processing_time_ms"] = np.mean(stats["processing_times"]) * 1000
    stats["min_fps"] = min(stats["fps_history"])
    stats["max_fps"] = max(stats["fps_history"])

    return stats


def print_statistics(stats: Dict) -> None:
    """Print formatted statistics."""
    print("\n" + "=" * 60)
    print("DETECTION STATISTICS")
    print("=" * 60)

    print(f"\nModel: {stats['model']}")
    print(f"Confidence Threshold: {stats['confidence_threshold']}")
    print(f"Frames Processed: {stats['total_frames']}")

    print(f"\nPerformance:")
    print(f"  Average FPS: {stats['avg_fps']:.1f}")
    print(f"  Min FPS: {stats['min_fps']:.1f}")
    print(f"  Max FPS: {stats['max_fps']:.1f}")
    print(f"  Avg Processing Time: {stats['avg_processing_time_ms']:.1f} ms")

    print(f"\nDetections:")
    print(f"  Total: {stats['total_detections']}")
    print(f"  Average per Frame: {stats['avg_detections_per_frame']:.1f}")

    print(f"\nClass Distribution:")
    total = sum(stats["class_counts"].values())
    for class_name, count in sorted(
        stats["class_counts"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        pct = 100 * count / total if total > 0 else 0
        print(f"  {class_name:15s}: {count:5d} ({pct:5.1f}%)")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="2D Object Detection on KITTI Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/KITTI",
        help="Path to KITTI dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/detection",
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="YOLOv8 model variant",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=-1,
        help="Number of frames to process (-1 for all frames)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame index",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=30,
        help="Output video FPS",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device (cuda/cpu)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video creation",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save individual frames as images",
    )

    args = parser.parse_args()

    # Check data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("\nTo download KITTI dataset, run:")
        print("  python scripts/download_kitti.py --quick")
        return 1

    output_dir = Path(args.output_dir)

    # Run detection pipeline
    try:
        stats = run_detection(
            data_dir=data_dir,
            output_dir=output_dir,
            model_name=args.model,
            confidence_threshold=args.confidence,
            num_frames=args.num_frames,
            start_frame=args.start_frame,
            create_video=not args.no_video,
            video_fps=args.video_fps,
            device=args.device,
            save_frames=args.save_frames,
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print statistics
    print_statistics(stats)

    # Save statistics
    stats_path = output_dir / "detection_stats.json"
    # Remove numpy arrays for JSON serialization
    json_stats = {
        k: v for k, v in stats.items()
        if k not in ["detections_per_frame", "processing_times", "fps_history"]
    }
    with open(stats_path, "w") as f:
        json.dump(json_stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_path}")

    if not args.no_video:
        # Find the actual video file created
        video_files = list(output_dir.glob("detection_output.*"))
        video_files = [f for f in video_files if f.suffix in ['.mp4', '.avi']]
        if video_files:
            print(f"Video saved to: {video_files[0]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
