#!/usr/bin/env python3
"""
2D Detection Evaluation on KITTI Dataset.

This script evaluates YOLOv8 detections against KITTI ground truth labels:
1. Calculates precision, recall, F1 score
2. Generates confusion matrix
3. Computes AP (Average Precision) per class
4. Analyzes detection quality by distance/size

Usage:
    python scripts/eval_2d_detection.py
    python scripts/eval_2d_detection.py --model yolov8x --num-frames 500
    python scripts/eval_2d_detection.py --iou-threshold 0.7

Output:
    outputs/metrics_2d.json - Evaluation metrics
    outputs/confusion_matrix.png - Confusion matrix visualization
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.kitti_loader import KITTILoader, Object3D
from perception2d.detector import ObjectDetector2D, Detection
from perception2d.postprocess import DetectionPostProcessor, compute_iou


# KITTI class mapping for evaluation
EVAL_CLASSES = ["Car", "Pedestrian", "Cyclist"]


def convert_kitti_to_detection(obj: Object3D) -> Detection:
    """
    Convert KITTI Object3D ground truth to Detection format.

    Args:
        obj: KITTI 3D object annotation.

    Returns:
        Detection object with 2D bbox.
    """
    return Detection(
        bbox=obj.bbox2d.copy(),
        class_id=-1,  # Ground truth doesn't have COCO ID
        class_name=obj.type,
        confidence=1.0,  # Ground truth has confidence 1.0
    )


def match_detections_to_ground_truth(
    detections: List[Detection],
    ground_truth: List[Detection],
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple], List[int], List[int]]:
    """
    Match detections to ground truth using IoU.

    Uses greedy matching: highest IoU pairs are matched first.

    Args:
        detections: Predicted detections.
        ground_truth: Ground truth annotations.
        iou_threshold: Minimum IoU for a valid match.

    Returns:
        matches: List of (det_idx, gt_idx, iou) tuples.
        unmatched_dets: Indices of unmatched detections (false positives).
        unmatched_gts: Indices of unmatched ground truths (false negatives).
    """
    if len(detections) == 0 or len(ground_truth) == 0:
        return [], list(range(len(detections))), list(range(len(ground_truth)))

    # Compute IoU matrix
    det_boxes = np.array([d.bbox for d in detections])
    gt_boxes = np.array([g.bbox for g in ground_truth])

    # Compute all pairwise IoUs
    iou_matrix = np.zeros((len(detections), len(ground_truth)))
    for i, det_box in enumerate(det_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(det_box, gt_box)

    # Greedy matching
    matches = []
    matched_dets = set()
    matched_gts = set()

    # Sort all IoU pairs by value (descending)
    pairs = []
    for i in range(len(detections)):
        for j in range(len(ground_truth)):
            if iou_matrix[i, j] >= iou_threshold:
                # Only match same class
                if detections[i].class_name == ground_truth[j].class_name:
                    pairs.append((iou_matrix[i, j], i, j))

    pairs.sort(reverse=True)

    for iou, det_idx, gt_idx in pairs:
        if det_idx not in matched_dets and gt_idx not in matched_gts:
            matches.append((det_idx, gt_idx, iou))
            matched_dets.add(det_idx)
            matched_gts.add(gt_idx)

    unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
    unmatched_gts = [i for i in range(len(ground_truth)) if i not in matched_gts]

    return matches, unmatched_dets, unmatched_gts


def compute_precision_recall(
    all_detections: List[List[Detection]],
    all_ground_truth: List[List[Detection]],
    iou_threshold: float = 0.5,
    class_name: Optional[str] = None,
) -> Dict:
    """
    Compute precision and recall across all frames.

    Args:
        all_detections: List of detection lists per frame.
        all_ground_truth: List of ground truth lists per frame.
        iou_threshold: IoU threshold for matching.
        class_name: Specific class to evaluate (None for all).

    Returns:
        Dictionary with precision, recall, F1, AP metrics.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # For AP calculation
    all_scores = []
    all_matches = []

    for dets, gts in zip(all_detections, all_ground_truth):
        # Filter by class if specified
        if class_name:
            dets = [d for d in dets if d.class_name == class_name]
            gts = [g for g in gts if g.class_name == class_name]

        matches, unmatched_dets, unmatched_gts = match_detections_to_ground_truth(
            dets, gts, iou_threshold
        )

        true_positives += len(matches)
        false_positives += len(unmatched_dets)
        false_negatives += len(unmatched_gts)

        # Track scores for AP
        for det_idx, gt_idx, iou in matches:
            all_scores.append(dets[det_idx].confidence)
            all_matches.append(1)  # True positive

        for det_idx in unmatched_dets:
            all_scores.append(dets[det_idx].confidence)
            all_matches.append(0)  # False positive

    # Compute metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Compute AP
    ap = compute_ap(all_scores, all_matches, true_positives + false_negatives)

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": ap,
    }


def compute_ap(
    scores: List[float],
    matches: List[int],
    num_gt: int,
) -> float:
    """
    Compute Average Precision using the 11-point interpolation method.

    Args:
        scores: Confidence scores for each detection.
        matches: 1 for true positive, 0 for false positive.
        num_gt: Total number of ground truth objects.

    Returns:
        Average Precision value.
    """
    if len(scores) == 0 or num_gt == 0:
        return 0.0

    # Sort by score descending
    sorted_indices = np.argsort(scores)[::-1]
    sorted_matches = np.array(matches)[sorted_indices]

    # Compute precision and recall at each threshold
    tp_cumsum = np.cumsum(sorted_matches)
    fp_cumsum = np.cumsum(1 - sorted_matches)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / num_gt

    # 11-point interpolation
    ap = 0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max()

    ap /= 11

    return ap


def compute_confusion_matrix(
    all_detections: List[List[Detection]],
    all_ground_truth: List[List[Detection]],
    classes: List[str],
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """
    Compute confusion matrix for object detection.

    Matrix[i, j] = number of GT class i predicted as class j.
    Extra row for false negatives (missed), extra column for false positives.

    Args:
        all_detections: Detection lists per frame.
        all_ground_truth: Ground truth lists per frame.
        classes: List of class names.
        iou_threshold: IoU threshold for matching.

    Returns:
        Confusion matrix as numpy array.
    """
    n_classes = len(classes)
    # Matrix: rows = GT class + "missed", cols = pred class + "background"
    matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)

    class_to_idx = {c: i for i, c in enumerate(classes)}

    for dets, gts in zip(all_detections, all_ground_truth):
        # Filter to evaluation classes
        dets = [d for d in dets if d.class_name in classes]
        gts = [g for g in gts if g.class_name in classes]

        matches, unmatched_dets, unmatched_gts = match_detections_to_ground_truth(
            dets, gts, iou_threshold
        )

        # True positives (matched)
        for det_idx, gt_idx, _ in matches:
            gt_class = gts[gt_idx].class_name
            pred_class = dets[det_idx].class_name

            gt_i = class_to_idx.get(gt_class, n_classes)
            pred_j = class_to_idx.get(pred_class, n_classes)

            if gt_class == pred_class:
                matrix[gt_i, pred_j] += 1
            else:
                # Class confusion (matched but wrong class)
                matrix[gt_i, pred_j] += 1

        # False negatives (missed GT)
        for gt_idx in unmatched_gts:
            gt_class = gts[gt_idx].class_name
            gt_i = class_to_idx.get(gt_class, n_classes)
            matrix[gt_i, n_classes] += 1  # Missed (background prediction)

        # False positives
        for det_idx in unmatched_dets:
            pred_class = dets[det_idx].class_name
            pred_j = class_to_idx.get(pred_class, n_classes)
            matrix[n_classes, pred_j] += 1  # Background as GT

    return matrix


def plot_confusion_matrix(
    matrix: np.ndarray,
    classes: List[str],
    output_path: Path,
    title: str = "Detection Confusion Matrix",
) -> None:
    """
    Save confusion matrix visualization.

    Args:
        matrix: Confusion matrix.
        classes: Class names.
        output_path: Output file path.
        title: Plot title.
    """
    n_classes = len(classes)
    labels = classes + ["Missed/BG"]

    # Create image
    cell_size = 80
    margin = 120
    width = cell_size * (n_classes + 1) + margin * 2
    height = cell_size * (n_classes + 1) + margin * 2

    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Normalize for coloring
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized = matrix / row_sums

    # Draw cells
    for i in range(n_classes + 1):
        for j in range(n_classes + 1):
            x = margin + j * cell_size
            y = margin + i * cell_size

            # Color based on normalized value
            if i == j and i < n_classes:
                # Diagonal (correct predictions) - green
                intensity = int(normalized[i, j] * 200)
                color = (200 - intensity, 200, 200 - intensity)
            elif i == n_classes or j == n_classes:
                # FP/FN row/column - red tint
                intensity = int(normalized[i, j] * 200)
                color = (200, 200 - intensity, 200 - intensity)
            else:
                # Off-diagonal (confusion) - yellow tint
                intensity = int(normalized[i, j] * 200)
                color = (200 - intensity, 200 - intensity, 200)

            cv2.rectangle(img, (x, y), (x + cell_size, y + cell_size), color, -1)
            cv2.rectangle(img, (x, y), (x + cell_size, y + cell_size), (100, 100, 100), 1)

            # Draw value
            value = matrix[i, j]
            text = str(value)
            font_scale = 0.5 if value < 1000 else 0.4
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tx = x + (cell_size - tw) // 2
            ty = y + (cell_size + th) // 2

            cv2.putText(
                img, text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA
            )

    # Draw labels
    for i, label in enumerate(labels):
        # Row labels (left)
        y = margin + i * cell_size + cell_size // 2 + 5
        cv2.putText(
            img, label, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

        # Column labels (top)
        x = margin + i * cell_size + 5
        cv2.putText(
            img, label, (x, margin - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    # Title
    cv2.putText(
        img, title, (width // 2 - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA
    )

    # Axis labels
    cv2.putText(
        img, "Ground Truth", (10, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA
    )
    cv2.putText(
        img, "Predicted", (width // 2 - 30, margin - 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA
    )

    # Save
    cv2.imwrite(str(output_path), img)


def run_evaluation(
    data_dir: Path,
    output_dir: Path,
    model_name: str = "yolov8m",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    num_frames: int = 100,
    device: str = "cuda",
) -> Dict:
    """
    Run full evaluation pipeline.

    Args:
        data_dir: KITTI dataset directory.
        output_dir: Output directory.
        model_name: YOLOv8 model variant.
        confidence_threshold: Detection confidence threshold.
        iou_threshold: IoU threshold for matching.
        num_frames: Number of frames to evaluate.
        device: Inference device.

    Returns:
        Dictionary with all metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    print(f"Loading KITTI dataset from {data_dir}...")
    loader = KITTILoader(data_dir, split="training", load_labels=True)
    total_available = len(loader)
    if num_frames <= 0:
        num_frames = total_available
    else:
        num_frames = min(num_frames, total_available)
    print(f"  Total frames available: {total_available}")
    print(f"  Evaluating on {num_frames} frames")

    print(f"\nInitializing {model_name} detector...")
    detector = ObjectDetector2D(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        device=device,
    )
    detector.warmup()

    processor = DetectionPostProcessor(
        min_confidence=confidence_threshold,
        min_box_size=(20, 20),
        nms_threshold=0.5,
    )

    # Collect all detections and ground truth
    all_detections = []
    all_ground_truth = []

    print(f"\nProcessing frames...")
    for i in range(num_frames):
        sample = loader[i]
        image = sample["image"]
        objects = sample.get("objects", [])

        # Run detection
        detections = detector.detect(image)
        detections = processor.process(detections)

        # Convert ground truth
        gt_detections = []
        for obj in objects:
            if obj.type in EVAL_CLASSES:
                gt_detections.append(convert_kitti_to_detection(obj))

        all_detections.append(detections)
        all_ground_truth.append(gt_detections)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames")

    # Compute metrics
    print("\nComputing metrics...")

    metrics = {
        "model": model_name,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "num_frames": num_frames,
        "classes": EVAL_CLASSES,
    }

    # Overall metrics
    overall = compute_precision_recall(
        all_detections, all_ground_truth, iou_threshold
    )
    metrics["overall"] = overall

    # Per-class metrics
    metrics["per_class"] = {}
    for class_name in EVAL_CLASSES:
        class_metrics = compute_precision_recall(
            all_detections, all_ground_truth, iou_threshold, class_name
        )
        metrics["per_class"][class_name] = class_metrics

    # Confusion matrix
    confusion = compute_confusion_matrix(
        all_detections, all_ground_truth, EVAL_CLASSES, iou_threshold
    )
    metrics["confusion_matrix"] = confusion.tolist()

    # Save confusion matrix visualization
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(confusion, EVAL_CLASSES, cm_path)
    print(f"  Confusion matrix saved to: {cm_path}")

    # Detection statistics
    total_gt = sum(len(gt) for gt in all_ground_truth)
    total_det = sum(len(det) for det in all_detections)
    metrics["total_ground_truth"] = total_gt
    metrics["total_detections"] = total_det

    return metrics


def print_metrics(metrics: Dict) -> None:
    """Print formatted evaluation metrics."""
    print("\n" + "=" * 70)
    print("2D DETECTION EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Model: {metrics['model']}")
    print(f"  Confidence Threshold: {metrics['confidence_threshold']}")
    print(f"  IoU Threshold: {metrics['iou_threshold']}")
    print(f"  Frames Evaluated: {metrics['num_frames']}")

    print(f"\nOverall Metrics:")
    overall = metrics["overall"]
    print(f"  Precision: {overall['precision']:.3f}")
    print(f"  Recall:    {overall['recall']:.3f}")
    print(f"  F1 Score:  {overall['f1']:.3f}")
    print(f"  AP:        {overall['ap']:.3f}")
    print(f"  TP: {overall['true_positives']}, FP: {overall['false_positives']}, FN: {overall['false_negatives']}")

    print(f"\nPer-Class Metrics:")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AP':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for class_name in metrics["classes"]:
        cm = metrics["per_class"][class_name]
        print(
            f"  {class_name:<12} "
            f"{cm['precision']:>10.3f} "
            f"{cm['recall']:>10.3f} "
            f"{cm['f1']:>10.3f} "
            f"{cm['ap']:>10.3f}"
        )

    # mAP
    map_score = np.mean([
        metrics["per_class"][c]["ap"] for c in metrics["classes"]
    ])
    print(f"\n  mAP@{metrics['iou_threshold']}: {map_score:.3f}")

    print(f"\nDetection Statistics:")
    print(f"  Total Ground Truth: {metrics['total_ground_truth']}")
    print(f"  Total Detections:   {metrics['total_detections']}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 2D Detection on KITTI",
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
        default="outputs/evaluation",
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
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=-1,
        help="Number of frames to evaluate (-1 for all frames)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device (cuda/cpu)",
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

    # Run evaluation
    try:
        metrics = run_evaluation(
            data_dir=data_dir,
            output_dir=output_dir,
            model_name=args.model,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou_threshold,
            num_frames=args.num_frames,
            device=args.device,
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print metrics
    print_metrics(metrics)

    # Save metrics
    metrics_path = output_dir / "metrics_2d.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
