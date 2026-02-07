#!/usr/bin/env python3
"""Verify KITTI dataset installation."""

import argparse
from pathlib import Path


def verify_kitti(data_dir: Path, split: str = "training") -> dict:
    """
    Verify KITTI dataset installation.

    Args:
        data_dir: Path to KITTI data directory.
        split: Dataset split to verify.

    Returns:
        Dictionary with verification results.
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    split_dir = data_dir / split

    # Check directories exist
    expected_dirs = {
        "image_2": "*.png",
        "velodyne": "*.bin",
        "calib": "*.txt",
    }

    if split == "training":
        expected_dirs["label_2"] = "*.txt"

    for dir_name, pattern in expected_dirs.items():
        dir_path = split_dir / dir_name

        if not dir_path.exists():
            results["valid"] = False
            results["errors"].append(f"Missing directory: {dir_path}")
            continue

        # Count files
        files = list(dir_path.glob(pattern))
        results["stats"][dir_name] = len(files)

        if len(files) == 0:
            results["valid"] = False
            results["errors"].append(f"No files found in {dir_path}")

    # Verify file counts match
    if results["stats"]:
        counts = list(results["stats"].values())
        if len(set(counts)) > 1:
            results["warnings"].append(
                f"File counts don't match: {results['stats']}"
            )

    # Verify expected counts for KITTI
    expected_counts = {
        "training": 7481,
        "testing": 7518,
    }

    if split in expected_counts:
        expected = expected_counts[split]
        for dir_name, count in results["stats"].items():
            if dir_name != "label_2" or split == "training":
                if count != expected:
                    results["warnings"].append(
                        f"{dir_name}: expected {expected} files, found {count}"
                    )

    # Sample file verification
    if results["stats"].get("image_2", 0) > 0:
        sample_image = next((split_dir / "image_2").glob("*.png"))
        try:
            import cv2
            img = cv2.imread(str(sample_image))
            if img is not None:
                h, w = img.shape[:2]
                results["stats"]["image_size"] = f"{w}x{h}"
            else:
                results["warnings"].append("Could not read sample image")
        except ImportError:
            results["warnings"].append("OpenCV not installed, skipping image check")

    if results["stats"].get("velodyne", 0) > 0:
        sample_lidar = next((split_dir / "velodyne").glob("*.bin"))
        try:
            import numpy as np
            points = np.fromfile(str(sample_lidar), dtype=np.float32).reshape(-1, 4)
            results["stats"]["sample_points"] = len(points)
        except Exception as e:
            results["warnings"].append(f"Could not read sample point cloud: {e}")

    return results


def print_results(results: dict, split: str) -> None:
    """Print verification results."""
    print(f"\n{'='*50}")
    print(f"KITTI Dataset Verification - {split}")
    print('='*50)

    if results["valid"]:
        print("\n[OK] Dataset structure is valid")
    else:
        print("\n[ERROR] Dataset structure has errors")

    print("\nStatistics:")
    for key, value in results["stats"].items():
        print(f"  {key}: {value}")

    if results["errors"]:
        print("\nErrors:")
        for error in results["errors"]:
            print(f"  [X] {error}")

    if results["warnings"]:
        print("\nWarnings:")
        for warning in results["warnings"]:
            print(f"  [!] {warning}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Verify KITTI dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/KITTI",
        help="Path to KITTI data directory",
    )
    parser.add_argument(
        "--split",
        choices=["training", "testing", "both"],
        default="both",
        help="Dataset split to verify",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("\nTo download the KITTI dataset, run:")
        print("  python scripts/download_kitti.py")
        return 1

    splits = ["training", "testing"] if args.split == "both" else [args.split]

    all_valid = True
    for split in splits:
        results = verify_kitti(data_dir, split)
        print_results(results, split)
        all_valid = all_valid and results["valid"]

    if all_valid:
        print("[OK] All verifications passed!")
        return 0
    else:
        print("[X] Some verifications failed. See errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
