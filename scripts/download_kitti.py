#!/usr/bin/env python3
"""
Download KITTI 3D Object Detection dataset.

Dataset: KITTI Vision Benchmark Suite - 3D Object Detection
URL: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

Components:
- Left color images (image_2): 12 GB
- Velodyne point clouds (velodyne): 29 GB
- Camera calibration (calib): 16 MB
- Training labels (label_2): 5 MB

Total compressed: ~41 GB (training only)
Total uncompressed: ~47 GB
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

# KITTI 3D Object Detection dataset URLs (official S3 mirrors)
KITTI_URLS = {
    "image_2": {
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
        "size": "12 GB",
        "description": "Left color images",
    },
    "velodyne": {
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip",
        "size": "29 GB",
        "description": "Velodyne point clouds",
    },
    "calib": {
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip",
        "size": "16 MB",
        "description": "Camera calibration files",
    },
    "label_2": {
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
        "size": "5 MB",
        "description": "Training labels",
    },
}


def check_url(url: str) -> bool:
    """Check if URL is accessible."""
    try:
        import requests
        response = requests.head(url, timeout=10, allow_redirects=True)
        return response.status_code == 200
    except Exception:
        # Try curl
        try:
            result = subprocess.run(
                ["curl", "-sI", "-o", "/dev/null", "-w", "%{http_code}", url],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return result.stdout.strip() == "200"
        except Exception:
            return False


def download_with_curl(url: str, output_path: Path, resume: bool = True) -> bool:
    """Download file using curl with resume support."""
    cmd = ["curl", "-L", "--progress-bar"]
    if resume and output_path.exists():
        cmd.extend(["-C", "-"])
    cmd.extend(["-o", str(output_path), url])

    try:
        result = subprocess.run(cmd)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_with_requests(url: str, output_path: Path, resume: bool = True) -> bool:
    """Download file using requests with progress bar and resume support."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("Please install: pip install requests tqdm")
        return False

    headers = {}
    mode = "wb"
    initial_size = 0

    if resume and output_path.exists():
        initial_size = output_path.stat().st_size
        headers["Range"] = f"bytes={initial_size}-"
        mode = "ab"

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)

        if response.status_code == 416:  # Range not satisfiable = file complete
            return True

        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        if initial_size > 0:
            total_size += initial_size

        with open(output_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=initial_size,
                unit="B",
                unit_scale=True,
                desc=output_path.name,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Download error: {e}")
        return False


def download_file(url: str, output_path: Path, max_retries: int = 3) -> bool:
    """Download a file with retry support."""
    print(f"\nDownloading: {output_path.name}")
    print(f"URL: {url}")

    for attempt in range(max_retries):
        if attempt > 0:
            print(f"Retry attempt {attempt + 1}/{max_retries}...")

        # Try curl first (better resume support)
        if download_with_curl(url, output_path):
            return True

        # Fallback to requests
        if download_with_requests(url, output_path):
            return True

    return False


def extract_zip(zip_path: Path, output_dir: Path, quick_mode: bool = False, num_samples: int = 100) -> bool:
    """Extract a zip file, optionally keeping only first N samples."""
    print(f"Extracting: {zip_path.name}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            if quick_mode:
                # Extract only first N samples
                members = zf.namelist()
                # Filter to get training files only and sort
                training_files = sorted([m for m in members if "training/" in m])

                # Group by directory and take first N from each
                extracted = set()
                for member in training_files:
                    parts = member.split("/")
                    if len(parts) >= 3:  # training/subdir/file
                        subdir = parts[1]
                        if subdir not in ["image_2", "velodyne", "calib", "label_2"]:
                            continue
                        # Count files in this subdir
                        count = sum(1 for e in extracted if f"training/{subdir}/" in e)
                        if count < num_samples:
                            zf.extract(member, output_dir)
                            extracted.add(member)
                    else:
                        # Extract directory entries
                        zf.extract(member, output_dir)

                print(f"  Extracted {len(extracted)} files (quick mode: {num_samples} samples)")
            else:
                zf.extractall(output_dir)
                print(f"  Extracted all files")
        return True
    except Exception as e:
        print(f"Extraction error: {e}")
        return False


def verify_urls() -> bool:
    """Verify all download URLs are accessible."""
    print("Verifying download URLs...")
    all_ok = True

    for name, info in KITTI_URLS.items():
        url = info["url"]
        print(f"  Checking {name}... ", end="", flush=True)
        if check_url(url):
            print(f"OK ({info['size']})")
        else:
            print("FAILED")
            all_ok = False

    return all_ok


def print_dataset_info():
    """Print dataset information."""
    print("\n" + "=" * 60)
    print("KITTI 3D Object Detection Dataset")
    print("=" * 60)
    print("\nComponents:")
    total_size = 0
    for name, info in KITTI_URLS.items():
        print(f"  - {name}: {info['description']} ({info['size']})")
    print("\nTraining set: 7,481 frames")
    print("Testing set: 7,518 frames (no labels)")
    print("\nTotal download size: ~41 GB")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Download KITTI 3D Object Detection dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_kitti.py                    # Download all components
  python download_kitti.py --quick            # Download only 100 samples (~500 MB)
  python download_kitti.py --components calib label_2  # Download only small files
  python download_kitti.py --verify           # Just verify URLs are accessible
        """
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/KITTI",
        help="Output directory for dataset (default: data/KITTI)",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=list(KITTI_URLS.keys()) + ["all"],
        default=["all"],
        help="Components to download (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: download full files but extract only 100 samples (~500 MB uncompressed)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to extract in quick mode (default: 100)",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep downloaded zip files after extraction",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify URLs are accessible, don't download",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print dataset information and exit",
    )

    args = parser.parse_args()

    # Print info and exit
    if args.info:
        print_dataset_info()
        return 0

    # Verify URLs only
    if args.verify:
        print_dataset_info()
        print()
        if verify_urls():
            print("\n[OK] All URLs are accessible!")
            return 0
        else:
            print("\n[ERROR] Some URLs are not accessible!")
            return 1

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine components to download
    if "all" in args.components:
        components = list(KITTI_URLS.keys())
    else:
        components = args.components

    # Print download plan
    print_dataset_info()
    print(f"\nDownload plan:")
    print(f"  Output directory: {output_dir.absolute()}")
    print(f"  Components: {', '.join(components)}")
    if args.quick:
        print(f"  Mode: QUICK ({args.num_samples} samples only)")
    else:
        print(f"  Mode: FULL (all 7,481 training frames)")
    print()

    # Verify URLs first
    if not verify_urls():
        print("\n[ERROR] URL verification failed. Check your internet connection.")
        return 1

    print()

    # Download and extract each component
    success_count = 0
    for component in components:
        info = KITTI_URLS[component]
        url = info["url"]
        zip_path = output_dir / f"data_object_{component}.zip"

        # Download
        if zip_path.exists():
            print(f"\n{zip_path.name} already exists, skipping download")
        else:
            if not download_file(url, zip_path):
                print(f"[ERROR] Failed to download {component}")
                continue

        # Extract
        if not extract_zip(zip_path, output_dir, args.quick, args.num_samples):
            print(f"[ERROR] Failed to extract {component}")
            continue

        # Cleanup
        if not args.keep_zip:
            print(f"Removing {zip_path.name}...")
            zip_path.unlink()

        success_count += 1

    # Summary
    print("\n" + "=" * 60)
    if success_count == len(components):
        print("[OK] Download complete!")
    else:
        print(f"[WARNING] {success_count}/{len(components)} components downloaded")

    print(f"\nDataset location: {output_dir.absolute()}")
    print("\nStructure:")
    print(f"  {output_dir}/")
    print("  +-- training/")
    print("  |   +-- image_2/     (PNG images)")
    print("  |   +-- velodyne/    (BIN point clouds)")
    print("  |   +-- calib/       (TXT calibration)")
    print("  |   +-- label_2/     (TXT labels)")
    print("  +-- testing/")
    print()
    print("Run 'python scripts/verify_data.py' to verify the installation.")

    return 0 if success_count == len(components) else 1


if __name__ == "__main__":
    sys.exit(main())
