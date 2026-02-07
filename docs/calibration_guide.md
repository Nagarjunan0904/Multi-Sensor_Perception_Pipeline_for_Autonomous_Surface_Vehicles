# Camera Calibration Guide

## Introduction

Camera calibration is essential for accurate 3D perception. This guide explains the calibration concepts and how they're used in the pipeline.

## Camera Intrinsic Parameters

The camera intrinsic matrix **K** maps 3D points in camera coordinates to 2D pixel coordinates:

```
    ┌ fx  0  cx ┐
K = │  0 fy  cy │
    └  0  0   1 ┘
```

Where:
- **fx, fy**: Focal length in pixels (fx = f/px, fy = f/py)
- **cx, cy**: Principal point (image center)
- **f**: Physical focal length
- **px, py**: Pixel dimensions

### KITTI Camera Parameters

For the KITTI left color camera (cam2):
- **fx, fy** ≈ 721 pixels
- **cx** ≈ 609 pixels
- **cy** ≈ 172 pixels
- Image size: 1242 × 375

## Projection Matrix

KITTI provides projection matrices P0-P3 for each camera:

```
P = K [R|t]
```

The 3×4 projection matrix directly maps homogeneous 3D points to 2D:

```
┌ u ┐     ┌ p00 p01 p02 p03 ┐ ┌ X ┐
│ v │ = s │ p10 p11 p12 p13 │ │ Y │
└ 1 ┘     └ p20 p21 p22 p23 ┘ │ Z │
                              └ 1 ┘
```

Where s is a scale factor (the resulting Z coordinate).

## Extrinsic Calibration

### LiDAR to Camera Transformation

The transformation from Velodyne LiDAR to camera coordinates:

```
P_cam = R * P_velo + T
```

KITTI provides this as `Tr_velo_to_cam` (3×4 matrix).

### Rectification

KITTI images are rectified, requiring an additional rotation:

```
P_rect = R0_rect * P_cam
```

### Complete Projection Pipeline

To project a LiDAR point to image coordinates:

```python
# 1. Transform to camera frame
P_cam = Tr_velo_to_cam @ [x, y, z, 1]

# 2. Apply rectification
P_rect = R0_rect @ P_cam[:3]

# 3. Project to image
p_img = P2 @ [P_rect, 1]

# 4. Normalize
u = p_img[0] / p_img[2]
v = p_img[1] / p_img[2]
```

## KITTI Calibration File Format

Each calibration file contains:

```
P0: [12 values] - Projection matrix for grayscale left
P1: [12 values] - Projection matrix for grayscale right
P2: [12 values] - Projection matrix for color left
P3: [12 values] - Projection matrix for color right
R0_rect: [9 values] - Rectification rotation
Tr_velo_to_cam: [12 values] - LiDAR to camera transform
Tr_imu_to_velo: [12 values] - IMU to LiDAR transform
```

## Inverse Projection (2D to 3D)

Given a 2D point (u, v) and depth d, recover 3D:

```python
X = (u - cx) * d / fx
Y = (v - cy) * d / fy
Z = d
```

This is the foundation of our depth-based 3D box generation.

## Calibration in the Pipeline

### Loading Calibration

```python
from src.calibration import CameraLiDARExtrinsics, CameraIntrinsics

# Load from KITTI file
extrinsics = CameraLiDARExtrinsics("path/to/calib.txt")

# Extract intrinsics from P2
intrinsics = CameraIntrinsics.from_projection_matrix(
    extrinsics.P2, width=1242, height=375
)
```

### Projecting Points

```python
from src.calibration import Projector

projector = Projector(intrinsics, extrinsics)

# Project LiDAR to image
points_2d, depths, mask = projector.project_lidar_to_image(
    points_lidar,
    filter_fov=True,
    filter_depth=True
)
```

### Getting Depth in 2D Box

```python
depth = projector.get_depth_in_box(
    points_lidar,
    box_2d=[x1, y1, x2, y2],
    method="median"  # or "mean", "closest"
)
```

## Common Issues

### Points Behind Camera
Always filter points with Z ≤ 0 (behind camera).

### Out-of-Image Points
Check that projected points are within image bounds.

### Calibration Mismatch
Ensure you're using the correct calibration file for each frame.

## References

- [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)
- Geiger et al., "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite", CVPR 2012
