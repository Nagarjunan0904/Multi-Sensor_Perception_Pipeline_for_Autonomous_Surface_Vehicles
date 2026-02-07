# Dataset Directory

This directory contains the KITTI dataset files. The actual data files are not tracked in git due to their size.

## KITTI 3D Object Detection Dataset

### Download Instructions

#### Option 1: Automated Download

```bash
python scripts/download_kitti.py
```

#### Option 2: Manual Download

1. Visit [KITTI Vision Benchmark - 3D Object Detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

2. Download the following files:
   - **Left color images** (12 GB): `data_object_image_2.zip`
   - **Velodyne point clouds** (29 GB): `data_object_velodyne.zip`
   - **Camera calibration files** (16 MB): `data_object_calib.zip`
   - **Training labels** (5 MB): `data_object_label_2.zip`

3. Extract all files into this directory:

```bash
cd data/KITTI
unzip data_object_image_2.zip
unzip data_object_velodyne.zip
unzip data_object_calib.zip
unzip data_object_label_2.zip
```

### Expected Directory Structure

After downloading and extracting, the structure should be:

```
data/KITTI/
├── training/
│   ├── image_2/           # 7,481 PNG images (1242 x 375)
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── velodyne/          # 7,481 point cloud files
│   │   ├── 000000.bin
│   │   ├── 000001.bin
│   │   └── ...
│   ├── calib/             # 7,481 calibration files
│   │   ├── 000000.txt
│   │   ├── 000001.txt
│   │   └── ...
│   └── label_2/           # 7,481 label files
│       ├── 000000.txt
│       ├── 000001.txt
│       └── ...
└── testing/
    ├── image_2/           # 7,518 PNG images
    ├── velodyne/          # 7,518 point cloud files
    └── calib/             # 7,518 calibration files
```

### Verify Installation

Run the verification script to ensure all files are correctly placed:

```bash
python scripts/verify_data.py
```

### Data Format

#### Images (`image_2/`)
- Format: PNG
- Resolution: 1242 x 375 pixels
- Color: RGB (stored as BGR in OpenCV)

#### Point Clouds (`velodyne/`)
- Format: Binary float32
- Each point: [x, y, z, reflectance] (4 floats = 16 bytes)
- Coordinate system: x=forward, y=left, z=up

#### Calibration (`calib/`)
- Text files with projection matrices
- P0, P1, P2, P3: Camera projection matrices
- R0_rect: Rectification matrix
- Tr_velo_to_cam: Velodyne to camera transformation

#### Labels (`label_2/`)
- One object per line
- Format: `class truncated occluded alpha bbox_2d dimensions location rotation_y`

### Storage Requirements

| Component | Size |
|-----------|------|
| Images (training) | ~12 GB |
| Images (testing) | ~12 GB |
| Point clouds (training) | ~29 GB |
| Point clouds (testing) | ~30 GB |
| Calibration | ~16 MB |
| Labels | ~5 MB |
| **Total** | **~83 GB** |

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{geiger2012cvpr,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {CVPR},
  year = {2012}
}
```
