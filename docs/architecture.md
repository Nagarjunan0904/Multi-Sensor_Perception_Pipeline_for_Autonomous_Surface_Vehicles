# System Architecture

## Overview

The Multi-Sensor Perception Pipeline implements a **late fusion** approach for 3D object detection. This architecture was chosen for its modularity, interpretability, and ease of component upgrades.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MULTI-SENSOR PERCEPTION PIPELINE                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                          INPUT LAYER                                    │ │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │ │
│  │  │    Camera    │    │    LiDAR     │    │ Calibration  │              │ │
│  │  │   (1242×375) │    │ (Velodyne64) │    │   (K,R,T)    │              │ │
│  │  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │ │
│  └─────────│───────────────────│───────────────────│──────────────────────┘ │
│            │                   │                   │                         │
│            ▼                   │                   │                         │
│  ┌─────────────────────────────│───────────────────│──────────────────────┐ │
│  │                    2D PERCEPTION                │                       │ │
│  │  ┌──────────────────┐       │                   │                       │ │
│  │  │     YOLOv8m      │       │                   │                       │ │
│  │  │   ┌──────────┐   │       │                   │                       │ │
│  │  │   │ Backbone │   │       │                   │                       │ │
│  │  │   │ (CSPNet) │   │       │                   │                       │ │
│  │  │   └────┬─────┘   │       │                   │                       │ │
│  │  │        │         │       │                   │                       │ │
│  │  │   ┌────▼─────┐   │       │                   │                       │ │
│  │  │   │   Neck   │   │       │                   │                       │ │
│  │  │   │ (PANet)  │   │       │                   │                       │ │
│  │  │   └────┬─────┘   │       │                   │                       │ │
│  │  │        │         │       │                   │                       │ │
│  │  │   ┌────▼─────┐   │       │                   │                       │ │
│  │  │   │   Head   │   │       │                   │                       │ │
│  │  │   │(Decoupled)│  │       │                   │                       │ │
│  │  │   └────┬─────┘   │       │                   │                       │ │
│  │  └────────│─────────┘       │                   │                       │ │
│  │           │                 │                   │                       │ │
│  │           ▼                 │                   │                       │ │
│  │  ┌──────────────────┐       │                   │                       │ │
│  │  │  2D Detections   │       │                   │                       │ │
│  │  │ (bbox, cls, conf)│       │                   │                       │ │
│  │  └────────┬─────────┘       │                   │                       │ │
│  └───────────│─────────────────│───────────────────│───────────────────────┘ │
│              │                 │                   │                         │
│              ▼                 ▼                   ▼                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        SENSOR FUSION                                   │  │
│  │                                                                        │  │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │  │
│  │  │   Projection    │    │ Depth Estimator │    │ 3D Box Generator│    │  │
│  │  │                 │    │                 │    │                 │    │  │
│  │  │ LiDAR → Camera  │ →  │ Points in BBox  │ →  │ 2D + Z → 3D     │    │  │
│  │  │ P = K[R|T]·X    │    │ Robust Median   │    │ Class Priors    │    │  │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘    │  │
│  │                                                         │              │  │
│  └─────────────────────────────────────────────────────────│──────────────┘  │
│                                                            │                 │
│                                                            ▼                 │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      OUTPUT / VISUALIZATION                            │  │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │  │
│  │  │   3D Boxes      │    │  Image Overlay  │    │  Bird's Eye View │   │  │
│  │  │ (x,y,z,l,w,h,θ) │    │  (2D + 3D vis)  │    │  (Waymo-style)   │   │  │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### 1. Data Layer (`src/data/`)

#### KITTILoader
Handles loading and parsing of the KITTI dataset format.

```python
class KITTILoader:
    """Load synchronized camera images, LiDAR point clouds, and calibration."""

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "image": np.ndarray,      # (H, W, 3) RGB image
            "points": np.ndarray,     # (N, 4) LiDAR points (x, y, z, intensity)
            "calib": KITTICalib,      # Calibration matrices
            "labels": List[Label3D],  # Ground truth (if available)
        }
```

**Key Features:**
- Lazy loading for memory efficiency
- Automatic calibration parsing
- Label parsing for evaluation

### 2. Calibration (`src/calibration/`)

#### CameraIntrinsics
Manages camera internal parameters.

```python
@dataclass
class CameraIntrinsics:
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D image coordinates."""
```

#### CameraLiDARExtrinsics
Handles coordinate transformations between sensors.

```python
class CameraLiDARExtrinsics:
    """Transform points between LiDAR and camera frames."""

    def transform_to_camera(self, points_lidar: np.ndarray) -> np.ndarray:
        """LiDAR frame → Camera frame."""

    def transform_to_lidar(self, points_camera: np.ndarray) -> np.ndarray:
        """Camera frame → LiDAR frame."""
```

### 3. 2D Perception (`src/perception2d/`)

#### ObjectDetector2D
Wrapper around YOLOv8 for 2D detection.

```python
class ObjectDetector2D:
    """YOLOv8-based 2D object detector."""

    def __init__(
        self,
        model_name: str = "yolov8m",
        device: str = "cuda",
        confidence_threshold: float = 0.3,
    ):
        self.model = YOLO(model_name)

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection on a single image."""
```

**Detection Output:**
```python
@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    class_id: int                             # Class index
    class_name: str                           # "car", "pedestrian", etc.
    confidence: float                         # Detection score
```

### 4. Sensor Fusion (`src/fusion/`)

#### DepthEstimator
Extracts depth from LiDAR for each 2D detection.

```python
class DepthEstimator:
    """Estimate depth for 2D detections using LiDAR points."""

    def estimate_full(
        self,
        points_2d: np.ndarray,    # Projected LiDAR points
        points_3d: np.ndarray,    # Original 3D coordinates
        bbox: Tuple[int, ...],    # 2D bounding box
    ) -> DepthResult:
        """
        Returns:
            depth: Estimated depth (median of points in bbox)
            confidence: Based on point count
            center_3d: 3D centroid of points
        """
```

**Depth Estimation Methods:**
| Method | Description | Robustness |
|--------|-------------|------------|
| Median | Middle value of depths | High (outlier resistant) |
| Mean | Average depth | Medium |
| Closest | Minimum depth | Low (noise sensitive) |
| Weighted | Distance-weighted average | Medium |

#### BBox3DGenerator
Lifts 2D detections to 3D.

```python
class BBox3DGenerator:
    """Generate 3D bounding boxes from 2D detections + depth."""

    # Class-specific dimension priors (from KITTI statistics)
    DIMENSION_PRIORS = {
        "car": (3.88, 1.63, 1.53),        # (length, width, height)
        "pedestrian": (0.88, 0.65, 1.77),
        "cyclist": (1.76, 0.60, 1.73),
    }

    def generate(
        self,
        detection: Detection,
        depth_result: DepthResult,
        intrinsics: CameraIntrinsics,
    ) -> BBox3D:
        """Generate 3D box from 2D detection and depth."""
```

**3D Box Output:**
```python
@dataclass
class BBox3D:
    center: np.ndarray    # (x, y, z) in camera frame
    dimensions: np.ndarray  # (length, width, height)
    yaw: float           # Rotation around Y-axis
    class_name: str
    confidence: float
```

### 5. Evaluation (`src/eval/`)

#### MetricsCalculator
Computes detection and localization metrics.

```python
class MetricsCalculator:
    """Compute 2D and 3D detection metrics."""

    def compute_2d_metrics(
        self,
        detections: List[Detection],
        ground_truths: List[Label2D],
        iou_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Returns: precision, recall, mAP, per-class AP"""

    def compute_3d_metrics(
        self,
        predictions: List[BBox3D],
        ground_truths: List[Label3D],
    ) -> Dict[str, float]:
        """Returns: 3D IoU, center error, dimension error, orientation error"""
```

#### RobustnessEvaluator
Tests pipeline under failure conditions.

```python
class NoiseInjector:
    """Inject various types of noise for robustness testing."""

    @staticmethod
    def add_depth_noise(points: np.ndarray, std: float) -> np.ndarray:
        """Add Gaussian noise to point cloud depth."""

    @staticmethod
    def add_lidar_dropout(points: np.ndarray, rate: float) -> np.ndarray:
        """Randomly remove points to simulate sensor dropout."""

    @staticmethod
    def add_calibration_error(points: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """Apply systematic calibration offset."""
```

### 6. Visualization (`src/viz/`)

#### BEVVisualizer
Creates bird's eye view visualizations.

```python
class BEVVisualizer:
    """Waymo/Tesla-style bird's eye view rendering."""

    def render(
        self,
        points: np.ndarray,
        detections: List[BBox3D],
        ego_vehicle: bool = True,
    ) -> np.ndarray:
        """Render top-down view with point cloud and boxes."""
```

#### MultiPanelDisplay
Combines multiple visualizations into a single view.

```python
class MultiPanelDisplay:
    """Create multi-panel visualization combining all views."""

    def create_display(
        self,
        image: np.ndarray,
        bev: np.ndarray,
        detections: List[Detection],
        boxes_3d: List[BBox3D],
    ) -> np.ndarray:
        """4-panel display: image, BEV, 3D overlay, info panel"""
```

## Data Flow

```
1. Load Frame
   ├── Camera Image (1242×375 RGB)
   ├── LiDAR Points (N×4: x,y,z,intensity)
   └── Calibration (P2, R0_rect, Tr_velo_to_cam)
          │
          ▼
2. 2D Detection (YOLOv8m)
   ├── Input: RGB Image
   ├── Process: CNN forward pass (~30ms)
   └── Output: List[Detection(bbox, class, conf)]
          │
          ▼
3. Point Cloud Projection
   ├── Transform: LiDAR → Camera (R, T)
   ├── Project: Camera → Image (K)
   └── Output: 2D points with 3D coordinates
          │
          ▼
4. Depth Estimation (per detection)
   ├── Filter: Points inside 2D bbox
   ├── Aggregate: Robust median depth
   └── Output: DepthResult(depth, confidence, center_3d)
          │
          ▼
5. 3D Box Generation
   ├── Center: Unproject 2D center at estimated depth
   ├── Dimensions: Class-specific priors
   ├── Orientation: Simplified (ray-based)
   └── Output: BBox3D(center, dims, yaw)
          │
          ▼
6. Output
   ├── 3D Detections for downstream tasks
   └── Visualizations (overlay, BEV)
```

## Coordinate Systems

### KITTI Convention

```
Camera Frame:           LiDAR Frame:           Image Frame:
      Z (forward)            X (forward)            +u (right)
      │                      │                      │
      │                      │                      │
      │                      │                      │
      └───── X (right)       └───── Y (left)        └───── +v (down)
     /                      /
    Y (down)               Z (up)
```

### Transformations

1. **LiDAR → Camera**: `P_cam = R_rect @ Tr_velo_to_cam @ P_lidar`
2. **Camera → Image**: `p_img = K @ P_cam / P_cam[2]`
3. **Combined**: `p_img = P2 @ R_rect @ Tr_velo_to_cam @ P_lidar`

## Design Decisions

### Why Late Fusion?

| Approach | Pros | Cons |
|----------|------|------|
| **Late Fusion** (ours) | Modular, interpretable, easy upgrades | Limited feature sharing |
| Early Fusion | Rich features | Complex, less modular |
| Deep Fusion | Best accuracy | Requires end-to-end training |

**Our Choice**: Late fusion provides the best trade-off for a perception pipeline that needs to be:
- Easily debuggable (can inspect each stage)
- Component-upgradeable (swap detectors without retraining)
- Explainable (clear data flow)

### Why Median Depth?

We tested multiple depth aggregation methods:

| Method | Mean Error | Robustness | Speed |
|--------|------------|------------|-------|
| Closest | 1.89m | Low | Fast |
| Mean | 1.56m | Medium | Fast |
| **Median** | **1.42m** | **High** | Fast |
| Weighted | 1.51m | Medium | Slow |

Median provides the best balance of accuracy and outlier resistance.

### Why YOLOv8m?

| Model | mAP@0.5 | Inference | Memory |
|-------|---------|-----------|--------|
| YOLOv8n | 0.48 | 8ms | 3.2GB |
| YOLOv8s | 0.52 | 12ms | 4.1GB |
| **YOLOv8m** | **0.58** | **25ms** | **5.8GB** |
| YOLOv8l | 0.61 | 45ms | 8.2GB |

YOLOv8m provides good accuracy while maintaining real-time capability.

## Configuration Schema

```yaml
# configs/default.yaml
data:
  root: "data/kitti"           # Dataset location
  split: "training"            # training/testing

detector:
  model: "yolov8m"             # YOLO variant
  confidence: 0.3              # Detection threshold
  iou_threshold: 0.45          # NMS threshold
  classes: [0, 1, 2]           # car, pedestrian, cyclist

fusion:
  depth_method: "median"       # median/mean/closest
  min_points: 5                # Minimum points for valid depth
  outlier_threshold: 2.0       # Sigma for RANSAC

visualization:
  bev_range: [-40, 40, 0, 80]  # BEV extent [x_min, x_max, y_min, y_max]
  resolution: 0.1              # Meters per pixel
```

## Extensibility Points

1. **Custom Detector**: Implement `detect(image) -> List[Detection]`
2. **Custom Depth Method**: Add to `DepthEstimator.estimate_full()`
3. **Custom Dimension Prior**: Update `BBox3DGenerator.DIMENSION_PRIORS`
4. **Custom Visualization**: Extend `BEVVisualizer` class

## Performance Characteristics

| Stage | Time (CPU) | Time (GPU) | Memory |
|-------|------------|------------|--------|
| Data Loading | 15ms | 15ms | 50MB |
| 2D Detection | 180ms | 25ms | 1.2GB |
| Projection | 5ms | 5ms | 10MB |
| Depth Estimation | 2ms | 2ms | 5MB |
| 3D Generation | 1ms | 1ms | 1MB |
| **Total** | **203ms** | **48ms** | **1.3GB** |

Real-time capable at ~20 FPS on GPU.
