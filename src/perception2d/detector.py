"""
YOLOv8 Object Detector for 2D perception.

This module provides a wrapper around Ultralytics YOLOv8 for detecting objects
relevant to autonomous driving (Cars, Pedestrians, Cyclists).

YOLOv8 Output Format:
=====================
YOLOv8 returns a list of Results objects, one per image. Each Results object contains:

1. results[i].boxes - Boxes object containing:
   - boxes.xyxy: (N, 4) tensor of [x1, y1, x2, y2] coordinates in pixels
   - boxes.xywh: (N, 4) tensor of [center_x, center_y, width, height]
   - boxes.xyxyn: (N, 4) normalized xyxy (0-1 range)
   - boxes.xywhn: (N, 4) normalized xywh (0-1 range)
   - boxes.conf: (N,) tensor of confidence scores
   - boxes.cls: (N,) tensor of class indices (COCO class IDs)
   - boxes.id: (N,) tensor of track IDs (if tracking enabled)
   - boxes.data: (N, 6+) raw tensor [x1, y1, x2, y2, conf, cls, ...]

2. results[i].orig_img - Original input image (numpy array)
3. results[i].orig_shape - Original image shape (height, width)
4. results[i].names - Dict mapping class ID to class name

COCO Class IDs (subset relevant to driving):
============================================
  0: person      -> Pedestrian
  1: bicycle     -> Cyclist
  2: car         -> Car
  3: motorcycle  -> Cyclist
  5: bus         -> Truck/Van
  7: truck       -> Truck

YOLOv8 Model Variants:
=====================
  yolov8n.pt - Nano:   3.2M params,  fastest, ~640 FPS on GPU
  yolov8s.pt - Small:  11.2M params, fast,   ~400 FPS on GPU
  yolov8m.pt - Medium: 25.9M params, balanced, ~180 FPS on GPU
  yolov8l.pt - Large:  43.7M params, accurate, ~120 FPS on GPU
  yolov8x.pt - XLarge: 68.2M params, most accurate, ~80 FPS on GPU

Coordinate Systems:
==================
  - Image: Origin at top-left, x increases right, y increases down
  - Box format: [x1, y1, x2, y2] where (x1, y1) is top-left, (x2, y2) is bottom-right
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class Detection:
    """
    Single 2D object detection result.

    This is the standardized output format for all detectors in the pipeline.

    Attributes:
        bbox: Bounding box coordinates [x1, y1, x2, y2] in pixels.
              (x1, y1) = top-left corner, (x2, y2) = bottom-right corner.
        class_id: Integer class ID from the detector (COCO ID for YOLO).
        class_name: Human-readable class name mapped to KITTI classes.
        confidence: Detection confidence score in range [0, 1].
        track_id: Optional tracking ID for multi-object tracking.
        metadata: Optional additional metadata (e.g., segmentation mask).
    """

    bbox: np.ndarray  # [x1, y1, x2, y2] in pixels
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure bbox is numpy array."""
        if not isinstance(self.bbox, np.ndarray):
            self.bbox = np.array(self.bbox, dtype=np.float32)

    @property
    def x1(self) -> float:
        """Left edge x coordinate."""
        return float(self.bbox[0])

    @property
    def y1(self) -> float:
        """Top edge y coordinate."""
        return float(self.bbox[1])

    @property
    def x2(self) -> float:
        """Right edge x coordinate."""
        return float(self.bbox[2])

    @property
    def y2(self) -> float:
        """Bottom edge y coordinate."""
        return float(self.bbox[3])

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Box area in pixels squared."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Box center (x, y) coordinates."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def aspect_ratio(self) -> float:
        """Width / height ratio."""
        return self.width / self.height if self.height > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "bbox": self.bbox.tolist(),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "track_id": self.track_id,
        }

    def __repr__(self) -> str:
        return (
            f"Detection({self.class_name}, "
            f"conf={self.confidence:.2f}, "
            f"bbox=[{self.x1:.0f}, {self.y1:.0f}, {self.x2:.0f}, {self.y2:.0f}])"
        )


class ObjectDetector2D:
    """
    YOLOv8-based 2D object detector with KITTI class mapping.

    This class wraps YOLOv8 to detect objects relevant to autonomous driving
    and maps COCO classes to KITTI-style classes (Car, Pedestrian, Cyclist).

    Usage:
        detector = ObjectDetector2D(model_name="yolov8m")
        detections = detector.detect(image)
        for det in detections:
            print(f"{det.class_name}: {det.confidence:.2f}")

    Attributes:
        model_name: YOLOv8 variant (yolov8n/s/m/l/x).
        confidence_threshold: Minimum confidence to keep detections.
        device: Inference device ('cuda', 'cpu', or device ID).
    """

    # ==========================================================================
    # COCO to KITTI Class Mapping
    # ==========================================================================
    # COCO has 80 classes, KITTI has 8 classes for 3D object detection.
    # We map relevant COCO classes to KITTI classes.
    #
    # KITTI Classes:
    #   - Car: passenger vehicles
    #   - Van: larger passenger vehicles
    #   - Truck: freight vehicles
    #   - Pedestrian: people walking
    #   - Person_sitting: seated people
    #   - Cyclist: people on bicycles/motorcycles
    #   - Tram: rail vehicles
    #   - Misc: other objects

    COCO_TO_KITTI: Dict[int, str] = {
        # COCO person -> KITTI Pedestrian
        0: "Pedestrian",

        # COCO bicycle, motorcycle -> KITTI Cyclist
        1: "Cyclist",    # bicycle
        3: "Cyclist",    # motorcycle

        # COCO car -> KITTI Car
        2: "Car",

        # COCO bus, truck -> KITTI Truck (or Van for bus)
        5: "Van",        # bus
        7: "Truck",      # truck
    }

    # Default classes to detect (COCO IDs)
    DEFAULT_CLASSES = [0, 1, 2, 3, 5, 7]

    # KITTI class colors for visualization (RGB)
    CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
        "Car": (0, 255, 127),
        "Pedestrian": (255, 82, 82),
        "Cyclist": (64, 156, 255),
        "Van": (0, 206, 209),
        "Truck": (255, 193, 37),
    }

    def __init__(
        self,
        model_name: str = "yolov8m",
        weights_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        classes: Optional[List[int]] = None,
        half_precision: bool = False,
    ):
        """
        Initialize YOLOv8 detector.

        Args:
            model_name: YOLOv8 variant to use:
                - 'yolov8n': Nano (fastest, least accurate)
                - 'yolov8s': Small
                - 'yolov8m': Medium (balanced)
                - 'yolov8l': Large
                - 'yolov8x': XLarge (slowest, most accurate)
            weights_path: Path to custom weights file. If None, downloads
                pretrained COCO weights automatically.
            confidence_threshold: Minimum confidence score [0, 1] to keep
                a detection. Lower = more detections, higher = fewer but
                more confident. Default 0.3 is good for recall.
            iou_threshold: IoU threshold for built-in NMS. Detections with
                IoU > threshold are suppressed. Default 0.45.
            device: Device for inference:
                - 'cuda': First GPU
                - 'cuda:0', 'cuda:1': Specific GPU
                - 'cpu': CPU inference
            classes: List of COCO class IDs to detect. None = use defaults.
            half_precision: Use FP16 inference (faster on supported GPUs).
        """
        self.model_name = model_name
        self.weights_path = Path(weights_path) if weights_path else None
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes if classes is not None else self.DEFAULT_CLASSES
        self.half_precision = half_precision

        # Model will be loaded lazily or explicitly
        self._model = None
        self._model_loaded = False

    def _load_model(self) -> None:
        """
        Load YOLOv8 model from weights.

        The model is loaded lazily on first inference to avoid GPU memory
        allocation during initialization.

        Raises:
            ImportError: If ultralytics package is not installed.
            FileNotFoundError: If custom weights file doesn't exist.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package required for YOLOv8. "
                "Install with: pip install ultralytics"
            )

        # Determine weights to load
        if self.weights_path and self.weights_path.exists():
            weights = str(self.weights_path)
        else:
            # Download pretrained weights
            weights = f"{self.model_name}.pt"

        # Load model
        self._model = YOLO(weights)

        # Move to device
        self._model.to(self.device)

        # Enable half precision if requested
        if self.half_precision and "cuda" in self.device:
            self._model.model.half()

        self._model_loaded = True

    @property
    def model(self):
        """Lazy-load model on first access."""
        if not self._model_loaded:
            self._load_model()
        return self._model

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        return_raw: bool = False,
    ) -> Union[List[Detection], Any]:
        """
        Run object detection on a single image.

        Args:
            image: Input image as numpy array. Accepts:
                - RGB image: (H, W, 3) uint8
                - BGR image: (H, W, 3) uint8 (OpenCV format)
                - Grayscale: (H, W) uint8 (will be converted)
            confidence_threshold: Override instance threshold for this call.
            return_raw: Return raw YOLOv8 Results object instead of
                parsed detections. Useful for debugging.

        Returns:
            List of Detection objects, or raw Results if return_raw=True.

        Example:
            >>> detector = ObjectDetector2D()
            >>> image = cv2.imread("image.jpg")
            >>> detections = detector.detect(image)
            >>> for det in detections:
            ...     print(f"{det.class_name}: {det.confidence:.2f}")
            Car: 0.92
            Pedestrian: 0.85

        YOLO Output Processing:
            1. Run YOLOv8 inference -> Returns Results object
            2. Extract boxes.xyxy (coordinates), boxes.conf (scores), boxes.cls (classes)
            3. Filter by KITTI-relevant classes using COCO_TO_KITTI mapping
            4. Create Detection objects with standardized format
        """
        conf = confidence_threshold or self.confidence_threshold

        # Run YOLOv8 inference
        # =====================
        # YOLOv8 internally handles:
        # - Image preprocessing (resize, normalize)
        # - Batching (even for single image)
        # - NMS (Non-Maximum Suppression)
        # - GPU/CPU transfer
        results = self.model(
            image,
            conf=conf,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False,  # Suppress console output
        )

        if return_raw:
            return results

        # Parse YOLOv8 results into Detection objects
        # =============================================
        detections = []

        for result in results:
            # result.boxes contains all detections for this image
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            # Iterate through each detection
            # boxes.xyxy: (N, 4) tensor of [x1, y1, x2, y2]
            # boxes.conf: (N,) tensor of confidence scores
            # boxes.cls: (N,) tensor of class IDs
            for i in range(len(boxes)):
                # Extract box coordinates (convert from tensor to numpy)
                bbox = boxes.xyxy[i].cpu().numpy().astype(np.float32)

                # Extract confidence score
                confidence = float(boxes.conf[i].cpu().numpy())

                # Extract class ID (COCO class index)
                class_id = int(boxes.cls[i].cpu().numpy())

                # Map COCO class to KITTI class
                class_name = self.COCO_TO_KITTI.get(class_id)

                # Skip classes we don't care about
                if class_name is None:
                    continue

                # Create standardized Detection object
                detection = Detection(
                    bbox=bbox,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                )

                detections.append(detection)

        return detections

    def detect_batch(
        self,
        images: List[np.ndarray],
        confidence_threshold: Optional[float] = None,
    ) -> List[List[Detection]]:
        """
        Run detection on a batch of images.

        Batch inference is more efficient than running detect() in a loop
        because it better utilizes GPU parallelism.

        Args:
            images: List of input images (can be different sizes).
            confidence_threshold: Override instance threshold.

        Returns:
            List of detection lists, one per input image.

        Example:
            >>> images = [cv2.imread(f) for f in image_files[:4]]
            >>> batch_results = detector.detect_batch(images)
            >>> for i, dets in enumerate(batch_results):
            ...     print(f"Image {i}: {len(dets)} detections")
        """
        conf = confidence_threshold or self.confidence_threshold

        # YOLOv8 handles batching automatically
        results = self.model(
            images,
            conf=conf,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False,
        )

        # Parse results for each image
        all_detections = []

        for result in results:
            detections = []
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy().astype(np.float32)
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())

                    class_name = self.COCO_TO_KITTI.get(class_id)
                    if class_name is None:
                        continue

                    detections.append(Detection(
                        bbox=bbox,
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                    ))

            all_detections.append(detections)

        return all_detections

    def warmup(self, image_size: Tuple[int, int] = (640, 480)) -> None:
        """
        Warm up the model with a dummy inference.

        First inference is slower due to CUDA kernel compilation and memory
        allocation. Call this before timing benchmarks.

        Args:
            image_size: (width, height) of dummy image.
        """
        dummy = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        self.detect(dummy)

    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get RGB color for visualization."""
        return self.CLASS_COLORS.get(class_name, (255, 255, 255))

    @property
    def class_names(self) -> List[str]:
        """List of detectable class names."""
        return list(set(self.COCO_TO_KITTI.values()))

    def __repr__(self) -> str:
        return (
            f"ObjectDetector2D("
            f"model={self.model_name}, "
            f"conf={self.confidence_threshold}, "
            f"device={self.device})"
        )


# Alias for backward compatibility
YOLOv8Detector = ObjectDetector2D
