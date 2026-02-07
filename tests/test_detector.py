"""
Tests for 2D object detector.

These tests verify:
1. Detector initialization
2. Detection output format
3. COCO to KITTI class mapping
4. Post-processing functions
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from perception2d.detector import Detection, ObjectDetector2D
from perception2d.postprocess import (
    apply_nms,
    filter_by_class,
    filter_by_confidence,
    filter_by_size,
    compute_iou,
    compute_iou_matrix,
    DetectionPostProcessor,
)


# =============================================================================
# Detection Class Tests
# =============================================================================

class TestDetection:
    """Tests for Detection dataclass."""

    def test_detection_creation(self):
        """Test creating a Detection object."""
        det = Detection(
            bbox=np.array([100, 200, 300, 400]),
            class_id=2,
            class_name="Car",
            confidence=0.85,
        )

        assert det.class_name == "Car"
        assert det.class_id == 2
        assert det.confidence == 0.85
        assert det.bbox.shape == (4,)

    def test_detection_properties(self):
        """Test Detection computed properties."""
        det = Detection(
            bbox=np.array([100, 200, 300, 400]),
            class_id=2,
            class_name="Car",
            confidence=0.85,
        )

        assert det.x1 == 100
        assert det.y1 == 200
        assert det.x2 == 300
        assert det.y2 == 400
        assert det.width == 200
        assert det.height == 200
        assert det.area == 40000
        assert det.center == (200, 300)
        assert det.aspect_ratio == 1.0

    def test_detection_bbox_conversion(self):
        """Test that bbox is converted to numpy array."""
        det = Detection(
            bbox=[100, 200, 300, 400],  # List input
            class_id=2,
            class_name="Car",
            confidence=0.85,
        )

        assert isinstance(det.bbox, np.ndarray)
        assert det.bbox.dtype == np.float32

    def test_detection_to_dict(self):
        """Test Detection to_dict method."""
        det = Detection(
            bbox=np.array([100, 200, 300, 400]),
            class_id=2,
            class_name="Car",
            confidence=0.85,
        )

        d = det.to_dict()

        assert "bbox" in d
        assert "class_id" in d
        assert "class_name" in d
        assert "confidence" in d
        assert d["class_name"] == "Car"
        assert isinstance(d["bbox"], list)

    def test_detection_repr(self):
        """Test Detection string representation."""
        det = Detection(
            bbox=np.array([100, 200, 300, 400]),
            class_id=2,
            class_name="Car",
            confidence=0.85,
        )

        repr_str = repr(det)
        assert "Car" in repr_str
        assert "0.85" in repr_str


# =============================================================================
# IoU Tests
# =============================================================================

class TestIoU:
    """Tests for IoU computation."""

    def test_iou_identical_boxes(self):
        """Test IoU of identical boxes is 1.0."""
        box = np.array([0, 0, 10, 10])
        iou = compute_iou(box, box)
        assert iou == 1.0

    def test_iou_no_overlap(self):
        """Test IoU of non-overlapping boxes is 0.0."""
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([20, 20, 30, 30])
        iou = compute_iou(box1, box2)
        assert iou == 0.0

    def test_iou_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        box1 = np.array([0, 0, 10, 10])
        box2 = np.array([5, 5, 15, 15])

        # Intersection: 5x5 = 25
        # Union: 100 + 100 - 25 = 175
        # IoU = 25/175 ≈ 0.143
        iou = compute_iou(box1, box2)
        assert abs(iou - 25/175) < 0.001

    def test_iou_contained_box(self):
        """Test IoU when one box contains another."""
        box1 = np.array([0, 0, 20, 20])  # 400 area
        box2 = np.array([5, 5, 15, 15])  # 100 area, fully inside

        # Intersection = 100, Union = 400
        iou = compute_iou(box1, box2)
        assert abs(iou - 100/400) < 0.001

    def test_iou_matrix(self):
        """Test vectorized IoU matrix computation."""
        boxes1 = np.array([
            [0, 0, 10, 10],
            [5, 5, 15, 15],
        ])
        boxes2 = np.array([
            [0, 0, 10, 10],
            [20, 20, 30, 30],
        ])

        iou_mat = compute_iou_matrix(boxes1, boxes2)

        assert iou_mat.shape == (2, 2)
        assert iou_mat[0, 0] == 1.0  # Identical
        assert iou_mat[0, 1] == 0.0  # No overlap
        assert iou_mat[1, 1] == 0.0  # No overlap


# =============================================================================
# NMS Tests
# =============================================================================

class TestNMS:
    """Tests for Non-Maximum Suppression."""

    def test_nms_empty_input(self):
        """Test NMS with empty input."""
        result = apply_nms([], iou_threshold=0.5)
        assert result == []

    def test_nms_single_detection(self):
        """Test NMS with single detection."""
        det = Detection(
            bbox=np.array([0, 0, 10, 10]),
            class_id=2,
            class_name="Car",
            confidence=0.9,
        )

        result = apply_nms([det], iou_threshold=0.5)
        assert len(result) == 1

    def test_nms_no_overlap(self):
        """Test NMS keeps all non-overlapping detections."""
        detections = [
            Detection(bbox=np.array([0, 0, 10, 10]), class_id=2, class_name="Car", confidence=0.9),
            Detection(bbox=np.array([20, 20, 30, 30]), class_id=2, class_name="Car", confidence=0.8),
        ]

        result = apply_nms(detections, iou_threshold=0.5)
        assert len(result) == 2

    def test_nms_overlapping_same_class(self):
        """Test NMS suppresses overlapping boxes of same class."""
        detections = [
            Detection(bbox=np.array([0, 0, 10, 10]), class_id=2, class_name="Car", confidence=0.9),
            Detection(bbox=np.array([1, 1, 11, 11]), class_id=2, class_name="Car", confidence=0.7),
        ]

        result = apply_nms(detections, iou_threshold=0.5)

        # Only highest confidence should remain
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_nms_overlapping_different_classes(self):
        """Test NMS keeps overlapping boxes of different classes."""
        detections = [
            Detection(bbox=np.array([0, 0, 10, 10]), class_id=2, class_name="Car", confidence=0.9),
            Detection(bbox=np.array([1, 1, 11, 11]), class_id=0, class_name="Pedestrian", confidence=0.8),
        ]

        result = apply_nms(detections, iou_threshold=0.5, class_agnostic=False)

        # Both should remain (different classes)
        assert len(result) == 2

    def test_nms_class_agnostic(self):
        """Test class-agnostic NMS."""
        detections = [
            Detection(bbox=np.array([0, 0, 10, 10]), class_id=2, class_name="Car", confidence=0.9),
            Detection(bbox=np.array([1, 1, 11, 11]), class_id=0, class_name="Pedestrian", confidence=0.8),
        ]

        result = apply_nms(detections, iou_threshold=0.5, class_agnostic=True)

        # Only one should remain (class-agnostic NMS)
        assert len(result) == 1


# =============================================================================
# Filter Tests
# =============================================================================

class TestFilters:
    """Tests for detection filtering functions."""

    @pytest.fixture
    def sample_detections(self):
        """Create sample detections for testing."""
        return [
            Detection(bbox=np.array([0, 0, 100, 100]), class_id=2, class_name="Car", confidence=0.9),
            Detection(bbox=np.array([50, 50, 150, 150]), class_id=0, class_name="Pedestrian", confidence=0.7),
            Detection(bbox=np.array([200, 200, 250, 250]), class_id=1, class_name="Cyclist", confidence=0.5),
            Detection(bbox=np.array([300, 300, 310, 310]), class_id=2, class_name="Car", confidence=0.3),
        ]

    def test_filter_by_class(self, sample_detections):
        """Test filtering by class."""
        result = filter_by_class(sample_detections, ["Car"])
        assert len(result) == 2
        assert all(d.class_name == "Car" for d in result)

    def test_filter_by_class_multiple(self, sample_detections):
        """Test filtering by multiple classes."""
        result = filter_by_class(sample_detections, ["Car", "Pedestrian"])
        assert len(result) == 3

    def test_filter_by_confidence(self, sample_detections):
        """Test filtering by confidence."""
        result = filter_by_confidence(sample_detections, threshold=0.6)
        assert len(result) == 2
        assert all(d.confidence >= 0.6 for d in result)

    def test_filter_by_size(self, sample_detections):
        """Test filtering by box size."""
        result = filter_by_size(sample_detections, min_width=50, min_height=50)
        assert len(result) == 2  # Only 100x100 boxes

    def test_filter_by_size_max(self, sample_detections):
        """Test filtering with max size."""
        result = filter_by_size(sample_detections, min_width=0, min_height=0, max_width=50, max_height=50)
        assert len(result) == 1  # Only 10x10 box


# =============================================================================
# Post-Processor Tests
# =============================================================================

class TestDetectionPostProcessor:
    """Tests for DetectionPostProcessor class."""

    def test_processor_creation(self):
        """Test creating a post-processor."""
        processor = DetectionPostProcessor(
            min_confidence=0.3,
            min_box_size=(20, 20),
            nms_threshold=0.5,
        )

        assert processor.min_confidence == 0.3
        assert processor.nms_threshold == 0.5

    def test_processor_process_empty(self):
        """Test processing empty list."""
        processor = DetectionPostProcessor()
        result = processor.process([])
        assert result == []

    def test_processor_process_all_steps(self):
        """Test processor applies all steps."""
        detections = [
            Detection(bbox=np.array([0, 0, 100, 100]), class_id=2, class_name="Car", confidence=0.9),
            Detection(bbox=np.array([1, 1, 101, 101]), class_id=2, class_name="Car", confidence=0.8),
            Detection(bbox=np.array([200, 200, 210, 210]), class_id=2, class_name="Car", confidence=0.2),
        ]

        processor = DetectionPostProcessor(
            min_confidence=0.3,
            min_box_size=(20, 20),
            nms_threshold=0.5,
        )

        result = processor.process(detections)

        # Should filter out low confidence and apply NMS
        assert len(result) == 1
        assert result[0].confidence == 0.9


# =============================================================================
# Detector Initialization Tests
# =============================================================================

class TestObjectDetector2DInit:
    """Tests for ObjectDetector2D initialization (without loading model)."""

    def test_detector_default_init(self):
        """Test detector initializes with defaults."""
        # Just test creation, not model loading
        detector = ObjectDetector2D.__new__(ObjectDetector2D)
        detector.model_name = "yolov8n"
        detector.confidence_threshold = 0.3
        detector.device = "cuda"
        detector._model = None
        detector._model_loaded = False

        assert detector.model_name == "yolov8n"
        assert detector.confidence_threshold == 0.3

    def test_detector_class_mapping(self):
        """Test COCO to KITTI class mapping."""
        mapping = ObjectDetector2D.COCO_TO_KITTI

        assert mapping[0] == "Pedestrian"  # person
        assert mapping[1] == "Cyclist"     # bicycle
        assert mapping[2] == "Car"         # car
        assert mapping[3] == "Cyclist"     # motorcycle
        assert mapping[5] == "Van"         # bus
        assert mapping[7] == "Truck"       # truck

    def test_detector_class_names(self):
        """Test getting class names."""
        detector = ObjectDetector2D.__new__(ObjectDetector2D)
        detector.COCO_TO_KITTI = ObjectDetector2D.COCO_TO_KITTI

        names = list(set(detector.COCO_TO_KITTI.values()))
        assert "Car" in names
        assert "Pedestrian" in names
        assert "Cyclist" in names


# =============================================================================
# Integration Tests (requires model)
# =============================================================================

@pytest.fixture
def detector():
    """Create detector (requires ultralytics package)."""
    try:
        return ObjectDetector2D(
            model_name="yolov8n",
            confidence_threshold=0.3,
            device="cpu",  # Use CPU for tests
        )
    except ImportError:
        pytest.skip("ultralytics package not installed")


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple test image (doesn't need to contain objects)
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestObjectDetector2DIntegration:
    """Integration tests that require model loading."""

    def test_detector_warmup(self, detector):
        """Test detector warmup."""
        detector.warmup(image_size=(320, 320))
        assert detector._model_loaded

    def test_detector_detect_empty_image(self, detector, sample_image):
        """Test detection on empty image."""
        detections = detector.detect(sample_image)

        assert isinstance(detections, list)
        # May or may not have detections on blank image

    def test_detector_detect_output_format(self, detector, sample_image):
        """Test that detections have correct format."""
        detections = detector.detect(sample_image)

        for det in detections:
            assert isinstance(det, Detection)
            assert hasattr(det, "bbox")
            assert hasattr(det, "class_id")
            assert hasattr(det, "class_name")
            assert hasattr(det, "confidence")
            assert det.bbox.shape == (4,)
            assert 0 <= det.confidence <= 1

    def test_detector_detect_batch(self, detector, sample_image):
        """Test batch detection."""
        images = [sample_image, sample_image]
        batch_results = detector.detect_batch(images)

        assert len(batch_results) == 2
        assert all(isinstance(r, list) for r in batch_results)


# Run tests with: pytest tests/test_detector.py -v
# Skip integration tests: pytest tests/test_detector.py -v -k "not Integration"
