"""Tests for object detection, tracker, and related schemas."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np

from src.models.schemas import Detection, Event
from src.core.config import Settings


class TestDetectionSchema:
    """Validate Detection schema constraints."""

    def test_valid_detection(self):
        d = Detection(bbox=[10.0, 20.0, 100.0, 200.0], confidence=0.95, class_id=0)
        assert d.bbox == [10.0, 20.0, 100.0, 200.0]
        assert d.confidence == 0.95
        assert d.class_id == 0

    def test_detection_bbox_exactly_four_values(self):
        d = Detection(bbox=[0, 0, 50, 50], confidence=0.5, class_id=1)
        assert len(d.bbox) == 4

    def test_detection_with_zero_confidence(self):
        d = Detection(bbox=[0, 0, 10, 10], confidence=0.0, class_id=0)
        assert d.confidence == 0.0

    def test_detection_with_max_confidence(self):
        d = Detection(bbox=[0, 0, 10, 10], confidence=1.0, class_id=0)
        assert d.confidence == 1.0

    def test_detection_negative_bbox_coordinates(self):
        d = Detection(bbox=[-10, -20, 100, 200], confidence=0.8, class_id=0)
        assert d.bbox[0] == -10

    def test_detection_high_class_id(self):
        d = Detection(bbox=[0, 0, 10, 10], confidence=0.9, class_id=79)
        assert d.class_id == 79


class TestEventSchema:
    """Validate Event schema constraints."""

    def test_valid_event(self):
        e = Event(
            id="abc123",
            label="person",
            confidence=0.93,
            timestamp="2026-01-01T00:00:00",
            snapshot="/tmp/snap.jpg",
        )
        assert e.id == "abc123"
        assert e.label == "person"

    def test_event_empty_label(self):
        e = Event(id="x", label="", confidence=0.5, timestamp="t", snapshot="s.jpg")
        assert e.label == ""

    def test_event_long_id(self):
        long_id = "a" * 256
        e = Event(id=long_id, label="car", confidence=0.7, timestamp="t", snapshot="s")
        assert len(e.id) == 256


class TestDetectorGetBoxes:
    """Test Detector.get_boxes with mocked YOLO results."""

    @staticmethod
    def _make_detector():
        mock_yolo_cls = MagicMock()
        mock_ultralytics = MagicMock()
        mock_ultralytics.YOLO = mock_yolo_cls
        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            if "src.core.detector" in sys.modules:
                del sys.modules["src.core.detector"]
            from src.core.detector import Detector
            return Detector()

    def test_empty_boxes_returns_empty_list(self):
        det = self._make_detector()
        result = MagicMock()
        result.boxes = None
        assert det.get_boxes(result) == []

    def test_single_box_parsed_correctly(self):
        det = self._make_detector()
        box = MagicMock()
        box.xyxy.__getitem__ = MagicMock(
            return_value=MagicMock(
                cpu=lambda: MagicMock(
                    numpy=lambda: MagicMock(
                        tolist=lambda: [10.0, 20.0, 100.0, 200.0]
                    )
                )
            )
        )
        conf_mock = MagicMock()
        conf_mock.__float__ = lambda self: 0.87
        box.conf.__getitem__ = MagicMock(return_value=conf_mock)

        cls_mock = MagicMock()
        cls_mock.__int__ = lambda self: 0
        box.cls.__getitem__ = MagicMock(return_value=cls_mock)

        result = MagicMock()
        result.boxes = [box]
        boxes = det.get_boxes(result)
        assert len(boxes) == 1
        assert boxes[0]["bbox"] == [10.0, 20.0, 100.0, 200.0]
        assert boxes[0]["confidence"] == 0.87
        assert boxes[0]["class_id"] == 0

    def test_multiple_boxes(self):
        det = self._make_detector()
        result = MagicMock()
        result.boxes = [MagicMock(), MagicMock(), MagicMock()]
        assert len(det.get_boxes(result)) == 3

    def test_no_boxes_attribute(self):
        det = self._make_detector()
        result = MagicMock()
        result.boxes = None
        assert det.get_boxes(result) == []


class TestDetectorInitialization:
    """Test Detector initialization with settings."""

    def test_detector_uses_settings_model(self):
        mock_yolo_cls = MagicMock()
        mock_ultralytics = MagicMock()
        mock_ultralytics.YOLO = mock_yolo_cls
        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            if "src.core.detector" in sys.modules:
                del sys.modules["src.core.detector"]
            from src.core.detector import Detector
            Detector()
            mock_yolo_cls.assert_called_once()

    def test_detector_model_loaded(self):
        mock_yolo_cls = MagicMock()
        mock_ultralytics = MagicMock()
        mock_ultralytics.YOLO = mock_yolo_cls
        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            if "src.core.detector" in sys.modules:
                del sys.modules["src.core.detector"]
            from src.core.detector import Detector
            det = Detector()
            assert det.model is not None


class TestSettings:
    """Test Settings configuration class."""

    def test_settings_can_be_instantiated(self):
        s = Settings()
        assert hasattr(s, "model_name")
        assert hasattr(s, "confidence_threshold")
        assert hasattr(s, "iou_threshold")
        assert hasattr(s, "snapshot_dir")
        assert hasattr(s, "db_path")
        assert hasattr(s, "target_classes")
        assert hasattr(s, "max_fps")

    def test_settings_types(self):
        s = Settings()
        assert isinstance(s.model_name, str)
        assert isinstance(s.confidence_threshold, float)
        assert isinstance(s.iou_threshold, float)
        assert isinstance(s.max_fps, int)
        assert isinstance(s.target_classes, list)

    def test_settings_default_model_is_yolov8n(self):
        s = Settings()
        assert s.model_name == "yolov8n.pt"

    def test_settings_default_confidence(self):
        s = Settings()
        assert 0.0 < s.confidence_threshold <= 1.0

    def test_settings_default_iou(self):
        s = Settings()
        assert 0.0 < s.iou_threshold <= 1.0


class TestTracker:
    """Test the track function with mocked YOLO."""

    def test_track_calls_model(self):
        mock_ultralytics = MagicMock()
        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            from src.core.tracker import track
            mock_model = MagicMock()
            mock_result = MagicMock()
            mock_model.track.return_value = [mock_result]
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = track(frame, mock_model)
            mock_model.track.assert_called_once()
            assert result is mock_result

    def test_track_returns_none_on_empty(self):
        mock_ultralytics = MagicMock()
        with patch.dict(sys.modules, {"ultralytics": mock_ultralytics}):
            from src.core.tracker import track
            mock_model = MagicMock()
            mock_model.track.return_value = []
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            assert track(frame, mock_model) is None
