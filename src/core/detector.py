"""YOLOv8 object detection wrapper."""

from ultralytics import YOLO
import numpy as np
from src.core.config import settings


class Detector:
    def __init__(self):
        self.model = YOLO(settings.model_name)

    def detect(self, frame: np.ndarray):
        results = self.model(
            frame,
            conf=settings.confidence_threshold,
            iou=settings.iou_threshold,
            classes=settings.target_classes,
            verbose=False,
        )
        return results[0]

    def get_boxes(self, result):
        if result.boxes is None:
            return []
        return [
            {
                "bbox": box.xyxy[0].cpu().numpy().tolist(),
                "confidence": float(box.conf[0]),
                "class_id": int(box.cls[0]),
            }
            for box in result.boxes
        ]
