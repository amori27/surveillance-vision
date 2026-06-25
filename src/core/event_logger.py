"""Event logging and snapshot management."""

import uuid
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from src.core.config import settings


class EventLogger:
    def __init__(self):
        self.snapshot_dir = Path(settings.snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, frame: np.ndarray, label: str, confidence: float):
        ts = datetime.utcnow().isoformat()
        snap_id = uuid.uuid4().hex[:8]
        path = self.snapshot_dir / f"{snap_id}.jpg"
        cv2.imwrite(str(path), frame)

        return {
            "id": snap_id,
            "label": label,
            "confidence": confidence,
            "timestamp": ts,
            "snapshot": str(path),
        }
