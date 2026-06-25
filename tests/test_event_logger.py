import numpy as np
from src.core.event_logger import EventLogger


def test_log_event_returns_dict():
    logger = EventLogger()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    event = logger.log_event(frame, "person", 0.95)
    assert event["label"] == "person"
    assert event["confidence"] == 0.95
    assert "id" in event
    assert "timestamp" in event
    assert "snapshot" in event
