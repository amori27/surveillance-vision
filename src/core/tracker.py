"""ByteTrack multi-object tracker adapter."""

from ultralytics import YOLO


def track(frame, model: YOLO):
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )
    return results[0] if results else None
