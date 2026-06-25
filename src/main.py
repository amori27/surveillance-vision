"""Main surveillance loop."""

import cv2
from src.core.config import settings
from src.core.detector import Detector
from src.core.event_logger import EventLogger


def process_stream(source: str | int = 0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open stream: {source}")
        return

    detector = Detector()
    logger = EventLogger()

    frame_delay = max(1, int(1000 / settings.max_fps))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)
        boxes = detector.get_boxes(result)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{box['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if box["confidence"] > 0.8:
                event = logger.log_event(frame, "person", box["confidence"])
                print(f"Event: {event['label']} @ {event['timestamp']}")

        cv2.imshow("Surveillance", frame)
        if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    process_stream(source)
