import os


class Settings:
    model_name: str = os.getenv("MODEL_NAME", "yolov8n.pt")
    confidence_threshold: float = float(
        os.getenv("CONFIDENCE_THRESHOLD", "0.5")
    )
    iou_threshold: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
    snapshot_dir: str = os.getenv("SNAPSHOT_DIR", "data/snapshots")
    db_path: str = os.getenv("DB_PATH", "data/events.db")
    target_classes: list[int] = [0]  # person
    max_fps: int = int(os.getenv("MAX_FPS", "15"))


settings = Settings()
