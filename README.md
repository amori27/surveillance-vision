# Surveillance Vision

> Intelligent monitoring with YOLOv8 detection, ByteTrack tracking, and automated incident logging. Supports RTSP IP cameras and local webcams.

> **Heads up before you install:** `ultralytics` pulls in PyTorch, and a plain `pip install -r requirements.txt` can quietly download gigabytes of CUDA wheels even on a CPU-only machine. If you just want to run the tests, you don't need any of that — they mock YOLO: `uv run --with pytest --with pytest-cov --with numpy --with pydantic --with opencv-python-headless pytest tests/ -q --cov=src`. For a CPU-only install, put the CPU wheel first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then install the rest of `requirements.txt`. Keep the default install if you actually want GPU inference.

## Features

- YOLOv8 object detection with configurable confidence/IoU
- ByteTrack multi-object tracking across frames
- RTSP camera stream support
- Automatic snapshot saving on high-confidence events
- Configurable via environment variables

## Requirements

- Python 3.8+
- Webcam or RTSP IP camera
- ~500MB free disk space for Ultralytics model weights

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Ultralytics (heavy ~300MB with YOLO model weights)
pip install ultralytics

# 3. Run with local webcam:
python -m src.main

# Or with an RTSP camera:
python -m src.main "rtsp://user:pass@camera:554/stream"
```

## License

MIT
