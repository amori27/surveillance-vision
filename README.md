# Surveillance Vision

[![CI/CD](https://github.com/amori27/surveillance-vision/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/amori27/surveillance-vision/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Intelligent monitoring with YOLOv8 detection, ByteTrack tracking, and automated incident logging. Supports RTSP IP cameras and local webcams.

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
