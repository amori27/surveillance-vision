# Surveillance Vision — Architecture

## Overview

Intelligent monitoring system using YOLOv8 for object detection, ByteTrack for multi-object tracking, and automatic incident logging with snapshot capture.

## Stack

| Layer | Technology |
|---|---|
| Detection | Ultralytics YOLOv8n |
| Tracking | ByteTrack |
| Stream | OpenCV (RTSP + local) |
| Storage | Local FS / S3 for snapshots |
| Logging | SQLAlchemy + SQLite |

## Data Flow

```
RTSP / Webcam ──> Frame Grabber ──> YOLOv8 ──> Detections ──> ByteTrack ──> Tracks
                                                                               │
                                                                        ┌──────▼──────┐
                                                                        │ Event Logger │
                                                                        │ + Snapshot   │
                                                                        └──────────────┘
```
