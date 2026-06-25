# API Reference

## `GET /health`

```json
{"status": "ok", "detections_today": 142}
```

## `GET /events`

List recorded events.

```json
[
  {"id": 1, "label": "person", "confidence": 0.93, "timestamp": "...", "snapshot": "snap_001.jpg"}
]
```

## `POST /stream`

Start monitoring an RTSP stream.

**Request:**
```json
{"url": "rtsp://camera:554/stream1"}
```
