from src.models.schemas import Detection, Event


def test_detection():
    d = Detection(bbox=[10, 20, 100, 200], confidence=0.9, class_id=0)
    assert len(d.bbox) == 4
    assert d.confidence == 0.9


def test_event():
    e = Event(id="abc", label="person", confidence=0.93,
              timestamp="2026-01-01", snapshot="snap.jpg")
    assert e.label == "person"
    assert e.id == "abc"
