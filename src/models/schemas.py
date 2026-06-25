from pydantic import BaseModel


class Detection(BaseModel):
    bbox: list[float]
    confidence: float
    class_id: int


class Event(BaseModel):
    id: str
    label: str
    confidence: float
    timestamp: str
    snapshot: str
