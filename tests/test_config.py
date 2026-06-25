from src.core.config import settings


def test_settings_defaults():
    assert settings.confidence_threshold == 0.5
    assert settings.iou_threshold == 0.45
    assert settings.target_classes == [0]
    assert settings.max_fps == 15
