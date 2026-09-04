import numpy as np

from uav_ican_3d.vision import ImageCVHungarianTracker


def test_cv_tracker_uses_hungarian_continuity_after_confidence_initialization() -> None:
    tracker = ImageCVHungarianTracker(1.0, 1.0, 0.99, 2)
    selected = tracker.step([[10.0, 10.0], [50.0, 50.0]], [0.9, 0.2], 0.0)
    assert selected == 0
    selected = tracker.step([[49.0, 50.0], [12.0, 10.0]], [0.99, 0.1], 1.0)
    assert selected == 1


def test_cv_tracker_accepts_empty_frames_without_rf_or_gt() -> None:
    tracker = ImageCVHungarianTracker(1.0, 1.0, 0.99, 1)
    assert tracker.step(np.empty((0, 2)), np.empty(0), 0.0) is None
