import inspect

from experiments.run_f4_association import _offline_subset
from uav_ican_3d.vision import ImageCVHungarianTracker, associate_candidates


def test_online_association_interfaces_exclude_gt_identity() -> None:
    tracker_parameters = set(inspect.signature(ImageCVHungarianTracker.step).parameters)
    proposed_parameters = set(inspect.signature(associate_candidates).parameters)
    assert "served_target_gt_id" not in tracker_parameters | proposed_parameters
    assert "user_gt_position" not in tracker_parameters | proposed_parameters
    assert "belief" not in tracker_parameters
    assert "rf" not in tracker_parameters


def test_subset_definition_is_explicitly_offline() -> None:
    assert "record" in inspect.signature(_offline_subset).parameters
