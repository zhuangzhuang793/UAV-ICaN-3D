import numpy as np
import yaml

from experiments.run_phase3_gate import _evaluate


def _phase3_config():
    with open("configs/quick.yaml", encoding="utf-8") as stream:
        return yaml.safe_load(stream)["phase3"]


def test_representative_target_is_inside_image() -> None:
    result = _evaluate(_phase3_config(), 60.0, 60.0, 1.0, 2.0, 0.5)
    assert 0.0 <= result.pixel_u < 640.0
    assert 0.0 <= result.pixel_v < 480.0


def test_vision_only_is_depth_rank_deficient() -> None:
    result = _evaluate(_phase3_config(), 60.0, 60.0, 1.0, 2.0, 0.5)
    assert result.vision_only_rank == 2


def test_joint_information_cannot_worsen_crlb() -> None:
    result = _evaluate(_phase3_config(), 60.0, 60.0, 1.0, 2.0, 0.5)
    assert result.peb_joint_m <= result.peb_rf_m + 1e-12
    assert result.zeb_joint_m <= result.zeb_rf_m + 1e-12

