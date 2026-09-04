import json
from pathlib import Path

import numpy as np
import yaml

from experiments.run_f2_efim import evaluate_cell


def test_formal_efim_cell_uses_frozen_anisotropic_covariance() -> None:
    config = yaml.safe_load(Path("configs/full_localization.yaml").read_text())
    calibration = json.loads(
        Path("results/full_localization/f1_visual_calibration.json").read_text()
    )
    covariance = np.asarray(calibration["selected_model"]["covariance_uv_px2"])
    result = evaluate_cell(
        config["f2"], covariance, "test", "nominal", 60.0, 60.0, 1.0, 1.0, 0.2
    )
    assert result.visual_available
    assert result.vision_only_rank < 3
    assert result.peb_joint_m <= result.peb_rf_m
    assert result.zeb_joint_m <= result.zeb_rf_m


def test_out_of_view_geometry_reduces_to_rf_only() -> None:
    config = yaml.safe_load(Path("configs/full_localization.yaml").read_text())
    phase = dict(config["f2"])
    phase["camera"] = dict(phase["camera"], width_px=100, height_px=100)
    covariance = np.eye(2) * 2.0
    result = evaluate_cell(
        phase, covariance, "test", "nominal", 120.0, 20.0, 1.0, 1.0, 0.2
    )
    assert not result.visual_available
    assert np.isclose(result.peb_joint_m, result.peb_rf_m)
    assert np.isclose(result.zeb_joint_m, result.zeb_rf_m)
