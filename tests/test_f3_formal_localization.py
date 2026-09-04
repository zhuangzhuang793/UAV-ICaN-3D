import json
from pathlib import Path

import yaml

from experiments.run_f3_observation_localization import _run_geometry


def test_f3_geometry_smoke_is_paired_and_finite() -> None:
    config = yaml.safe_load(Path("configs/full_localization.yaml").read_text())
    protocol = json.loads(
        Path("results/full_localization/f2_representative_geometries.json").read_text()
    )
    calibration = json.loads(
        Path("results/full_localization/f1_visual_calibration.json").read_text()
    )
    phase3 = dict(config["f3"])
    phase3["monte_carlo_trials"] = 4
    summary, trials = _run_geometry(
        (
            config["f2"],
            phase3,
            protocol["cells"][0],
            {
                "rf_parameters": protocol["rf_parameters"],
                "camera_covariance_uv_px2": calibration["selected_model"][
                    "covariance_uv_px2"
                ],
            },
            44,
        )
    )
    assert summary["trials"] == 4
    assert len(trials) == 4
    assert all(row.rf_nees >= 0.0 and row.joint_nees >= 0.0 for row in trials)
