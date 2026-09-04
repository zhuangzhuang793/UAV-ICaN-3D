"""Mechanically evaluate the pre-registered F8 localization criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def run(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["outputs"]["root"])
    phase = config["f8"]
    f7 = json.loads((output_root / "f7_summary.json").read_text())
    f4 = json.loads((output_root / "f4_association_summary.json").read_text())
    if f7["status"] != "COMPLETE" or f4["status"] != "PASS":
        raise RuntimeError("F8 requires complete F7 and passing F4 artifacts")
    coverage_min, coverage_max = map(float, phase["preferred_coverage_range"])
    checks = {
        "minimum_3d_rmse_reduction": {
            "target": float(phase["minimum_3d_rmse_reduction"]),
            "observed": float(f7["primary_3d_rmse_reduction"]),
            "pass": float(f7["primary_3d_rmse_reduction"])
            >= float(phase["minimum_3d_rmse_reduction"]),
        },
        "positive_95ci_lower_bound": {
            "target": 0.0,
            "observed": float(f7["bootstrap"]["improvement_95ci_lower"]),
            "pass": float(f7["bootstrap"]["improvement_95ci_lower"]) > 0.0,
        },
        "target_z_rmse_reduction": {
            "target": float(phase["target_z_rmse_reduction"]),
            "observed": float(f7["primary_z_rmse_reduction"]),
            "pass": float(f7["primary_z_rmse_reduction"])
            >= float(phase["target_z_rmse_reduction"]),
        },
        "preferred_joint_coverage": {
            "target": [coverage_min, coverage_max],
            "observed": float(
                f7["methods"]["rf_vision_shared_pose"]["coverage_95"]
            ),
            "pass": coverage_min
            <= float(f7["methods"]["rf_vision_shared_pose"]["coverage_95"])
            <= coverage_max,
        },
        "hard_association_accuracy": {
            "target": float(phase["target_hard_association_accuracy"]),
            "observed": float(f4["proposed_hard_accuracy"]),
            "pass": float(f4["proposed_hard_accuracy"])
            >= float(phase["target_hard_association_accuracy"]),
        },
    }
    passed = all(check["pass"] for check in checks.values())
    result = {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "f7_hard_association_seeded_diagnostic": float(f7["association"]["hard"]),
        "decision_rule": "all pre-registered criteria must pass",
        "next_phase_allowed": bool(passed),
    }
    (output_root / "f8_success_criteria.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("F8 GATE: FAIL; preserve results and stop before F9")
    print("F8 GATE: PASS")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/full_localization.yaml"))
    run(parser.parse_args().config)
