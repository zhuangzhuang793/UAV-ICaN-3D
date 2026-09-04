"""Run the deterministic Phase 0 coordinate-system quick gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from uav_ican_3d.geometry import PinholeCamera, RigidTransform


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0.0:
        matrix[:, 0] *= -1.0
    return matrix


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "quick":
        raise RuntimeError("Phase 0 gate refuses to run unless mode is 'quick'")

    phase = config["phase0"]
    trials = int(phase["random_trials"])
    if not 1 <= trials <= 10:
        raise ValueError("phase0.random_trials must be between 1 and 10")

    rng = np.random.default_rng(int(config["seed"]))
    worst_transform_error = 0.0
    for _ in range(trials):
        transform_world_body = RigidTransform(
            _random_rotation(rng), rng.uniform(-100.0, 100.0, size=3)
        )
        point_world = rng.uniform(-200.0, 200.0, size=3)
        recovered_world = transform_world_body.apply_point(
            transform_world_body.inverse().apply_point(point_world)
        )
        worst_transform_error = max(
            worst_transform_error, float(np.max(np.abs(recovered_world - point_world)))
        )

    camera = PinholeCamera(**phase["camera"])
    manual_point_camera = np.array([1.0, 2.0, 10.0])
    expected_pixel = np.array([360.0, 324.0])
    actual_pixel = camera.project_world(manual_point_camera, RigidTransform.identity())
    projection_error = float(np.max(np.abs(actual_pixel - expected_pixel)))

    transform_tolerance = float(phase["transform_tolerance"])
    projection_tolerance = float(phase["projection_tolerance"])
    print(f"mode=quick trials={trials}")
    print(f"worst_world_body_world_error={worst_transform_error:.3e}")
    print(f"manual_projection_error={projection_error:.3e}")
    if worst_transform_error > transform_tolerance:
        raise AssertionError("world -> body -> world error exceeds tolerance")
    if projection_error > projection_tolerance:
        raise AssertionError("hand-computed camera projection does not match")
    print("PHASE 0 QUICK GATE: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/quick.yaml"))
    run(parser.parse_args().config)

