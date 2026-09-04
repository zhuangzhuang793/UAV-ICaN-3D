from pathlib import Path

import numpy as np
import yaml

from experiments.generate_cosys_full_localization import _sequence_split, _trajectory


CONFIG = Path("configs/full_localization.yaml")


def test_full_sequences_are_disjoint_and_have_expected_size() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    phase = config["f0"]
    split_sets = [set(values) for values in phase["sequence_splits"].values()]
    assert sum(map(len, split_sets)) == 12
    assert not any(
        first & second
        for index, first in enumerate(split_sets)
        for second in split_sets[index + 1 :]
    )
    assert sum(map(len, split_sets)) * phase["frames_per_sequence"] == 4800
    for split, sequence_ids in phase["sequence_splits"].items():
        for sequence_id in sequence_ids:
            assert _sequence_split(phase, sequence_id) == split


def test_full_trajectory_is_deterministic_and_camera_stays_above_target() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    profile = config["f0"]["sequence_profiles"]["seq00"]
    first = _trajectory(37, 400, profile, config["seed"])
    second = _trajectory(37, 400, profile, config["seed"])
    for name in first[0]:
        assert np.allclose(first[0][name][0], second[0][name][0])
        assert first[0][name][1] == second[0][name][1]
    served = first[0]["ServedVehicle"][0]
    camera = first[1]
    camera_roll, camera_pitch, _ = first[2]
    assert camera[2] < served[2]
    assert np.linalg.norm(camera[:2] - served[:2]) > 0.0
    assert camera_roll == 0.0
    assert camera_pitch < 0.0


def test_formal_protocol_keeps_quick_outputs_separate() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["mode"] == "full_localization"
    assert config["outputs"]["root"] == "results/full_localization"
    assert config["f0"]["detector"]["training_fraction"] == 1.0
    assert config["f0"]["detector"]["epochs"] <= 20
    assert config["f0"]["detector"]["patience"] == 5
