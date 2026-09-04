"""Run the final timestamped end-to-end QUICK closed-loop smoke test."""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from uav_ican_3d.control import (
    propagate_position_to_angles,
    robust_mpc_step,
    sample_trajectory_distribution,
    select_robust_beam,
)
from uav_ican_3d.geometry import PinholeCamera, RigidTransform
from uav_ican_3d.localization import (
    SRSWaveformConfig,
    WaveformRFEstimator,
    estimate_position_map,
    predict_rf_observation,
    simulate_los_srs,
    wrap_angle,
)
from uav_ican_3d.prediction import BeliefTrajectoryPredictor, update_position_belief
from uav_ican_3d.simulation import ROTATION_AIRSIM_CAMERA_OPENCV
from uav_ican_3d.types import BeliefState, RFObservation, VisualObservation
from uav_ican_3d.vision import associate_candidates, project_joint_rf_belief_to_image


BODY_ROTATION_WORLD = np.array(
    [
        [np.sqrt(0.5), 0.0, -np.sqrt(0.5)],
        [0.0, -1.0, 0.0],
        [-np.sqrt(0.5), 0.0, -np.sqrt(0.5)],
    ]
)


@dataclass(frozen=True)
class OnlineObservation:
    timestamp_s: float
    received_srs: np.ndarray
    visual_candidates_uv: np.ndarray


@dataclass(frozen=True)
class EvaluationTruth:
    ue_position_world_m: np.ndarray
    uav_position_world_m: np.ndarray
    true_range_m: float


class QuickClosedLoopEnvironment:
    """Own ground truth and expose only sensor outputs to the online stack."""

    def __init__(
        self,
        waveform_config: SRSWaveformConfig,
        camera: PinholeCamera,
        body_camera: RigidTransform,
        visual_covariance: np.ndarray,
        snr_db: float,
        seed: int,
    ) -> None:
        self.waveform_config = waveform_config
        self.camera = camera
        self.body_camera = body_camera
        self.visual_covariance = visual_covariance
        self.snr_db = snr_db
        self.rng = np.random.default_rng(seed)
        self.ue_position = np.array([20.0, 5.0, 0.0])
        self.ue_heading = 0.20
        self.ue_speed_mps = 5.0
        self.ue_turn_rate_rps = 0.10
        self.uav_position = np.array([0.0, 0.0, 50.0])

    def observe(self, timestamp_s: float) -> tuple[OnlineObservation, EvaluationTruth]:
        true_world_body = RigidTransform(BODY_ROTATION_WORLD, self.uav_position)
        rf_truth = predict_rf_observation(self.ue_position, true_world_body)
        received = simulate_los_srs(
            *rf_truth, self.snr_db, self.waveform_config, self.rng
        )
        distractors = [
            self.ue_position + np.array([12.0, -9.0, 0.0]),
            self.ue_position + np.array([-10.0, 11.0, 0.0]),
        ]
        targets = [self.ue_position, *distractors]
        candidates = []
        for target in targets:
            pixel = self.camera.project_world(
                target, true_world_body.compose(self.body_camera)
            )
            candidates.append(
                pixel
                + self.rng.multivariate_normal(np.zeros(2), self.visual_covariance)
            )
        candidates.append(np.array([60.0, 420.0]))
        candidates_array = np.asarray(candidates)[self.rng.permutation(len(candidates))]
        observation = OnlineObservation(timestamp_s, received, candidates_array)
        truth = EvaluationTruth(
            self.ue_position.copy(), self.uav_position.copy(), float(rf_truth[0])
        )
        return observation, truth

    def advance(self, velocity_world_mps: np.ndarray, dt_s: float) -> None:
        velocity = self.ue_speed_mps * np.array(
            [np.cos(self.ue_heading), np.sin(self.ue_heading), 0.0]
        )
        self.ue_position = self.ue_position + velocity * dt_s
        self.ue_heading += self.ue_turn_rate_rps * dt_s
        actuation_noise = self.rng.normal(0.0, 0.02, 3)
        self.uav_position = self.uav_position + velocity_world_mps * dt_s + actuation_noise


def _load_predictor(path: Path) -> BeliefTrajectoryPredictor:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    predictor = BeliefTrajectoryPredictor(
        checkpoint["history_steps"],
        checkpoint["horizon_steps"],
        checkpoint["hidden_size"],
        checkpoint["dt_s"],
        use_covariance=True,
    )
    predictor.load_state_dict(checkpoint["state_dict"])
    predictor.eval()
    return predictor


def _waveform_estimator(config: SRSWaveformConfig) -> WaveformRFEstimator:
    return WaveformRFEstimator(
        config,
        np.arange(15.0, 80.01, 0.1),
        np.deg2rad(np.arange(-60.0, 60.01, 0.5)),
        np.deg2rad(np.arange(-20.0, 45.01, 0.5)),
    )


def _online_localize(
    observation: OnlineObservation,
    nominal_world_body: RigidTransform,
    estimator: WaveformRFEstimator,
    rf_bias: np.ndarray,
    rf_covariance: np.ndarray,
    pose_covariance: np.ndarray,
    camera: PinholeCamera,
    body_camera: RigidTransform,
    visual_covariance: np.ndarray,
    association_confidence: float,
) -> tuple[object, object, bool]:
    measurement = estimator.estimate(observation.received_srs).vector - rf_bias
    measurement[1:] = wrap_angle(measurement[1:])
    rf_observation = RFObservation(
        measurement[0],
        measurement[1],
        measurement[2],
        rf_covariance,
        observation.timestamp_s,
    )
    rf_estimate = estimate_position_map(
        rf_observation,
        nominal_world_body,
        RigidTransform.identity(),
        pose_covariance,
    )
    image_belief = project_joint_rf_belief_to_image(
        rf_estimate.position_world_m,
        rf_estimate.joint_covariance,
        nominal_world_body,
        body_camera,
        camera,
        pose_delta_mean=rf_estimate.pose_delta,
    )
    association = associate_candidates(
        observation.visual_candidates_uv, image_belief, association_confidence
    )
    if association.selected_index is None:
        return rf_estimate, rf_estimate, False
    visual_observation = VisualObservation(
        observation.visual_candidates_uv[association.selected_index],
        visual_covariance,
        observation.timestamp_s,
    )
    joint_estimate = estimate_position_map(
        rf_observation,
        nominal_world_body,
        RigidTransform.identity(),
        pose_covariance,
        camera,
        body_camera,
        visual_observation,
        rf_estimate.position_world_m,
    )
    return rf_estimate, joint_estimate, True


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "quick":
        raise RuntimeError("Phase 10 gate refuses to run unless mode is 'quick'")
    phase = config["phase10"]
    rf_calibration = json.loads(
        Path(phase["waveform_calibration_json"]).read_text(encoding="utf-8")
    )
    rf_bias = np.asarray(rf_calibration["bias_range_azimuth_elevation"])
    rf_covariance = np.asarray(rf_calibration["covariance_range_azimuth_elevation"])
    predictor = _load_predictor(Path(phase["predictor_checkpoint"]))
    waveform_config = SRSWaveformConfig()
    estimator = _waveform_estimator(waveform_config)
    camera = PinholeCamera(320.0, 320.0, 320.0, 240.0)
    body_camera = RigidTransform(ROTATION_AIRSIM_CAMERA_OPENCV, np.zeros(3))
    visual_covariance = np.asarray(phase["visual_covariance_px2"], dtype=float)
    pose_covariance = np.diag(
        [float(phase["pose_position_std_m"]) ** 2] * 3
        + [np.deg2rad(float(phase["pose_attitude_std_deg"])) ** 2] * 3
    )
    environment = QuickClosedLoopEnvironment(
        waveform_config,
        camera,
        body_camera,
        visual_covariance,
        float(phase["waveform_snr_db"]),
        int(config["seed"]) + 10,
    )
    rng = np.random.default_rng(int(config["seed"]) + 100)
    nominal_uav_position = np.array([0.0, 0.0, 50.0])
    tracker: BeliefState | None = None
    history: deque[BeliefState] = deque(maxlen=predictor.history_steps)
    azimuth_grid = np.deg2rad(np.arange(-60.0, 60.1, 15.0))
    elevation_grid = np.deg2rad(np.arange(-15.0, 45.1, 15.0))
    rows: list[dict[str, float | int]] = []
    true_ranges = []
    initial_uav_position = nominal_uav_position.copy()
    dt_s = float(phase["slow_dt_s"])

    for step in range(int(phase["episode_steps"])):
        timestamp_s = (step + 1) * dt_s
        observation, truth = environment.observe(timestamp_s)
        nominal_world_body = RigidTransform(BODY_ROTATION_WORLD, nominal_uav_position)
        rf_estimate, joint_estimate, associated = _online_localize(
            observation,
            nominal_world_body,
            estimator,
            rf_bias,
            rf_covariance,
            pose_covariance,
            camera,
            body_camera,
            visual_covariance,
            float(phase["association_confidence"]),
        )
        if tracker is None:
            tracker = BeliefState(
                np.concatenate((joint_estimate.position_world_m, np.zeros(3))),
                np.block(
                    [
                        [joint_estimate.position_covariance, np.zeros((3, 3))],
                        [np.zeros((3, 3)), np.eye(3) * 4.0],
                    ]
                ),
                timestamp_s,
            )
            for _ in range(predictor.history_steps):
                history.append(tracker)
        else:
            tracker = update_position_belief(
                tracker,
                joint_estimate.position_world_m,
                joint_estimate.position_covariance,
                timestamp_s,
            )
            history.append(tracker)
        means = torch.from_numpy(
            np.stack([belief.mean for belief in history]).astype(np.float32)
        ).unsqueeze(0)
        covariances = torch.from_numpy(
            np.stack([belief.covariance for belief in history]).astype(np.float32)
        ).unsqueeze(0)
        with torch.no_grad():
            predicted_mean, predicted_std = predictor(means, covariances)
        predicted_mean_numpy = predicted_mean[0].numpy().astype(float)
        predicted_std_numpy = predicted_std[0].numpy().astype(float)
        trajectory_samples = sample_trajectory_distribution(
            predicted_mean_numpy,
            predicted_std_numpy,
            int(phase["trajectory_samples"]),
            rng,
        )
        decision = robust_mpc_step(
            nominal_uav_position,
            trajectory_samples,
            dt_s,
            float(phase["maximum_uav_speed_mps"]),
            tuple(float(value) for value in phase["altitude_bounds_m"]),
        )
        nominal_uav_position = nominal_uav_position + decision.velocity_world_mps * dt_s
        world_array = RigidTransform(BODY_ROTATION_WORLD, nominal_uav_position)
        angular_belief = propagate_position_to_angles(
            tracker.mean[:3], tracker.covariance[:3, :3], world_array
        )
        for fast_step in range(int(phase["fast_beam_updates_per_step"])):
            beam = select_robust_beam(
                angular_belief,
                azimuth_grid,
                elevation_grid,
                rng,
                sample_count=128,
            )
            rows.append(
                {
                    "timestamp_s": timestamp_s + fast_step * dt_s / 2.0,
                    "step": step,
                    "fast_step": fast_step,
                    "rf_localization_error_m": float(
                        np.linalg.norm(rf_estimate.position_world_m - truth.ue_position_world_m)
                    ),
                    "joint_localization_error_m": float(
                        np.linalg.norm(joint_estimate.position_world_m - truth.ue_position_world_m)
                    ),
                    "associated": int(associated),
                    "prediction_uncertainty_m": float(np.mean(predicted_std_numpy)),
                    "mpc_score": decision.score,
                    "uav_x_m": float(nominal_uav_position[0]),
                    "uav_y_m": float(nominal_uav_position[1]),
                    "uav_z_m": float(nominal_uav_position[2]),
                    "uav_speed_mps": float(np.linalg.norm(decision.velocity_world_mps)),
                    "beam_azimuth_rad": beam.azimuth_rad,
                    "beam_elevation_rad": beam.elevation_rad,
                    "beam_aperture": beam.aperture_size,
                    "beam_robust_gain": beam.robust_gain,
                    "true_range_evaluation_only_m": truth.true_range_m,
                }
            )
        true_ranges.append(truth.true_range_m)
        environment.advance(decision.velocity_world_mps, dt_s)

    numeric = np.array([[float(value) for value in row.values()] for row in rows])
    associations = len({row["step"] for row in rows if row["associated"]})
    slow_updates = len({row["step"] for row in rows})
    fast_updates = len(rows)
    uav_motion = float(np.linalg.norm(nominal_uav_position - initial_uav_position))
    geometry_feedback = uav_motion > 0.0 and float(np.ptp(true_ranges)) > 0.0
    altitude_bounds = tuple(float(value) for value in phase["altitude_bounds_m"])
    constraints_ok = all(
        row["uav_speed_mps"] <= float(phase["maximum_uav_speed_mps"]) + 1e-9
        and altitude_bounds[0] <= row["uav_z_m"] <= altitude_bounds[1]
        for row in rows
    )
    gate = phase["gate"]
    passed = (
        associations >= int(gate["minimum_associations"])
        and slow_updates >= int(gate["minimum_predictions"])
        and slow_updates >= int(gate["minimum_slow_updates"])
        and fast_updates >= int(gate["minimum_fast_updates"])
        and np.all(np.isfinite(numeric))
        and constraints_ok
        and geometry_feedback
    )
    result_path = Path(phase["results_csv"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    result_path.write_text(
        ",".join(columns)
        + "\n"
        + "\n".join(",".join(str(row[column]) for column in columns) for row in rows)
        + "\n"
    )
    rf_rmse = float(
        np.sqrt(np.mean([row["rf_localization_error_m"] ** 2 for row in rows[::2]]))
    )
    joint_rmse = float(
        np.sqrt(np.mean([row["joint_localization_error_m"] ** 2 for row in rows[::2]]))
    )
    status = "PASS" if passed else "FAIL"
    Path("docs/PHASE10_DECISION.md").write_text(
        f"""# Phase 10 final end-to-end smoke test

Status: **{status}**

The short online loop connected SRS reception, waveform range/2-D-AoA estimation, 3-D RF belief,
image projection, served-target association, joint localization, six-state belief tracking,
probabilistic trajectory prediction, slow robust MPC, and fast 3-D UPA beam selection.

- Waveform/localization/prediction/slow-control steps: `{slow_updates}`
- Successful visual associations: `{associations}/{slow_updates}`
- Fast beam updates: `{fast_updates}`
- RF / joint localization RMSE: `{rf_rmse:.3f} / {joint_rmse:.3f}` m
- Mean logged prediction uncertainty:
  `{np.mean([row['prediction_uncertainty_m'] for row in rows]):.3f}` m
- Nominal UAV motion: `{uav_motion:.3f}` m
- New geometry changed later observations: `{geometry_feedback}`
- All timestamps/values finite and constraints satisfied: `{constraints_ok}`

`QuickClosedLoopEnvironment` alone owns UE/UAV truth and emits only complex SRS samples plus noisy
visual candidates. `_online_localize`, the tracker, predictor, MPC, and beam selector receive no
truth. Ground truth appears only in the evaluation logger columns explicitly suffixed
`evaluation_only` or in localization-error calculations after the online outputs exist.
""",
        encoding="utf-8",
    )
    print(f"online_steps={slow_updates} associations={associations} fast_updates={fast_updates}")
    print(f"rf_rmse_m={rf_rmse:.3f} joint_rmse_m={joint_rmse:.3f}")
    print(f"uav_motion_m={uav_motion:.3f} geometry_feedback={geometry_feedback}")
    print(f"all_finite={np.all(np.isfinite(numeric))} constraints_ok={constraints_ok}")
    print(f"PHASE 10 QUICK GATE: {status}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/quick.yaml"))
    run(parser.parse_args().config)
