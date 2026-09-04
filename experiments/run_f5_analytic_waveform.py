"""Formal analytic-LoS waveform benchmark for delay and 2-D AoA estimation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

from uav_ican_3d.localization import (
    SRSWaveformConfig,
    WaveformRFEstimator,
    predict_rf_observation,
    simulate_los_srs,
    wrap_angle,
)
from uav_ican_3d.simulation import complementarity_scene


def _grid(specification: list[float]) -> np.ndarray:
    start, stop, step = map(float, specification)
    return np.arange(start, stop + 0.5 * step, step)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _waveform_components(config: dict) -> tuple[SRSWaveformConfig, WaveformRFEstimator]:
    phase = config["f5"]
    waveform = SRSWaveformConfig(
        carrier_hz=float(config["system"]["carrier_hz"]),
        subcarrier_count=int(phase["subcarrier_count"]),
        subcarrier_spacing_hz=float(phase["subcarrier_spacing_hz"]),
        upa_rows=int(config["system"]["upa_shape"][0]),
        upa_columns=int(config["system"]["upa_shape"][1]),
    )
    estimator = WaveformRFEstimator(
        waveform,
        _grid(phase["range_grid_m"]),
        np.deg2rad(_grid(phase["azimuth_grid_deg"])),
        np.deg2rad(_grid(phase["elevation_grid_deg"])),
    )
    return waveform, estimator


def _estimate_only_waveforms(
    estimator: WaveformRFEstimator, received: np.ndarray
) -> np.ndarray:
    """Online boundary: accepts only complex samples and the preconfigured estimator."""

    return np.asarray([estimate.vector for estimate in estimator.estimate_batch(received)])


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("mode") != "full_localization":
        raise RuntimeError("F5 requires mode: full_localization")
    phase = config["f5"]
    if int(phase["waveform_realizations"]) != 200:
        raise RuntimeError("F5 requires exactly 200 waveforms per SNR and geometry")
    waveform, estimator = _waveform_components(config)
    output_root = Path(config["outputs"]["root"])
    protocol = json.loads((output_root / "f2_representative_geometries.json").read_text())
    if len(protocol["cells"]) != 8:
        raise RuntimeError("F5 requires the eight frozen representative geometries")
    batch_size = int(phase["estimator_batch_size"])
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    nominal_errors: list[np.ndarray] = []
    for cell in protocol["cells"]:
        target, world_body, body_array, _, _ = complementarity_scene(
            config["f2"], float(cell["altitude_m"]), float(cell["horizontal_distance_m"])
        )
        truth = predict_rf_observation(target, world_body.compose(body_array))
        if not (
            estimator.range_grid_m[0] <= truth[0] <= estimator.range_grid_m[-1]
            and estimator.azimuth_grid_rad[0] <= truth[1] <= estimator.azimuth_grid_rad[-1]
            and estimator.elevation_grid_rad[0] <= truth[2] <= estimator.elevation_grid_rad[-1]
        ):
            raise RuntimeError(f"cell {cell['cell_id']} truth lies outside an estimator grid")
        for snr_index, snr_db in enumerate(map(float, phase["snr_db"])):
            seed = int(config["seed"]) + 50000 + int(cell["cell_id"]) * 100 + snr_index
            rng = np.random.default_rng(seed)
            estimates: list[np.ndarray] = []
            count = int(phase["waveform_realizations"])
            for start in range(0, count, batch_size):
                current = min(batch_size, count - start)
                received = np.stack(
                    [
                        simulate_los_srs(
                            float(truth[0]),
                            float(truth[1]),
                            float(truth[2]),
                            snr_db,
                            waveform,
                            rng,
                        )
                        for _ in range(current)
                    ]
                )
                estimates.extend(_estimate_only_waveforms(estimator, received))
            estimate_array = np.asarray(estimates)
            errors = estimate_array - truth
            errors[:, 1] = wrap_angle(errors[:, 1])
            covariance = np.cov(errors, rowvar=False, ddof=1)
            bias = np.mean(errors, axis=0)
            if not np.all(np.isfinite(covariance)) or np.linalg.eigvalsh(covariance)[0] < -1e-12:
                raise AssertionError("F5 empirical covariance is non-finite or indefinite")
            if np.isclose(snr_db, float(phase["nominal_snr_db"])):
                nominal_errors.extend(errors)
            for realization, error in enumerate(errors):
                detail_rows.append(
                    {
                        "cell_id": int(cell["cell_id"]),
                        "altitude_m": float(cell["altitude_m"]),
                        "horizontal_distance_m": float(cell["horizontal_distance_m"]),
                        "snr_db": snr_db,
                        "realization": realization,
                        "seed": seed,
                        "range_error_m": float(error[0]),
                        "azimuth_error_deg": float(np.rad2deg(error[1])),
                        "elevation_error_deg": float(np.rad2deg(error[2])),
                    }
                )
            summary_rows.append(
                {
                    "cell_id": int(cell["cell_id"]),
                    "altitude_m": float(cell["altitude_m"]),
                    "horizontal_distance_m": float(cell["horizontal_distance_m"]),
                    "snr_db": snr_db,
                    "realizations": count,
                    "range_rmse_m": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
                    "azimuth_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(errors[:, 1] ** 2)))),
                    "elevation_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(errors[:, 2] ** 2)))),
                    "bias_range_m": float(bias[0]),
                    "bias_azimuth_rad": float(bias[1]),
                    "bias_elevation_rad": float(bias[2]),
                    **{
                        f"cov_{row}{column}": float(covariance[row, column])
                        for row in range(3)
                        for column in range(3)
                    },
                }
            )
            print(
                f"cell={cell['cell_id']} snr={snr_db:g}dB "
                f"range={summary_rows[-1]['range_rmse_m']:.3f}m "
                f"az={summary_rows[-1]['azimuth_rmse_deg']:.3f}deg "
                f"el={summary_rows[-1]['elevation_rmse_deg']:.3f}deg",
                flush=True,
            )

    _write_csv(output_root / "f5_waveform_errors.csv", detail_rows)
    _write_csv(output_root / "f5_waveform_summary.csv", summary_rows)
    aggregate_rows: list[dict] = []
    for snr_db in map(float, phase["snr_db"]):
        selected = [row for row in detail_rows if np.isclose(row["snr_db"], snr_db)]
        aggregate_rows.append(
            {
                "snr_db": snr_db,
                "waveforms": len(selected),
                "range_rmse_m": float(
                    np.sqrt(np.mean([row["range_error_m"] ** 2 for row in selected]))
                ),
                "azimuth_rmse_deg": float(
                    np.sqrt(np.mean([row["azimuth_error_deg"] ** 2 for row in selected]))
                ),
                "elevation_rmse_deg": float(
                    np.sqrt(np.mean([row["elevation_error_deg"] ** 2 for row in selected]))
                ),
            }
        )
    _write_csv(output_root / "f5_waveform_snr_curve.csv", aggregate_rows)
    snr_values = [row["snr_db"] for row in aggregate_rows]
    correlations = {
        metric: float(spearmanr(snr_values, [row[metric] for row in aggregate_rows]).statistic)
        for metric in ("range_rmse_m", "azimuth_rmse_deg", "elevation_rmse_deg")
    }
    nominal_error_array = np.asarray(nominal_errors)
    nominal_bias = np.mean(nominal_error_array, axis=0)
    nominal_covariance = np.cov(nominal_error_array, rowvar=False, ddof=1)
    threshold = float(phase["gate"]["minimum_error_snr_spearman_magnitude"])
    passed = (
        all(correlation <= -threshold for correlation in correlations.values())
        and all(
            aggregate_rows[-1][metric] < aggregate_rows[0][metric]
            for metric in ("range_rmse_m", "azimuth_rmse_deg", "elevation_rmse_deg")
        )
        and np.all(np.isfinite(nominal_covariance))
        and np.linalg.eigvalsh(nominal_covariance)[0] > 0.0
    )
    estimator_source = Path("src/uav_ican_3d/localization/rf_waveform.py")
    result = {
        "status": "PASS" if passed else "FAIL",
        "geometries": 8,
        "snr_levels": list(map(float, phase["snr_db"])),
        "waveforms_per_cell_snr": int(phase["waveform_realizations"]),
        "total_waveforms": len(detail_rows),
        "snr_error_spearman": correlations,
        "nominal_snr_db": float(phase["nominal_snr_db"]),
        "nominal_bias_range_azimuth_elevation": nominal_bias.tolist(),
        "nominal_covariance_range_azimuth_elevation": nominal_covariance.tolist(),
        "estimator_source": str(estimator_source),
        "estimator_source_sha256": _sha256(estimator_source),
        "estimator_online_inputs": ["complex_received_waveform", "known_reference"],
    }
    (output_root / "f5_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise AssertionError("F5-A GATE: FAIL")
    print("F5-A GATE: PASS")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/full_localization.yaml"))
    run(parser.parse_args().config)
