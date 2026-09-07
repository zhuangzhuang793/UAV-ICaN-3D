"""Audit Sionna multipath without per-channel received-power normalization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from experiments.run_f5_analytic_waveform import _estimate_only_waveforms, _waveform_components
from experiments.run_f6_sionna import _truth
from uav_ican_3d.localization import (
    sionna_received_power_gain,
    synthesize_sionna_waveform,
    trace_sionna_channel,
    wrap_angle,
)
from uav_ican_3d.localization.rf_realism import link_budget_for_reference_snr


def _dbm_to_watts(value: float) -> float:
    return 1e-3 * 10.0 ** (value / 10.0)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(settings_path: Path) -> dict:
    settings = yaml.safe_load(settings_path.read_text())
    base = yaml.safe_load(Path(settings["base_config"]).read_text())
    output_root = Path(settings["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    waveform, estimator = _waveform_components(base)
    link = settings["link_budget"]
    budget = link_budget_for_reference_snr(
        float(link["reference_range_m"]),
        float(link["reference_snr_db"]),
        waveform,
        tx_gain_dbi=float(link["tx_gain_dbi"]),
        rx_gain_dbi=float(link["rx_gain_dbi"]),
        receiver_noise_figure_db=float(link["receiver_noise_figure_db"]),
        system_loss_db=float(link["system_loss_db"]),
    )
    transmit_total_w = _dbm_to_watts(budget.tx_power_total_dbm)
    transmit_per_subcarrier_w = transmit_total_w / waveform.subcarrier_count
    noise_per_subcarrier_w = _dbm_to_watts(
        budget.noise_power_per_subcarrier_dbm(waveform)
    )
    protocol = json.loads(
        (Path(base["outputs"]["root"]) / "f2_representative_geometries.json").read_text()
    )
    f6 = base["f6"]
    sionna = settings["sionna"]
    rows: list[dict] = []
    summaries: list[dict] = []
    for reflections in (False, bool(sionna["include_reflections"])):
        label = "los" if not reflections else "multipath"
        for cell in protocol["cells"]:
            seed = int(settings["seed"]) + int(sionna["seed_offset"]) + int(cell["cell_id"])
            channel = trace_sionna_channel(
                Path(f6["scene_asset"]),
                float(base["system"]["carrier_hz"]),
                [float(cell["horizontal_distance_m"]), 0.0, float(f6["target_height_m"])],
                [0.0, 0.0, float(cell["altitude_m"])],
                max_depth=(
                    int(f6["los_max_depth"])
                    if not reflections
                    else int(f6["multipath_max_depth"])
                ),
                reflections=reflections,
                samples_per_source=int(f6["samples_per_source"]),
                max_paths_per_source=int(f6["max_paths_per_source"]),
                synthetic_array=bool(f6["synthetic_array"]),
                seed=seed,
                diffuse_reflections=reflections and bool(sionna["diffuse_reflections"]),
                refraction=reflections and bool(sionna["refraction"]),
                diffraction=reflections and bool(sionna["diffraction"]),
                edge_diffraction=reflections and bool(sionna["edge_diffraction"]),
            )
            raw_gain = sionna_received_power_gain(channel, waveform)
            received_power_w = transmit_per_subcarrier_w * raw_gain
            effective_snr_db = 10.0 * np.log10(received_power_w / noise_per_subcarrier_w)
            truth = _truth(cell, float(f6["target_height_m"]))
            count = int(sionna["realizations_per_geometry"])
            rng = np.random.default_rng(seed + 900_000)
            received = np.stack(
                [
                    synthesize_sionna_waveform(
                        channel,
                        0.0,
                        waveform,
                        rng,
                        normalize_received_power=False,
                        transmit_power_per_subcarrier_w=transmit_per_subcarrier_w,
                        noise_power_per_subcarrier_w=noise_per_subcarrier_w,
                    )
                    for _ in range(count)
                ]
            )
            estimates = _estimate_only_waveforms(estimator, received)
            errors = estimates - truth
            errors[:, 1] = wrap_angle(errors[:, 1])
            for realization, error in enumerate(errors):
                rows.append(
                    {
                        "channel": label,
                        "cell_id": int(cell["cell_id"]),
                        "realization": realization,
                        "range_m": float(truth[0]),
                        "path_count": int(channel.delays_s.size),
                        "raw_power_gain": raw_gain,
                        "effective_snr_db": effective_snr_db,
                        "range_error_m": float(error[0]),
                        "azimuth_error_deg": float(np.rad2deg(error[1])),
                        "elevation_error_deg": float(np.rad2deg(error[2])),
                    }
                )
            summary = {
                "channel": label,
                "cell_id": int(cell["cell_id"]),
                "range_m": float(truth[0]),
                "path_count": int(channel.delays_s.size),
                "raw_power_gain": raw_gain,
                "effective_snr_db": effective_snr_db,
                "range_rmse_m": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
                "azimuth_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(errors[:, 1] ** 2)))),
                "elevation_rmse_deg": float(
                    np.rad2deg(np.sqrt(np.mean(errors[:, 2] ** 2)))
                ),
            }
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)
    _write_csv(output_root / "sionna_absolute_power_errors.csv", rows)
    _write_csv(output_root / "sionna_absolute_power_cells.csv", summaries)
    aggregate = []
    for label in ("los", "multipath"):
        selected = [row for row in rows if row["channel"] == label]
        aggregate.append(
            {
                "channel": label,
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
    result = {
        "status": "COMPLETE",
        "scope": "eight frozen representative geometries; not full AirSim trajectories",
        "normalization_removed": True,
        "transmit_power_total_dbm": budget.tx_power_total_dbm,
        "noise_power_per_subcarrier_dbm": budget.noise_power_per_subcarrier_dbm(waveform),
        "propagation": {
            "los": True,
            "specular_reflection": bool(sionna["include_reflections"]),
            "diffuse_reflection": bool(sionna["diffuse_reflections"]),
            "refraction": bool(sionna["refraction"]),
            "diffraction": bool(sionna["diffraction"]),
            "edge_diffraction": bool(sionna["edge_diffraction"]),
        },
        "aggregate": aggregate,
        "limitation": (
            "The available Sionna scene is a ground plane plus one wall, "
            "not the Cosys-AirSim map mesh."
        ),
    }
    (output_root / "sionna_absolute_power_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rf_realism_ladder.yaml")
    )
    run(parser.parse_args().config)
