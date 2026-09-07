"""Create post-F8 diagnostic plots for the full data split and F7 trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


PRIMARY_METHODS = ("rf_only_full_3d", "rf_vision_shared_pose")
METHOD_STYLE = {
    "rf_only_full_3d": ("#d95f02", "RF-only"),
    "rf_vision_shared_pose": ("#1b73b3", "RF+Vision"),
}
SPLIT_COLORS = {
    "calibration": "#2ca02c",
    "validation": "#ff9f1c",
    "test": "#6f42c1",
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_manifests(config: dict) -> tuple[list[dict], dict[tuple[str, int], np.ndarray]]:
    root = Path(config["f0"]["output_root"])
    split_rows: list[dict] = []
    truth_by_frame: dict[tuple[str, int], np.ndarray] = {}
    sequence_ids = [f"seq{index:02d}" for index in range(12)]
    for sequence_id in sequence_ids:
        records = [
            json.loads(line)
            for line in (root / sequence_id / "manifest.jsonl").read_text().splitlines()
            if line
        ]
        if len(records) != 400:
            raise RuntimeError(f"{sequence_id} does not contain 400 frames")
        for record in records:
            target = np.asarray(
                record["evaluation_only"]["antenna_phase_centers_world_m"]["ServedVehicle"],
                dtype=float,
            )
            uav = np.asarray(record["transform_world_body"]["t_W_B"], dtype=float)
            relative = target - uav
            frame_index = int(record["frame_index"])
            truth_by_frame[(sequence_id, frame_index)] = target
            split_rows.append(
                {
                    "sequence_id": sequence_id,
                    "split": record["split"],
                    "frame_index": frame_index,
                    "timestamp_s": float(record["timestamp_s"]),
                    "target_x_m": float(target[0]),
                    "target_y_m": float(target[1]),
                    "target_z_m": float(target[2]),
                    "uav_x_m": float(uav[0]),
                    "uav_y_m": float(uav[1]),
                    "uav_z_m": float(uav[2]),
                    "horizontal_separation_m": float(np.linalg.norm(relative[:2])),
                    "vertical_separation_m": float(abs(relative[2])),
                    "range_m": float(np.linalg.norm(relative)),
                }
            )
    if len(split_rows) != 4800:
        raise RuntimeError("full split must contain exactly 4800 frames")
    return split_rows, truth_by_frame


def _load_f7(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as source:
        for raw in csv.DictReader(source):
            if raw["method"] not in PRIMARY_METHODS:
                continue
            rows.append(
                {
                    "sequence_id": raw["sequence_id"],
                    "frame_index": int(raw["frame_index"]),
                    "waveform_seed": int(raw["waveform_seed"]),
                    "method": raw["method"],
                    "estimate": np.array(
                        [raw["estimate_x_m"], raw["estimate_y_m"], raw["estimate_z_m"]],
                        dtype=float,
                    ),
                    "error_3d_m": float(raw["error_3d_m"]),
                    "association_correct": int(raw["association_correct"]),
                }
            )
    if len(rows) != 3200 * 5 * 2:
        raise RuntimeError("F7 primary trajectory records are incomplete")
    return rows


def _aggregate_f7(
    f7_rows: list[dict], truth_by_frame: dict[tuple[str, int], np.ndarray]
) -> list[dict]:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in f7_rows:
        grouped[(row["sequence_id"], row["frame_index"], row["method"])].append(row)
    output: list[dict] = []
    for sequence_id in sorted({row["sequence_id"] for row in f7_rows}):
        for frame_index in range(400):
            truth = truth_by_frame[(sequence_id, frame_index)]
            row: dict[str, float | int | str] = {
                "sequence_id": sequence_id,
                "frame_index": frame_index,
                "gt_x_m": float(truth[0]),
                "gt_y_m": float(truth[1]),
                "gt_z_m": float(truth[2]),
            }
            for method, prefix in (
                ("rf_only_full_3d", "rf"),
                ("rf_vision_shared_pose", "fused"),
            ):
                values = grouped[(sequence_id, frame_index, method)]
                estimates = np.asarray([value["estimate"] for value in values])
                errors = np.asarray([value["error_3d_m"] for value in values])
                for axis, name in enumerate("xyz"):
                    row[f"{prefix}_{name}_mean_m"] = float(np.mean(estimates[:, axis]))
                    row[f"{prefix}_{name}_q10_m"] = float(np.quantile(estimates[:, axis], 0.10))
                    row[f"{prefix}_{name}_q90_m"] = float(np.quantile(estimates[:, axis], 0.90))
                row[f"{prefix}_error_3d_median"] = float(np.median(errors))
                row[f"{prefix}_error_3d_q10_m"] = float(np.quantile(errors, 0.10))
                row[f"{prefix}_error_3d_q90_m"] = float(np.quantile(errors, 0.90))
            output.append(row)
    return output


def _training_history(source: Path, output: Path) -> list[dict]:
    with source.open(encoding="utf-8", newline="") as stream:
        rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(stream)]
    _write_csv(output, rows)
    return rows


def _plot_process_overview(
    split_rows: list[dict], training_rows: list[dict], output_root: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    epochs = np.asarray([int(row["epoch"]) for row in training_rows])
    for key, label, color in (
        ("train/box_loss", "train box", "#1f77b4"),
        ("train/cls_loss", "train cls", "#2ca02c"),
        ("val/box_loss", "val box", "#ff7f0e"),
        ("val/cls_loss", "val cls", "#d62728"),
    ):
        axes[0, 0].plot(epochs, [float(row[key]) for row in training_rows], label=label, color=color)
    axes[0, 0].set(title="LoRA detector loss history", xlabel="Epoch", ylabel="Loss")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(ncol=2, fontsize=8)

    for key, label, color in (
        ("metrics/precision(B)", "Precision", "#9467bd"),
        ("metrics/recall(B)", "Recall", "#8c564b"),
        ("metrics/mAP50(B)", "mAP50", "#17becf"),
        ("metrics/mAP50-95(B)", "mAP50-95", "#e377c2"),
    ):
        axes[0, 1].plot(epochs, [float(row[key]) for row in training_rows], label=label, color=color)
    axes[0, 1].set(title="Independent VisDrone validation metrics", xlabel="Epoch", ylabel="Metric")
    axes[0, 1].set_ylim(0.0, 0.6)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(ncol=2, fontsize=8)

    by_sequence: dict[str, list[dict]] = defaultdict(list)
    for row in split_rows:
        by_sequence[row["sequence_id"]].append(row)
    for sequence_id, rows in sorted(by_sequence.items()):
        split = str(rows[0]["split"])
        color = SPLIT_COLORS[split]
        axes[1, 0].plot(
            [row["target_x_m"] for row in rows],
            [row["target_y_m"] for row in rows],
            color=color,
            alpha=0.75,
            linewidth=1.2,
        )
        axes[1, 0].plot(
            [row["uav_x_m"] for row in rows],
            [row["uav_y_m"] for row in rows],
            color=color,
            alpha=0.65,
            linewidth=1.0,
            linestyle="--",
        )
        axes[1, 0].text(rows[0]["uav_x_m"], rows[0]["uav_y_m"], sequence_id, fontsize=7)
    axes[1, 0].set(
        title="AirSim world trajectories (solid target, dashed UAV)",
        xlabel="World x (m)",
        ylabel="World y (m)",
        aspect="equal",
    )
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(
        handles=[
            Line2D([0], [0], color=color, label=split.title())
            for split, color in SPLIT_COLORS.items()
        ],
        fontsize=8,
    )

    for sequence_id, rows in sorted(by_sequence.items()):
        split = str(rows[0]["split"])
        axes[1, 1].plot(
            [row["horizontal_separation_m"] for row in rows],
            [row["vertical_separation_m"] for row in rows],
            color=SPLIT_COLORS[split],
            alpha=0.65,
            linewidth=1.0,
        )
        middle = rows[len(rows) // 2]
        axes[1, 1].text(
            middle["horizontal_separation_m"],
            middle["vertical_separation_m"],
            sequence_id,
            fontsize=7,
        )
    axes[1, 1].set(
        title="Relative geometry coverage by complete sequence",
        xlabel="Horizontal UAV-target separation (m)",
        ylabel="Vertical separation (m)",
    )
    axes[1, 1].grid(alpha=0.25)
    figure.suptitle("FULL training and split diagnostics (post-F8, exploratory)", fontsize=15)
    figure.savefig(output_root / "full_process_overview.pdf", bbox_inches="tight")
    figure.savefig(output_root / "full_process_overview.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _trajectory_arrays(
    sequence_id: str,
    f7_rows: list[dict],
    truth_by_frame: dict[tuple[str, int], np.ndarray],
) -> tuple[np.ndarray, dict[str, dict[int, np.ndarray]], dict[str, dict[int, np.ndarray]]]:
    truth = np.asarray([truth_by_frame[(sequence_id, frame)] for frame in range(400)])
    trajectories: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    errors: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for method in PRIMARY_METHODS:
        method_rows = [
            row for row in f7_rows if row["sequence_id"] == sequence_id and row["method"] == method
        ]
        for seed in sorted({row["waveform_seed"] for row in method_rows}):
            seed_rows = sorted(
                [row for row in method_rows if row["waveform_seed"] == seed],
                key=lambda row: row["frame_index"],
            )
            trajectories[method][seed] = np.asarray([row["estimate"] for row in seed_rows])
            errors[method][seed] = np.asarray([row["error_3d_m"] for row in seed_rows])
    return truth, trajectories, errors


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def _plot_test_trajectories(
    f7_rows: list[dict], truth_by_frame: dict[tuple[str, int], np.ndarray], output_root: Path
) -> None:
    sequence_ids = [f"seq{index:02d}" for index in range(4, 12)]
    overview, overview_axes = plt.subplots(4, 2, figsize=(13, 18), constrained_layout=True)
    with PdfPages(output_root / "f7_trajectory_details.pdf") as pdf:
        for sequence_id, overview_axis in zip(sequence_ids, overview_axes.flat, strict=True):
            truth, trajectories, errors = _trajectory_arrays(sequence_id, f7_rows, truth_by_frame)
            overview_axis.plot(truth[:, 0], truth[:, 1], color="black", linewidth=2.0, label="GT")
            for method in PRIMARY_METHODS:
                color, label = METHOD_STYLE[method]
                stacked = np.stack(list(trajectories[method].values()))
                for trajectory in stacked:
                    overview_axis.plot(
                        trajectory[:, 0], trajectory[:, 1], color=color, alpha=0.09, linewidth=0.6
                    )
                mean = np.mean(stacked, axis=0)
                overview_axis.plot(mean[:, 0], mean[:, 1], color=color, linewidth=1.3, label=label)
            rf_errors = np.concatenate(list(errors["rf_only_full_3d"].values()))
            fused_errors = np.concatenate(list(errors["rf_vision_shared_pose"].values()))
            gain = (_rmse(rf_errors) - _rmse(fused_errors)) / _rmse(rf_errors)
            overview_axis.set_title(f"{sequence_id}: 3D RMSE gain {gain:.1%}")
            overview_axis.set_xlabel("World x (m)")
            overview_axis.set_ylabel("World y (m)")
            overview_axis.set_aspect("equal")
            overview_axis.grid(alpha=0.2)
            if sequence_id == "seq04":
                overview_axis.legend(fontsize=8)

            detail, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
            axes[0].plot(truth[:, 0], truth[:, 1], color="black", linewidth=2.2, label="GT")
            frames = np.arange(400)
            axes[1].plot(frames, truth[:, 2], color="black", linewidth=2.2, label="GT")
            for method in PRIMARY_METHODS:
                color, label = METHOD_STYLE[method]
                stacked = np.stack(list(trajectories[method].values()))
                error_stacked = np.stack(list(errors[method].values()))
                for trajectory in stacked:
                    axes[0].plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=0.10, linewidth=0.6)
                    axes[1].plot(frames, trajectory[:, 2], color=color, alpha=0.10, linewidth=0.6)
                mean = np.mean(stacked, axis=0)
                axes[0].plot(mean[:, 0], mean[:, 1], color=color, linewidth=1.4, label=label)
                axes[1].plot(frames, mean[:, 2], color=color, linewidth=1.2, label=label)
                median = np.median(error_stacked, axis=0)
                lower, upper = np.quantile(error_stacked, [0.10, 0.90], axis=0)
                axes[2].plot(frames, median, color=color, linewidth=1.2, label=label)
                axes[2].fill_between(frames, lower, upper, color=color, alpha=0.16)
            axes[0].set(title="XY trajectory", xlabel="World x (m)", ylabel="World y (m)")
            axes[0].set_aspect("equal")
            axes[1].set(title="Vertical trajectory", xlabel="Frame", ylabel="World z (m)")
            axes[2].set(title="3D error (median, 10–90% across seeds)", xlabel="Frame", ylabel="Error (m)")
            for axis in axes:
                axis.grid(alpha=0.22)
                axis.legend(fontsize=8)
            detail.suptitle(
                f"{sequence_id} — all five waveform seeds (faint) and seed mean (bold)", fontsize=14
            )
            pdf.savefig(detail, bbox_inches="tight")
            plt.close(detail)
    overview.suptitle(
        "F7 held-out trajectories: all 3200 frames × five waveform seeds\n"
        "Faint lines are individual seeds; bold lines are seed means",
        fontsize=15,
    )
    overview.savefig(output_root / "f7_trajectory_overview.pdf", bbox_inches="tight")
    overview.savefig(output_root / "f7_trajectory_overview.png", dpi=180, bbox_inches="tight")
    plt.close(overview)


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["outputs"]["root"]) / "diagnostics"
    output_root.mkdir(parents=True, exist_ok=True)
    split_rows, truth_by_frame = _load_manifests(config)
    f7_rows = _load_f7(Path(config["outputs"]["root"]) / "f7_frame_results.csv")
    aggregate_rows = _aggregate_f7(f7_rows, truth_by_frame)
    training_rows = _training_history(
        Path("runs/obb/runs/full_localization_lora/formal_r4_a8/results.csv"),
        output_root / "detector_training_history.csv",
    )
    _write_csv(output_root / "full_split_trajectory_source.csv", split_rows)
    _write_csv(output_root / "f7_trajectory_source.csv", aggregate_rows)
    _plot_process_overview(split_rows, training_rows, output_root)
    _plot_test_trajectories(f7_rows, truth_by_frame, output_root)
    summary = {
        "status": "COMPLETE",
        "classification": "post-F8 exploratory diagnostic; not formal F9",
        "air_sim_frames": len(split_rows),
        "test_frames": len(aggregate_rows),
        "raw_primary_results": len(f7_rows),
        "waveform_seeds_per_frame": 5,
        "sequence_ids": [f"seq{index:02d}" for index in range(12)],
        "test_sequence_ids": [f"seq{index:02d}" for index in range(4, 12)],
        "figures": [
            "full_process_overview.pdf",
            "f7_trajectory_overview.pdf",
            "f7_trajectory_details.pdf",
        ],
    }
    (output_root / "trajectory_visualization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/full_localization.yaml"))
    run(parser.parse_args().config)
